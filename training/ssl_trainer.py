"""
training/ssl_trainer.py

Runs the masked-reconstruction SSL / domain-adaptation stage on the
Indian coastal dataset ONLY (never touches MARIDA). A small held-out
fraction of the unlabeled data is used purely to pick the best
checkpoint / drive early stopping -- there are no labels involved at
any point in this stage.
"""

import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from models.ssl_model import MaskedAutoencoderSSL
from training.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from utils.logger import CSVEpochLogger, build_console_logger
from utils.system import get_gpu_memory_mb


class SSLTrainer:
    def __init__(
        self,
        model: MaskedAutoencoderSSL,
        train_dataset,
        cfg: dict,
        device: torch.device,
        run_dir: str,
        val_fraction: float = 0.05,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.run_dir = run_dir

        n_val = max(1, int(len(train_dataset) * val_fraction))
        n_train = len(train_dataset) - n_val
        generator = torch.Generator().manual_seed(cfg.get("seed", 42))
        self.train_subset, self.val_subset = random_split(
            train_dataset, [n_train, n_val], generator=generator
        )

        self.train_loader = DataLoader(
            self.train_subset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            drop_last=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_subset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.05)
        )
        self.epochs = cfg["epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.epochs), eta_min=cfg.get("min_lr", 1e-6)
        )
        self.use_amp = cfg.get("amp", True) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip = cfg.get("grad_clip", 1.0)

        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.csv_logger = CSVEpochLogger(os.path.join(run_dir, "logs", "ssl_train.csv"))
        self.logger = build_console_logger("ssl_trainer", os.path.join(run_dir, "logs", "ssl_console.log"))

        self.early_stopping = EarlyStopping(
            patience=cfg.get("early_stopping_patience", 15), mode="min"
        )
        self.start_epoch = 0
        self.best_val_loss = float("inf")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss, n_batches = 0.0, 0
        for view_a, view_b in loader:
            # Both augmented views are valid MAE inputs; we reconstruct
            # both per batch to make full use of every SSL sample.
            imgs = torch.cat([view_a, view_b], dim=0).to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, _, _ = self.model(imgs)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def resume(self, ckpt_path: str) -> None:
        state = load_checkpoint(ckpt_path, self.model, self.optimizer, self.scheduler, self.scaler)
        self.start_epoch = state["epoch"] + 1
        self.best_val_loss = state["best_metric"]
        if "early_stopping" in state:
            self.early_stopping.load_state_dict(state["early_stopping"])
        self.logger.info(f"Resumed SSL training from epoch {self.start_epoch} "
                          f"(best_val_loss={self.best_val_loss:.5f})")

    def fit(self) -> str:
        """Returns the path to the best SSL checkpoint."""
        best_path = os.path.join(self.ckpt_dir, "ssl_best.pth")
        last_path = os.path.join(self.ckpt_dir, "ssl_last.pth")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()
            epoch_time = time.time() - t0

            improved = self.early_stopping.step(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch_time_sec": round(epoch_time, 2),
                "gpu_memory_mb": round(get_gpu_memory_mb(), 1),
            }
            self.csv_logger.log(row)
            self.logger.info(
                f"[SSL] epoch {epoch:03d}/{self.epochs} "
                f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"lr={row['learning_rate']:.2e} time={epoch_time:.1f}s"
            )

            extra = {"early_stopping": self.early_stopping.state_dict()}
            save_checkpoint(last_path, self.model, self.optimizer, self.scheduler, self.scaler,
                             epoch, self.best_val_loss, extra)
            if improved:
                save_checkpoint(best_path, self.model, self.optimizer, self.scheduler, self.scaler,
                                 epoch, self.best_val_loss, extra)

            if self.early_stopping.should_stop:
                self.logger.info(f"[SSL] Early stopping triggered at epoch {epoch}.")
                break

        if not os.path.exists(best_path):
            save_checkpoint(best_path, self.model, self.optimizer, self.scheduler, self.scaler,
                             self.start_epoch, self.best_val_loss)
        return best_path
