"""
training/segmentation_trainer.py

Trains the SegmentationModel on MARIDA train split, validates on the
official MARIDA val split every epoch, tracks mean IoU for best-
checkpoint selection / early stopping, and NEVER looks at the test
split (that is handled separately by main.py's `run_test` step, using
the best checkpoint this trainer produced).
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import SegmentationMetrics
from training.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from utils.logger import CSVEpochLogger, build_console_logger
from utils.system import get_gpu_memory_mb


class SegmentationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        loss_fn: torch.nn.Module,
        cfg: dict,
        device: torch.device,
        run_dir: str,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.run_dir = run_dir
        self.loss_fn = loss_fn.to(device)

        self.train_loader = DataLoader(
            train_dataset, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg.get("num_workers", 4), drop_last=True, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=cfg.get("val_batch_size", cfg["batch_size"]), shuffle=False,
            num_workers=cfg.get("num_workers", 4), pin_memory=True,
        )

        backbone_lr_mult = cfg.get("encoder_lr_mult", 0.1)
        encoder_params = list(self.model.encoder.parameters()) if hasattr(self.model, "encoder") else []
        encoder_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.model.parameters() if id(p) not in encoder_param_ids]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": cfg["lr"] * backbone_lr_mult},
                {"params": other_params, "lr": cfg["lr"]},
            ],
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        self.epochs = cfg["epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.epochs), eta_min=cfg.get("min_lr", 1e-6)
        )
        self.use_amp = cfg.get("amp", True) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip = cfg.get("grad_clip", 1.0)

        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.train_csv = CSVEpochLogger(os.path.join(run_dir, "logs", "segmentation_train.csv"))
        self.val_csv = CSVEpochLogger(os.path.join(run_dir, "logs", "segmentation_val.csv"))
        self.logger = build_console_logger("seg_trainer", os.path.join(run_dir, "logs", "segmentation_console.log"))

        self.early_stopping = EarlyStopping(
            patience=cfg.get("early_stopping_patience", 15), mode="max"
        )
        self.start_epoch = 0
        self.best_val_miou = -1.0

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for images, masks in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple:
        self.model.eval()
        metrics = SegmentationMetrics()
        total_loss, n_batches = 0.0, 0
        for images, masks in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
            metrics.update(logits, masks)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1), metrics.compute()

    def resume(self, ckpt_path: str) -> None:
        state = load_checkpoint(ckpt_path, self.model, self.optimizer, self.scheduler, self.scaler)
        self.start_epoch = state["epoch"] + 1
        self.best_val_miou = state["best_metric"]
        if "early_stopping" in state:
            self.early_stopping.load_state_dict(state["early_stopping"])
        self.logger.info(f"Resumed segmentation training from epoch {self.start_epoch} "
                          f"(best_val_miou={self.best_val_miou:.4f})")

    def fit(self) -> str:
        """Returns path to the best segmentation checkpoint (by val mean IoU)."""
        best_path = os.path.join(self.ckpt_dir, "segmentation_best.pth")
        last_path = os.path.join(self.ckpt_dir, "segmentation_last.pth")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_loss, val_metrics = self._validate()
            self.scheduler.step()
            epoch_time = time.time() - t0

            improved = self.early_stopping.step(val_metrics["mean_iou"])
            if val_metrics["mean_iou"] > self.best_val_miou:
                self.best_val_miou = val_metrics["mean_iou"]

            lr = self.optimizer.param_groups[-1]["lr"]
            self.train_csv.log({
                "epoch": epoch, "train_loss": train_loss, "learning_rate": lr,
                "epoch_time_sec": round(epoch_time, 2), "gpu_memory_mb": round(get_gpu_memory_mb(), 1),
            })
            self.val_csv.log({
                "epoch": epoch,
                "validation_loss": val_loss,
                "pixel_accuracy": val_metrics["pixel_accuracy"],
                "mean_iou": val_metrics["mean_iou"],
                "macro_f1": val_metrics["macro_f1"],
                "mean_dice": val_metrics["mean_dice"],
                "marine_debris_iou": val_metrics["marine_debris_iou"],
                "marine_debris_f1": val_metrics["marine_debris_f1"],
                "marine_debris_precision": val_metrics["marine_debris_precision"],
                "marine_debris_recall": val_metrics["marine_debris_recall"],
                "learning_rate": lr,
                "epoch_time_sec": round(epoch_time, 2),
                "gpu_memory_mb": round(get_gpu_memory_mb(), 1),
            })

            self.logger.info(
                f"[SEG] epoch {epoch:03d}/{self.epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} mIoU={val_metrics['mean_iou']:.4f} "
                f"mDice={val_metrics['mean_dice']:.4f} MD_IoU={val_metrics['marine_debris_iou']:.4f} "
                f"time={epoch_time:.1f}s"
            )

            extra = {"early_stopping": self.early_stopping.state_dict()}
            save_checkpoint(last_path, self.model, self.optimizer, self.scheduler, self.scaler,
                             epoch, self.best_val_miou, extra)
            if improved:
                save_checkpoint(best_path, self.model, self.optimizer, self.scheduler, self.scaler,
                                 epoch, self.best_val_miou, extra)

            if self.early_stopping.should_stop:
                self.logger.info(f"[SEG] Early stopping triggered at epoch {epoch}.")
                break

        if not os.path.exists(best_path):
            save_checkpoint(best_path, self.model, self.optimizer, self.scheduler, self.scaler,
                             self.start_epoch, self.best_val_miou)
        return best_path
