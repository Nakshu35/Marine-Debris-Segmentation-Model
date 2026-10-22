import os
from typing import Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    extra: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
    }
    if extra:
        state.update(extra)
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # atomic write to avoid truncated checkpoints on crash


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    map_location: str = "cpu",
) -> dict:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    return state


class EarlyStopping:
    """Stops training when a metric has not improved for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        assert mode in ("max", "min")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        improved = (metric > self.best + self.min_delta) if self.mode == "max" else (metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def state_dict(self) -> dict:
        return {"best": self.best, "counter": self.counter, "should_stop": self.should_stop}

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]
