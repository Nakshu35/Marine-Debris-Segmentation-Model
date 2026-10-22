from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.marida_dataset import IGNORE_INDEX, NUM_CLASSES

_EPS = 1e-8


def compute_class_weights(class_pixel_counts: torch.Tensor, method: str = "inverse_sqrt") -> torch.Tensor:
    counts = class_pixel_counts.float().clamp(min=1.0)
    total = counts.sum()
    if method == "none":
        return torch.ones_like(counts)
    if method == "inverse":
        w = total / (NUM_CLASSES * counts)
    elif method == "inverse_sqrt":
        w = torch.sqrt(total / (NUM_CLASSES * counts))
    else:
        raise ValueError(f"Unknown class-weight method '{method}'")
    return w / w.mean()


def _per_pixel_ce(logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor], ignore_index: int):
    """Returns (ce, w, valid): unweighted per-pixel CE, per-pixel weight (0 on ignored pixels)."""
    logits = logits.float()
    valid = targets != ignore_index
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    ce = F.cross_entropy(logits, safe_targets, reduction="none")
    if weight is not None:
        w = weight.to(device=ce.device, dtype=ce.dtype)[safe_targets]
    else:
        w = torch.ones_like(ce)
    w = w * valid.to(ce.dtype)
    return ce, w, valid


class MaskedCrossEntropy(nn.Module):
    """Same result as nn.CrossEntropyLoss(weight, ignore_index) but returns 0 (not NaN)
    when no pixel is valid."""

    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce, w, _ = _per_pixel_ce(logits, targets, self.weight, self.ignore_index)
        return (ce * w).sum() / w.sum().clamp(min=_EPS)


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)  # (B,C,H,W)
        valid = targets != self.ignore_index
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        targets_1h = F.one_hot(safe_targets, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()

        probs = probs * valid
        targets_1h = targets_1h * valid

        dims = (0, 2, 3)
        intersection = (probs * targets_1h).sum(dims)
        union = probs.sum(dims) + targets_1h.sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_class.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce, w, _ = _per_pixel_ce(logits, targets, self.weight, self.ignore_index)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return (focal * w).sum() / w.sum().clamp(min=_EPS)


class CombinedLoss(nn.Module):
    """Weighted sum of a "hard" per-pixel term (CE or Focal) and Dice."""

    def __init__(
        self,
        mode: str = "ce_dice",
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_gamma: float = 2.0,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.mode = mode
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        self.ce = MaskedCrossEntropy(weight=class_weights, ignore_index=ignore_index)
        self.focal = FocalLoss(gamma=focal_gamma, weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "ce":
            return self.ce(logits, targets)
        if self.mode == "dice":
            return self.dice(logits, targets)
        if self.mode == "focal":
            return self.focal(logits, targets)
        if self.mode == "ce_dice":
            return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
        if self.mode == "focal_dice":
            return self.ce_weight * self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)
        raise ValueError(f"Unknown loss mode '{self.mode}'")


def build_loss(cfg: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    return CombinedLoss(
        mode=cfg.get("mode", "ce_dice"),
        class_weights=class_weights,
        ce_weight=cfg.get("ce_weight", 1.0),
        dice_weight=cfg.get("dice_weight", 1.0),
        focal_gamma=cfg.get("focal_gamma", 2.0),
    )