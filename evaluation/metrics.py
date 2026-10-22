from typing import Dict

import numpy as np
import torch

from data.marida_dataset import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


class SegmentationMetrics:
    """Accumulates a confusion matrix batch-by-batch and derives all metrics from it."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion[:] = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        targets = targets.detach().cpu().numpy().reshape(-1)
        valid = targets != self.ignore_index
        preds, targets = preds[valid], targets[valid]
        idx = targets.astype(np.int64) * self.num_classes + preds.astype(np.int64)
        self.confusion += np.bincount(idx, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def get_confusion_matrix(self) -> np.ndarray:
        return self.confusion.copy()

    def compute(self) -> Dict:
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        eps = 1e-10
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        dice = f1.copy()
        pixel_acc = tp.sum() / (cm.sum() + eps)

        # Only average over classes present in GT or predictions
        present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0

        result = {
            "pixel_accuracy": float(pixel_acc),
            "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "mean_dice": float(dice[present].mean()) if present.any() else 0.0,
            "mean_precision": float(precision[present].mean()) if present.any() else 0.0,
            "mean_recall": float(recall[present].mean()) if present.any() else 0.0,
            "per_class_iou": {CLASS_NAMES[i]: float(iou[i]) for i in range(self.num_classes)},
            "per_class_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(self.num_classes)},
            "per_class_dice": {CLASS_NAMES[i]: float(dice[i]) for i in range(self.num_classes)},
            "per_class_precision": {CLASS_NAMES[i]: float(precision[i]) for i in range(self.num_classes)},
            "per_class_recall": {CLASS_NAMES[i]: float(recall[i]) for i in range(self.num_classes)},
        }

        md_idx = CLASS_NAMES.index("Marine Debris")
        result["marine_debris_iou"] = float(iou[md_idx])
        result["marine_debris_f1"] = float(f1[md_idx])
        result["marine_debris_precision"] = float(precision[md_idx])
        result["marine_debris_recall"] = float(recall[md_idx])
        return result
