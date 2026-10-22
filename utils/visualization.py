import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# B4 (red), B3 (green), B2 (blue) indices in the 11-band stack
_RGB_BAND_IDX = {"B4": 3, "B3": 2, "B2": 1}

# Fixed 11-class color palette for reproducible figures
CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [255, 165, 0],
        [255, 255, 0],
        [0, 255, 0],
        [128, 128, 128],
        [0, 0, 255],
        [139, 69, 19],
        [255, 0, 255],
        [0, 255, 255],
        [173, 216, 230],
    ],
    dtype=np.uint8,
)


def s2_to_rgb(image: np.ndarray) -> np.ndarray:
    """Converts (11, H, W) reflectance array to (H, W, 3) uint8 RGB using a 2-98 percentile stretch."""
    rgb = np.stack([image[_RGB_BAND_IDX[b]] for b in ("B4", "B3", "B2")], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], (2, 98))
        if hi <= lo:
            hi = lo + 1e-6
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def mask_to_color(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Converts (H, W) class id array to (H, W, 3) uint8 color image."""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ignore_index
    color[valid] = CLASS_COLORS[mask[valid] % len(CLASS_COLORS)]
    return color


def save_sanity_figure(image_rgb: np.ndarray, title: str, out_path: str, mask: Optional[np.ndarray] = None) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 2 if mask is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"{title} (RGB)")
    axes[0].axis("off")
    if mask is not None:
        axes[1].imshow(mask_to_color(mask))
        axes[1].set_title(f"{title} (labels)")
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_prediction_overlay(
    image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray,
    out_path: str, ignore_index: int = 255,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rgb = s2_to_rgb(image)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")
    axes[1].imshow(mask_to_color(gt_mask, ignore_index)); axes[1].set_title("Ground truth"); axes[1].axis("off")
    axes[2].imshow(mask_to_color(pred_mask, ignore_index)); axes[2].set_title("Prediction"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_confusion_matrix_figure(cm: np.ndarray, class_names, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title("Normalized confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
