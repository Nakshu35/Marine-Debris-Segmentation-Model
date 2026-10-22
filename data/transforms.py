from typing import Tuple

import torch
import torch.nn.functional as F

# Fixed 11-band channel order used throughout the project
BAND_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
NUM_BANDS = len(BAND_ORDER)


def paired_random_flip(image: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Horizontal and vertical flip with p=0.5, applied identically to image and mask."""
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=[-1])
        if mask is not None:
            mask = torch.flip(mask, dims=[-1])
    if torch.rand(1).item() < 0.5:
        image = torch.flip(image, dims=[-2])
        if mask is not None:
            mask = torch.flip(mask, dims=[-2])
    return image, mask


def paired_random_rotate90(image: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random 90-degree rotation, applied identically to image and mask."""
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        image = torch.rot90(image, k, dims=[-2, -1])
        if mask is not None:
            mask = torch.rot90(mask, k, dims=[-2, -1])
    return image, mask


def paired_random_crop(image: torch.Tensor, mask: torch.Tensor, crop_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random crop of the same region from image (C,H,W) and mask (H,W)."""
    _, h, w = image.shape
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        if mask is not None:
            mask = F.pad(mask.unsqueeze(0).float(), (0, pad_w, 0, pad_h), mode="constant", value=255).squeeze(0).long()
        h, w = image.shape[-2], image.shape[-1]
    top = int(torch.randint(0, h - crop_size + 1, (1,)).item())
    left = int(torch.randint(0, w - crop_size + 1, (1,)).item())
    image = image[:, top:top + crop_size, left:left + crop_size]
    if mask is not None:
        mask = mask[top:top + crop_size, left:left + crop_size]
    return image, mask


def spectral_jitter(image: torch.Tensor, max_gain: float = 0.05, max_bias: float = 0.02) -> torch.Tensor:
    """Per-band multiplicative gain + additive bias. SSL only."""
    c = image.shape[0]
    gain = 1.0 + (torch.rand(c, 1, 1, device=image.device) * 2 - 1) * max_gain
    bias = (torch.rand(c, 1, 1, device=image.device) * 2 - 1) * max_bias
    return image * gain + bias


def gaussian_noise(image: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Additive Gaussian noise. SSL only."""
    return image + torch.randn_like(image) * sigma


def random_band_dropout(image: torch.Tensor, p: float = 0.05) -> torch.Tensor:
    """Zeros out entire bands with probability p. SSL only."""
    c = image.shape[0]
    keep = (torch.rand(c, device=image.device) > p).float().view(c, 1, 1)
    return image * keep


class SSLAugmentation:
    """Augmentation pipeline for SSL — produces two independent views of one patch."""

    def __init__(
        self,
        crop_size: int = 224,
        spectral_jitter_prob: float = 0.5,
        noise_prob: float = 0.3,
        band_dropout_prob: float = 0.1,
    ):
        self.crop_size = crop_size
        self.spectral_jitter_prob = spectral_jitter_prob
        self.noise_prob = noise_prob
        self.band_dropout_prob = band_dropout_prob

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image, _ = paired_random_crop(image, None, self.crop_size)
        image, _ = paired_random_flip(image, None)
        image, _ = paired_random_rotate90(image, None)
        if torch.rand(1).item() < self.spectral_jitter_prob:
            image = spectral_jitter(image)
        if torch.rand(1).item() < self.noise_prob:
            image = gaussian_noise(image)
        if torch.rand(1).item() < self.band_dropout_prob:
            image = random_band_dropout(image)
        return image.clamp(0.0, 1.5)


class SegmentationAugmentation:
    """Spatial-only augmentation applied identically to image and mask."""

    def __init__(self, crop_size: int = 256, train: bool = True):
        self.crop_size = crop_size
        self.train = train

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        image, mask = paired_random_crop(image, mask, self.crop_size)
        if self.train:
            image, mask = paired_random_flip(image, mask)
            image, mask = paired_random_rotate90(image, mask)
        return image, mask
