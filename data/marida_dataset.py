import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from data.transforms import NUM_BANDS, SegmentationAugmentation

IGNORE_INDEX = 255
NUM_CLASSES = 11

CLASS_NAMES = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Sargassum",
    "Natural Organic Material",
    "Ship",
    "Clouds",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
]

# Original MARIDA label ids (1-15) -> remapped class ids (0-10)
ORIGINAL_TO_NEW = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 6,  # Waves -> Marine Water
    13: 6,  # Cloud Shadows -> Marine Water
    14: 6,  # Wakes -> Marine Water
    15: 6,  # Mixed Water -> Marine Water
}

# 256-entry lookup table for fast vectorized label remapping
_REMAP_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for orig, new in ORIGINAL_TO_NEW.items():
    _REMAP_LUT[orig] = new


def remap_label_array(raw_labels: np.ndarray) -> np.ndarray:
    raw = raw_labels.astype(np.int64)
    raw = np.clip(raw, 0, 255)
    return _REMAP_LUT[raw]


def _read_split_file(split_path: str) -> List[str]:
    with open(split_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


_IMAGE_EXTS = (".tif", ".tiff", ".TIF", ".TIFF")

# Known filename prefixes that MARIDA uses but split files sometimes omit
_KNOWN_PREFIXES = ("S2_",)


def _build_patch_index(patches_dir: str) -> Dict[str, Tuple[str, str]]:
    """Walks patches_dir and returns {patch_name: (image_path, label_path)}."""
    index: Dict[str, Tuple[str, str]] = {}
    for dirpath, _dirnames, filenames in os.walk(patches_dir):
        label_files = {}
        for fname in filenames:
            stem, ext = os.path.splitext(fname)
            if ext in _IMAGE_EXTS and stem.endswith("_cl"):
                label_files[stem[: -len("_cl")]] = os.path.join(dirpath, fname)

        for fname in filenames:
            stem, ext = os.path.splitext(fname)
            if ext not in _IMAGE_EXTS or stem.endswith("_cl"):
                continue
            if stem not in label_files:
                continue
            pair = (os.path.join(dirpath, fname), label_files[stem])
            index[stem] = pair
            for prefix in _KNOWN_PREFIXES:
                if stem.startswith(prefix):
                    index.setdefault(stem[len(prefix):], pair)
    return index


def _lookup_patch(patch_index: Dict[str, Tuple[str, str]], name: str) -> Optional[Tuple[str, str]]:
    if name in patch_index:
        return patch_index[name]
    for prefix in _KNOWN_PREFIXES:
        if (prefix + name) in patch_index:
            return patch_index[prefix + name]
    return None


class MARIDADataset(Dataset):
    """MARIDA supervised segmentation dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str,
        crop_size: int = 256,
        splits_subdir: str = "splits",
        patches_subdir: str = "patches",
        validate_on_init: bool = True,
    ):
        assert split in ("train", "val", "test"), split
        self.root_dir = root_dir
        self.split = split
        self.patches_dir = os.path.join(root_dir, patches_subdir)

        split_file = os.path.join(root_dir, splits_subdir, f"{split}_X.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: '{split_file}'.")
        self.patch_names = _read_split_file(split_file)

        patch_index = _build_patch_index(self.patches_dir)
        if not patch_index:
            raise FileNotFoundError(f"No image/label pairs found under '{self.patches_dir}'.")

        self.image_paths, self.label_paths = [], []
        missing = []
        for name in self.patch_names:
            found = _lookup_patch(patch_index, name)
            if found:
                self.image_paths.append(found[0])
                self.label_paths.append(found[1])
            else:
                missing.append(name)

        if missing:
            print(f"[WARNING] {len(missing)} patches not found, e.g. {missing[:3]}")
        if not self.image_paths:
            raise FileNotFoundError(f"No usable patches for split '{split}' under '{self.patches_dir}'.")

        self.aug = SegmentationAugmentation(crop_size=crop_size, train=(split == "train"))
        if validate_on_init:
            self._validate_sample(self.image_paths[0], self.label_paths[0])

    def _validate_sample(self, img_path: str, lbl_path: str) -> None:
        with rasterio.open(img_path) as src:
            if src.count != NUM_BANDS:
                raise ValueError(f"'{img_path}' has {src.count} bands, expected {NUM_BANDS}.")
            if not np.isfinite(src.read().astype(np.float32)).all():
                raise ValueError(f"Non-finite values in '{img_path}'.")
        with rasterio.open(lbl_path) as src:
            bad = [v for v in np.unique(src.read(1)) if v not in ORIGINAL_TO_NEW and v != 0]
            if bad:
                raise ValueError(f"Unexpected label ids {bad} in '{lbl_path}'.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        with rasterio.open(self.image_paths[idx]) as src:
            img = src.read().astype(np.float32) 
        img = np.nan_to_num(img, nan=0.0, posinf=1.5, neginf=0.0)
        with rasterio.open(self.label_paths[idx]) as src:
            raw_lbl = src.read(1)
        return img, remap_label_array(raw_lbl)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, lbl = self._load_pair(idx)
        img_t = torch.from_numpy(img)
        lbl_t = torch.from_numpy(lbl.astype(np.int64))
        img_t, lbl_t = self.aug(img_t, lbl_t)
        return img_t, lbl_t.long()

    def compute_class_pixel_counts(self, max_samples: Optional[int] = None) -> np.ndarray:
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        n = len(self.image_paths) if max_samples is None else min(max_samples, len(self.image_paths))
        for i in range(n):
            with rasterio.open(self.label_paths[i]) as src:
                lbl = remap_label_array(src.read(1))
            valid = lbl != IGNORE_INDEX
            counts += np.bincount(lbl[valid], minlength=NUM_CLASSES)[:NUM_CLASSES]
        return counts

    def print_statistics(self) -> None:
        counts = self.compute_class_pixel_counts()
        total = counts.sum()
        ignored = sum(
            int((rasterio.open(p).read(1) == 0).sum()) for p in self.label_paths
        )
        print(f"--- MARIDA [{self.split}] ---")
        print(f"  samples: {len(self.image_paths)}  ignored pixels: {ignored}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {i:2d} {name:<26s} {counts[i]:>10d}  ({100.0 * counts[i] / max(total, 1):5.2f}%)")