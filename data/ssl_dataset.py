import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from data.transforms import NUM_BANDS, SSLAugmentation


class IndianCoastalSSLDataset(Dataset):
    """Unlabeled Sentinel-2 patches for SSL pretraining. Returns two augmented views per sample."""

    def __init__(
        self,
        root_dir: str,
        crop_size: int = 224,
        file_list: Optional[List[str]] = None,
        validate_on_init: bool = True,
    ):
        self.root_dir = root_dir
        self.files = list(file_list) if file_list is not None else sorted(
            glob.glob(os.path.join(root_dir, "*.tif")) +
            glob.glob(os.path.join(root_dir, "*.tiff"))
        )
        if not self.files:
            raise FileNotFoundError(f"No .tif/.tiff files found under '{root_dir}'.")

        self.aug = SSLAugmentation(crop_size=crop_size)
        if validate_on_init:
            self._validate_sample(self.files[0])

    def _validate_sample(self, path: str) -> None:
        with rasterio.open(path) as src:
            if src.count != NUM_BANDS:
                raise ValueError(f"'{path}' has {src.count} bands, expected {NUM_BANDS}.")
            arr = src.read()
            if not np.issubdtype(arr.dtype, np.unsignedinteger) and arr.dtype != np.uint16:
                print(f"[WARNING] '{path}' dtype={arr.dtype}, expected uint16.")
            if not np.isfinite(arr.astype(np.float32)).all():
                raise ValueError(f"Non-finite values in '{path}'.")

    def __len__(self) -> int:
        return len(self.files)

    def _load_raw(self, path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        arr = arr / 10000.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.5, neginf=0.0)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self._load_raw(self.files[idx])
        return self.aug(raw.clone()), self.aug(raw.clone())

    def get_stats(self, n_samples: int = 50) -> dict:
        """Per-band mean/std over a random subsample."""
        n_samples = min(n_samples, len(self.files))
        idxs = np.random.choice(len(self.files), n_samples, replace=False)
        means, stds = [], []
        for i in idxs:
            arr = self._load_raw(self.files[i]).numpy()
            means.append(arr.mean(axis=(1, 2)))
            stds.append(arr.std(axis=(1, 2)))
        return {
            "num_files": len(self.files),
            "per_band_mean": np.mean(means, axis=0).tolist(),
            "per_band_std": np.mean(stds, axis=0).tolist(),
        }
