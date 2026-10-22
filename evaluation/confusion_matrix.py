import csv
import os

import numpy as np

from data.marida_dataset import CLASS_NAMES


def save_confusion_matrix_csv(cm: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GT \\ Pred"] + CLASS_NAMES)
        for i, row in enumerate(cm):
            writer.writerow([CLASS_NAMES[i]] + row.tolist())
