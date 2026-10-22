import csv
import logging
import os
import sys
import time
from typing import Dict, List


def build_console_logger(name: str, log_file: str) -> logging.Logger:
    """Console + file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class CSVEpochLogger:
    """Appends one dict per epoch to a CSV file."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._fieldnames: List[str] = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as f:
                header = next(csv.reader(f), None)
                if header:
                    self._fieldnames = header

    def log(self, row: Dict) -> None:
        row = dict(row)
        row.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

        if not self._fieldnames:
            self._fieldnames = list(row.keys())

        new_keys = [k for k in row.keys() if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            if file_exists:
                self._rewrite_with_new_header()
                file_exists = True

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})

    def _rewrite_with_new_header(self) -> None:
        with open(self.csv_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in self._fieldnames})
