"""Tiny on-disk array cache so expensive DL / detector work is never repeated."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from unfakeable.utils.io import ensure_dir


class ArrayCache:
    def __init__(self, root):
        self.root = ensure_dir(root)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npy"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def load(self, key: str) -> np.ndarray:
        return np.load(self.path(key))

    def save(self, key: str, arr: np.ndarray) -> None:
        tmp = self.root / f"{key}.tmp.npy"
        np.save(tmp, arr)
        tmp.replace(self.path(key))  # atomic: an interrupted run never leaves a half-written entry
