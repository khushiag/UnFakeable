"""Assemble the shared feature matrix from per-video blocks and define the named feature sets."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

FEATURE_SETS: Dict[str, list] = {
    "spatial": ["spatial"],  # ablation baseline: EfficientNet only
    "temporal": ["temporal"],  # temporal encoder only (diagnostic)
    "fused": ["spatial", "temporal"],  # spatial-temporal fusion
    "handcrafted": ["handcrafted"],  # LBP/HOG/spectrum baseline, no CNN
}


def build_matrix(spatial: Optional[np.ndarray], temporal: Optional[np.ndarray],
                 handcrafted: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Concatenate available blocks column-wise and return ``(X, {block: (start, end)})``."""
    parts, blocks, start = [], {}, 0
    for name, arr in (("spatial", spatial), ("temporal", temporal), ("handcrafted", handcrafted)):
        if arr is None:
            continue
        parts.append(np.asarray(arr, dtype=np.float32))
        blocks[name] = (start, start + arr.shape[1])
        start += arr.shape[1]
    if not parts:
        raise ValueError("no feature blocks supplied")
    return np.concatenate(parts, axis=1), blocks


def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    """(N,T,D) -> (N,D) spatial video descriptor."""
    return embeddings.mean(axis=1)
