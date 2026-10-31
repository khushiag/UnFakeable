"""Assemble the shared feature matrices from per-video blocks and define the named feature sets.

Two views of the same features are kept, with IDENTICAL column layout (``blocks``):
* frame matrix: one row per sampled frame (spatial + handcrafted of that frame, temporal of its video);
* video matrix: the mean of a video's frame rows (temporal is constant per video, so it is unchanged).
With ``train.granularity=frame`` the classifiers learn from frame rows and a video's score is the mean of its
frames' fake probabilities; with ``video`` they learn from video rows (the original design).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

FEATURE_SETS: Dict[str, list] = {
    "spatial": ["spatial"],  # ablation baseline: CNN appearance features only
    "temporal": ["temporal"],  # temporal encoder only (diagnostic)
    "fused": ["spatial", "temporal"],  # spatial-temporal fusion
    "handcrafted": ["handcrafted"],  # LBP/HOG/spectrum baseline, no CNN
    "all": ["spatial", "temporal", "handcrafted"],  # every block together
}


def build_matrix(spatial: Optional[np.ndarray], temporal: Optional[np.ndarray],
                 handcrafted: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Concatenate available 2-D blocks column-wise and return ``(X, {block: (start, end)})``."""
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


def build_frame_matrix(emb: np.ndarray, temporal: Optional[np.ndarray], hand_frames: Optional[np.ndarray]
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[int, int]]]:
    """emb (N,T,D), temporal (N,H) | None, hand_frames (N,T,K) | None -> (Xf (N*T, ...), video_index (N*T,), blocks)."""
    n, t = emb.shape[:2]
    tem = None if temporal is None else np.repeat(temporal, t, axis=0)
    hand = None if hand_frames is None else hand_frames.reshape(n * t, -1)
    Xf, blocks = build_matrix(emb.reshape(n * t, -1), tem, hand)
    return Xf, np.repeat(np.arange(n), t), blocks


def video_means(Xf: np.ndarray, video_index: np.ndarray, n_videos: int) -> np.ndarray:
    """Average frame rows per video -> (n_videos, D)."""
    sums = np.zeros((n_videos, Xf.shape[1]), dtype=np.float64)
    np.add.at(sums, video_index, Xf)
    counts = np.bincount(video_index, minlength=n_videos)[:, None]
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    """(N,T,D) -> (N,D) spatial video descriptor."""
    return embeddings.mean(axis=1)
