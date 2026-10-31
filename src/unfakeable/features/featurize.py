"""Turn aligned face crops into feature rows with the SAME column layout as training, and score videos.

Used by robustness testing, single-file inference and the dashboard, so all three treat a video exactly the way
the evaluation did.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from unfakeable.features.handcrafted import handcrafted_frame_features
from unfakeable.models.classical import fake_scores


def featurize_crops(crops: Sequence[np.ndarray], ext, enc, hand_cfg, blocks: Dict[str, Tuple[int, int]],
                    need_temporal: bool, need_hand: bool) -> Tuple[np.ndarray, np.ndarray]:
    """List of (T,H,W,3) crop stacks -> (frame matrix (sum T, D), video index). Unused blocks stay zero."""
    width = max(b for _, b in blocks.values())
    rows: List[np.ndarray] = []
    vidx: List[np.ndarray] = []
    for i, c in enumerate(crops):
        emb = ext.embed(c)  # (T, D_spatial)
        X = np.zeros((len(c), width), dtype=np.float32)
        a, b = blocks["spatial"]
        if emb.shape[1] != b - a:
            raise RuntimeError(f"backbone gives {emb.shape[1]} features per frame but the model was trained on "
                               f"{b - a}; the run was made with a different spatial.backbone_source/layers")
        X[:, a:b] = emb
        if need_temporal and enc is not None and "temporal" in blocks:
            if len(c) < 3:
                raise ValueError("the temporal/fused feature sets need a video (>= 3 frames); "
                                 "use the spatial or handcrafted feature set for single images")
            a, b = blocks["temporal"]
            X[:, a:b] = enc.transform(emb[None])[0]
        if need_hand and "handcrafted" in blocks:
            a, b = blocks["handcrafted"]
            X[:, a:b] = handcrafted_frame_features(c, hand_cfg)
        rows.append(X)
        vidx.append(np.full(len(c), i))
    return np.concatenate(rows), np.concatenate(vidx)


def video_level(Xf: np.ndarray, vidx: np.ndarray, n: int) -> np.ndarray:
    from unfakeable.features.assembler import video_means

    return video_means(Xf, vidx, n)


def score_videos(model, transformer, Xf: np.ndarray, vidx: np.ndarray, n: int, granularity: str
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Fake probability per video and per frame.

    frame granularity: every frame is classified, the video score is the mean frame probability.
    video granularity: frames are averaged into one feature row first, then classified once.
    """
    if granularity == "frame":
        pf = fake_scores(model, transformer.transform(Xf))
        sums = np.bincount(vidx, weights=pf, minlength=n)
        counts = np.bincount(vidx, minlength=n)
        return sums / np.maximum(counts, 1), pf
    if granularity == "video":
        pv = fake_scores(model, transformer.transform(video_level(Xf, vidx, n)))
        return pv, pv[vidx]
    raise ValueError(f"train.granularity must be 'frame' or 'video', got {granularity!r}")
