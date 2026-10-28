"""Video/image -> fixed-length sequence of aligned face crops (uint8, RGB)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from unfakeable.config import PreprocCfg
from unfakeable.preprocessing.align import align_crop
from unfakeable.preprocessing.detectors import Detection
from unfakeable.preprocessing.frames import read_frames
from unfakeable.utils.log import get_logger

log = get_logger(__name__)


class PreprocessError(RuntimeError):
    pass


@dataclass
class FaceSequence:
    frames: np.ndarray  # (T, S, S, 3) uint8 RGB, aligned crops
    frame_idx: np.ndarray  # indices of the sampled source frames
    detected: np.ndarray  # (T,) bool: True where a face was found in that frame itself
    sources: List[str]  # detector name per frame ('' where filled from a neighbour)

    @property
    def n_detected(self) -> int:
        return int(self.detected.sum())


class VideoFacePreprocessor:
    def __init__(self, cfg: PreprocCfg, detector):
        self.cfg, self.detector = cfg, detector

    def from_frames(self, frames: Sequence[np.ndarray], idx: np.ndarray) -> FaceSequence:
        cfg = self.cfg
        dets: List[Optional[Detection]] = self.detector.detect(frames)
        crops: List[Optional[np.ndarray]] = [
            align_crop(f, d, cfg.face_size, cfg.margin)[0] if d is not None else None
            for f, d in zip(frames, dets)
        ]
        detected = np.array([c is not None for c in crops])
        ratio = detected.mean() if len(detected) else 0.0
        if detected.sum() == 0 or ratio < cfg.min_face_ratio:
            raise PreprocessError(f"face found in {int(detected.sum())}/{len(frames)} frames "
                                  f"(< min_face_ratio={cfg.min_face_ratio})")
        valid_t = np.flatnonzero(detected)
        for t in range(len(crops)):  # fill misses with the temporally nearest detected crop
            if crops[t] is None:
                crops[t] = crops[int(valid_t[np.argmin(np.abs(valid_t - t))])]
        return FaceSequence(np.stack(crops).astype(np.uint8), np.asarray(idx), detected,
                            [d.source if d is not None else "" for d in dets])

    def process(self, path: str) -> FaceSequence:
        frames, idx = read_frames(path, self.cfg.n_frames)
        return self.from_frames(frames, idx)
