"""Fixed-count, uniformly spaced frame sampling from videos (or a single image treated as one frame)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VideoReadError(RuntimeError):
    pass


def sample_indices(n_total: int, k: int) -> np.ndarray:
    """``k`` uniformly spaced indices over ``[0, n_total-1]`` (repeats allowed when ``n_total < k``)."""
    if n_total <= 0:
        raise VideoReadError("video has no frames")
    return np.round(np.linspace(0, n_total - 1, k)).astype(int)


def _collect(path: str, idx: np.ndarray) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"cannot open {path}")
    wanted, out, i, last = set(idx.tolist()), {}, 0, int(idx.max())
    while i <= last and cap.grab():  # grab() decodes without the colour conversion -> cheap skip
        if i in wanted:
            ok, bgr = cap.retrieve()
            if ok:
                out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def _count_by_decoding(path: str) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def _imread(path: Path):
    # cv2.imread fails on some Windows paths with non-ASCII characters; imdecode does not
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def read_frames(path: str, k: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return ``k`` RGB frames and their indices. Images yield a single frame."""
    p = Path(path)
    if not p.exists():
        raise VideoReadError(f"file not found: {path}")
    if p.suffix.lower() in IMAGE_EXTS:
        bgr = _imread(p)
        if bgr is None:
            raise VideoReadError(f"cannot read image {path}")
        return [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)], np.array([0])
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise VideoReadError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    for attempt in (0, 1):
        if attempt == 1 or n <= 0:  # container metadata missing/wrong -> count by decoding
            n = _count_by_decoding(str(p))
        idx = sample_indices(n, k)
        got = _collect(str(p), idx)
        if all(int(i) in got for i in idx):
            return [got[int(i)] for i in idx], idx
    raise VideoReadError(f"could not decode requested frames from {path}")
