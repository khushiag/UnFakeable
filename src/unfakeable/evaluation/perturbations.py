"""Compression/blur perturbations applied to aligned face crops (uint8 RGB), used for robustness tests."""
from __future__ import annotations

from typing import Callable, List, Tuple

import cv2
import numpy as np


def jpeg(frames: np.ndarray, quality: int) -> np.ndarray:
    out = []
    for f in frames:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        out.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return np.stack(out)


def gaussian_blur(frames: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([cv2.GaussianBlur(f, (0, 0), sigmaX=float(sigma)) for f in frames])


def downscale(frames: np.ndarray, factor: float) -> np.ndarray:
    """Down-sample by ``factor`` then back up to the original size (resolution loss)."""
    h, w = frames.shape[1:3]
    small = (max(8, int(round(w * factor))), max(8, int(round(h * factor))))
    return np.stack([cv2.resize(cv2.resize(f, small, interpolation=cv2.INTER_AREA), (w, h),
                                interpolation=cv2.INTER_LINEAR) for f in frames])


def build_conditions(jpeg_q: List[int], blur_s: List[float], down_f: List[float]
                     ) -> List[Tuple[str, str, float, Callable[[np.ndarray], np.ndarray]]]:
    """[(label, kind, severity, fn)]; severity increases with degradation so curves read left-to-right."""
    conds = [("clean", "clean", 0.0, lambda x: x)]
    conds += [(f"jpeg_q{q}", "jpeg", 100.0 - q, (lambda x, q=q: jpeg(x, q))) for q in jpeg_q]
    conds += [(f"blur_s{s}", "blur", float(s), (lambda x, s=s: gaussian_blur(x, s))) for s in blur_s]
    conds += [(f"down_x{f}", "downscale", 1.0 / f, (lambda x, f=f: downscale(x, f))) for f in down_f]
    return conds
