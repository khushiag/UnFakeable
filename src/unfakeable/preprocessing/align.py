"""Eye-level alignment and square cropping of a detected face in a single affine warp."""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from unfakeable.preprocessing.detectors import Detection


def alignment_matrix(det: Detection, out_size: int, margin: float) -> np.ndarray:
    """2x3 affine matrix mapping original-frame pixels to the aligned ``out_size`` crop."""
    x1, y1, x2, y2 = det.box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    angle = 0.0
    if det.left_eye is not None and det.right_eye is not None:
        (ax, ay), (bx, by) = sorted([det.left_eye, det.right_eye], key=lambda p: p[0])
        angle = math.degrees(math.atan2(by - ay, bx - ax))
    m = cv2.getRotationMatrix2D((cx, cy), angle, out_size / side)
    m[0, 2] += out_size / 2.0 - cx
    m[1, 2] += out_size / 2.0 - cy
    return m


def align_crop(rgb: np.ndarray, det: Detection, out_size: int, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    m = alignment_matrix(det, out_size, margin)
    crop = cv2.warpAffine(rgb, m, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return crop, m
