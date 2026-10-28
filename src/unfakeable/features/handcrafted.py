"""Handcrafted classical-ML baseline features (no CNN involved).

* LBP (uniform, grid-pooled histograms): micro-texture statistics; blending and re-synthesis tend to change the
  local texture of skin relative to untouched footage.
* HOG: gradient/edge structure; sensitive to blending seams and over-smoothed regions.
* Radial log-power spectrum: summarises the 2-D Fourier spectrum in concentric rings; up-sampling and
  generator layers are known to leave spectral signatures that this captures.

Each descriptor is computed per frame and averaged over frames.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

from unfakeable.config import HandcraftedCfg


def _lbp_hist(gray: np.ndarray, cfg: HandcraftedCfg) -> np.ndarray:
    n_bins = cfg.lbp_points + 2
    lbp = local_binary_pattern(gray, cfg.lbp_points, cfg.lbp_radius, method="uniform").astype(np.int64)
    h, w = lbp.shape
    gs, feats = cfg.lbp_grid, []
    for i in range(gs):
        for j in range(gs):
            cell = lbp[i * h // gs:(i + 1) * h // gs, j * w // gs:(j + 1) * w // gs]
            feats.append(np.bincount(cell.ravel(), minlength=n_bins)[:n_bins] / max(cell.size, 1))
    return np.concatenate(feats)


def _radial_spectrum(gray: np.ndarray, bins: int) -> np.ndarray:
    n = gray.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float64) * win))))
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - n / 2.0, xx - n / 2.0)
    idx = np.minimum((r / (n / 2.0) * bins).astype(int), bins - 1)
    sums = np.bincount(idx.ravel(), weights=spec.ravel(), minlength=bins)
    cnts = np.bincount(idx.ravel(), minlength=bins)
    return sums / np.maximum(cnts, 1)


def frame_features(rgb: np.ndarray, cfg: HandcraftedCfg) -> np.ndarray:
    gray = cv2.cvtColor(cv2.resize(rgb, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2GRAY)
    hog_v = hog(gray, orientations=9, pixels_per_cell=(cfg.hog_cell, cfg.hog_cell), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True)
    return np.concatenate([_lbp_hist(gray, cfg), hog_v, _radial_spectrum(gray, cfg.spectrum_bins)]).astype(np.float32)


def handcrafted_features(frames: np.ndarray, cfg: HandcraftedCfg) -> np.ndarray:
    """(T,H,W,3) uint8 -> (D,) float32 (mean over frames)."""
    return np.mean([frame_features(f, cfg) for f in frames], axis=0).astype(np.float32)


def handcrafted_dim(cfg: HandcraftedCfg) -> int:
    return int(frame_features(np.zeros((cfg.image_size, cfg.image_size, 3), np.uint8), cfg).shape[0])


def handcrafted_group_sizes(cfg: HandcraftedCfg) -> List[tuple]:
    """(name, size) of each descriptor, in concatenation order (used for reporting)."""
    lbp = cfg.lbp_grid ** 2 * (cfg.lbp_points + 2)
    total = handcrafted_dim(cfg)
    return [("lbp", lbp), ("hog", total - lbp - cfg.spectrum_bins), ("spectrum", cfg.spectrum_bins)]
