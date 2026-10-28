"""Grad-CAM on the frozen EfficientNet-B0 backbone.

The real/fake decision is made by SVM / Random Forest / XGBoost on PCA-reduced features, which are not
differentiable with respect to pixels. Grad-CAM therefore needs a differentiable target. We use a *linear probe*:
a logistic regression fitted on the TRAINING split's mean-pooled spatial embeddings (standardised). Its logit is a
linear function of the pooled backbone features, so Grad-CAM highlights the facial regions whose backbone features
push that linear read-out towards "fake".

Interpretation (as stated in the report): these heatmaps show which regions drive the *spatial features* towards the
fake direction. They are supporting evidence, not an explanation of the SVM/RF/XGBoost decision itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearProbe:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, seed: int = 42) -> "LinearProbe":
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=seed)
        lr.fit(sc.transform(X), y)
        return cls(sc.mean_.astype(np.float32), sc.scale_.astype(np.float32), lr.coef_[0].astype(np.float32),
                   float(lr.intercept_[0]))

    def logit_np(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept

    def logit_torch(self, emb: torch.Tensor) -> torch.Tensor:
        dev = emb.device
        mean = torch.as_tensor(self.mean, device=dev)
        scale = torch.as_tensor(self.scale, device=dev)
        coef = torch.as_tensor(self.coef, device=dev)
        return ((emb - mean) / scale) @ coef + self.intercept


def gradcam_maps(extractor, probe: LinearProbe, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(T,H,W,3) uint8 crops -> ((T,H,W) heatmaps in [0,1], (T,) probe fake-probabilities)."""
    model = extractor.model
    with torch.inference_mode(False), torch.enable_grad():
        x = extractor.to_tensor(frames)
        with torch.no_grad():
            fm = model.feature_map(x)
        fm = fm.detach().clone().requires_grad_(True)
        logit = probe.logit_torch(model.pool(fm))
        logit.sum().backward()
        weights = fm.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * fm).sum(dim=1)).detach().cpu().numpy()
        probs = torch.sigmoid(logit).detach().cpu().numpy()
    h, w = frames.shape[1:3]
    out = np.zeros((len(frames), h, w), np.float32)
    for i, c in enumerate(cam):
        c = cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC)
        c = np.clip(c, 0, None)
        out[i] = c / c.max() if c.max() > 0 else c
    return out, probs


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * rgb.astype(np.float32) + alpha * heat, 0, 255).astype(np.uint8)
