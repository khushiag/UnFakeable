"""Pretrained EfficientNet-B0 (ImageNet) embedding extractor. The network is used ONLY as a frozen feature extractor."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from unfakeable.preprocessing.detectors import ensure_file
from unfakeable.utils.device import resolve_device
from unfakeable.utils.io import sha256_file
from unfakeable.utils.log import get_logger

log = get_logger(__name__)

EMBED_DIM = 1280
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# first 8 hex chars of the sha256 of the lukemelas EfficientNet-B0 checkpoint (encoded in its file name)
_LUKEMELAS_SHA8 = "355c32eb"


class _TorchvisionB0(torch.nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        self.m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)

    def feature_map(self, x):  # (B,1280,7,7) - last conv block, used by Grad-CAM
        return self.m.features(x)

    def pool(self, fm):
        return torch.flatten(self.m.avgpool(fm), 1)

    def forward(self, x):
        return self.pool(self.feature_map(x))


class _LukemelasB0(torch.nn.Module):
    def __init__(self, weights_path: Path):
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        self.m = EfficientNet.from_name("efficientnet-b0")
        sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        res = self.m.load_state_dict(sd, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch: missing={res.missing_keys[:3]} unexpected={res.unexpected_keys[:3]}")

    def feature_map(self, x):
        return self.m.extract_features(x)

    def pool(self, fm):
        return torch.flatten(self.m._avg_pooling(fm), 1)

    def forward(self, x):
        return self.pool(self.feature_map(x))


def load_backbone(source: str, weights_dir: str, weights_url: str) -> Tuple[torch.nn.Module, str]:
    """Return ``(module, source_actually_used)``. ``auto`` tries torchvision weights first, then the GitHub mirror."""
    order = ["torchvision", "efficientnet_pytorch"] if source == "auto" else [source]
    last: Exception = RuntimeError("no backbone source attempted")
    for src in order:
        try:
            if src == "torchvision":
                return _TorchvisionB0(pretrained=True).eval(), src
            if src == "efficientnet_pytorch":
                path = Path(weights_dir) / "efficientnet-b0-355c32eb.pth"
                if not ensure_file(path, weights_url, timeout=120):
                    raise RuntimeError(f"weights unavailable at {path}")
                if sha256_file(path)[:8] != _LUKEMELAS_SHA8:
                    raise RuntimeError(f"checksum mismatch for {path}")
                return _LukemelasB0(path).eval(), src
            if src == "random":
                log.warning("backbone_source=random: UNTRAINED weights. For unit tests only; never for experiments.")
                return _TorchvisionB0(pretrained=False).eval(), src
            raise ValueError(f"unknown backbone_source {src!r}")
        except Exception as exc:
            log.warning("Backbone source %s failed: %s", src, exc)
            last = exc
    raise RuntimeError(f"No usable EfficientNet-B0 weights ({last}).")


class SpatialExtractor:
    """(T,H,W,3) uint8 face crops -> (T,1280) float32 embeddings. Weights are frozen (never trained)."""

    dim = EMBED_DIM

    def __init__(self, source: str, weights_dir: str, weights_url: str, batch_size: int = 64,
                 device: str = "auto", input_size: int = 224):
        self.device = torch.device(resolve_device(device))
        self.model, self.source_used = load_backbone(source, weights_dir, weights_url)
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.batch_size, self.input_size = batch_size, input_size
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear",
                              antialias=True, align_corners=False)
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def embed(self, frames: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(frames), self.batch_size):
            out.append(self.model(self.to_tensor(frames[i:i + self.batch_size])).float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
