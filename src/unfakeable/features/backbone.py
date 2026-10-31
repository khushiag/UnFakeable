"""Frozen pretrained image backbones used ONLY as feature extractors (no weights are ever trained here).

Backbones
* EfficientNet-B0 (ImageNet): ``torchvision`` weights, or the ``efficientnet_pytorch`` GitHub copy as a fallback.
* CLIP ViT-L/14 (OpenAI weights through ``open_clip``): ``clip_vitl14``. Frozen CLIP features are a strong,
  well-documented basis for fake-image detection with a simple classifier on top (Ojha et al., CVPR 2023).

Feature layout per frame (``spatial.layers``)
* ``final``: the globally pooled last feature map (1280-d for EfficientNet, 768-d for CLIP).
* ``multi`` (EfficientNet only): ``final`` followed by the mean and standard deviation of every channel of four
  intermediate blocks. Global ImageNet features mostly describe *what* is in the picture (identity, pose, scene);
  blending seams, smoothing and up-sampling traces live in low/mid-level texture, which these statistics keep.
  The first ``final_dim`` values are always the final pooled features (Grad-CAM relies on that).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from unfakeable.preprocessing.detectors import ensure_file
from unfakeable.utils.device import resolve_device
from unfakeable.utils.io import sha256_file
from unfakeable.utils.log import get_logger

log = get_logger(__name__)

EMBED_DIM = 1280  # EfficientNet-B0 final pooled dimension
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# first 8 hex chars of the sha256 of the lukemelas EfficientNet-B0 checkpoint (encoded in its file name)
_LUKEMELAS_SHA8 = "355c32eb"
CLIP_SOURCES = ("clip_vitl14", "clip_random")


def _mean_std(fm: torch.Tensor) -> List[torch.Tensor]:
    flat = torch.flatten(fm, 2)
    return [flat.mean(-1), flat.std(-1, unbiased=False)]


class _TorchvisionB0(torch.nn.Module):
    final_dim = 1280
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    _TAPS = (2, 3, 4, 5)  # 24, 40, 80, 112 channels at 56/28/14/14 px

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

    def forward_multi(self, x):
        taps = []
        for i, layer in enumerate(self.m.features):
            x = layer(x)
            if i in self._TAPS:
                taps += _mean_std(x)
        return torch.cat([self.pool(x)] + taps, dim=1)


class _LukemelasB0(torch.nn.Module):
    final_dim = 1280
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    _TAPS = ("reduction_2", "reduction_3", "reduction_4", "reduction_5")

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

    def forward_multi(self, x):
        ep = self.m.extract_endpoints(x)
        taps = []
        for k in self._TAPS:
            taps += _mean_std(ep[k])
        return torch.cat([self.forward(x)] + taps, dim=1)


class _ClipViTL14(torch.nn.Module):
    final_dim = 768
    mean, std = _CLIP_MEAN, _CLIP_STD

    def __init__(self, pretrained: bool):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("CLIP backbone needs `pip install open_clip_torch`") from exc
        model = open_clip.create_model("ViT-L-14", pretrained="openai" if pretrained else None)
        self.visual = model.visual

    def forward(self, x):
        return self.visual(x)


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
            if src == "clip_vitl14":
                log.info("Loading CLIP ViT-L/14 (first time downloads ~1.7 GB from Hugging Face)")
                return _ClipViTL14(pretrained=True).eval(), src
            if src in ("random", "clip_random"):  # fixed seed: the same "random" weights on every load
                log.warning("backbone_source=%s: UNTRAINED weights. For unit tests only; never for experiments.", src)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(0)
                    net = _ClipViTL14(pretrained=False) if src == "clip_random" else _TorchvisionB0(pretrained=False)
                return net.eval(), src
            raise ValueError(f"unknown backbone_source {src!r}")
        except Exception as exc:
            log.warning("Backbone source %s failed: %s", src, exc)
            last = exc
    raise RuntimeError(f"No usable backbone weights ({last}).")


class SpatialExtractor:
    """(T,H,W,3) uint8 face crops -> (T,D) float32 per-frame features. Weights are frozen (never trained)."""

    def __init__(self, source: str, weights_dir: str, weights_url: str, batch_size: int = 64,
                 device: str = "auto", input_size: int = 224, layers: str = "final"):
        if layers not in ("final", "multi"):
            raise ValueError(f"spatial.layers must be 'final' or 'multi', got {layers!r}")
        self.device = torch.device(resolve_device(device))
        self.model, self.source_used = load_backbone(source, weights_dir, weights_url)
        if layers == "multi" and not hasattr(self.model, "forward_multi"):
            log.warning("spatial.layers=multi is only available for EfficientNet; using final features for %s",
                        self.source_used)
            layers = "final"
        self.layers = layers
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.final_dim = int(self.model.final_dim)
        self.supports_gradcam = hasattr(self.model, "feature_map")
        self.batch_size, self.input_size = batch_size, input_size
        self._mean = torch.tensor(self.model.mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(self.model.std, device=self.device).view(1, 3, 1, 1)
        self._amp = self.device.type == "cuda" and self.source_used in CLIP_SOURCES  # fp16 keeps ViT-L in 6 GB

    @property
    def cache_tag(self) -> str:
        """Identifies the exact feature definition (cached embeddings are keyed by it)."""
        return self.source_used if self.layers == "final" else f"{self.source_used}-multi"

    def to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear",
                              antialias=True, align_corners=False)
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def embed(self, frames: np.ndarray) -> np.ndarray:
        fn = self.model.forward_multi if self.layers == "multi" else self.model.forward
        out = []
        for i in range(0, len(frames), self.batch_size):
            x = self.to_tensor(frames[i:i + self.batch_size])
            with torch.autocast("cuda", dtype=torch.float16, enabled=self._amp):
                y = fn(x)
            out.append(y.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
