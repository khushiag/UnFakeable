"""Device selection. ``auto`` = the NVIDIA GPU when PyTorch can see one, otherwise the CPU."""
from __future__ import annotations

from unfakeable.utils.log import get_logger

log = get_logger(__name__)
_REPORTED = set()


def resolve_device(name: str) -> str:
    """Return a concrete torch device string ('cuda', 'cuda:0', 'cpu') for a config value."""
    import torch

    name = (name or "auto").lower()
    if name == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    elif name.startswith("cuda") and not torch.cuda.is_available():
        log.warning("Config asks for %s but PyTorch cannot see a CUDA GPU; falling back to CPU. "
                    "Install the CUDA build of PyTorch (see README, step 2).", name)
        dev = "cpu"
    else:
        dev = name
    if dev not in _REPORTED:
        _REPORTED.add(dev)
        if dev.startswith("cuda"):
            log.info("Using GPU: %s", torch.cuda.get_device_name(0))
        else:
            log.info("Using CPU for deep-learning feature extraction")
    return dev
