import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_file(path, n_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(n_bytes), b""):
            h.update(chunk)
    return h.hexdigest()
