"""Typed, YAML-backed configuration. Every path, frame count, subset size and hyperparameter lives here."""
from __future__ import annotations

import copy
import dataclasses as dc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

import yaml


class ConfigError(ValueError):
    pass


DEFAULT_CLASSIFIERS: Dict[str, Dict[str, Any]] = {
    "svm_rbf": {
        "search": "grid",
        "space": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1]},
    },
    "random_forest": {
        "search": "random",
        "n_iter": 20,
        "space": {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 8, 16, 32],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.3],
        },
    },
    "xgboost": {
        "search": "random",
        "n_iter": 30,
        "space": {
            "n_estimators": [200, 400, 800],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1.0, 5.0, 10.0],
        },
    },
}


@dc.dataclass
class PathsCfg:
    artifacts_dir: str = "artifacts"
    cache_dir: str = "cache"
    weights_dir: str = "weights"
    ffpp_root: str = "data/FaceForensics++"
    celebdf_root: str = "data/Celeb-DF-v2"
    mediapipe_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    )
    efficientnet_weights_url: str = (
        "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
    )


@dc.dataclass
class DataCfg:
    ffpp_compression: str = "c23"
    ffpp_fake_methods: List[str] = dc.field(
        default_factory=lambda: ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    )
    ffpp_max_real: Optional[int] = 100  # None = use every real video
    ffpp_max_fake: Optional[int] = None  # None = match real count when balance=True
    balance: bool = True
    test_size: float = 0.2  # fraction of source groups held out inside FF++
    celebdf_use_official_test_list: bool = True
    celebdf_max_per_class: Optional[int] = 60


@dc.dataclass
class PreprocCfg:
    n_frames: int = 16
    face_size: int = 224
    margin: float = 0.3
    detect_max_side: int = 640
    mtcnn_min_confidence: float = 0.9
    use_mediapipe_fallback: bool = True
    min_face_ratio: float = 0.5  # videos with fewer detected frames are dropped (and logged)
    device: str = "auto"  # auto = GPU if available


@dc.dataclass
class SpatialCfg:
    backbone_source: str = "auto"  # auto | torchvision | efficientnet_pytorch | clip_vitl14 | random (tests only)
    layers: str = "multi"  # multi = final features + texture statistics of 4 intermediate blocks | final
    batch_size: int = 64
    device: str = "auto"


@dc.dataclass
class TemporalCfg:
    enabled: bool = True
    cell: str = "lstm"  # lstm | gru (lighter alternative)
    input_mode: str = "diff"  # diff = frame-to-frame embedding change, raw = embeddings
    hidden_size: int = 128
    num_layers: int = 1
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: str = "auto"


@dc.dataclass
class HandcraftedCfg:
    enabled: bool = True
    image_size: int = 128
    lbp_points: int = 8
    lbp_radius: int = 1
    lbp_grid: int = 4
    hog_cell: int = 32
    spectrum_bins: int = 64


@dc.dataclass
class FusionCfg:
    n_components: Optional[float] = 64  # int = PCA components, 0<float<1 = variance ratio, null = no PCA
    block_weighting: str = "sqrt_dim"  # sqrt_dim | none


@dc.dataclass
class TrainCfg:
    granularity: str = "frame"  # frame = classify every frame, average per video | video = one row per video
    frames_per_video: int = 8  # frame rows per TRAINING video (evenly spaced); evaluation always uses all frames


@dc.dataclass
class TuneCfg:
    cv_folds: int = 5
    n_jobs: int = 1
    classifiers: Dict[str, Dict[str, Any]] = dc.field(default_factory=lambda: copy.deepcopy(DEFAULT_CLASSIFIERS))


@dc.dataclass
class EvalCfg:
    permutation_repeats: int = 5
    shap_enabled: bool = True
    shap_max_samples: int = 100
    top_n_features: int = 20


@dc.dataclass
class GradcamCfg:
    enabled: bool = True
    videos_per_class: int = 3  # per evaluation set
    frames_per_video: int = 4
    alpha: float = 0.45  # heatmap opacity in the overlays


@dc.dataclass
class RobustCfg:
    enabled: bool = True
    jpeg_qualities: List[int] = dc.field(default_factory=lambda: [90, 70, 50, 30])
    blur_sigmas: List[float] = dc.field(default_factory=lambda: [1.0, 2.0, 3.0])
    downscale_factors: List[float] = dc.field(default_factory=lambda: [0.5, 0.25])
    max_videos_per_set: int = 100
    feature_sets: List[str] = dc.field(default_factory=lambda: ["spatial", "fused"])


@dc.dataclass
class Config:
    run_name: str = "default"
    seed: int = 42
    note: str = ""  # printed at the top of the generated report
    log_level: str = "INFO"
    paths: PathsCfg = dc.field(default_factory=PathsCfg)
    data: DataCfg = dc.field(default_factory=DataCfg)
    preproc: PreprocCfg = dc.field(default_factory=PreprocCfg)
    spatial: SpatialCfg = dc.field(default_factory=SpatialCfg)
    temporal: TemporalCfg = dc.field(default_factory=TemporalCfg)
    handcrafted: HandcraftedCfg = dc.field(default_factory=HandcraftedCfg)
    fusion: FusionCfg = dc.field(default_factory=FusionCfg)
    feature_sets: List[str] = dc.field(default_factory=lambda: ["spatial", "fused", "handcrafted"])
    train: TrainCfg = dc.field(default_factory=TrainCfg)
    tuning: TuneCfg = dc.field(default_factory=TuneCfg)
    eval: EvalCfg = dc.field(default_factory=EvalCfg)
    gradcam: GradcamCfg = dc.field(default_factory=GradcamCfg)
    robustness: RobustCfg = dc.field(default_factory=RobustCfg)

    # ---- derived locations -------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.paths.artifacts_dir) / self.run_name

    def prep_fingerprint(self) -> str:
        p = self.preproc
        return short_hash(
            {"n_frames": p.n_frames, "face_size": p.face_size, "margin": p.margin,
             "conf": p.mtcnn_min_confidence, "mp": p.use_mediapipe_fallback, "side": p.detect_max_side}
        )

    def handcrafted_fingerprint(self) -> str:
        return short_hash(dc.asdict(self.handcrafted))

    def to_dict(self) -> Dict[str, Any]:
        return dc.asdict(self)


def short_hash(obj: Any, n: int = 10) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def _build(cls: Any, data: Dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in dc.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        kwargs[key] = _build(hints[key], value) if dc.is_dataclass(hints[key]) else value
    return cls(**kwargs)


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def apply_overrides(data: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML (so ``null``, ``[1,2]`` and ``0.5`` work)."""
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must look like key.sub=value, got: {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Defaults <- YAML file (deep-merged) <- command-line overrides."""
    data = dc.asdict(Config())
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            _deep_update(data, yaml.safe_load(fh) or {})
    apply_overrides(data, overrides)
    return _build(Config, data)
