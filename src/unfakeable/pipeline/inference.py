"""Single-file inference (used by the CLI ``predict`` command and the dashboard backend).

Loads only artifacts written by the training stages; the same preprocessing, frozen backbone, temporal encoder,
transformer and classifier that produced the reported results are reused.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict

import cv2
import joblib
import numpy as np

from unfakeable.config import load_config
from unfakeable.features.assembler import FEATURE_SETS, mean_pool
from unfakeable.features.backbone import SpatialExtractor
from unfakeable.features.handcrafted import handcrafted_features
from unfakeable.features.temporal import TemporalEncoder
from unfakeable.models.classical import fake_scores
from unfakeable.preprocessing.detectors import build_detector
from unfakeable.preprocessing.video_preprocessor import VideoFacePreprocessor
from unfakeable.utils.io import load_json


def _png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


class UnFakeablePredictor:
    def __init__(self, run_dir: str, feature_set: str = "fused", classifier: str = "svm_rbf"):
        run = Path(run_dir)
        if not (run / "config_used.yaml").exists():
            raise FileNotFoundError(f"{run} is not a finished training run (config_used.yaml missing)")
        self.run = run
        self.cfg = load_config(str(run / "config_used.yaml"))
        self.feature_set, self.classifier = feature_set, classifier
        self.blocks = {k: tuple(v) for k, v in load_json(run / "features" / "blocks.json").items()}
        meta = load_json(run / "features" / "feature_meta.json")
        self.backbone_source = meta["backbone_source_used"]  # must match the one used for training
        self.tf = joblib.load(run / "models" / feature_set / "transformer.joblib")
        self.model = joblib.load(run / "models" / feature_set / f"{classifier}.joblib")
        pc = self.cfg.preproc
        self.pre = VideoFacePreprocessor(pc, build_detector(
            pc.device, pc.mtcnn_min_confidence, pc.detect_max_side, pc.use_mediapipe_fallback,
            Path(self.cfg.paths.weights_dir) / "blaze_face_short_range.tflite", self.cfg.paths.mediapipe_model_url))
        self.ext = SpatialExtractor(self.backbone_source, self.cfg.paths.weights_dir,
                                    self.cfg.paths.efficientnet_weights_url, self.cfg.spatial.batch_size,
                                    self.cfg.spatial.device, pc.face_size)
        enc_path = run / "features" / "temporal_encoder.pt"
        self.enc = TemporalEncoder.load(enc_path, self.cfg.temporal.device) if enc_path.exists() else None
        self.need = FEATURE_SETS[feature_set]
        probe_path = run / "models" / "gradcam_probe.joblib"
        self.probe = joblib.load(probe_path) if probe_path.exists() else None

    def predict(self, path: str, explain: bool = False, max_cam_frames: int = 4) -> Dict[str, object]:
        seq = self.pre.process(path)
        if "temporal" in self.need and len(seq.frames) < 3:
            raise ValueError(f"feature set '{self.feature_set}' needs a video (>= 3 frames); "
                             "use the spatial or handcrafted feature set for single images")
        X = np.zeros((1, max(b for _, b in self.blocks.values())), dtype=np.float32)
        emb = self.ext.embed(seq.frames)[None]
        a, b = self.blocks["spatial"]
        X[:, a:b] = mean_pool(emb)
        if "temporal" in self.need:
            a, b = self.blocks["temporal"]
            X[:, a:b] = self.enc.transform(emb)
        if "handcrafted" in self.need:
            a, b = self.blocks["handcrafted"]
            X[:, a:b] = handcrafted_features(seq.frames, self.cfg.handcrafted)[None]
        Xt = self.tf.transform(X)
        p = float(fake_scores(self.model, Xt)[0])
        pred = int(self.model.predict(Xt)[0])
        out = {"prob_fake": p, "label": "fake" if pred == 1 else "real", "pred": pred,
               "frames_with_face": seq.n_detected, "frames_total": int(len(seq.frames)),
               "feature_set": self.feature_set, "classifier": self.classifier,
               "backbone_source": self.backbone_source}
        if explain:
            out["faces"] = [_png_b64(f) for f in seq.frames[np.round(np.linspace(0, len(seq.frames) - 1,
                                                                                   min(max_cam_frames, len(seq.frames)))).astype(int)]]
            if self.probe is not None:
                from unfakeable.evaluation.gradcam import gradcam_maps, overlay

                sel = np.round(np.linspace(0, len(seq.frames) - 1, min(max_cam_frames, len(seq.frames)))).astype(int)
                cams, probs = gradcam_maps(self.ext, self.probe, seq.frames[sel])
                out["gradcam"] = [_png_b64(overlay(seq.frames[s], c, self.cfg.gradcam.alpha))
                                  for s, c in zip(sel, cams)]
                out["probe_prob_fake_per_frame"] = [float(v) for v in probs]
        return out
