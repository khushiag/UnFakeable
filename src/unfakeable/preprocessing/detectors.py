"""Face detectors: MTCNN (primary) with a MediaPipe fallback, behind one batch interface."""
from __future__ import annotations

import importlib.util
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from unfakeable.utils.log import get_logger

log = get_logger(__name__)
Point = Tuple[float, float]


@dataclass
class Detection:
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in original-frame pixels
    score: float
    left_eye: Optional[Point]
    right_eye: Optional[Point]
    source: str


def _shrink(frame: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s < 1.0:
        frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return frame, s


class MTCNNDetector:
    name = "mtcnn"

    def __init__(self, device: str = "auto", min_confidence: float = 0.9, max_side: int = 640):
        self.device, self.min_confidence, self.max_side = device, min_confidence, max_side
        self._mtcnn = None

    def _lazy(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN  # weights ship inside the facenet-pytorch package

            from unfakeable.utils.device import resolve_device

            self._mtcnn = MTCNN(keep_all=True, device=resolve_device(self.device), post_process=False)
        return self._mtcnn

    def detect(self, frames: Sequence[np.ndarray]) -> List[Optional[Detection]]:
        mtcnn, out = self._lazy(), []
        for frame in frames:
            small, s = _shrink(frame, self.max_side)
            boxes, probs, lms = mtcnn.detect(small, landmarks=True)
            if boxes is None:
                out.append(None)
                continue
            boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
            probs = np.asarray(probs, dtype=float).reshape(-1)
            lms = np.asarray(lms, dtype=float).reshape(len(boxes), 5, 2)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            score = np.where(probs >= self.min_confidence, area * probs, -1.0)
            i = int(np.argmax(score))
            if score[i] < 0:
                out.append(None)
                continue
            b, lm = boxes[i] / s, lms[i] / s
            out.append(Detection(tuple(b), float(probs[i]), tuple(lm[0]), tuple(lm[1]), self.name))
        return out


def parse_mediapipe_detections(dets, width: int, height: int, source: str = "mediapipe") -> Optional[Detection]:
    """Pick the best MediaPipe detection (score x area). Coordinates are in the pixels of the image passed in.

    MediaPipe BlazeFace keypoints 0 and 1 are the two eyes (subject's right/left); the aligner sorts by x so the
    order does not matter.
    """
    best, best_val = None, -1.0
    for d in dets:
        bb = d.bounding_box
        score = float(d.categories[0].score) if d.categories else 0.0
        val = score * bb.width * bb.height
        if val > best_val:
            best, best_val = d, val
    if best is None:
        return None
    bb = best.bounding_box
    kps = best.keypoints or []
    eyes = [(kp.x * width, kp.y * height) for kp in kps[:2]] if len(kps) >= 2 else [None, None]
    score = float(best.categories[0].score) if best.categories else 0.0
    return Detection((bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height),
                     score, eyes[0], eyes[1], source)


class MediaPipeDetector:
    name = "mediapipe"

    def __init__(self, model_path: str, min_confidence: float = 0.5, max_side: int = 640):
        self.model_path, self.min_confidence, self.max_side = model_path, min_confidence, max_side
        self._det = None

    def _lazy(self):
        if self._det is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            opts = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
                min_detection_confidence=self.min_confidence)
            self._det = vision.FaceDetector.create_from_options(opts)
        return self._det

    def detect(self, frames: Sequence[np.ndarray]) -> List[Optional[Detection]]:
        import mediapipe as mp

        det, out = self._lazy(), []
        for frame in frames:
            small, s = _shrink(frame, self.max_side)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
            parsed = parse_mediapipe_detections(det.detect(img).detections, small.shape[1], small.shape[0])
            if parsed is None:
                out.append(None)
                continue
            b = tuple(v / s for v in parsed.box)
            scale = lambda p: None if p is None else (p[0] / s, p[1] / s)  # noqa: E731
            out.append(Detection(b, parsed.score, scale(parsed.left_eye), scale(parsed.right_eye), self.name))
        return out


class FallbackDetector:
    """Run ``primary`` on every frame, then ``secondary`` only on frames the primary missed.

    A failure inside the secondary detector is logged once and the secondary is switched off, so a broken
    optional dependency can never make an otherwise good video fail.
    """

    def __init__(self, primary, secondary=None):
        self.primary, self.secondary = primary, secondary

    def detect(self, frames: Sequence[np.ndarray]) -> List[Optional[Detection]]:
        res = list(self.primary.detect(frames))
        missing = [i for i, r in enumerate(res) if r is None]
        if missing and self.secondary is not None:
            try:
                for i, r in zip(missing, self.secondary.detect([frames[i] for i in missing])):
                    res[i] = r
            except Exception as exc:  # noqa: BLE001
                log.warning("Fallback detector disabled after error: %s", exc)
                self.secondary = None
        return res


def ensure_file(path: Path, url: str, timeout: int = 30) -> bool:
    """Download ``url`` to ``path`` if absent. Returns False (and logs) when the download is impossible."""
    if path.exists() and path.stat().st_size > 0:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))  # noqa: S310 (fixed, config-provided URL)
        return path.stat().st_size > 0
    except Exception as exc:  # network blocked, 404, ...
        log.warning("Could not download %s (%s)", url, exc)
        if path.exists():
            path.unlink()
        return False


def build_detector(device: str, min_confidence: float, max_side: int, use_mediapipe: bool,
                   mediapipe_model: Path, mediapipe_url: str) -> FallbackDetector:
    secondary = None
    if use_mediapipe:
        if importlib.util.find_spec("mediapipe") is None:
            log.info("MediaPipe not installed - running MTCNN only (optional fallback, see README).")
        elif ensure_file(mediapipe_model, mediapipe_url):
            secondary = MediaPipeDetector(str(mediapipe_model), max_side=max_side)
        else:
            log.warning("MediaPipe fallback DISABLED: model file unavailable at %s. Place "
                        "blaze_face_short_range.tflite there to enable it.", mediapipe_model)
    return FallbackDetector(MTCNNDetector(device, min_confidence, max_side), secondary)
