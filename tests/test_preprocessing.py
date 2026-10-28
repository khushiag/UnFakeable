from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from unfakeable.config import PreprocCfg
from unfakeable.preprocessing.align import alignment_matrix, align_crop
from unfakeable.preprocessing.detectors import Detection, FallbackDetector, parse_mediapipe_detections
from unfakeable.preprocessing.frames import VideoReadError, read_frames, sample_indices
from unfakeable.preprocessing.video_preprocessor import PreprocessError, VideoFacePreprocessor


def _video(path, n=30, size=64):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (size, size))
    for i in range(n):
        f = np.full((size, size, 3), (i * 8) % 255, np.uint8)
        w.write(f)
    w.release()


def test_sample_indices():
    assert sample_indices(100, 5).tolist() == [0, 25, 50, 74, 99]
    idx = sample_indices(100, 16)
    assert len(idx) == 16 and idx[0] == 0 and idx[-1] == 99 and (np.diff(idx) > 0).all()
    assert len(sample_indices(3, 8)) == 8  # fewer frames than requested -> repeats, still fixed length
    with pytest.raises(VideoReadError):
        sample_indices(0, 4)


def test_read_frames_fixed_count_and_order(tmp_path):
    p = tmp_path / "v.mp4"
    _video(p, n=30)
    frames, idx = read_frames(str(p), 8)
    assert len(frames) == 8 and frames[0].shape == (64, 64, 3)
    means = [f.mean() for f in frames]
    assert all(np.diff(means) > 0)  # frames come from increasing time points, in order
    cv2.imwrite(str(tmp_path / "i.png"), np.zeros((20, 20, 3), np.uint8))
    f1, i1 = read_frames(str(tmp_path / "i.png"), 8)
    assert len(f1) == 1 and i1.tolist() == [0]
    with pytest.raises(VideoReadError):
        read_frames(str(tmp_path / "missing.mp4"), 4)


def test_read_frames_path_with_spaces(tmp_path):
    d = tmp_path / "Sem V" / "Mini Project FML"
    d.mkdir(parents=True)
    _video(d / "clip one.mp4", n=12)
    frames, _ = read_frames(str(d / "clip one.mp4"), 4)
    assert len(frames) == 4


def test_alignment_levels_the_eyes():
    det = Detection((100, 100, 200, 220), 0.99, (120.0, 140.0), (180.0, 170.0), "t")  # right eye lower => tilted
    m = alignment_matrix(det, 224, 0.3)
    pts = np.array([det.left_eye, det.right_eye])
    out = pts @ m[:, :2].T + m[:, 2]
    assert abs(out[1, 1] - out[0, 1]) < 1e-6  # eyes on the same row after alignment
    assert abs(out[:, 1].mean() - 112) < 40  # and near the crop centre
    crop, _ = align_crop(np.zeros((300, 300, 3), np.uint8), det, 224, 0.3)
    assert crop.shape == (224, 224, 3) and crop.dtype == np.uint8


def test_alignment_without_landmarks_does_not_rotate():
    det = Detection((10, 10, 110, 110), 0.9, None, None, "t")
    m = alignment_matrix(det, 100, 0.0)
    assert np.allclose(m[:, :2], np.eye(2), atol=1e-9)


def test_parse_mediapipe_detections():
    kp = lambda x, y: SimpleNamespace(x=x, y=y)  # noqa: E731
    mk = lambda ox, oy, w, h, s: SimpleNamespace(  # noqa: E731
        bounding_box=SimpleNamespace(origin_x=ox, origin_y=oy, width=w, height=h),
        categories=[SimpleNamespace(score=s)], keypoints=[kp(0.4, 0.4), kp(0.6, 0.4), kp(0.5, 0.5)])
    d = parse_mediapipe_detections([mk(0, 0, 10, 10, 0.99), mk(20, 20, 50, 60, 0.9)], 200, 100)
    assert d.box == (20, 20, 70, 80) and d.left_eye == (80.0, 40.0) and d.right_eye == (120.0, 40.0)
    assert parse_mediapipe_detections([], 10, 10) is None


class _Stub:
    def __init__(self, results, name="a"):
        self.results, self.name, self.calls = results, name, 0

    def detect(self, frames):
        self.calls += len(frames)
        return list(self.results[:len(frames)])


class _Broken:
    def detect(self, frames):
        raise RuntimeError("optional dependency exploded")


def test_fallback_runs_secondary_only_on_misses():
    d = lambda s: Detection((0, 0, 10, 10), 1.0, None, None, s)  # noqa: E731
    primary, secondary = _Stub([d("mtcnn"), None, d("mtcnn"), None]), _Stub([d("mediapipe")] * 4, "b")
    out = FallbackDetector(primary, secondary).detect([np.zeros((20, 20, 3), np.uint8)] * 4)
    assert [o.source for o in out] == ["mtcnn", "mediapipe", "mtcnn", "mediapipe"] and secondary.calls == 2


def test_broken_fallback_never_fails_a_video():
    d = Detection((0, 0, 10, 10), 1.0, None, None, "mtcnn")
    fb = FallbackDetector(_Stub([d, None]), _Broken())
    out = fb.detect([np.zeros((20, 20, 3), np.uint8)] * 2)
    assert out[0] is d and out[1] is None and fb.secondary is None


def test_preprocessor_fills_gaps_and_rejects_faceless():
    cfg = PreprocCfg(n_frames=4, face_size=32, min_face_ratio=0.5)
    frames = [np.full((100, 100, 3), 40 * (i + 1), np.uint8) for i in range(4)]
    det = lambda: Detection((20, 20, 80, 80), 1.0, (35, 45), (65, 45), "mtcnn")  # noqa: E731
    pre = VideoFacePreprocessor(cfg, _Stub([det(), None, det(), det()]))
    seq = pre.from_frames(frames, np.arange(4))
    assert seq.frames.shape == (4, 32, 32, 3) and seq.detected.tolist() == [True, False, True, True]
    assert np.array_equal(seq.frames[1], seq.frames[0])  # tie between neighbours resolves to the earlier one
    with pytest.raises(PreprocessError):
        VideoFacePreprocessor(cfg, _Stub([None, None, None, det()])).from_frames(frames, np.arange(4))


@pytest.mark.slow
def test_mtcnn_detects_real_face():
    from skimage import data

    from unfakeable.preprocessing.detectors import MTCNNDetector

    det = MTCNNDetector("cpu", 0.9, 640).detect([data.astronaut()])[0]
    assert det is not None and det.source == "mtcnn" and det.left_eye is not None
    x1, y1, x2, y2 = det.box
    assert 150 < x1 < 220 and 40 < y1 < 90 and 240 < x2 < 300  # astronaut's face region
