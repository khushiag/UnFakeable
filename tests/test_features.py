import numpy as np
import pytest

from unfakeable.config import HandcraftedCfg, TemporalCfg
from unfakeable.features.assembler import build_matrix, mean_pool
from unfakeable.features.backbone import SpatialExtractor
from unfakeable.features.handcrafted import handcrafted_dim, handcrafted_features
from unfakeable.features.store import ArrayCache
from unfakeable.features.temporal import TemporalEncoder, fit_temporal_encoder, prepare_sequences
from unfakeable.features.transformer import BlockScalerPCA


# ------------------------------------------------------------------ transformer: fit on train only
def test_transformer_fits_only_on_train_rows():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, (80, 12))
    Xte = rng.normal(5, 3, (30, 12))  # deliberately shifted: must not influence the fit
    blocks = {"a": (0, 8), "b": (8, 12)}
    tf = BlockScalerPCA(blocks, ["a", "b"], n_components=5, random_state=0).fit(Xtr)
    assert np.allclose(tf.scaler_.mean_, Xtr.mean(0))
    assert tf.pca_.n_components_ == 5
    before = tf.pca_.components_.copy()
    _ = tf.transform(Xte)
    assert np.array_equal(before, tf.pca_.components_)  # transform never refits
    tf2 = BlockScalerPCA(blocks, ["a", "b"], n_components=5, random_state=0).fit(np.vstack([Xtr, Xte]))
    assert not np.allclose(tf.scaler_.mean_, tf2.scaler_.mean_)


def test_transformer_block_selection_and_weighting():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (50, 10))
    blocks = {"a": (0, 6), "b": (6, 10)}
    tf = BlockScalerPCA(blocks, ["b"], n_components=None).fit(X)
    assert tf.transform(X).shape == (50, 4) and set(tf.block_of_feature_) == {"b"}
    both = BlockScalerPCA(blocks, ["a", "b"], n_components=None, block_weighting="sqrt_dim").fit(X)
    Z = both.transform(X)
    # each block contributes equal total variance (= 1 after sqrt-dim weighting of unit-variance features)
    assert np.isclose(Z[:, :6].var(0).sum(), 1.0, atol=0.05) and np.isclose(Z[:, 6:].var(0).sum(), 1.0, atol=0.05)
    with pytest.raises(ValueError):
        BlockScalerPCA(blocks, ["a"], block_weighting="bogus").fit(X)


def test_transformer_clips_components_and_variance_mode():
    X = np.random.default_rng(2).normal(size=(10, 30))
    assert BlockScalerPCA(None, None, n_components=64).fit(X).n_features_out_ <= 9
    assert 0 < BlockScalerPCA(None, None, n_components=0.9).fit(X).n_features_out_ <= 9


def test_build_matrix_blocks():
    X, b = build_matrix(np.zeros((4, 5)), None, np.ones((4, 3)))
    assert X.shape == (4, 8) and b == {"spatial": (0, 5), "handcrafted": (5, 8)}
    assert mean_pool(np.arange(24, dtype=float).reshape(2, 3, 4)).shape == (2, 4)


def test_array_cache_roundtrip(tmp_path):
    c = ArrayCache(tmp_path / "c")
    assert not c.exists("k")
    c.save("k", np.arange(5))
    assert c.exists("k") and c.load("k").tolist() == [0, 1, 2, 3, 4]


# ------------------------------------------------------------------ temporal encoder
def _seqs(n=40, t=8, d=16, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 1, d))
    return (base + 0.3 * rng.normal(size=(n, t, d))).astype(np.float32)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_temporal_shapes_determinism_roundtrip(tmp_path, cell):
    cfg = TemporalCfg(cell=cell, hidden_size=12, epochs=4, batch_size=16, device="cpu")
    x = _seqs()
    e1, h1 = fit_temporal_encoder(x, cfg, seed=3)
    e2, _ = fit_temporal_encoder(x, cfg, seed=3)
    z1, z2 = e1.transform(x), e2.transform(x)
    assert z1.shape == (40, 12) and np.allclose(z1, z2, atol=1e-6)  # same seed -> same features
    e1.save(tmp_path / "t.pt")
    z3 = TemporalEncoder.load(tmp_path / "t.pt").transform(x)
    assert np.allclose(z1, z3, atol=1e-6)
    assert h1[-1] < h1[0]  # the autoencoder actually trains


def test_temporal_is_label_free_and_validates_input():
    import inspect

    assert "y" not in inspect.signature(fit_temporal_encoder).parameters  # no labels can reach the encoder
    with pytest.raises(ValueError):
        prepare_sequences(np.zeros((2, 2, 4)), "diff")
    with pytest.raises(ValueError):
        prepare_sequences(np.zeros((2, 5, 4)), "nope")
    with pytest.raises(ValueError):
        fit_temporal_encoder(_seqs(), TemporalCfg(cell="rnn", epochs=1, device="cpu"), 0)
    assert prepare_sequences(np.arange(24, dtype=float).reshape(1, 6, 4), "diff").shape == (1, 5, 4)


def test_temporal_features_reflect_dynamics():
    """Static clips vs flickering clips must map to clearly different latents (diff input removes content)."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 1, 16))
    static = np.repeat(base, 8, axis=1) + 0.01 * rng.normal(size=(60, 8, 16))
    flick = np.repeat(base, 8, axis=1) + 1.0 * rng.normal(size=(60, 8, 16))
    enc, _ = fit_temporal_encoder(np.concatenate([static[:30], flick[:30]]).astype(np.float32),
                                  TemporalCfg(hidden_size=8, epochs=40, batch_size=20, device="cpu"), seed=0)
    zs, zf = enc.transform(static[30:].astype(np.float32)), enc.transform(flick[30:].astype(np.float32))
    gap = np.linalg.norm(zs.mean(0) - zf.mean(0))
    spread = 0.5 * (zs.std(0).mean() + zf.std(0).mean())
    assert gap > 3 * spread


# ------------------------------------------------------------------ handcrafted
def test_handcrafted_dim_deterministic_and_sensitive():
    cfg = HandcraftedCfg()
    assert handcrafted_dim(cfg) == 548  # 160 LBP + 324 HOG + 64 spectrum (report Table 3)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    f1, f2 = handcrafted_features(frames, cfg), handcrafted_features(frames, cfg)
    assert f1.shape == (handcrafted_dim(cfg),) and np.array_equal(f1, f2) and np.isfinite(f1).all()
    import cv2

    blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 3) for f in frames])
    assert np.linalg.norm(handcrafted_features(blurred, cfg) - f1) > 0.1  # descriptors respond to blur


# ------------------------------------------------------------------ backbone + Grad-CAM
def test_spatial_extractor_shapes_with_random_weights():
    ext = SpatialExtractor("random", "weights", "", batch_size=3, device="cpu", input_size=224)
    out = ext.embed(np.random.default_rng(0).integers(0, 255, (5, 96, 96, 3), dtype=np.uint8))  # also resizes
    assert out.shape == (5, 1280) and out.dtype == np.float32 and np.isfinite(out).all()
    assert ext.source_used == "random"


def test_gradcam_maps_and_overlay():
    from unfakeable.evaluation.gradcam import LinearProbe, gradcam_maps, overlay

    ext = SpatialExtractor("random", "weights", "", batch_size=4, device="cpu", input_size=224)
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (6, 224, 224, 3), dtype=np.uint8)
    emb = ext.embed(frames)
    probe = LinearProbe.fit(emb, np.array([0, 1, 0, 1, 0, 1]))
    cams, probs = gradcam_maps(ext, probe, frames[:2])
    assert cams.shape == (2, 224, 224) and cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert np.allclose(probs, 1 / (1 + np.exp(-probe.logit_np(emb[:2]))), atol=1e-3)  # same read-out as numpy
    ov = overlay(frames[0], cams[0])
    assert ov.shape == frames[0].shape and ov.dtype == np.uint8
