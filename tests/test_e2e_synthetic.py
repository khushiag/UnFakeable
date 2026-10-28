"""End-to-end plumbing test on SYNTHETIC clips (not deepfakes). Uses UNTRAINED backbone weights for speed, so it checks
that every stage runs and artifacts are coherent, never that detection works."""
import pandas as pd
import pytest

from unfakeable.config import load_config
from unfakeable.data.synthetic import make_synthetic_celebdf, make_synthetic_ffpp
from unfakeable.pipeline import stages as S
from unfakeable.pipeline.inference import UnFakeablePredictor
from unfakeable.utils.io import load_json

pytestmark = pytest.mark.slow


def test_full_pipeline_runs_and_is_leak_free(tmp_path):
    make_synthetic_ffpp(str(tmp_path / "ffpp"), n_pairs=6, seed=0, n_frames=10, methods=("Deepfakes",))
    make_synthetic_celebdf(str(tmp_path / "celeb"), n_real=5, n_fake=5, seed=1, n_frames=10)
    cfg = load_config(None, [
        f"paths.ffpp_root={tmp_path / 'ffpp'}", f"paths.celebdf_root={tmp_path / 'celeb'}",
        f"paths.artifacts_dir={tmp_path / 'art'}", f"paths.cache_dir={tmp_path / 'cache'}",
        "paths.weights_dir=weights", "run_name=e2e", "data.ffpp_fake_methods=[Deepfakes]", "data.ffpp_max_real=12",
        "data.test_size=0.25", "data.celebdf_use_official_test_list=false", "data.celebdf_max_per_class=5",
        "preproc.n_frames=6", "preproc.device=cpu", "spatial.device=cpu", "temporal.device=cpu",
        "spatial.backbone_source=random", "temporal.hidden_size=8", "temporal.epochs=3",
        "fusion.n_components=6", "feature_sets=[spatial,fused,handcrafted]", "tuning.cv_folds=3",
        "tuning.classifiers.svm_rbf.space={C: [1], gamma: [scale]}",
        "tuning.classifiers.random_forest={search: random, n_iter: 1, space: {n_estimators: [30]}}",
        "tuning.classifiers.xgboost={search: random, n_iter: 1, space: {n_estimators: [20], max_depth: [2]}}",
        "eval.shap_max_samples=10", "eval.permutation_repeats=2", "robustness.max_videos_per_set=8",
        "robustness.jpeg_qualities=[50]", "robustness.blur_sigmas=[]", "robustness.downscale_factors=[]",
        "robustness.feature_sets=[spatial,fused]", "gradcam.videos_per_class=1", "gradcam.frames_per_video=2"])
    allm = S.stage_manifest(cfg)
    # leakage guard: every FF++ source group sits entirely in train or test
    ff = allm[allm.dataset == "ffpp"]
    assert ff.groupby("group").split.nunique().max() == 1 and set(ff.split) == {"train", "test"}
    assert set(allm[allm.dataset == "celebdf"].split) == {"external"}

    ok = S.stage_preprocess(cfg)
    assert len(ok) >= 0.9 * len(allm)
    S.stage_features(cfg)
    S.stage_train(cfg)

    met = pd.read_csv(cfg.run_dir / "metrics" / "metrics_summary.csv")
    assert set(met.feature_set) == {"spatial", "fused", "handcrafted"}
    assert set(met.classifier) == {"svm_rbf", "random_forest", "xgboost"} and set(met.eval_set) == {"internal_test", "external"}
    for fs, g in met.groupby("feature_set"):  # IDENTICAL feature matrix for the 3 classifiers
        assert g.train_matrix_sha256.nunique() == 1
    assert met.roc_auc.between(0, 1).all()
    # no classifier ever saw the external set: external rows exist only as predictions
    pred = pd.read_csv(cfg.run_dir / "predictions" / "fused__svm_rbf__external.csv")
    assert set(pred.dataset) == {"celebdf"} and pred.score_fake.between(0, 1).all()

    S.stage_explain(cfg)
    assert (cfg.run_dir / "explain" / "block_attribution.csv").exists()
    gc = S.stage_gradcam(cfg)
    assert set(gc.eval_set) == {"internal_test", "external"}
    assert (cfg.run_dir / "plots" / "gradcam__external.png").exists()
    rb = S.stage_robustness(cfg)
    assert {"clean", "jpeg_q50"} <= set(rb.condition)
    clean = rb[(rb.condition == "clean") & (rb.eval_set == "internal_test")]
    assert len(clean) == 2 * 3  # 2 feature sets x 3 classifiers
    report = S.stage_report(cfg)
    assert report.exists() and "Held-out metrics" in report.read_text()

    pred_fn = UnFakeablePredictor(str(cfg.run_dir), "fused", "random_forest")
    r = pred_fn.predict(allm[allm.dataset == "celebdf"].iloc[0].path, explain=True)
    assert 0.0 <= r["prob_fake"] <= 1.0 and r["label"] in ("real", "fake")
    assert len(r["gradcam"]) == 4 and r["gradcam"][0].startswith("data:image/png;base64,")
    assert load_json(cfg.run_dir / "features" / "feature_meta.json")["backbone_source_used"] == "random"

    # dashboard backend serves the same run
    from fastapi.testclient import TestClient

    from unfakeable.app.server import create_app

    client = TestClient(create_app(str(cfg.run_dir)))
    s = client.get("/api/summary").json()
    assert s["metrics"] and "fused" in s["models"] and any(p.startswith("roc__") for p in s["plots"])
    assert client.get(f"/api/plots/{s['plots'][0]}").status_code == 200
    assert client.get("/api/plots/..%2Fconfig_used.yaml").status_code == 404
    path = allm[allm.dataset == "celebdf"].iloc[0].path
    with open(path, "rb") as fh:
        resp = client.post("/api/predict", files={"file": ("clip.mp4", fh, "video/mp4")},
                           data={"feature_set": "spatial", "classifier": "xgboost"})
    assert resp.status_code == 200 and resp.json()["classifier"] == "xgboost"
    assert client.get("/").status_code == 200
