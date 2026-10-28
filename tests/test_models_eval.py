import numpy as np
import pytest
from sklearn.base import clone

from unfakeable.config import TuneCfg
from unfakeable.evaluation import explain as ex
from unfakeable.evaluation import perturbations as pt
from unfakeable.evaluation.metrics import compute_metrics
from unfakeable.features.transformer import BlockScalerPCA
from unfakeable.models.classical import CLASSIFIER_NAMES, fake_scores, make_classifier
from unfakeable.models.tuning import tune_classifier
from unfakeable.utils.io import sha256_array


def _blobs(n=120, d=20, informative=(0, 8), seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    X = rng.normal(size=(n, d))
    X[:, informative[0]:informative[1]] += y[:, None] * 1.5
    groups = np.repeat(np.arange(n // 2), 2)  # two videos per source group
    return X, y, groups


def _tiny_cfg():
    return TuneCfg(cv_folds=3, classifiers={
        "svm_rbf": {"search": "grid", "space": {"C": [1, 10], "gamma": ["scale", 0.01]}},
        "random_forest": {"search": "random", "n_iter": 2, "space": {"n_estimators": [50], "max_depth": [4, None]}},
        "xgboost": {"search": "random", "n_iter": 2, "space": {"n_estimators": [30], "max_depth": [2, 3]}}})


def test_metrics_match_hand_computation():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    s = np.array([0.1, 0.7, 0.2, 0.9, 0.8, 0.4, 0.6, 0.3])
    m = compute_metrics(y, p, s)
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]  # [[TN, FP], [FN, TP]]
    assert m["accuracy"] == 0.75 and m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75
    assert m["roc_auc"] == pytest.approx(14 / 16)  # 14 of 16 (fake, real) score pairs ordered correctly
    assert compute_metrics([0, 0], [0, 0], [0.1, 0.2])["roc_auc"] is None


@pytest.mark.parametrize("name", CLASSIFIER_NAMES)
def test_tuning_and_identical_matrix_for_all_classifiers(name):
    X, y, g = _blobs()
    tpl = BlockScalerPCA({"a": (0, 12), "b": (12, 20)}, ["a", "b"], n_components=6, random_state=0)
    tr, te = np.arange(len(y)) < 90, np.arange(len(y)) >= 90
    tf = clone(tpl).fit(X[tr])
    Xtr, Xte = tf.transform(X[tr]), tf.transform(X[te])
    h = sha256_array(Xtr)
    res = tune_classifier(name, X[tr], y[tr], g[tr], tpl, _tiny_cfg(), seed=0)
    assert set(res.best_params) <= set(_tiny_cfg().classifiers[name]["space"])
    assert res.cv_metrics["cv_roc_auc_mean"] > 0.85 and res.n_splits == 3
    clf = make_classifier(name, res.best_params, seed=0, probability=True).fit(Xtr, y[tr])
    assert sha256_array(Xtr) == h  # fitting never mutates the shared matrix
    s = fake_scores(clf, Xte)
    assert s.shape == (30,) and ((0 <= s) & (s <= 1)).all()
    assert compute_metrics(y[te], clf.predict(Xte), s)["roc_auc"] > 0.85


def test_tuning_is_seed_reproducible():
    X, y, g = _blobs()
    tpl = BlockScalerPCA(None, None, n_components=5, random_state=0)
    a = tune_classifier("random_forest", X, y, g, tpl, _tiny_cfg(), seed=1)
    b = tune_classifier("random_forest", X, y, g, tpl, _tiny_cfg(), seed=1)
    assert a.best_params == b.best_params and a.cv_metrics == b.cv_metrics


def test_block_attribution_finds_the_informative_block():
    X, y, g = _blobs(n=200, d=20, informative=(0, 8))  # signal only in block "a"
    blocks = {"a": (0, 8), "b": (8, 20)}
    tf = BlockScalerPCA(blocks, ["a", "b"], n_components=10, random_state=0).fit(X)
    clf = make_classifier("random_forest", {"n_estimators": 200}, seed=0).fit(tf.transform(X), y)
    share = ex.block_attribution(tf, clf.feature_importances_)
    assert pytest.approx(sum(share.values()), abs=1e-6) == 1.0 and share["a"] > share["b"]
    top = ex.top_original_features(tf, clf.feature_importances_, 5)
    assert len(top) == 5 and top.importance.is_monotonic_decreasing
    perm = ex.permutation_component_importance(clf, tf.transform(X), y, 2, 0)
    assert perm.shape == (10,)


def test_shap_tree_returns_one_value_per_feature():
    pytest.importorskip("shap")
    X, y, _ = _blobs()
    for name in ("random_forest", "xgboost"):
        clf = make_classifier(name, {"n_estimators": 20}, seed=0).fit(X, y)
        sv = ex.shap_values_tree(clf, X, 40, 0)
        assert sv is not None and sv.shape == (X.shape[1],) and (sv >= 0).all()


def test_perturbations_shapes_and_effects():
    rng = np.random.default_rng(0)
    fr = rng.integers(0, 255, (2, 64, 64, 3), dtype=np.uint8)
    for out in (pt.jpeg(fr, 20), pt.gaussian_blur(fr, 2.0), pt.downscale(fr, 0.5)):
        assert out.shape == fr.shape and out.dtype == np.uint8 and not np.array_equal(out, fr)
    assert pt.gaussian_blur(fr, 3.0).var() < fr.var()
    assert pt.jpeg(fr, 10).astype(float).std() <= pt.jpeg(fr, 95).astype(float).std() * 1.05
    conds = pt.build_conditions([90], [1.0], [0.5])
    assert [c[0] for c in conds] == ["clean", "jpeg_q90", "blur_s1.0", "down_x0.5"]
    assert np.array_equal(conds[0][3](fr), fr)
