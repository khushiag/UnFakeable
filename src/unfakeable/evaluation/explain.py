"""Feature analysis: native importances, permutation importance, block attribution through PCA, and SHAP."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from unfakeable.utils.log import get_logger

log = get_logger(__name__)


def native_importance(model) -> Optional[np.ndarray]:
    imp = getattr(model, "feature_importances_", None)
    return None if imp is None else np.asarray(imp, dtype=float)


def permutation_component_importance(model, X, y, n_repeats: int, seed: int) -> np.ndarray:
    """Mean drop in ROC-AUC when each transformed feature is shuffled (works for the RBF-SVM too)."""
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return r.importances_mean


def _per_original_feature(transformer, importance: np.ndarray) -> np.ndarray:
    imp = np.clip(np.asarray(importance, dtype=float), 0, None)
    if transformer.pca_ is not None:
        return (imp[:, None] * transformer.pca_.components_ ** 2).sum(axis=0)
    return imp


def block_attribution(transformer, importance: np.ndarray) -> Dict[str, float]:
    """Share of model importance attributable to each feature block (spatial / temporal / handcrafted).

    PCA components are mapped back to the scaled input features through their squared loadings, so a component's
    importance is split across original features in proportion to how much each contributes to it.
    """
    per_feature = _per_original_feature(transformer, importance)
    total = per_feature.sum()
    if total <= 0:
        return {b: float("nan") for b in np.unique(transformer.block_of_feature_)}
    return {b: float(per_feature[transformer.block_of_feature_ == b].sum() / total)
            for b in np.unique(transformer.block_of_feature_)}


def top_original_features(transformer, importance: np.ndarray, top_n: int) -> pd.DataFrame:
    per_feature = _per_original_feature(transformer, importance)
    order = np.argsort(per_feature)[::-1][:top_n]
    return pd.DataFrame({"block": transformer.block_of_feature_[order],
                         "feature_index_in_block": [int(transformer.selected_idx_[i]) for i in order],
                         "importance": per_feature[order]})


def shap_values_tree(model, X: np.ndarray, max_samples: int, seed: int) -> Optional[np.ndarray]:
    """mean-|SHAP| per transformed feature for RF/XGBoost; returns None (with a log line) if SHAP is unavailable."""
    try:
        import shap
    except ImportError:
        log.warning("shap not installed; skipping SHAP")
        return None
    rng = np.random.default_rng(seed)
    Xs = X[rng.permutation(len(X))[:max_samples]]
    try:
        sv = shap.TreeExplainer(model).shap_values(Xs)
    except Exception as exc:
        log.warning("SHAP failed for %s: %s", type(model).__name__, exc)
        return None
    if isinstance(sv, list):  # older shap: one array per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes) or (classes, n, features) depending on shap version
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    return np.abs(sv).mean(axis=0)
