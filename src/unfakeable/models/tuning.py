"""Hyperparameter search with source-group-stratified CV. The scaler/PCA is refit inside every fold."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from unfakeable.config import TuneCfg
from unfakeable.models.classical import make_classifier
from unfakeable.utils.log import get_logger

log = get_logger(__name__)

SCORING = {"roc_auc": "roc_auc", "accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}


@dataclass
class TuneResult:
    best_params: Dict[str, Any]
    cv_metrics: Dict[str, float]  # mean/std of the selected candidate over folds
    cv_results: pd.DataFrame
    n_splits: int


def tune_classifier(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, transformer, cfg: TuneCfg,
                    seed: int) -> TuneResult:
    spec = cfg.classifiers[name]
    n_splits = min(cfg.cv_folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("need at least 2 source groups for grouped cross-validation")
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    spw = n_neg / max(n_pos, 1)
    est = make_classifier(name, seed=seed, n_jobs=cfg.n_jobs, probability=False, scale_pos_weight=spw)
    space = {f"clf__{k}": v for k, v in spec["space"].items()}
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with tempfile.TemporaryDirectory() as tmp:  # caches the fold-wise scaler/PCA across candidates
        pipe = Pipeline([("prep", clone(transformer)), ("clf", est)], memory=tmp)
        # SVC itself is single-threaded, so its candidates/folds run in parallel; RF/XGBoost already use every
        # core inside each fit (n_jobs on the model), so their search stays sequential to avoid oversubscription.
        search_jobs = cfg.n_jobs if name == "svm_rbf" else 1
        common = dict(scoring=SCORING, refit=False, cv=cv, n_jobs=search_jobs, error_score="raise")
        if spec.get("search", "random") == "grid":
            search = GridSearchCV(pipe, space, **common)
        else:
            n_iter = min(int(spec.get("n_iter", 20)), len(ParameterGrid(space)))
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, random_state=seed, **common)
        search.fit(X, y, groups=groups)
    res = pd.DataFrame(search.cv_results_)
    best = int(res["rank_test_roc_auc"].to_numpy().argmin())
    params = {k.replace("clf__", "", 1): v for k, v in search.cv_results_["params"][best].items()}
    metrics = {}
    for m in SCORING:
        metrics[f"cv_{m}_mean"] = float(res.loc[best, f"mean_test_{m}"])
        metrics[f"cv_{m}_std"] = float(res.loc[best, f"std_test_{m}"])
    log.info("%s tuned over %d candidates, CV AUC=%.4f +/- %.4f, params=%s", name, len(res),
             metrics["cv_roc_auc_mean"], metrics["cv_roc_auc_std"], params)
    return TuneResult(params, metrics, res, n_splits)
