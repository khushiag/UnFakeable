"""The three classical decision makers. All consume the SAME transformed feature matrix."""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

CLASSIFIER_NAMES = ("svm_rbf", "random_forest", "xgboost")

# scikit-learn 1.9 deprecates SVC(probability=True) (Platt scaling) in favour of CalibratedClassifierCV; it still
# works until 1.11, and requirements.txt pins scikit-learn < 1.11, so the notice is silenced here.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*", category=FutureWarning)


def make_classifier(name: str, params: Optional[Dict[str, Any]] = None, seed: int = 42, n_jobs: int = 1,
                    probability: bool = True, scale_pos_weight: float = 1.0):
    params = dict(params or {})
    if name == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced", probability=probability, random_state=seed,
                   cache_size=500, **params)
    if name == "random_forest":
        params.setdefault("n_estimators", 400)
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                             n_jobs=n_jobs, random_state=seed, scale_pos_weight=scale_pos_weight, **params)
    raise ValueError(f"unknown classifier {name!r}; expected one of {CLASSIFIER_NAMES}")


def fake_scores(model, X):
    """Probability of the positive/fake class, shape (n,)."""
    return model.predict_proba(X)[:, 1]
