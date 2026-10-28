from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def compute_metrics(y_true, y_pred, y_score: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Fake = positive class. ``confusion_matrix`` is [[TN, FP], [FN, TP]]."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out: Dict[str, object] = {
        "n": int(len(y_true)), "n_real": int((y_true == 0).sum()), "n_fake": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean([(y_pred[y_true == c] == c).mean() for c in (0, 1) if (y_true == c).any()])),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": cm.tolist(),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out
