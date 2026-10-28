"""Leakage-safe feature transformer: block selection -> StandardScaler -> block weighting -> PCA.

Everything is fitted inside ``fit`` and therefore only ever sees the data passed to ``fit`` (the training split, or
the training portion of a CV fold when used inside a ``Pipeline``).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class BlockScalerPCA(BaseEstimator, TransformerMixin):
    def __init__(self, blocks: Optional[Dict[str, Tuple[int, int]]] = None, use: Optional[Sequence[str]] = None,
                 n_components=None, block_weighting: str = "sqrt_dim", random_state: int = 0):
        self.blocks = blocks
        self.use = use
        self.n_components = n_components
        self.block_weighting = block_weighting
        self.random_state = random_state

    def _select(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_idx_]

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        blocks = self.blocks or {"all": (0, X.shape[1])}
        use = list(self.use) if self.use else list(blocks)
        idx: List[int] = []
        owner: List[str] = []
        for name in use:
            a, b = blocks[name]
            idx.extend(range(a, b))
            owner.extend([name] * (b - a))
        self.selected_idx_ = np.asarray(idx)
        self.block_of_feature_ = np.asarray(owner)
        Xs = self._select(X)
        self.scaler_ = StandardScaler().fit(Xs)
        self.weights_ = np.ones(Xs.shape[1])
        if self.block_weighting == "sqrt_dim":
            for name in use:
                m = self.block_of_feature_ == name
                self.weights_[m] = 1.0 / np.sqrt(m.sum())
        elif self.block_weighting != "none":
            raise ValueError(f"block_weighting must be 'sqrt_dim' or 'none', got {self.block_weighting!r}")
        Z = self.scaler_.transform(Xs) * self.weights_
        self.pca_ = None
        if self.n_components is not None:
            nc = self.n_components
            if isinstance(nc, (int, np.integer)) or float(nc) >= 1:
                nc = int(max(1, min(int(nc), Z.shape[0] - 1, Z.shape[1])))
            self.pca_ = PCA(n_components=nc, random_state=self.random_state).fit(Z)
        self.n_features_out_ = self.pca_.n_components_ if self.pca_ is not None else Z.shape[1]
        return self

    def transform(self, X):
        Z = self.scaler_.transform(self._select(np.asarray(X, dtype=np.float64))) * self.weights_
        return self.pca_.transform(Z) if self.pca_ is not None else Z

    def feature_names_out(self) -> List[str]:
        if self.pca_ is not None:
            return [f"pc{i + 1}" for i in range(self.n_features_out_)]
        return [f"{b}_{i}" for i, b in enumerate(self.block_of_feature_)]
