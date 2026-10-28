from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from unfakeable.utils.io import ensure_dir  # noqa: E402

PALETTE = {"svm_rbf": "#1f4e9c", "random_forest": "#1fa39a", "xgboost": "#f08c2e"}


def _save(fig, path) -> str:
    ensure_dir(Path(path).parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_roc(curves: Dict[str, tuple], title: str, path) -> str:
    """curves: {label: (y_true, y_score)}"""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for label, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=label, color=PALETTE.get(label))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_confusions(cms: Dict[str, Sequence[Sequence[int]]], title: str, path) -> str:
    fig, axes = plt.subplots(1, len(cms), figsize=(3.4 * len(cms), 3.4), squeeze=False)
    for ax, (label, cm) in zip(axes[0], cms.items()):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "fake"], yticklabels=["real", "fake"],
               xlabel="predicted", ylabel="true", title=label)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def plot_grouped_bars(df: pd.DataFrame, x: str, hue: str, y: str, title: str, path, ylim=(0, 1.05)) -> str:
    pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.6 * len(pivot) + 3, 4))
    pivot.plot(kind="bar", ax=ax, rot=0, color=[PALETTE.get(c, None) for c in pivot.columns])
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7)
    ax.set(ylabel=y, title=title, ylim=ylim)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_barh(names: Sequence[str], values: Sequence[float], title: str, path) -> str:
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(names) + 1.5))
    ax.barh(list(names)[::-1], list(values)[::-1])
    ax.set(title=title)
    return _save(fig, path)


def plot_robustness(df: pd.DataFrame, metric: str, title: str, path) -> str:
    """df columns: condition, severity (numeric), kind, label."""
    kinds = list(df["kind"].unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.8), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        sub = df[df["kind"] == kind]
        for label, g in sub.groupby("label"):
            g = g.sort_values("severity")
            ax.plot(g["severity"], g[metric], marker="o", label=label)
        ax.set(title=kind, xlabel="severity", ylabel=metric)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def plot_image_grid(rows: Sequence[Sequence[np.ndarray]], row_titles: Sequence[str], title: str, path) -> str:
    """Grid of RGB images (one row per video)."""
    n_r, n_c = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.1 * n_c, 2.25 * n_r), squeeze=False)
    for i, (imgs, rt) in enumerate(zip(rows, row_titles)):
        for j in range(n_c):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(imgs):
                ax.imshow(imgs[j])
        axes[i][0].set_title(rt, fontsize=7, loc="left")
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)
