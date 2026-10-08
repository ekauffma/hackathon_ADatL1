from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def savefig(fig, path: Path, dpi: int = 160):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[PLOT] saved {path}")


def plot_data_distributions(data, feature_names=("Et", "eta", "phi"), outpath=None):
    """Histogram each (scaled) input feature over the real objects in an EventData."""
    x = data.x.reshape(len(data.x), data.mask.shape[1], -1)[data.mask]  # (n_objects_total, n_features)
    fig, axes = plt.subplots(1, x.shape[1], figsize=(4 * x.shape[1], 3.5))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.hist(x[:, i], bins=100, histtype="step")
        ax.set_xlabel(f"{feature_names[i]} (scaled)")
        ax.set_yscale("log")
    fig.tight_layout()
    if outpath:
        savefig(fig, outpath)
    return fig


def plot_loss_curve(history: Dict[str, List[Tuple[int, float]]], outpath=None):
    """history: {"train": [(epoch, loss), ...], "val": [(epoch, loss), ...]}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, points in history.items():
        if points:
            epochs, losses = zip(*points)
            ax.plot(epochs, losses, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    if outpath:
        savefig(fig, outpath)
    return fig


def plot_score_distributions(scores: Dict[str, np.ndarray], title: str = "", outpath=None):
    """scores: {sample name: anomaly scores}, including "background"."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_scores = np.concatenate(list(scores.values()))
    bins = np.linspace(*np.quantile(all_scores, [0.0, 0.999]), 80)
    for name, s in scores.items():
        ax.hist(s, bins=bins, histtype="step", density=True, label=name,
                linewidth=2 if name == "background" else 1)
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8)
    if outpath:
        savefig(fig, outpath)
    return fig


def plot_roc(rocs: Dict[str, Tuple[np.ndarray, np.ndarray, float]], acceptance: float = None,
             title: str = "", outpath=None):
    """rocs: {signal name: (background acceptance, signal efficiency, auc)}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    if acceptance is not None:
        ax.axvline(acceptance, color="gray", linestyle="--", label=f"acceptance = {acceptance:g}")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_xlabel("background (zero-bias) acceptance")
    ax.set_ylabel("signal efficiency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    if outpath:
        savefig(fig, outpath)
    return fig


def plot_signal_efficiency(eff: Dict[str, Dict[str, float]], acceptance: float, outpath=None):
    """eff: {model name: {signal name: efficiency}} as a grouped bar chart."""
    models = list(eff)
    signals = list(next(iter(eff.values())))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(signals)), 4.5))
    for i, m in enumerate(models):
        ax.bar(np.arange(len(signals)) + i * width, [eff[m][s] for s in signals], width, label=m)
    ax.set_xticks(np.arange(len(signals)) + 0.4 - width / 2)
    ax.set_xticklabels(signals, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"signal efficiency at {acceptance:g} acceptance")
    ax.legend()
    if outpath:
        savefig(fig, outpath)
    return fig
