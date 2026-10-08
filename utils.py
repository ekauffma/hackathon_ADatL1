import argparse
import os
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


class CreateFolder(argparse.Action):
    """argparse action: create the folder (and parents) if it does not exist yet."""

    def __call__(self, parser, namespace, values, option_string=None):
        os.makedirs(values, exist_ok=True)
        setattr(namespace, self.dest, values)


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_str)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA device is not available."
    elif device.type == "mps":
        assert torch.backends.mps.is_available(), "MPS device is not available."
    return device


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: how many evaluations to wait after the last improvement.
            min_delta: minimum change in the monitored quantity to count as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


def count_parameters(model) -> int:
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0


@torch.no_grad()
def compute_scores(model, loader, device="cpu") -> np.ndarray:
    """Anomaly score of every event in a loader (works for torch and sklearn models)."""
    if isinstance(model, torch.nn.Module):
        model.eval()
        scores = [model(x.to(device), mask.to(device)).cpu() for x, mask in loader]
        return torch.cat(scores).numpy()
    x, mask = (t.numpy() for t in loader.dataset.tensors)
    return model.score(x, mask)


# ---------------------------------------------------------------------------
# Metrics. In all of them, background = normal (zero-bias) events, signal = anomalies.
# ---------------------------------------------------------------------------
def _labels_and_scores(bkg_scores, sig_scores):
    y = np.r_[np.zeros(len(bkg_scores)), np.ones(len(sig_scores))]
    return y, np.r_[bkg_scores, sig_scores]


def roc(bkg_scores: np.ndarray, sig_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (background acceptance, signal efficiency, AUC)."""
    y, s = _labels_and_scores(bkg_scores, sig_scores)
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(roc_auc_score(y, s))


def threshold_at_acceptance(bkg_scores: np.ndarray, acceptance: float) -> float:
    """Score threshold that keeps the given fraction of background events."""
    return float(np.quantile(bkg_scores, 1.0 - acceptance))


def efficiency_at_acceptance(bkg_scores: np.ndarray, sig_scores: np.ndarray, acceptance: float) -> float:
    """Fraction of signal events above the threshold that keeps `acceptance` of the background.

    This is how a trigger is judged: the rate of normal events it may keep is fixed
    by the bandwidth budget, and we want to catch as much signal as possible within it.
    """
    thr = threshold_at_acceptance(bkg_scores, acceptance)
    return float(np.mean(sig_scores > thr))
