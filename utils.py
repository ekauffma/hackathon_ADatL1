import argparse
import os
from typing import Tuple

import numpy as np
from qkeras.utils import load_qmodel
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras

from models.autoencoder import anomaly_score
from models.registry import class_to_config
from models.sklearn_wrapper import SklearnModelWrapper


class CreateFolder(argparse.Action):
    """argparse action: create the folder (and parents) if it does not exist yet."""

    def __call__(self, parser, namespace, values, option_string=None):
        os.makedirs(values, exist_ok=True)
        setattr(namespace, self.dest, values)


def is_keras(model) -> bool:
    return isinstance(model, keras.Model)


def count_parameters(model) -> int:
    if is_keras(model):
        return int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return 0


def input_type_of(model) -> str:
    """The input type ("flat" or "objects") a trained model expects."""
    if is_keras(model):
        return "flat" if len(model.input_shape) == 2 else "objects"
    return class_to_config(type(model))[1]["input_type"]


def compute_scores(model, data) -> np.ndarray:
    """Anomaly score of every event in an EventData (works for Keras and sklearn models)."""
    if is_keras(model):
        return anomaly_score(model, data.x, data.mask)
    return model.score(data.x, data.mask)


def load_model(path):
    """Load a model saved by train.py: .h5 (Keras / QKeras) or .pkl (sklearn)."""
    if str(path).endswith(".pkl"):
        return SklearnModelWrapper.load(path)
    return load_qmodel(str(path), compile=False)  # also loads plain Keras models


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
