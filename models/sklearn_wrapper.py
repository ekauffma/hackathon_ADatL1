"""Classical (non-deep-learning) anomaly detectors from scikit-learn.

Like the Keras models, ``score(x, mask)`` returns one anomaly score per event,
higher = more anomalous.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .registry import register


class SklearnModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, x: np.ndarray, mask: np.ndarray):
        print(f"Training {type(self).__name__} on {x.shape[0]} events with {x.shape[1]} features")
        self.model.fit(x)
        return self

    def score(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


@register("isolation-forest", input_type="flat", framework="sklearn")
class IsolationForestModel(SklearnModelWrapper):
    """Isolation forest: anomalies are easier to separate from the rest with random cuts."""

    def __init__(self, input_shape, n_estimators: int = 100):
        super().__init__(IsolationForest(n_estimators=n_estimators, n_jobs=-1, random_state=42))

    def score(self, x, mask):
        return -self.model.score_samples(x)  # sklearn: lower = more abnormal


@register("pca", input_type="flat", framework="sklearn")
class PCAModel(SklearnModelWrapper):
    """PCA reconstruction error: a linear 'autoencoder'."""

    def __init__(self, input_shape, n_components: int = 8):
        super().__init__(PCA(n_components=n_components, random_state=42))

    def score(self, x, mask):
        x_hat = self.model.inverse_transform(self.model.transform(x))
        return ((x_hat - x) ** 2).mean(axis=1)
