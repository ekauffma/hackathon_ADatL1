"""Keras / QKeras anomaly detection models.

Every model here is an autoencoder: a plain Keras model that maps an event ``x`` to its
reconstruction ``x_hat``. train.py minimises the masked reconstruction error on normal
(zero-bias) events, so the model learns to reconstruct normal events well, and events it
reconstructs badly get a high anomaly score (see ``masked_mse``).

Constructors receive ``input_shape``: the shape of one event without the batch dimension,
e.g. (117,) for "flat" input or (39, 3) for "objects" input.
"""

import math

import numpy as np
import tensorflow as tf
from qkeras import QActivation, QDense, quantized_bits, quantized_relu
from tensorflow import keras
from tensorflow.keras import layers

from .registry import register


def expand_mask(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Broadcast the per-object mask (N, n_objects) to one entry per value of ``x``."""
    if x.ndim == 2:  # flat input
        return np.repeat(mask, x.shape[1] // mask.shape[1], axis=1)
    return np.broadcast_to(mask[..., None], x.shape)


def make_target(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Training target for ``masked_mse``: the input and its mask, stacked on the last axis."""
    return np.concatenate([x, expand_mask(mask, x).astype(x.dtype)], axis=-1)


def masked_mse(target, x_hat):
    """Per-event mean squared error, ignoring padded objects.

    Keras losses only receive (y_true, y_pred), so the mask travels inside ``target``
    (built by ``make_target``).
    """
    x, mask = tf.split(target, 2, axis=-1)
    x_hat = tf.reshape(x_hat, tf.shape(x))
    err = tf.reshape(tf.square(x_hat - x) * mask, (tf.shape(x)[0], -1))
    n = tf.reshape(mask, (tf.shape(x)[0], -1))
    return tf.reduce_sum(err, axis=1) / tf.maximum(tf.reduce_sum(n, axis=1), 1.0)


def anomaly_score(model, x: np.ndarray, mask: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Masked reconstruction error of every event: higher = more anomalous."""
    x_hat = model.predict(x, batch_size=batch_size, verbose=0)
    return masked_mse(make_target(x, mask), x_hat).numpy()


@register("dense-ae", input_type="flat")
def dense_autoencoder(input_shape, latent_dim: int = 8):
    """A small fully connected autoencoder: input -> 32 -> 16 -> 8 -> 16 -> 32 -> input."""
    n_in = math.prod(input_shape)
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(latent_dim, name="latent"),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_in),
    ], name="dense_ae")


@register("tiny-ae", input_type="flat")
def tiny_autoencoder(input_shape, latent_dim: int = 4):
    """A deliberately tiny autoencoder, closer to what fits on a trigger FPGA."""
    n_in = math.prod(input_shape)
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Dense(16, activation="relu"),
        layers.Dense(latent_dim, name="latent"),
        layers.Dense(16, activation="relu"),
        layers.Dense(n_in),
    ], name="tiny_ae")


@register("tiny-qae", input_type="flat")
def tiny_quantized_autoencoder(input_shape, latent_dim: int = 4, bits: int = 8):
    """tiny-ae with quantized weights and activations (QKeras), ready for hls4ml.

    ``quantized_bits(bits, 0)`` stores each weight with ``bits`` bits, 0 of them before the binary point.
    """
    n_in = math.prod(input_shape)

    def qdense(units, name=None):
        q = quantized_bits(bits, 0, alpha=1)
        return QDense(units, kernel_quantizer=q, bias_quantizer=q, name=name)

    return keras.Sequential([
        keras.Input(shape=input_shape),
        qdense(16), QActivation(quantized_relu(bits)),
        qdense(latent_dim, name="latent"),
        qdense(16), QActivation(quantized_relu(bits)),
        qdense(n_in),
    ], name="tiny_qae")


# TODO(participants): add your own models here, e.g. a variational autoencoder, a model
# with "objects" input that treats each particle separately (deep sets / transformer), ...
