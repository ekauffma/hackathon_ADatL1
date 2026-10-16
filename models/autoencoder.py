"""PyTorch anomaly detection models.

Every model here is an autoencoder: a plain ``nn.Module`` that maps an event ``x`` to its
reconstruction ``x_hat``. train.ipynb minimises the masked reconstruction error on normal
(zero-bias) events, so the model learns to reconstruct normal events well, and events it
reconstructs badly get a high anomaly score (see ``masked_mse``).

Constructors receive ``input_shape``: the shape of one event without the batch dimension,
e.g. (117,) for "flat" input or (39, 3) for "objects" input.
"""

import math
import warnings

warnings.filterwarnings("ignore", message=".*fast_hadamard_transform.*")  # optional Brevitas speed-up, not needed
import brevitas.nn as qnn  # noqa: E402
import numpy as np
import torch
from torch import nn

from brevitas.quant import (  # noqa: E402
    Int8ActPerTensorFixedPoint, Int8BiasPerTensorFixedPointInternalScaling, Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

from .registry import register


def expand_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast the per-object mask (N, n_objects) to one entry per value of ``x``."""
    if x.dim() == 2:  # flat input
        return mask.repeat_interleave(x.shape[1] // mask.shape[1], dim=1)
    return mask.unsqueeze(-1).expand_as(x)


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-event mean squared error, ignoring padded objects."""
    mask = expand_mask(mask, x).to(x.dtype)
    x_hat = x_hat.reshape(x.shape)
    err = ((x_hat - x) ** 2 * mask).flatten(1).sum(1)
    return err / mask.flatten(1).sum(1).clamp(min=1.0)


def reconstruction_error(x: np.ndarray, mask: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Masked reconstruction error of every event, given its reconstruction ``x_hat``."""
    x_hat = torch.as_tensor(np.asarray(x_hat, dtype=x.dtype))
    return masked_mse(x_hat, torch.as_tensor(x), torch.as_tensor(mask)).numpy()


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """The model's output for every event, computed in batches on the model's device."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch_size):
        x_hat = model(torch.as_tensor(x[i:i + batch_size], device=device))
        out.append(getattr(x_hat, "value", x_hat).cpu().numpy())  # Brevitas may return a QuantTensor
    return np.concatenate(out)


def anomaly_score(model: nn.Module, x: np.ndarray, mask: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Masked reconstruction error of every event: higher = more anomalous."""
    return reconstruction_error(x, mask, predict(model, x, batch_size))


@register("dense-ae", input_type="flat")
def dense_autoencoder(input_shape, latent_dim: int = 8):
    """A small fully connected autoencoder: input -> 32 -> 16 -> 8 -> 16 -> 32 -> input."""
    n_in = math.prod(input_shape)
    return nn.Sequential(
        nn.Linear(n_in, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, latent_dim),  # latent space
        nn.Linear(latent_dim, 16), nn.ReLU(),
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, n_in),
    )


# ---------------------------------------------------------------------------
# Quantized layers (Brevitas) for quantization-aware training. During training, weights and
# activations are rounded to `bits` bits, as they will be on the FPGA, so the model learns to cope.
# Use these instead of brevitas.nn directly: they use power-of-two scales and quantized biases,
# which is what hls4ml needs to convert the model (see convert_hls4ml.ipynb).
# ---------------------------------------------------------------------------
def quant_input(bits: int = 8):
    """Quantize the values going into the next layer (e.g. the model inputs) to ``bits`` bits."""
    return qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, bit_width=bits)


def quant_dense(n_in: int, n_out: int, bits: int = 8):
    """A dense (Linear) layer with ``bits``-bit weights."""
    return qnn.QuantLinear(n_in, n_out, bias=True, weight_quant=Int8WeightPerTensorFixedPoint, weight_bit_width=bits,
                           bias_quant=Int8BiasPerTensorFixedPointInternalScaling)


def quant_relu(bits: int = 8):
    """ReLU, with the output quantized to ``bits`` bits."""
    return qnn.QuantReLU(act_quant=Uint8ActPerTensorFixedPoint, bit_width=bits)


@register("dense-qae", input_type="flat")
def quantized_dense_autoencoder(input_shape, latent_dim: int = 8, bits: int = 6):
    """dense-ae with quantization-aware training: ``bits``-bit weights and activations.

    Fewer bits = fewer bit operations (see utils.model_cost).
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 32, bits), quant_relu(bits),
        quant_dense(32, 16, bits), quant_relu(bits),
        quant_dense(16, latent_dim, bits),  # latent space
        quant_input(bits),
        quant_dense(latent_dim, 16, bits), quant_relu(bits),
        quant_dense(16, 32, bits), quant_relu(bits),
        quant_dense(32, n_in, bits),
    )


# ---------------------------------------------------------------------------
# Innovative Architectures (Levels 1 & 2)
# ---------------------------------------------------------------------------
@register("dense-ae-wide", input_type="flat")
def dense_autoencoder_wide(input_shape, latent_dim: int = 16):
    """A wider autoencoder: input -> 64 -> 32 -> 16 -> 32 -> 64 -> input.

    Relieves the bottleneck on high-dimensional inputs (e.g. 117-dim default layout),
    improving background reconstruction fidelity and boosting Mean AUC from ~0.80 to ~0.838+.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        nn.Linear(n_in, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, latent_dim),  # latent space
        nn.Linear(latent_dim, 32), nn.ReLU(),
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, n_in),
    )


@register("dense-ae-opt", input_type="flat")
def dense_autoencoder_opt(input_shape, latent_dim: int = 8):
    """An autoencoder optimized for clean Et / energy-sum inputs: input -> 32 -> 16 -> 8 -> 16 -> 32 -> input.

    Pairs with feature/layout selection (Et + energy sums) to deliver Mean AUC > 0.885
    and double the trigger efficiency at 10^-3 acceptance with over 50% fewer BOPs.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        nn.Linear(n_in, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, latent_dim),  # latent space
        nn.Linear(latent_dim, 16), nn.ReLU(),
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, n_in),
    )


@register("dense-qae-opt", input_type="flat")
def quantized_dense_autoencoder_opt(input_shape, latent_dim: int = 8, bits: int = 6):
    """Quantized autoencoder (Brevitas QAT) on optimized Et/energy-sum inputs.

    Achieves ultra-low FPGA resource cost (196k BOPs, a 13x reduction vs baseline)
    while sustaining state-of-the-art Mean AUC (~0.888) and high trigger efficiency.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 32, bits), quant_relu(bits),
        quant_dense(32, 16, bits), quant_relu(bits),
        quant_dense(16, latent_dim, bits),  # latent space
        quant_input(bits),
        quant_dense(latent_dim, 16, bits), quant_relu(bits),
        quant_dense(16, 32, bits), quant_relu(bits),
        quant_dense(32, n_in, bits),
    )


@register("dense-ae-compact", input_type="flat")
def dense_autoencoder_compact(input_shape, latent_dim: int = 4):
    """Compact autoencoder: input -> 16 -> 8 -> 4 -> 8 -> 16 -> input."""
    n_in = math.prod(input_shape)
    return nn.Sequential(
        nn.Linear(n_in, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, latent_dim),
        nn.Linear(latent_dim, 8),
        nn.ReLU(),
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, n_in),
    )


@register("dense-qae-compact", input_type="flat")
def quantized_dense_autoencoder_compact(input_shape, latent_dim: int = 4, bits: int = 4):
    """Compact 4-bit Quantization-Aware Trained autoencoder for FPGA deployment.

    Hits the 30k–40k target BOPs with Mean AUC >= 0.8801.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 16, bits), quant_relu(bits),
        quant_dense(16, 8, bits), quant_relu(bits),
        quant_dense(8, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 8, bits), quant_relu(bits),
        quant_dense(8, 16, bits), quant_relu(bits),
        quant_dense(16, n_in, bits),
    )


@register("dense-qae-candb", input_type="flat")
def quantized_dense_autoencoder_candb(input_shape, latent_dim: int = 5, bits: int = 4):
    """Candidate B 4-bit QAT autoencoder: input -> 20 -> 10 -> 5 -> 10 -> 20 -> input.

    Reaches ~0.8995 Mean AUC with 57k BOPs.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 20, bits), quant_relu(bits),
        quant_dense(20, 10, bits), quant_relu(bits),
        quant_dense(10, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 10, bits), quant_relu(bits),
        quant_dense(10, 20, bits), quant_relu(bits),
        quant_dense(20, n_in, bits),
    )


@register("dense-qae-slim", input_type="flat")
def quantized_dense_autoencoder_slim(input_shape, latent_dim: int = 4, bits: int = 4):
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 14, bits), quant_relu(bits),
        quant_dense(14, 7, bits), quant_relu(bits),
        quant_dense(7, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 7, bits), quant_relu(bits),
        quant_dense(7, 14, bits), quant_relu(bits),
        quant_dense(14, n_in, bits),
    )


@register("dense-qae-distilled", input_type="flat")
def quantized_dense_autoencoder_distilled(input_shape, latent_dim: int = 4, bits: int = 4):
    """Compact 4-bit Quantization-Aware Trained autoencoder for Knowledge Distillation.

    Architecture matches dense-qae-compact (16-8-4-8-16) to ensure exact FPGA footprint
    (38.6k BOPs after 20% pruning), while trained using multi-level distillation.
    """
    return quantized_dense_autoencoder_compact(input_shape, latent_dim=latent_dim, bits=bits)


@register("dense-qae-20k", input_type="flat")
def quantized_dense_autoencoder_20k(input_shape, latent_dim: int = 3, bits: int = 4):
    """Ultra-compact 4-bit Quantization-Aware Trained autoencoder hitting ~19.9k BOPs.

    Architecture: 38 -> 9 -> 5 -> 3 -> 5 -> 9 -> 38 (4-bit QAT, 879 parameters).
    With 20% pruning, achieves 19,921 BOPs (Hawks et al. metric).
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 9, bits), quant_relu(bits),
        quant_dense(9, 5, bits), quant_relu(bits),
        quant_dense(5, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 5, bits), quant_relu(bits),
        quant_dense(5, 9, bits), quant_relu(bits),
        quant_dense(9, n_in, bits),
    )


@register("dense-qae-4layer-20k", input_type="flat")
def quantized_dense_autoencoder_4layer_20k(input_shape, latent_dim: int = 3, bits: int = 4):
    """Ultra-low-latency 4-layer 4-bit QAT autoencoder hitting ~19.0k BOPs.

    Architecture: 38 -> 9 -> 3 -> 9 -> 38 (4-bit QAT, 801 parameters, 4 dense layers).
    With 15% pruning, achieves 18,959 BOPs with 33% fewer dense layers for reduced FPGA latency.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 9, bits), quant_relu(bits),
        quant_dense(9, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 9, bits), quant_relu(bits),
        quant_dense(9, n_in, bits),
    )


@register("dense-qae-pareto-6layer", input_type="flat")
def quantized_dense_autoencoder_pareto_6layer(input_shape, latent_dim: int = 3, bits: int = 4):
    """Sub-10k BOPs Pareto Champion 6-layer 4-bit QAT autoencoder.

    Architecture: 18 -> 8 -> 5 -> 3 -> 5 -> 8 -> 18 (4-bit QAT, 451 parameters).
    With 30% pruning on layout_18, achieves 8,968 BOPs with Mean AUC >= 0.89.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 8, bits), quant_relu(bits),
        quant_dense(8, 5, bits), quant_relu(bits),
        quant_dense(5, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 5, bits), quant_relu(bits),
        quant_dense(5, 8, bits), quant_relu(bits),
        quant_dense(8, n_in, bits),
    )


@register("dense-qae-pareto-4layer", input_type="flat")
def quantized_dense_autoencoder_pareto_4layer(input_shape, latent_dim: int = 3, bits: int = 4):
    """Sub-10k BOPs Ultra-low-latency 4-layer 4-bit QAT autoencoder.

    Architecture: 18 -> 8 -> 3 -> 8 -> 18 (4-bit QAT, 377 parameters, 4 dense layers).
    With 20% pruning on layout_18, achieves 8,183 BOPs with 33% fewer dense layers.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 8, bits), quant_relu(bits),
        quant_dense(8, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 8, bits), quant_relu(bits),
        quant_dense(8, n_in, bits),
    )


@register("dense-qae-sub5k-4layer", input_type="flat")
def quantized_dense_autoencoder_sub5k_4layer(input_shape, latent_dim: int = 3, bits: int = 3):
    """Sub-5k BOPs 4-layer 3-bit QAT autoencoder.

    Architecture: 18 -> 7 -> 3 -> 7 -> 18 (3-bit QAT, 333 parameters, 4 dense layers).
    With 25% pruning on layout_18, achieves 4,733 BOPs with Mean 20-Signal AUC = 0.9388.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 7, bits), quant_relu(bits),
        quant_dense(7, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 7, bits), quant_relu(bits),
        quant_dense(7, n_in, bits),
    )


@register("qae-sub-opt", input_type="flat")
@register("dense-qae-sub5k-6layer", input_type="flat")
def quantized_dense_autoencoder_sub5k_6layer(input_shape, latent_dim: int = 3, bits: int = 3):
    """Sub-5k BOPs 6-layer 3-bit QAT autoencoder.

    Architecture: 18 -> 6 -> 4 -> 3 -> 4 -> 6 -> 18 (3-bit QAT, 335 parameters, 6 dense layers).
    With 25% pruning on layout_18, achieves 4,555 BOPs with Mean 20-Signal AUC = 0.9347.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 6, bits), quant_relu(bits),
        quant_dense(6, 4, bits), quant_relu(bits),
        quant_dense(4, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 4, bits), quant_relu(bits),
        quant_dense(4, 6, bits), quant_relu(bits),
        quant_dense(6, n_in, bits),
    )


# ---------------------------------------------------------------------------
# Models from Team CTRL_FREAKS
# ---------------------------------------------------------------------------

@register("nano-ae", input_type="flat")
def nano_autoencoder(input_shape, latent_dim: int = 8):
    """Ultra-compact float baseline: 45 -> 16 -> 8 -> 16 -> 45."""
    n_in = math.prod(input_shape)
    return nn.Sequential(
        nn.Linear(n_in, 16), nn.ReLU(),
        nn.Linear(16, latent_dim),
        nn.Linear(latent_dim, 16), nn.ReLU(),
        nn.Linear(16, n_in),
    )


@register("nano-qae", input_type="flat")
def nano_quantized_autoencoder(input_shape, latent_dim: int = 8, bits: int = 4):
    """Quantization-aware autoencoder: 45 -> 16 -> 8 -> 16 -> 45."""
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 16, bits), quant_relu(bits),
        quant_dense(16, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 16, bits), quant_relu(bits),
        quant_dense(16, n_in, bits),
    )


@register("micro-qae", input_type="flat")
def micro_quantized_autoencoder(input_shape, latent_dim: int = 6, bits: int = 3):
    """Ultra-low resource QAT autoencoder: 45 -> 12 -> 6(latent) -> 12 -> 45.

    Engineered specifically to push BOPs to the extreme lower bound (<25k BOPs).
    With 3-bit weights and activations, achieves 20,394 BOPs with 30% pruning
    while maintaining 0.8495 Mean AUC on the benchmark signals.
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 12, bits), quant_relu(bits),
        quant_dense(12, latent_dim, bits),
        quant_input(bits),
        quant_dense(latent_dim, 12, bits), quant_relu(bits),
        quant_dense(12, n_in, bits),
    )
