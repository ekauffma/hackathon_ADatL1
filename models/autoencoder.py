"""PyTorch anomaly detection models.

Every model takes ``(x, mask)`` and returns one anomaly score per event, where higher
means more anomalous. train.py minimises the mean score over normal (zero-bias) events,
so an autoencoder learns to reconstruct normal events well, and events it reconstructs
badly get a high score.

Constructors receive ``input_shape``: the shape of one event without the batch dimension,
e.g. (117,) for "flat" input or (39, 3) for "objects" input.
"""

import math

import torch
from torch import nn

from .registry import register


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-event mean squared error, ignoring padded objects."""
    if x.dim() == 2:  # flat input: expand the mask to one entry per feature
        mask = mask.repeat_interleave(x.shape[1] // mask.shape[1], dim=1)
    else:
        mask = mask.unsqueeze(-1).expand_as(x)
    mask = mask.float()
    err = (x_hat - x) ** 2 * mask
    return err.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1)


@register("dense-ae", input_type="flat")
class DenseAutoencoder(nn.Module):
    """A small fully connected autoencoder: input -> 32 -> 16 -> 8 -> 16 -> 32 -> input."""

    def __init__(self, input_shape, latent_dim: int = 8):
        super().__init__()
        n_in = math.prod(input_shape)
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, n_in),
        )

    def forward(self, x, mask):
        x_hat = self.decoder(self.encoder(x))
        return masked_mse(x_hat, x, mask)


@register("tiny-ae", input_type="flat")
class TinyAutoencoder(nn.Module):
    """A deliberately tiny autoencoder, closer to what fits on a trigger FPGA."""

    def __init__(self, input_shape, latent_dim: int = 4):
        super().__init__()
        n_in = math.prod(input_shape)
        self.encoder = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, n_in))

    def forward(self, x, mask):
        x_hat = self.decoder(self.encoder(x))
        return masked_mse(x_hat, x, mask)


# TODO(participants): add your own models here, e.g. a variational autoencoder, a model
# with "objects" input that treats each particle separately (deep sets / transformer), ...
