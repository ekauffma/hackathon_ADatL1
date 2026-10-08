"""Turn a data config (config/*.yml) into arrays ready for training and evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import dataset

REPO = Path(__file__).resolve().parent


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _root(cfg: dict) -> Path:
    root = Path(cfg.get("root", "."))
    return root if root.is_absolute() else REPO / root


def download_config(config_path: str):
    """Download every [sample, split] that a config refers to."""
    cfg = load_config(config_path)
    entries = cfg["train"] + cfg["val"] + cfg["test"]["background"] + cfg["test"]["signals"]
    for sample, split in entries:
        dataset.download([sample], splits=[split], root=_root(cfg))


def preprocess(x: np.ndarray, mask: np.ndarray, scaling: Optional[dict]) -> np.ndarray:
    """Apply the scaling from the config to real objects; padded slots stay 0."""
    x = x.copy()
    if scaling is not None:
        if scaling.get("log_et", False):
            x[..., 0] = np.log1p(x[..., 0])
        mean = np.asarray(scaling["mean"], dtype=np.float32)
        std = np.asarray(scaling["std"], dtype=np.float32)
        x = (x - mean) / std
    x[~mask] = 0.0
    return x


def load_arrays(
    entries: List[List[str]], cfg: dict, max_events: Optional[int], apply_scaling: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate a list of [sample, split] entries.

    Returns ``x`` (N, n_objects, n_features), ``mask`` (N, n_objects) and the
    integer ``label`` of each event (0 = zero bias, <0 = simulated normal, >0 = signal).
    """
    xs, masks, labels = [], [], []
    for sample, split in entries:
        events = dataset.load_events(
            sample, split, max_events=max_events, physical_units=True, root=_root(cfg)
        )
        x, mask = dataset.to_padded(events, cfg["input"]["layout"], cfg["input"]["features"])
        xs.append(x)
        masks.append(mask)
        labels.append(np.asarray(events.label, dtype=np.int64))
    x, mask, label = np.concatenate(xs), np.concatenate(masks), np.concatenate(labels)
    if apply_scaling:
        x = preprocess(x, mask, cfg.get("scaling"))
    return x, mask, label


def _to_input(x: np.ndarray, input_type: str) -> np.ndarray:
    if input_type == "flat":
        return x.reshape(len(x), -1)
    if input_type == "objects":
        return x
    raise ValueError(f"Unknown input_type: {input_type}")


@dataclass
class EventData:
    """Model inputs for a set of events.

    ``x`` is (N, n_objects * n_features) for "flat" input or (N, n_objects, n_features) for
    "objects" input. ``mask`` is (N, n_objects): True for real objects, False for padding.
    """

    x: np.ndarray
    mask: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


def make_dataset(x, mask, input_type: str = "flat") -> EventData:
    return EventData(_to_input(x, input_type).astype(np.float32), mask.astype(bool))


def create_datasets(
    config_path: str,
    input_type: str = "flat",  # "flat": (N, n_objects * n_features), "objects": (N, n_objects, n_features)
    apply_scaling: bool = True,
    load_test: bool = False,
) -> Tuple[EventData, EventData, Optional[Dict[str, EventData]]]:
    """Build the train / val datasets, and optionally one test dataset per sample.

    The test datasets are keyed by sample name: the background first, then each signal.
    """
    cfg = load_config(config_path)
    max_events = cfg.get("max_events", {})

    def load(entries, n):
        x, mask, _ = load_arrays(entries, cfg, n, apply_scaling)
        return make_dataset(x, mask, input_type)

    train = load(cfg["train"], max_events.get("train"))
    val = load(cfg["val"], max_events.get("val"))

    test = None
    if load_test:
        test = {}
        n = max_events.get("test")
        test["background"] = load(cfg["test"]["background"], n)
        for sample, split in cfg["test"]["signals"]:
            test[sample] = load([[sample, split]], n)

    return train, val, test


if __name__ == "__main__":
    # Example usage: python dataloader.py
    train, val, _ = create_datasets("config/baseline.yml")

    print(f"Training events:   {len(train):,}  x shape = {train.x.shape}, mask shape = {train.mask.shape}")
    print(f"Validation events: {len(val):,}")
