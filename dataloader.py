"""Turn a data config (config/*.yml) into PyTorch DataLoaders."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

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


def make_loader(x, mask, input_type: str = "flat", batch_size: int = 1024, shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(_to_input(x, input_type), dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.bool),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def create_dataloaders(
    config_path: str,
    batch_size: int = 1024,
    shuffle: bool = True,
    input_type: str = "flat",  # "flat": (N, n_objects * n_features), "objects": (N, n_objects, n_features)
    apply_scaling: bool = True,
    load_test: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[Dict[str, DataLoader]]]:
    """Build train / val loaders, and optionally one test loader per sample.

    Each batch is ``(x, mask)``. ``mask`` is True for real objects, False for padding.
    The test loaders are keyed by sample name: the background first, then each signal.
    """
    cfg = load_config(config_path)
    max_events = cfg.get("max_events", {})

    def loader(entries, n, shuffle_):
        x, mask, _ = load_arrays(entries, cfg, n, apply_scaling)
        return make_loader(x, mask, input_type, batch_size, shuffle_)

    train_loader = loader(cfg["train"], max_events.get("train"), shuffle)
    val_loader = loader(cfg["val"], max_events.get("val"), False)

    test_loaders = None
    if load_test:
        test_loaders = {}
        n = max_events.get("test")
        test_loaders["background"] = loader(cfg["test"]["background"], n, False)
        for sample, split in cfg["test"]["signals"]:
            test_loaders[sample] = loader([[sample, split]], n, False)

    return train_loader, val_loader, test_loaders


if __name__ == "__main__":
    # Example usage: python dataloader.py
    train_loader, val_loader, _ = create_dataloaders("config/baseline.yml", batch_size=64)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    for batch_idx, (x, mask) in enumerate(train_loader):
        print(f"Batch {batch_idx}: x shape = {x.shape}, mask shape = {mask.shape}")
        if batch_idx == 2:
            break
