"""Reading the CMS Level-1 Trigger anomaly detection dataset.

The dataset lives on the Hugging Face Hub at ``CERN/anomaly_detection_cmsl1t``.
Every sample (zero-bias data, simulated background, simulated signals) is stored
as parquet files under ``data/<sample>/<split>-NNNNN-of-NNNNN.parquet``, with one
row per collision event. Object collections (muons, jets, e-gammas, taus) are
jagged lists; energy sums (ET, HT, MET, ...) are lists of length one.

Typical use::

    import dataset
    dataset.download(["ZB_run396102", "WtoTauto3Mu"], splits=["train", "valid"])
    events = dataset.load_events("ZB_run396102", "train", max_events=100_000)
    x, mask = dataset.to_padded(events)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ID = "CERN/anomaly_detection_cmsl1t"
# Folder that contains the dataset's "data/" directory (a full download of the Hub
# repository has the same layout). Defaults to this repository: files go to ./data/<sample>/.
DEFAULT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Catalogue of samples and their integer labels (from the dataset card).
#   0  : zero-bias collision data (the "normal" data)
#   <0 : simulated normal data
#   >0 : simulated anomalies / signals
# ---------------------------------------------------------------------------
ZEROBIAS = ["ZB_run396102", "ZB_run398183"]
BACKGROUND_SIM = ["SingleNeutrino_E-10-gun"]
SIGNALS = [
    "GluGluHTo2B_Par-MH-125",
    "GluGluHto2G_Par-MH-125",
    "GluGluHto2G_Par-MH-90",
    "GluGluHto2Tau_Par-MH-125",
    "GluGlutoHHto2B2WtoLNu2Q_Par-c2-0-kl-1-kt-1",
    "HHHto4B2Tau_Par-c3-0-d4-0",
    "HHHto6B_Par-c3-0-d4-0",
    "SUSYGluGluToBBHTo2B_Par-M-1200",
    "SUSYGluGluToBBHToBB_Par-M-120",
    "SUSYGluGluToBBHToBB_Par-M-350",
    "SUSYGluGluToBBHToBB_Par-M-600",
    "TTHTo2C_Par-MH-125",
    "TTHto2B_Par-MH-125",
    "VBFHTo2C_Par-MH-125",
    "VBFHto2B_Par-MH-125",
    "VBFHto2Tau_Par-MH-125",
    "WtoTauto3Mu",
    "ggH-suep-decay",
    "haa-4b-ma15",
    "smj-case-A",
]
ALL_SAMPLES = ZEROBIAS + BACKGROUND_SIM + SIGNALS

# Only the zero-bias samples have a train split; simulations have valid/test.
SPLITS = ("train", "valid", "test")

# ---------------------------------------------------------------------------
# Column naming. Raw columns are "<collection>_<branch>", e.g. "jets_jetIEt".
# We map the kinematic branches to common names Et / eta / phi.
# ---------------------------------------------------------------------------
OBJECTS = ("muons", "jets", "egammas", "taus")
ENERGY_SUMS = ("ET", "HT", "MET", "MHT", "FET", "FHT")

KINEMATIC_BRANCHES = {
    "muons": {"Et": "muonIEt", "eta": "muonIEtaAtVtx", "phi": "muonIPhiAtVtx"},
    "jets": {"Et": "jetIEt", "eta": "jetIEta", "phi": "jetIPhi"},
    "egammas": {"Et": "egIEt", "eta": "egIEta", "phi": "egIPhi"},
    "taus": {"Et": "tauIEt", "eta": "tauIEta", "phi": "tauIPhi"},
    "MET": {"Et": "Et", "phi": "phi"},
    "MHT": {"Et": "Et", "phi": "phi"},
    "FET": {"Et": "Et", "phi": "phi"},
    "FHT": {"Et": "Et", "phi": "phi"},
    "ET": {"Et": "Et"},
    "HT": {"Et": "Et"},
}

# Hardware-integer -> physical unit conversion (GeV, pseudorapidity, radians).
CALO_ETA = 0.0870 / 2
CALO_PHI = 2 * np.pi / 144
MUON_ETA = 0.0870 / 8
MUON_PHI = 2 * np.pi / 576
SCALES = {
    "muons": {"Et": 0.5, "eta": MUON_ETA, "phi": MUON_PHI},
    "jets": {"Et": 0.5, "eta": CALO_ETA, "phi": CALO_PHI},
    "egammas": {"Et": 0.5, "eta": CALO_ETA, "phi": CALO_PHI},
    "taus": {"Et": 0.5, "eta": CALO_ETA, "phi": CALO_PHI},
    **{s: {"Et": 0.5, "phi": CALO_PHI} for s in ("MET", "MHT", "FET", "FHT")},
    **{s: {"Et": 0.5} for s in ("ET", "HT")},
}


# ---------------------------------------------------------------------------
# Locating / downloading files
# ---------------------------------------------------------------------------
def _patterns(samples: Iterable[str], splits: Iterable[str], seeds: bool) -> list[str]:
    pats = []
    for s in samples:
        if s not in ALL_SAMPLES:
            raise ValueError(f"Unknown sample {s!r}. See dataset.ALL_SAMPLES.")
        for sp in splits:
            pats.append(f"data/{s}/{sp}-*.parquet")
            if seeds:
                pats.append(f"data/{s}/seeds/{sp}-*.parquet")
    return pats


def download(
    samples: Sequence[str] = ALL_SAMPLES,
    splits: Sequence[str] = SPLITS,
    seeds: bool = False,
    root: str | Path = DEFAULT_ROOT,
) -> Path:
    """Download (a subset of) the dataset from the Hugging Face Hub.

    Files that already exist locally are skipped, so this is safe to re-run.

    Args:
        samples: which samples to fetch, e.g. ``["ZB_run396102", "WtoTauto3Mu"]``.
        splits: any of ``"train"``, ``"valid"``, ``"test"``.
        seeds: also fetch the ``seeds`` files (decisions of the standard
            trigger algorithms, for benchmarking against the current menu).
        root: folder in which ``data/<sample>/`` is created (default: this repository).
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=_patterns(samples, splits, seeds),
    )
    return Path(root)


def files(sample: str, split: str, seeds: bool = False, root: str | Path = DEFAULT_ROOT) -> list[Path]:
    """Sorted list of local parquet files for a sample/split."""
    folder = Path(root) / "data" / sample
    if seeds:
        folder = folder / "seeds"
    found = sorted(folder.glob(f"{split}-*.parquet"))
    if not found:
        raise FileNotFoundError(
            f"No files for {sample}/{split} (seeds={seeds}) under {folder}. "
            f"Run dataset.download([{sample!r}], splits=[{split!r}]) first."
        )
    return found


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_table(
    sample: str,
    split: str,
    columns: Sequence[str] | None = None,
    max_events: int | None = None,
    seeds: bool = False,
    root: str | Path = DEFAULT_ROOT,
) -> pa.Table:
    """Read raw parquet rows into a pyarrow Table (optionally only the first N)."""
    batches, n = [], 0
    for f in files(sample, split, seeds=seeds, root=root):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=columns, batch_size=65_536):
            if max_events is not None and n + batch.num_rows > max_events:
                batch = batch.slice(0, max_events - n)
            batches.append(batch)
            n += batch.num_rows
            if max_events is not None and n >= max_events:
                return pa.Table.from_batches(batches)
    return pa.Table.from_batches(batches)


def load_events(
    sample: str,
    split: str,
    max_events: int | None = None,
    drop_removed: bool = True,
    physical_units: bool = False,
    root: str | Path = DEFAULT_ROOT,
) -> ak.Array:
    """Load events as an awkward array with one record per collection.

    The result has fields ``muons``, ``jets``, ``egammas``, ``taus`` (jagged,
    each with ``Et``/``eta``/``phi``), the energy sums ``ET``, ``HT``, ``MET``,
    ``MHT``, ``FET``, ``FHT`` (one entry each), plus ``label``, ``L1bit`` and
    ``order``. Calorimeter objects are Et-ordered; muons are sorted here too.

    Args:
        drop_removed: drop events with ``order == -1`` (removed by the standard
            preprocessing because they are saturated).
        physical_units: convert hardware integers to GeV / eta / radians.
    """
    columns = [
        f"{coll}_{raw}"
        for coll, branches in KINEMATIC_BRANCHES.items()
        for raw in branches.values()
    ] + ["label", "L1bit", "order"]
    table = read_table(sample, split, columns=columns, max_events=max_events, root=root)
    raw = ak.from_arrow(table)

    out = {}
    for coll, branches in KINEMATIC_BRANCHES.items():
        rec = {}
        for name, branch in branches.items():
            values = raw[f"{coll}_{branch}"]
            if physical_units:
                if coll == "muons" and name == "Et":
                    values = values - 1  # hardware 0 means "no muon"
                values = values * SCALES[coll][name]
            rec[name] = values
        rec = ak.zip(rec)
        if coll == "muons":
            rec = rec[ak.argsort(rec.Et, ascending=False)]
        out[coll] = rec
    for field in ("label", "L1bit", "order"):
        out[field] = raw[field]

    events = ak.Array(out)
    if drop_removed:
        events = events[events.order >= 0]
    return events


def load_seeds(sample: str, split: str, max_events: int | None = None, root: str | Path = DEFAULT_ROOT):
    """Load the standard trigger algorithm decisions as a pandas DataFrame.

    Row i of the seeds matches row i of :func:`read_table` for the same
    sample/split (before any filtering), and both carry the ``order`` column.
    """
    return read_table(sample, split, max_events=max_events, seeds=True, root=root).to_pandas()


# ---------------------------------------------------------------------------
# Converting to fixed-size arrays for ML
# ---------------------------------------------------------------------------
# Default layout: 1 MET + 4 muons + 10 jets + 12 e-gammas + 12 taus = 39 slots.
DEFAULT_LAYOUT = {"MET": 1, "muons": 4, "jets": 10, "egammas": 12, "taus": 12}


def to_padded(
    events: ak.Array,
    layout: dict[str, int] = DEFAULT_LAYOUT,
    features: Sequence[str] = ("Et", "eta", "phi"),
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate each collection and stack into a dense array.

    Args:
        events: output of :func:`load_events`.
        layout: collection -> number of slots to keep (highest-Et first).
        features: per-object features; a collection lacking one (e.g. ``eta``
            for MET) gets ``fill_value``.

    Returns:
        ``x`` of shape ``(n_events, sum(layout.values()), len(features))`` and a
        boolean ``mask`` of shape ``(n_events, sum(layout.values()))`` that is
        True where a real object is present.
    """
    xs, masks = [], []
    for coll, n in layout.items():
        objs = ak.pad_none(events[coll], n, clip=True, axis=1)
        present = ak.to_numpy(~ak.is_none(objs, axis=1))
        cols = []
        for feat in features:
            if feat in objs.fields:
                col = ak.to_numpy(ak.fill_none(objs[feat], fill_value)).astype(np.float32)
            else:
                col = np.full(present.shape, fill_value, dtype=np.float32)
            cols.append(col)
        xs.append(np.stack(cols, axis=-1))
        masks.append(present)
    return np.concatenate(xs, axis=1), np.concatenate(masks, axis=1)


if __name__ == "__main__":
    # Quick check: python dataset.py
    events = load_events("ZB_run396102", "valid", max_events=1000, physical_units=True)
    print(f"{len(events)} events, fields: {events.fields}")
    x, mask = to_padded(events)
    print(f"x: {x.shape}, mask: {mask.shape}, objects in first event: {mask[0].sum()}")
