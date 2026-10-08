"""Evaluate trained models: how much signal does each catch at a fixed zero-bias acceptance?

Edit MODELS / CONFIG below, then run:  python evaluate.py
"""

import csv
from pathlib import Path

import torch

from dataloader import create_dataloaders
from models import registry
from models.sklearn_wrapper import SklearnModelWrapper
from plotting import plot_roc, plot_score_distributions, plot_signal_efficiency
from utils import compute_scores, efficiency_at_acceptance, get_device, roc

# (display name, path to a model saved by train.py)
MODELS = [
    ("dense-ae", "output/baseline/best_model.pth"),
    ("tiny-ae", "output/tiny/best_model.pth"),
    ("pca", "output/pca/best_model.pkl"),
]
# The test samples (background + signals) come from this config.
CONFIG = "config/baseline.yml"

# Fraction of zero-bias events the trigger is allowed to keep.
# TODO(organisers): choose the operating point for the hackathon.
ACCEPTANCE = 1e-3

device = get_device("auto")
outdir = Path("output/evaluation")


def load_model(path: str):
    if str(path).endswith(".pkl"):
        return SklearnModelWrapper.load(path)
    return torch.load(path, map_location=device, weights_only=False).to(device)


def main():
    test_loaders = {}  # loaded once per input type, then reused
    efficiencies, rows = {}, []

    for model_name, model_path in MODELS:
        if not Path(model_path).exists():
            print(f"[SKIP] {model_path} not found")
            continue

        print(f"\n[INFO] Loading model from {model_path}...")
        model = load_model(model_path)
        architecture, model_cfg = registry.class_to_config(type(model))
        input_type = model_cfg["input_type"]
        print(f"[INFO] Evaluating {architecture} with input_type={input_type}...")

        if input_type not in test_loaders:
            _, _, test_loaders[input_type] = create_dataloaders(
                CONFIG, shuffle=False, input_type=input_type, load_test=True
            )
        loaders = test_loaders[input_type]

        scores = {name: compute_scores(model, loader, device) for name, loader in loaders.items()}
        bkg = scores["background"]

        rocs, efficiencies[model_name] = {}, {}
        for signal in scores:
            if signal == "background":
                continue
            rocs[signal] = roc(bkg, scores[signal])
            eff = efficiency_at_acceptance(bkg, scores[signal], ACCEPTANCE)
            efficiencies[model_name][signal] = eff
            rows.append({"model": model_name, "signal": signal, "auc": rocs[signal][2], "efficiency": eff})
            print(f"[METRIC] {model_name:15s} {signal:35s} AUC = {rocs[signal][2]:.4f}   "
                  f"efficiency @ {ACCEPTANCE:g} = {eff:.4f}")

        plot_score_distributions(scores, title=model_name, outpath=outdir / f"scores_{model_name}.png")
        plot_roc(rocs, ACCEPTANCE, title=model_name, outpath=outdir / f"roc_{model_name}.png")

    if efficiencies:
        plot_signal_efficiency(efficiencies, ACCEPTANCE, outpath=outdir / "signal_efficiency.png")
        with open(outdir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "signal", "auc", "efficiency"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"[DONE] Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
