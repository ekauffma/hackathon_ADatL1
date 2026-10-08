"""Evaluate trained models: how much signal does each catch at a fixed zero-bias acceptance?

Edit MODELS / CONFIG below, then run:  python evaluate.py
"""

import csv
from pathlib import Path

from dataloader import create_datasets
from plotting import plot_roc, plot_score_distributions, plot_signal_efficiency
from utils import compute_scores, efficiency_at_acceptance, input_type_of, load_model, roc

# (display name, path to a model saved by train.py)
MODELS = [
    ("dense-ae", "output/baseline/best_model.h5"),
    ("tiny-ae", "output/tiny/best_model.h5"),
    ("tiny-qae", "output/tiny-q/best_model.h5"),
    ("pca", "output/pca/best_model.pkl"),
]
# The test samples (background + signals) come from this config.
CONFIG = "config/baseline.yml"

# Fraction of zero-bias events the trigger is allowed to keep.
# TODO(organisers): choose the operating point for the hackathon.
ACCEPTANCE = 1e-3

outdir = Path("output/evaluation")


def main():
    test_sets = {}  # loaded once per input type, then reused
    efficiencies, rows = {}, []

    for model_name, model_path in MODELS:
        if not Path(model_path).exists():
            print(f"[SKIP] {model_path} not found")
            continue

        print(f"\n[INFO] Loading model from {model_path}...")
        model = load_model(model_path)
        input_type = input_type_of(model)
        print(f"[INFO] Evaluating {model_name} with input_type={input_type}...")

        if input_type not in test_sets:
            _, _, test_sets[input_type] = create_datasets(CONFIG, input_type=input_type, load_test=True)

        scores = {name: compute_scores(model, data) for name, data in test_sets[input_type].items()}
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
