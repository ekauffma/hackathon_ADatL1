import argparse
from pathlib import Path

import yaml
from tensorflow import keras

from dataloader import create_datasets
from models.autoencoder import make_target, masked_mse
from models.registry import REGISTRY
from utils import CreateFolder, count_parameters


def train_keras(model, train, val, args):
    # The loss is the reconstruction error of each event, so minimising it teaches
    # the autoencoder to reconstruct normal events.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), loss=masked_mse)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(args.output / "best_model.h5"), save_best_only=True),
    ]
    fit = model.fit(
        train.x, make_target(train.x, train.mask),
        validation_data=(val.x, make_target(val.x, val.mask)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    epochs = range(1, len(fit.history["loss"]) + 1)
    history = {
        "train": list(zip(epochs, fit.history["loss"])),
        "val": list(zip(epochs, fit.history["val_loss"])),
    }
    return float(min(fit.history["val_loss"])), history


def train_sklearn(model, train, args):
    model.fit(train.x, train.mask)
    model.save(args.output / "best_model.pkl")
    return None, {}


def main(args):
    model_cfg = REGISTRY[args.model]
    input_type = model_cfg["input_type"]

    train, val, _ = create_datasets(args.data_config, input_type=input_type)

    if args.plot_data:
        from plotting import plot_data_distributions
        plot_data_distributions(train, outpath=args.output / "input_distributions.png")

    input_shape = train.x.shape[1:]
    model = model_cfg["build"](input_shape)
    if model_cfg["framework"] == "keras":
        model.summary()
    n_params = count_parameters(model)
    print(f"Number of trainable parameters: {n_params}")

    if model_cfg["framework"] == "keras":
        best_val_loss, history = train_keras(model, train, val, args)
    else:
        best_val_loss, history = train_sklearn(model, train, args)

    if history:
        from plotting import plot_loss_curve
        plot_loss_curve(history, outpath=args.output / "loss.png")

    # Record what was trained, so evaluate.py and your teammates know what's in this folder.
    with open(args.output / "train_info.yml", "w") as f:
        yaml.safe_dump({
            "model": args.model,
            "data_config": str(args.data_config),
            "n_params": n_params,
            "best_val_loss": best_val_loss,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        }, f)
    print(f"[DONE] Wrote outputs to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an anomaly detection model on L1 trigger data")

    # Dataset and architecture
    parser.add_argument("--data-config", "-c", type=str, default="config/baseline.yml")
    parser.add_argument(
        "--model", "-m", type=str, default="dense-ae", choices=list(REGISTRY.keys()), help="Model architecture."
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", "-bs", type=int, default=1024, help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")

    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default=Path("output/"),
        help="Output directory for results"
    )
    parser.add_argument("--plot-data", action="store_true", help="Plot input distributions after loading")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    main(args)
