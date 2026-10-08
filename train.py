import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from dataloader import create_dataloaders
from models.registry import REGISTRY
from utils import CreateFolder, EarlyStopping, count_parameters, get_device


@torch.no_grad()
def validate(model, val_loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, mask in val_loader:
        scores = model(x.to(device), mask.to(device))
        total += scores.sum().item()
        n += len(scores)
    model.train()
    return total / n


def train_torch(model, train_loader, val_loader, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience)
    model.to(device)
    model.train()

    history = {"train": [], "val": []}
    global_step, best_val_loss = 0, float("inf")
    for epoch in range(args.epochs):
        running_loss, running_n = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{args.epochs}", unit="batch")
        for x, mask in pbar:
            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad()
            # The model returns one anomaly score per event; for an autoencoder this is the
            # reconstruction error, so minimising its mean teaches it to reconstruct normal events.
            loss = model(x, mask).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_n += 1
            global_step += 1
            pbar.set_postfix({"running loss": f"{running_loss / running_n:.4f}"})

            if args.eval_freq > 0 and global_step % args.eval_freq == 0:
                history["train"].append((global_step, running_loss / running_n))
                val_loss = validate(model, val_loader, device)
                history["val"].append((global_step, val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model, args.output / "best_model.pth")

        val_loss = validate(model, val_loader, device)
        history["val"].append((global_step, val_loss))
        print(f"Epoch {epoch + 1}: train loss {running_loss / running_n:.4f}, val loss {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, args.output / "best_model.pth")
        if early_stopping.step(val_loss):
            print("Early stopping: validation loss has not improved.")
            break

    return best_val_loss, history


def train_sklearn(model, train_loader, args):
    x, mask = (t.numpy() for t in train_loader.dataset.tensors)
    model.fit(x, mask)
    model.save(args.output / "best_model.pkl")
    return None, {}


def main(args):
    model_cfg = REGISTRY[args.model]
    input_type = model_cfg["input_type"]
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders(
        args.data_config,
        batch_size=args.batch_size,
        shuffle=True,
        input_type=input_type,
    )

    if args.plot_data:
        from plotting import plot_data_distributions
        plot_data_distributions(train_loader, outpath=args.output / "input_distributions.png")

    input_shape = tuple(train_loader.dataset.tensors[0].shape[1:])
    model = model_cfg["class"](input_shape)
    print(model)
    n_params = count_parameters(model)
    print(f"Number of trainable parameters: {n_params}")

    if model_cfg["framework"] == "torch":
        best_val_loss, history = train_torch(model, train_loader, val_loader, args, device)
    else:
        best_val_loss, history = train_sklearn(model, train_loader, args)

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
    parser.add_argument(
        "--eval-freq", "-ef", type=int, default=500,
        help="Also evaluate every N training steps (besides every epoch). Deactivated if set to 0."
    )
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")

    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use (cuda is faster)"
    )
    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default=Path("output/"),
        help="Output directory for results"
    )
    parser.add_argument("--plot-data", action="store_true", help="Plot input distributions after loading")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    main(args)
