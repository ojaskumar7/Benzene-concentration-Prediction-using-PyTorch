"""Train the benzene regressor and save its state dict.

Usage:
    python -m src.train
    python -m src.train --epochs 200 --hidden-units 256 --hidden-layers 2
"""

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import r2_score
from torch import nn

try:  # imported as a package: python -m src.train
    from .data import DEFAULT_DATA_PATH, load_clean
    from .features import Datasets, build_datasets
    from .model import RegressionModel, save_checkpoint
except ImportError:  # run as a script: python src/train.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import DEFAULT_DATA_PATH, load_clean
    from src.features import Datasets, build_datasets
    from src.model import RegressionModel, save_checkpoint


def train(
    model: nn.Module,
    data: Datasets,
    epochs: int = 200,
    lr: float = 0.01,
    log_every: int = 10,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Full-batch training with per-epoch train and test loss tracking."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train, y_train = data.X_train.to(device), data.y_train.to(device)
    X_test, y_test = data.X_test.to(device), data.y_test.to(device)

    history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        model.train()
        loss = loss_fn(model(X_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_loss = loss_fn(model(X_test), y_test)

        if epoch % log_every == 0 or epoch == epochs - 1:
            history["epoch"].append(epoch)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(test_loss.item())
            print(
                f"Epoch: {epoch} | Train loss: {loss.item():.4f} "
                f"| Test loss: {test_loss.item():.4f}"
            )

    return history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-units", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/regression_model.pth"),
        help="Where to save the trained state dict.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    df = load_clean(args.data)
    data = build_datasets(df, test_size=args.test_size, seed=args.seed)
    print(f"Rows: {len(df)} | Features: {len(data.feature_names)} | Device: {device}")

    model = RegressionModel(
        len(data.feature_names),
        1,
        hidden_units=args.hidden_units,
        num_hidden_layers=args.hidden_layers,
    ).to(device)

    train(model, data, epochs=args.epochs, lr=args.lr, log_every=args.log_every, device=device)

    model.eval()
    with torch.inference_mode():
        predictions = model(data.X_test.to(device)).cpu().numpy()
    print(f"Final test R^2: {r2_score(data.y_test.numpy(), predictions):.4f}")

    save_checkpoint(
        args.output,
        model,
        scaler=data.scaler,
        feature_names=data.feature_names,
    )
    print(f"Saved checkpoint with scaler to {args.output}")


if __name__ == "__main__":
    main()
