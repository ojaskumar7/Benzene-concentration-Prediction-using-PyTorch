"""Evaluate a saved checkpoint on the held-out test split.

Usage:
    python -m src.evaluate
    python -m src.evaluate --model "models/Regression Model.pth" --hidden-layers 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

try:  # imported as a package: python -m src.evaluate
    from .data import DEFAULT_DATA_PATH, load_clean
    from .features import build_datasets
    from .model import (
        DEFAULT_HIDDEN_LAYERS,
        DEFAULT_HIDDEN_UNITS,
        DEFAULT_MODEL_PATH,
        load_model,
    )
except ImportError:  # run as a script: python src/evaluate.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import DEFAULT_DATA_PATH, load_clean
    from src.features import build_datasets
    from src.model import (
        DEFAULT_HIDDEN_LAYERS,
        DEFAULT_HIDDEN_UNITS,
        DEFAULT_MODEL_PATH,
        load_model,
    )


def compute_metrics(y_true, predictions) -> dict[str, float]:
    """Return MSE, RMSE and R2 for regression predictions."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    mse = float(mean_squared_error(y_true, predictions))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, predictions)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hidden-units", type=int, default=DEFAULT_HIDDEN_UNITS)
    parser.add_argument("--hidden-layers", type=int, default=DEFAULT_HIDDEN_LAYERS)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_clean(args.data)
    data = build_datasets(df, test_size=args.test_size, seed=args.seed)
    model = load_model(
        args.model,
        input_features=len(data.feature_names),
        hidden_units=args.hidden_units,
        num_hidden_layers=args.hidden_layers,
        device=device,
    )

    with torch.inference_mode():
        predictions = model(data.X_test.to(device)).cpu().numpy()
    y_true = data.y_test.numpy()

    metrics = compute_metrics(y_true, predictions)
    print(f"Model: {args.model}")
    print(f"Test rows: {len(y_true)} | Features: {len(data.feature_names)}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R^2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
