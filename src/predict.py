"""Score raw sensor rows from a CSV with a trained checkpoint.

Usage:
    python -m src.predict --input rows.csv --output predictions.csv
    python -m src.predict --input rows.csv --sep ";" --decimal ","
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

try:  # imported as a package: python -m src.predict
    from .data import load_clean
    from .features import build_datasets
    from .model import DEFAULT_MODEL_PATH, Checkpoint, load_checkpoint
except ImportError:  # run as a script: python src/predict.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import load_clean
    from src.features import build_datasets
    from src.model import DEFAULT_MODEL_PATH, Checkpoint, load_checkpoint

PREDICTION_COLUMN = "Predicted C6H6(GT)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV of raw, unscaled sensor rows."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Where to write the scored CSV.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sep", default=",", help="Input column separator (default: ',').")
    parser.add_argument("--decimal", default=".", help="Input decimal mark (default: '.').")
    return parser.parse_args(argv)


def load_for_prediction(path: Path) -> Checkpoint:
    """Load a checkpoint, rebuilding the training scaler for weights-only files."""
    checkpoint = load_checkpoint(path)
    if checkpoint.scaler is not None:
        return checkpoint

    # The shipped checkpoint predates scaler persistence; its scaler is the one
    # fit on the default training split, so rebuild it deterministically.
    data = build_datasets(load_clean())
    print(f"{path} has no stored scaler; using the default training split's scaler.")
    return Checkpoint(
        model=checkpoint.model,
        scaler=data.scaler,
        feature_names=checkpoint.feature_names or data.feature_names,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = load_for_prediction(args.model)

    rows = pd.read_csv(args.input, sep=args.sep, decimal=args.decimal)
    try:
        predictions = checkpoint.predict_raw(rows)
    except (KeyError, ValueError) as error:
        raise SystemExit(f"{args.input}: {error}") from None

    scored = rows.copy()
    scored[PREDICTION_COLUMN] = predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)

    print(f"Scored {len(predictions)} rows from {args.input}")
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
