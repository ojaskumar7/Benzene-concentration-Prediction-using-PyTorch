"""PyTorch model for benzene concentration regression."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Regression Model.pth"

# The shipped checkpoint is a 2-hidden-block, 256-unit network over 11 features.
DEFAULT_HIDDEN_UNITS = 256
DEFAULT_HIDDEN_LAYERS = 2

# Scaler statistics are stored as tensors so the checkpoint still loads with
# torch.load(weights_only=True) instead of unpickling a sklearn object.
SCALER_MEAN_KEY = "scaler_mean"
SCALER_SCALE_KEY = "scaler_scale"


class RegressionModel(nn.Module):
    """Fully connected regressor: input -> hidden blocks -> single output.

    ``num_hidden_layers`` counts the hidden-to-hidden blocks on top of the
    input projection, matching the notebook's experiment table. With
    ``num_hidden_layers=1`` the stack is Linear(input, units) -> ReLU ->
    Linear(units, units) -> ReLU -> Linear(units, output); the default of 2
    adds one more block and reproduces the shipped checkpoint's four Linear
    layers.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_units: int = DEFAULT_HIDDEN_UNITS,
        num_hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_units = hidden_units
        self.num_hidden_layers = num_hidden_layers

        layers: list[nn.Module] = [nn.Linear(input_features, hidden_units), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers.append(nn.Linear(hidden_units, output_features))
        # Kept as one Sequential so saved state-dict keys stay "layers_stacked.*".
        self.layers_stacked = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers_stacked(x)


@dataclass(frozen=True)
class Checkpoint:
    """A loaded model together with the preprocessing it was trained with."""

    model: RegressionModel
    scaler: StandardScaler | None = None
    feature_names: list[str] | None = None

    def predict_raw(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict from unscaled sensor values, returning a 1-D array.

        A dataframe is reordered to the training feature order; the stored
        scaler is applied before the forward pass.
        """
        if self.scaler is None:
            raise ValueError(
                "checkpoint has no scaler; retrain with src.train to store one"
            )

        if isinstance(features, pd.DataFrame):
            frame = features
            if self.feature_names is not None:
                missing = [name for name in self.feature_names if name not in frame.columns]
                if missing:
                    raise KeyError(f"missing feature columns: {missing}")
                frame = frame[self.feature_names]
            values = frame.to_numpy(dtype=np.float64)
        else:
            values = np.asarray(features, dtype=np.float64)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        scaled = self.scaler.transform(values)
        device = next(self.model.parameters()).device
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(tensor)
        return predictions.cpu().numpy().ravel()


def save_checkpoint(
    path: Path,
    model: RegressionModel,
    scaler: StandardScaler | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """Save weights, architecture, feature order, and scaler statistics."""
    payload: dict = {
        "state_dict": model.state_dict(),
        "input_features": model.input_features,
        "output_features": model.output_features,
        "hidden_units": model.hidden_units,
        "num_hidden_layers": model.num_hidden_layers,
    }
    if scaler is not None:
        payload[SCALER_MEAN_KEY] = torch.as_tensor(scaler.mean_, dtype=torch.float64)
        payload[SCALER_SCALE_KEY] = torch.as_tensor(scaler.scale_, dtype=torch.float64)
    if feature_names is not None:
        payload["feature_names"] = list(feature_names)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path = DEFAULT_MODEL_PATH,
    input_features: int | None = None,
    output_features: int | None = None,
    hidden_units: int | None = None,
    num_hidden_layers: int | None = None,
    device: str = "cpu",
) -> Checkpoint:
    """Load a checkpoint, inferring the architecture when it is not recorded.

    Plain state dicts (such as the shipped checkpoint) are still supported;
    they simply carry no scaler.
    """
    payload = torch.load(path, map_location=device, weights_only=True)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        stored = (
            payload.get("input_features"),
            payload.get("output_features"),
            payload.get("hidden_units"),
            payload.get("num_hidden_layers"),
        )
        scaler = _scaler_from_payload(payload)
        feature_names = list(payload["feature_names"]) if "feature_names" in payload else None
    else:
        state_dict = payload
        stored = (None, None, None, None)
        scaler = None
        feature_names = None

    inferred = _architecture_from_state_dict(state_dict)
    resolved = [
        override if override is not None else recorded if recorded is not None else guess
        for override, recorded, guess in zip(
            (input_features, output_features, hidden_units, num_hidden_layers),
            stored,
            inferred,
        )
    ]

    model = RegressionModel(*resolved)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return Checkpoint(model=model, scaler=scaler, feature_names=feature_names)


def load_model(
    path: Path = DEFAULT_MODEL_PATH,
    input_features: int | None = None,
    output_features: int | None = None,
    hidden_units: int | None = None,
    num_hidden_layers: int | None = None,
    device: str = "cpu",
) -> RegressionModel:
    """Build the architecture and load a saved state dict into eval mode."""
    return load_checkpoint(
        path,
        input_features=input_features,
        output_features=output_features,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        device=device,
    ).model


def _architecture_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, int, int, int]:
    """Recover (input, output, hidden units, hidden layers) from the weights."""
    shapes = [tuple(value.shape) for key, value in state_dict.items() if key.endswith(".weight")]
    input_features = shapes[0][1]
    hidden_units = shapes[0][0]
    output_features = shapes[-1][0]
    # The stack is an input projection plus num_hidden_layers blocks plus an output.
    num_hidden_layers = len(shapes) - 2
    return input_features, output_features, hidden_units, num_hidden_layers


def _scaler_from_payload(payload: dict) -> StandardScaler | None:
    if SCALER_MEAN_KEY not in payload or SCALER_SCALE_KEY not in payload:
        return None
    scaler = StandardScaler()
    scaler.mean_ = payload[SCALER_MEAN_KEY].cpu().numpy()
    scaler.scale_ = payload[SCALER_SCALE_KEY].cpu().numpy()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler
