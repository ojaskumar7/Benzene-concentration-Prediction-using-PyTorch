"""PyTorch model for benzene concentration regression."""

from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Regression Model.pth"

# The shipped checkpoint is a 2-hidden-block, 256-unit network over 11 features.
DEFAULT_HIDDEN_UNITS = 256
DEFAULT_HIDDEN_LAYERS = 2


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
        layers: list[nn.Module] = [nn.Linear(input_features, hidden_units), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers.append(nn.Linear(hidden_units, output_features))
        # Kept as one Sequential so saved state-dict keys stay "layers_stacked.*".
        self.layers_stacked = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers_stacked(x)


def load_model(
    path: Path = DEFAULT_MODEL_PATH,
    input_features: int = 11,
    output_features: int = 1,
    hidden_units: int = DEFAULT_HIDDEN_UNITS,
    num_hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
    device: str = "cpu",
) -> RegressionModel:
    """Build the architecture and load a saved state dict into eval mode."""
    model = RegressionModel(
        input_features,
        output_features,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
    )
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
