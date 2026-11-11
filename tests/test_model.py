import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.model import (
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_HIDDEN_UNITS,
    DEFAULT_MODEL_PATH,
    RegressionModel,
    load_checkpoint,
    load_model,
    save_checkpoint,
)

FEATURES = [f"feature_{index}" for index in range(11)]
SHIPPED = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(), reason="shipped checkpoint is not available"
)


def fitted_scaler() -> StandardScaler:
    values = np.random.default_rng(0).normal(size=(50, len(FEATURES)))
    return StandardScaler().fit(values)


def test_forward_shape():
    model = RegressionModel(len(FEATURES), 1, hidden_units=16, num_hidden_layers=1)

    assert model(torch.zeros(4, len(FEATURES))).shape == (4, 1)


def test_default_architecture_has_two_hidden_blocks():
    model = RegressionModel(len(FEATURES), 1)
    linears = [layer for layer in model.layers_stacked if isinstance(layer, torch.nn.Linear)]

    assert [tuple(layer.weight.shape) for layer in linears] == [
        (DEFAULT_HIDDEN_UNITS, len(FEATURES)),
        (DEFAULT_HIDDEN_UNITS, DEFAULT_HIDDEN_UNITS),
        (DEFAULT_HIDDEN_UNITS, DEFAULT_HIDDEN_UNITS),
        (1, DEFAULT_HIDDEN_UNITS),
    ]


@SHIPPED
def test_shipped_checkpoint_loads_with_inferred_architecture():
    checkpoint = load_checkpoint(DEFAULT_MODEL_PATH)

    assert checkpoint.model.training is False
    assert checkpoint.model.hidden_units == DEFAULT_HIDDEN_UNITS
    assert checkpoint.model.num_hidden_layers == DEFAULT_HIDDEN_LAYERS
    # The shipped file predates scaler persistence: weights only.
    assert checkpoint.scaler is None
    assert checkpoint.feature_names is None


@SHIPPED
def test_architecture_mismatch_is_rejected():
    with pytest.raises(RuntimeError):
        load_model(DEFAULT_MODEL_PATH, hidden_units=16, num_hidden_layers=1)


def test_checkpoint_round_trip_preserves_scaler_and_features(tmp_path):
    scaler = fitted_scaler()
    model = RegressionModel(len(FEATURES), 1, hidden_units=32, num_hidden_layers=1)
    path = tmp_path / "model.pth"

    save_checkpoint(path, model, scaler=scaler, feature_names=FEATURES)
    checkpoint = load_checkpoint(path)

    assert checkpoint.feature_names == FEATURES
    assert checkpoint.scaler is not None
    assert checkpoint.model.hidden_units == 32
    assert checkpoint.model.num_hidden_layers == 1
    np.testing.assert_allclose(checkpoint.scaler.mean_, scaler.mean_)
    np.testing.assert_allclose(checkpoint.scaler.scale_, scaler.scale_)


def test_predict_raw_matches_manual_scaling(tmp_path):
    scaler = fitted_scaler()
    model = RegressionModel(len(FEATURES), 1, hidden_units=16, num_hidden_layers=1)
    path = tmp_path / "model.pth"
    save_checkpoint(path, model, scaler=scaler, feature_names=FEATURES)
    checkpoint = load_checkpoint(path)

    raw = pd.DataFrame(
        np.random.default_rng(1).normal(size=(5, len(FEATURES))), columns=FEATURES
    )
    scaled = scaler.transform(raw[FEATURES].to_numpy())
    with torch.inference_mode():
        expected = model(torch.tensor(scaled, dtype=torch.float32)).numpy().ravel()

    np.testing.assert_allclose(checkpoint.predict_raw(raw), expected, rtol=1e-5)
    # Column order is recovered from the stored feature names.
    shuffled = raw[list(reversed(FEATURES))]
    np.testing.assert_allclose(checkpoint.predict_raw(shuffled), expected, rtol=1e-5)
    # A single raw row is accepted and returns a 1-D array.
    assert checkpoint.predict_raw(raw.iloc[0].to_numpy()).shape == (1,)


def test_predict_raw_requires_the_stored_scaler(tmp_path):
    model = RegressionModel(len(FEATURES), 1)
    path = tmp_path / "weights_only.pth"
    torch.save(model.state_dict(), path)

    checkpoint = load_checkpoint(path)
    with pytest.raises(ValueError, match="no scaler"):
        checkpoint.predict_raw(np.zeros((1, len(FEATURES))))
