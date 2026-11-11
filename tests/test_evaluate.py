import numpy as np
import pytest
import torch

from src.data import load_clean
from src.evaluate import compute_metrics
from src.features import build_datasets
from src.model import DEFAULT_MODEL_PATH, load_model


def test_perfect_predictions_score_zero_loss_and_unit_r2():
    values = np.array([1.0, 2.0, 3.0])

    metrics = compute_metrics(values, values)

    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0)


def test_known_errors_produce_expected_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    predictions = np.array([1.0, 2.0, 5.0])  # errors 0, 0, -2

    metrics = compute_metrics(y_true, predictions)

    assert metrics["mse"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(4 / 3))
    # Predicting worse than the mean gives a negative R2.
    assert metrics["r2"] == pytest.approx(-1.0)


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="shipped checkpoint is not available")
def test_shipped_checkpoint_matches_documented_metrics():
    data = build_datasets(load_clean())
    model = load_model(DEFAULT_MODEL_PATH)

    with torch.inference_mode():
        predictions = model(data.X_test).numpy()
    metrics = compute_metrics(data.y_test.numpy(), predictions)

    assert len(predictions) == 1510
    assert metrics["mse"] == pytest.approx(0.0486, abs=5e-4)
    assert metrics["r2"] == pytest.approx(0.9992, abs=5e-4)
