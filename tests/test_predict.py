import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.data import load_clean
from src.features import build_datasets
from src.model import RegressionModel, load_checkpoint, save_checkpoint
from src.predict import PREDICTION_COLUMN, load_for_prediction, main

FEATURES = [
    "CO(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]


def write_bundle(tmp_path):
    raw = np.random.default_rng(0).normal(size=(40, len(FEATURES)))
    scaler = StandardScaler().fit(raw)
    model = RegressionModel(len(FEATURES), 1, hidden_units=16, num_hidden_layers=1)
    path = tmp_path / "model.pth"
    save_checkpoint(path, model, scaler=scaler, feature_names=FEATURES)
    return path


def write_rows(tmp_path, rows: pd.DataFrame, **csv_options):
    path = tmp_path / "rows.csv"
    rows.to_csv(path, index=False, **csv_options)
    return path


def test_predict_writes_one_score_per_row(tmp_path):
    model_path = write_bundle(tmp_path)
    rows = pd.DataFrame(
        np.random.default_rng(1).normal(size=(3, len(FEATURES))), columns=FEATURES
    )
    input_path = write_rows(tmp_path, rows)
    output_path = tmp_path / "predictions.csv"

    main(["--input", str(input_path), "--output", str(output_path), "--model", str(model_path)])

    scored = pd.read_csv(output_path)
    assert list(scored.columns) == [*FEATURES, PREDICTION_COLUMN]
    assert len(scored) == 3

    expected = load_checkpoint(model_path).predict_raw(rows)
    np.testing.assert_allclose(scored[PREDICTION_COLUMN].to_numpy(), expected, rtol=1e-6)


def test_predict_accepts_reordered_and_semicolon_files(tmp_path):
    model_path = write_bundle(tmp_path)
    rows = pd.DataFrame(
        np.random.default_rng(2).normal(size=(2, len(FEATURES))), columns=FEATURES
    )
    expected = load_checkpoint(model_path).predict_raw(rows)

    shuffled = rows[list(reversed(FEATURES))]
    input_path = write_rows(tmp_path, shuffled, sep=";", decimal=",")
    output_path = tmp_path / "predictions.csv"

    main(
        [
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", str(model_path),
            "--sep", ";",
            "--decimal", ",",
        ]
    )

    scored = pd.read_csv(output_path)
    np.testing.assert_allclose(scored[PREDICTION_COLUMN].to_numpy(), expected, rtol=1e-6)


def test_predict_reports_missing_feature_columns(tmp_path):
    model_path = write_bundle(tmp_path)
    rows = pd.DataFrame(
        np.random.default_rng(3).normal(size=(1, len(FEATURES) - 1)),
        columns=FEATURES[:-1],
    )
    input_path = write_rows(tmp_path, rows)

    with pytest.raises(SystemExit, match="missing feature columns"):
        main(["--input", str(input_path), "--model", str(model_path)])


def test_weights_only_checkpoint_falls_back_to_training_scaler(tmp_path):
    data = build_datasets(load_clean())
    model = RegressionModel(len(data.feature_names), 1, hidden_units=16, num_hidden_layers=1)
    path = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), path)

    checkpoint = load_for_prediction(path)

    assert checkpoint.scaler is not None
    assert checkpoint.feature_names == data.feature_names
