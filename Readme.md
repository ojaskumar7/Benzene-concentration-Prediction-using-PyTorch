# Air Quality Prediction: Deep Learning with PyTorch

Predicting **benzene (C6H6)** concentration from inexpensive environmental sensors with a PyTorch neural network, trained on the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality).

Real benzene sensors are expensive and need regular calibration. This project builds a "virtual sensor": given readings from temperature, humidity, CO, and other low-cost sensors, it estimates benzene concentration so air quality can be monitored at a fraction of the hardware cost.

## Project structure

```
Air Quality Data/   UCI dataset (AirQualityUCI.csv is what the code reads)
Images/             per-experiment prediction and residual plots
models/             checkpoints, including the shipped "Regression Model.pth"
notebooks/          original analysis notebook (EDA, experiments, plots)
src/                packaged pipeline
  data.py           loading and cleaning
  features.py       feature/target split, scaling, tensors
  model.py          network definition, checkpoint load/save, raw prediction
  train.py          training entry point
  evaluate.py       evaluation entry point
tests/              pytest suite for cleaning, checkpoints, and metrics
requirements.txt
requirements-dev.txt
pytest.ini
```

The notebook is the narrative record of the analysis and keeps its own copy of the code; `src/` is the reusable version with command line entry points.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data pipeline

1. Read the CSV with `;` separators and `,` decimals.
2. Drop `Date`, `Time`, the two empty unnamed columns, and the trailing blank rows.
3. Drop `NMHC(GT)`, which is missing for roughly 90% of the records.
4. Replace the `-200` placeholder with `NaN`.
5. Linearly interpolate gaps of up to 4 samples, then drop rows that are still incomplete — data quality over quantity.

That leaves **7,549 rows × 11 features**, with `C6H6(GT)` as the target. The data is split 80/20 before scaling, and `StandardScaler` is fit on the training split only, so the test set never influences preprocessing.

## Model

A fully connected regressor with the shape `11 → 256 → 256 → 256 → 1` and ReLU after each hidden layer, trained with MSE loss and Adam (`lr = 0.01`) for 200 full-batch epochs.

The shipped checkpoint `models/Regression Model.pth` is this configuration. In the notebook's experiment table the same shape is labelled "2 hidden layers", because its count excludes the input projection.

## Experiments

The notebook compares ten architectures, each trained once for 200 epochs. Losses are MSE on the training and test splits.

| Hidden layers | Units | Train loss | Test loss | R² |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 16 | 0.5522 | 0.6084 | 0.9893 |
| 1 | 32 | 0.2018 | 0.2503 | 0.9908 |
| 1 | 64 | 0.1765 | 0.2196 | 0.9961 |
| 1 | 128 | 0.0753 | 0.1112 | 0.9977 |
| 1 | 256 | 0.0424 | 0.0583 | 0.9991 |
| 2 | 16 | 0.6260 | 0.7220 | 0.9921 |
| 2 | 32 | 0.0964 | 0.1239 | 0.9962 |
| 2 | 64 | 0.1011 | 0.1195 | 0.9967 |
| 2 | 128 | 0.0628 | 0.0863 | 0.9986 |
| **2** | **256** | **0.0308** | **0.0471** | **0.9992** |

**Choice:** the 2 × 256 model had the lowest test loss, so it is the one that was saved. The 2 × 32 model is a much lighter alternative if inference cost matters more than the last few points of accuracy.

Training is stochastic, so these single runs will not reproduce exactly.

## Test results

Latest evaluation of the shipped checkpoint on the held-out split (1,510 rows):

```
$ python -m src.evaluate
Model: models/Regression Model.pth
Test rows: 1510 | Features: 11
MSE: 0.0486
RMSE: 0.2204
R^2: 0.9992
```

Residuals on the same split: mean absolute error **0.1259**, **99.3%** of predictions within ±1 mg/m³ of the actual value, largest error 3.08 on a target range of 0.20–50.70 mg/m³.

A freshly trained checkpoint (seed 42, 200 epochs) reaches test MSE 0.0779, RMSE 0.2791, and R² 0.9987. Re-run the commands below to refresh these numbers.

## Usage

Train a model (writes `models/regression_model.pth` by default, leaving the shipped checkpoint untouched):

```bash
python -m src.train
python -m src.train --epochs 200 --lr 0.01 --hidden-units 256 --hidden-layers 2
```

Evaluate a checkpoint:

```bash
python -m src.evaluate
python -m src.evaluate --model models/regression_model.pth
```

Both modules also run as plain scripts: `python src/train.py` and `python src/evaluate.py`.

| Flag | Default | Used by |
| :--- | :--- | :--- |
| `--data` | `Air Quality Data/AirQualityUCI.csv` | train, evaluate |
| `--epochs` | `200` | train |
| `--lr` | `0.01` | train |
| `--hidden-units` | `256` | train, evaluate |
| `--hidden-layers` | `2` | train, evaluate |
| `--test-size` | `0.2` | train, evaluate |
| `--seed` | `42` | train, evaluate |
| `--output` | `models/regression_model.pth` | train |
| `--model` | `models/Regression Model.pth` | evaluate |

`--hidden-layers` follows the notebook's convention, so the default of 2 matches the shipped checkpoint.

## Using the saved model

Checkpoints written by `src/train.py` store the weights, architecture, feature order, and scaler statistics, so raw sensor values can be predicted directly:

```python
import pandas as pd
from src.model import load_checkpoint

checkpoint = load_checkpoint("models/regression_model.pth")
row = pd.DataFrame([{
    "CO(GT)": 2.6, "PT08.S1(CO)": 1360.0, "PT08.S2(NMHC)": 1046.0,
    "NOx(GT)": 166.0, "PT08.S3(NOx)": 1056.0, "NO2(GT)": 113.0,
    "PT08.S4(NO2)": 1692.0, "PT08.S5(O3)": 1268.0, "T": 13.6,
    "RH": 48.9, "AH": 0.7578,
}])
benzene = checkpoint.predict_raw(row)  # shape (1,), mg/m³
```

`predict_raw` applies the stored scaler and reorders columns by the training feature order, so column order in the dataframe does not matter. A NumPy array is accepted too.

When only the model object is needed, use the loader directly:

```python
from src.model import load_model

model = load_model("models/Regression Model.pth")  # eval mode, CPU; architecture inferred
```

The shipped `models/Regression Model.pth` predates the bundle format and holds weights only, so it loads fine but `predict_raw` raises `ValueError` for it. Retrain to produce a checkpoint that carries the scaler.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

14 tests cover the cleaning rules (including long-gap handling), checkpoint round-trips with scaler and architecture inference, rejection of a mismatched architecture, and the metric formulas. The suite reads the committed dataset and checkpoint, so it runs offline.

## Notes and limitations

- Training is full-batch and runs on CPU; 200 epochs take well under a minute for this dataset.
- The shipped checkpoint stores weights only; checkpoints produced by `src/train.py` also carry the scaler, feature order, and architecture.
- Metrics vary between runs because weight initialization and training are stochastic.

## Dataset

UCI Machine Learning Repository, *Air Quality* (ID 360): <https://archive.ics.uci.edu/dataset/360/air+quality>

## Tech stack

Python, PyTorch, pandas, NumPy, scikit-learn, Matplotlib, Jupyter.
