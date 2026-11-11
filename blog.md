# Building a Virtual Benzene Sensor with PyTorch

Benzene (C6H6) is one of the more useful things to measure in urban air: it is a known carcinogen, and traffic is a major source. It is also awkward to measure. Reference analysers are expensive, need maintenance, and do not scale to dense city-wide deployments.

This project asks a narrower, more practical question: **can a small neural network estimate benzene concentration from sensors that are already cheap and widely deployed?**

The repository trains a PyTorch regressor on the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality), packages the notebook pipeline into reusable modules, and ships a checkpoint that can score raw sensor rows from the command line. This post walks through the data work, the model, the results, and — more importantly — what the headline metric does and does not prove.

## The dataset

The UCI Air Quality dataset contains hourly responses from an array of metal-oxide chemical sensors, plus reference measurements, recorded in an Italian city between March 2004 and April 2005. There are 9,357 valid hourly records.

The target is `C6H6(GT)`, the benzene concentration in mg/m³. The inputs are eleven features: carbon monoxide, nitrogen oxides, non-methane hydrocarbons, ozone, temperature, relative humidity, and absolute humidity, several of them measured through specific sensor channels such as `PT08.S2(NMHC)`.

Two quirks drive most of the cleaning code:

1. Missing measurements are encoded as the sentinel value `-200`, not as empty cells.
2. Some sensor channels are missing for most of the year. `NMHC(GT)` is absent in roughly 90% of the rows.

## Cleaning, and why the order matters

The pipeline in `src/data.py` follows the notebook step for step:

```python
df = df.drop(columns=["Date", "Time", "Unnamed: 15", "Unnamed: 16"]).dropna()
df = df.drop(columns=["NMHC(GT)"])
df = df.replace(-200, np.nan)
df = df.interpolate(method="linear", limit=4)
df = df.dropna()
```

A few decisions are worth calling out.

**Drop, then impute, then drop again.** The trailing blank rows are removed first, the hopeless `NMHC(GT)` column is discarded, and only then are `-200` values turned into `NaN`. Doing it in the other order would have meant interpolating across rows that contain no data at all.

**Interpolate, but only a little.** Hourly air quality moves smoothly, so filling an hour from its neighbours is reasonable. Filling a six-hour outage with a straight line is fabrication. The `limit=4` argument allows short gaps to be filled and leaves longer ones to be dropped by the final `dropna()`. That is a deliberate trade of quantity for quality: interpolation across a long outage would create confident-looking rows with no information in them.

After cleaning, the frame holds **7,549 rows and 11 features**, plus the target.

## Splitting before scaling

The train/test split happens *before* the scaler is fit, so the test set never influences the mean or variance used during preprocessing:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
```

This is the standard defence against preprocessing leakage. It matters more than usual here because the features have wildly different ranges — CO in single digits, sensor responses in the thousands — and a scaler fit on the full dataset would quietly leak test-set statistics into training.

## The model

The network is deliberately boring: fully connected layers with ReLU activations, one output unit, no regularisation tricks.

```python
class RegressionModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=256, num_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(input_features, hidden_units), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers.append(nn.Linear(hidden_units, output_features))
        self.layers_stacked = nn.Sequential(*layers)
```

The shipped checkpoint is `11 → 256 → 256 → 256 → 1`: an input projection, two hidden-to-hidden blocks, and a linear output. Training uses MSE loss, Adam with a learning rate of `0.01`, 200 full-batch epochs, and a fixed seed. On this dataset that takes seconds on a CPU.

Full batch is a conscious simplification. With ~6,000 training rows, minibatching buys little in speed and adds a knob to tune; the notebook used full batch, and the packaged code kept it.

## Experiments

The notebook compared ten architectures, one run each, 200 epochs apiece. A subset of the results:

| Hidden layers | Units | Train MSE | Test MSE | R² |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 64 | 0.1765 | 0.2196 | 0.9961 |
| 1 | 256 | 0.0424 | 0.0583 | 0.9991 |
| 2 | 128 | 0.0628 | 0.0863 | 0.9986 |
| **2** | **256** | **0.0308** | **0.0471** | **0.9992** |

The 2 × 256 configuration had the best test loss and was the one saved. The 2 × 32 model is a reasonable alternative if inference cost matters: it is far smaller and still lands around R² 0.996.

Training is stochastic. These are single runs, and rerunning them moves the numbers by a few percent.

## Results

Evaluating the shipped checkpoint on the held-out 20%:

```
Test rows: 1510 | Features: 11
MSE: 0.0486
RMSE: 0.2204
R^2: 0.9992
```

On the same split, mean absolute error is **0.126 mg/m³**, **99.3%** of predictions land within ±1 mg/m³ of the reference value, and the largest error is 3.08 mg/m³. For a target that ranges from 0.2 to 50.7 mg/m³, that is a good absolute result, not just a good ratio.

## What R² 0.999 doesn't tell you

A number that high invites suspicion, and it should. Three checks put it in context.

**First, is it just a linear relationship?** A plain `LinearRegression` on the same scaled features reaches MSE 1.4442 and R² 0.9753. The network cuts error by roughly 30×, so depth is doing real work.

**Second, does it survive time?** A random split puts neighbouring hours on both sides of the boundary, which flatters any model on time-series data. Retraining on the first 80% of hours and testing on the last 20% gives the network **MSE 0.1049, R² 0.9973** — a small degradation, and still far ahead of the linear model's 2.5021 / 0.9364 on the same split.

| Split | Model | MSE | R² |
| :--- | :--- | ---: | ---: |
| Random 80/20 | PyTorch 2 × 256 | **0.0486** | **0.9992** |
| Random 80/20 | Linear regression | 1.4442 | 0.9753 |
| Chronological | PyTorch 2 × 256 | **0.1049** | **0.9973** |
| Chronological | Linear regression | 2.5021 | 0.9364 |

**Third, where is the signal coming from?** `PT08.S2(NMHC)` correlates **0.982** with benzene, and a one-feature linear model on it alone reaches R² 0.96. Without that sensor, the remaining features still reach 0.95 linearly. So the "cheap virtual sensor" idea works — but the dominant predictor is a sensor that already targets the same hydrocarbon family. If the goal were replacing benzene measurement specifically, the interesting experiment is the no-`PT08.S2` model, not the eleven-feature one.

The honest summary: the model generalises better than a linear baseline, including out of time, but the problem is intrinsically easy in this dataset, and the metric alone does not establish that the model would hold up in a different city or season.

## Packaging the notebook

The original analysis lived in one long notebook. The `src/` package turns it into pieces that can be imported, tested, and run:

```
src/data.py       load + clean
src/features.py   split, scale, tensor conversion
src/model.py      architecture, checkpoint I/O, raw prediction
src/train.py      training CLI
src/evaluate.py   metrics CLI
src/predict.py    raw-CSV scoring CLI
```

The training script writes a checkpoint bundle rather than a bare state dict. Alongside the weights it stores the architecture, the feature order, and the scaler's mean and scale. Storing the scaler as tensors rather than a pickled scikit-learn object means the file still loads with `torch.load(..., weights_only=True)`, which avoids arbitrary unpickling:

```python
checkpoint = load_checkpoint("models/regression_model.pth")
benzene = checkpoint.predict_raw(raw_sensor_rows)   # unscaled values in, mg/m³ out
```

That closes a gap that bites most notebook-to-script conversions: a model file that is useless without reconstructing the exact preprocessing that produced it.

The suite has 18 pytest tests covering the cleaning rules, checkpoint round-trips, architecture inference, metric formulas, and the predict CLI. Checking the real dataset's cleaned shape was worth it: the test caught a miscount in my own assumption before it reached the docs.

## Reproducing it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

python -m src.train                 # 200 epochs, saves models/regression_model.pth
python -m src.evaluate              # scores the shipped checkpoint
python -m src.predict --input rows.csv --output predictions.csv
python -m pytest                    # 18 tests
```

## Lessons

1. **High R² is a claim, not a conclusion.** Compare against the simplest baseline and split by time before believing it.
2. **Sentinel values are silent.** `-200` looks like a number to every loader; treat it explicitly.
3. **Interpolation is a loan, not a gift.** Bound how far you are willing to fill, or you are training on invented data.
4. **Checkpoints should carry their preprocessing.** Weights without scaler and feature order are not a deployable model.
5. **The notebook is the lab, the package is the product.** Both are worth keeping, but only one should be imported.

The code, tests, and documentation are in the [Benzene concentration prediction repository](https://github.com/ojaskumar7/Benzene-concentration-Prediction-using-PyTorch).
