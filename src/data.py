"""Loading and cleaning of the UCI Air Quality dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "Air Quality Data" / "AirQualityUCI.csv"

# -200 is the dataset's placeholder for a missing measurement.
MISSING_VALUE = -200
# Timestamps plus the two trailing empty columns.
DROPPED_COLUMNS = ["Date", "Time", "Unnamed: 15", "Unnamed: 16"]
# Roughly 90% of this column is missing, so it carries no usable signal.
DROPPED_FEATURE = "NMHC(GT)"
# Only short gaps are interpolated; longer ones are dropped instead of invented.
INTERPOLATION_LIMIT = 4
TARGET = "C6H6(GT)"


def load_raw(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, which uses ';' separators and ',' decimals."""
    return pd.read_csv(path, sep=";", decimal=",")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe, in the same order as the notebook.

    1. Drop timestamp and empty columns, then the trailing blank rows.
    2. Drop the almost entirely missing NMHC(GT) column.
    3. Treat the -200 placeholder as a missing value.
    4. Linearly interpolate gaps of up to INTERPOLATION_LIMIT samples.
    5. Drop rows that still contain missing values.
    """
    df = df.drop(columns=DROPPED_COLUMNS).dropna()
    df = df.drop(columns=DROPPED_FEATURE)
    df = df.replace(MISSING_VALUE, np.nan)
    df = df.interpolate(method="linear", limit=INTERPOLATION_LIMIT)
    return df.dropna()


def load_clean(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and clean the dataset in one step."""
    return clean(load_raw(path))
