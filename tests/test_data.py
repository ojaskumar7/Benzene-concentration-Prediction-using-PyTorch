import numpy as np
import pandas as pd
import pytest

from src.data import (
    DROPPED_COLUMNS,
    DROPPED_FEATURE,
    MISSING_VALUE,
    TARGET,
    clean,
    load_clean,
)


def make_raw_frame() -> pd.DataFrame:
    """A tiny frame carrying the same artifacts as the real CSV."""
    return pd.DataFrame(
        {
            "Date": ["2004-03-10"] * 4,
            "Time": ["18.00.00", "19.00.00", "20.00.00", "21.00.00"],
            "CO(GT)": [2.6, MISSING_VALUE, 2.0, np.nan],
            "PT08.S1(CO)": [1360.0, 1292.0, 1402.0, np.nan],
            "NMHC(GT)": [MISSING_VALUE] * 4,
            "C6H6(GT)": [11.9, 9.4, 8.0, np.nan],
            "Unnamed: 15": [np.nan] * 4,
            "Unnamed: 16": [np.nan] * 4,
        }
    )


def make_gap_frame(gap_size: int) -> pd.DataFrame:
    n = gap_size + 4
    return pd.DataFrame(
        {
            "Date": ["2004-03-10"] * n,
            "Time": [f"{hour:02d}.00.00" for hour in range(n)],
            "CO(GT)": [2.0, 2.1] + [np.nan] * gap_size + [2.4, 2.5],
            "PT08.S1(CO)": np.linspace(1300.0, 1400.0, n),
            "NMHC(GT)": [MISSING_VALUE] * n,
            "C6H6(GT)": np.linspace(10.0, 11.0, n),
            "Unnamed: 15": [np.nan] * n,
            "Unnamed: 16": [np.nan] * n,
        }
    )


def test_clean_drops_unusable_columns_and_blank_rows():
    cleaned = clean(make_raw_frame())

    assert list(cleaned.columns) == ["CO(GT)", "PT08.S1(CO)", TARGET]
    assert len(cleaned) == 3
    for column in [*DROPPED_COLUMNS, DROPPED_FEATURE]:
        assert column not in cleaned.columns


def test_clean_replaces_placeholder_and_interpolates_short_gaps():
    cleaned = clean(make_raw_frame())

    assert (cleaned == MISSING_VALUE).sum().sum() == 0
    assert not cleaned.isna().any().any()
    # -200 becomes NaN and is filled from its neighbours 2.6 and 2.0.
    assert cleaned["CO(GT)"].tolist() == pytest.approx([2.6, 2.3, 2.0])


def test_clean_leaves_no_nan_even_for_long_gaps():
    cleaned = clean(make_gap_frame(gap_size=6))

    assert not cleaned.isna().any().any()
    # Gaps longer than the interpolation limit are dropped, not invented.
    assert len(cleaned) < len(make_gap_frame(gap_size=6))


def test_cleaned_dataset_has_expected_shape():
    df = load_clean()

    # 11 sensor features plus the benzene target.
    assert df.shape == (7549, 12)
    assert TARGET in df.columns
    assert not df.isna().any().any()
    assert (df == MISSING_VALUE).sum().sum() == 0
