"""Feature/target split, scaling and tensor conversion."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import TARGET


@dataclass(frozen=True)
class Datasets:
    """Tensors for training and testing, plus the fitted scaler and feature order."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    feature_names: list[str]
    scaler: StandardScaler


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Use benzene as the target and every other column as a feature."""
    return df.drop(columns=[TARGET]), df[TARGET]


def build_datasets(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Datasets:
    """Split, scale and convert the cleaned dataframe into PyTorch tensors.

    The split happens before scaling so the test set never influences the
    scaler.
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return Datasets(
        X_train=feature_tensor(X_train_scaled),
        y_train=target_tensor(y_train),
        X_test=feature_tensor(X_test_scaled),
        y_test=target_tensor(y_test),
        feature_names=list(X.columns),
        scaler=scaler,
    )


def feature_tensor(values) -> torch.Tensor:
    """Convert feature values into a float32 tensor."""
    return torch.tensor(np.asarray(values), dtype=torch.float32)


def target_tensor(values) -> torch.Tensor:
    """Convert target values into a float32 tensor shaped [n_samples, 1]."""
    return feature_tensor(values).reshape(-1, 1)
