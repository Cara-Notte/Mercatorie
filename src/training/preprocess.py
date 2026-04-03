from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.training.feature_engineering import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


@dataclass
class TrainingSplit:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    train_dates: pd.Series
    valid_dates: pd.Series
    test_dates: pd.Series


def time_split(df: pd.DataFrame, target_column: str = "target_7d_class") -> TrainingSplit:
    model_df = df.copy().sort_values(["date", "commodity"]).reset_index(drop=True)

    if target_column not in model_df.columns:
        raise ValueError(f"Missing target column '{target_column}' in training dataset")

    X = model_df[FEATURE_COLUMNS]
    y = model_df[target_column]
    dates = model_df["date"]

    unique_dates = np.array(sorted(dates.unique()))
    train_cutoff = unique_dates[int(len(unique_dates) * 0.70)]
    valid_cutoff = unique_dates[int(len(unique_dates) * 0.85)]

    train_mask = dates <= train_cutoff
    valid_mask = (dates > train_cutoff) & (dates <= valid_cutoff)
    test_mask = dates > valid_cutoff

    return TrainingSplit(
        X_train=X.loc[train_mask].copy(),
        X_valid=X.loc[valid_mask].copy(),
        X_test=X.loc[test_mask].copy(),
        y_train=y.loc[train_mask].copy(),
        y_valid=y.loc[valid_mask].copy(),
        y_test=y.loc[test_mask].copy(),
        train_dates=dates.loc[train_mask].copy(),
        valid_dates=dates.loc[valid_mask].copy(),
        test_dates=dates.loc[test_mask].copy(),
    )


def build_tree_preprocessor() -> ColumnTransformer:
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    binary_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        [
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ]
    )
