from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-6

CATEGORICAL_FEATURES = ["commodity"]
BINARY_FEATURES = ["is_observed_source", "is_month_start", "is_month_end", "high_vol_regime"]
NUMERIC_FEATURES = [
    "year",
    "price_idr",
    "price_lag_1d",
    "price_lag_7d",
    "price_lag_30d",
    "price_change_1d_pct",
    "price_change_7d_pct",
    "price_change_30d_pct",
    "rolling_mean_7d",
    "rolling_mean_30d",
    "rolling_std_30d",
    "price_vs_ma30_pct",
    "volatility_30d_pct",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "dow_sin",
    "dow_cos",
    "price_zscore_30d",
    "ma_gap_7_30_pct",
    "lag_gap_1_7_pct",
    "lag_gap_7_30_pct",
    "momentum_gap_7_30_pct",
    "volatility_x_momentum",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

RAW_REQUIRED_COLUMNS = [
    "date",
    "commodity",
    "is_observed_source",
    "is_month_start",
    "is_month_end",
    "year",
    "month",
    "week_of_year",
    "day_of_week",
    "price_idr",
    "price_lag_1d",
    "price_lag_7d",
    "price_lag_30d",
    "price_change_1d_pct",
    "price_change_7d_pct",
    "price_change_30d_pct",
    "rolling_mean_7d",
    "rolling_mean_30d",
    "rolling_std_30d",
    "price_vs_ma30_pct",
    "volatility_30d_pct",
]

TARGET_VALUE_COLUMNS = {
    7: "target_7d_inflation_pct",
    30: "target_30d_inflation_pct",
}
TARGET_CLASS_COLUMNS = {
    7: "target_7d_class",
    30: "target_30d_class",
}


def _make_inflation_class(value: float, stable_band: float = 1.0) -> str:
    if value > stable_band:
        return "Inflation"
    if value < -stable_band:
        return "Deflation"
    return "Stable"


def validate_raw_columns(df: pd.DataFrame) -> None:
    """Validate the feature-ready raw input contract.

    Training and inference currently require feature-ready input where upstream columns
    (lags, rolling means/std, pct changes) already exist.
    """
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns for feature engineering: {missing}")


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build shared engineered features for both training and inference.

    Upstream responsibility is explicit: callers must provide feature-ready raw columns
    such as price lags, rolling statistics, and percentage-change columns.
    """
    validate_raw_columns(df)

    engineered = df.copy().sort_values(["commodity", "date"]).reset_index(drop=True)

    # INSERT NOTEBOOK-DERIVED UPSTREAM FEATURE LOGIC HERE

    engineered["month_sin"] = np.sin(2 * np.pi * engineered["month"] / 12)
    engineered["month_cos"] = np.cos(2 * np.pi * engineered["month"] / 12)
    engineered["week_sin"] = np.sin(2 * np.pi * engineered["week_of_year"] / 52)
    engineered["week_cos"] = np.cos(2 * np.pi * engineered["week_of_year"] / 52)
    engineered["dow_sin"] = np.sin(2 * np.pi * engineered["day_of_week"] / 7)
    engineered["dow_cos"] = np.cos(2 * np.pi * engineered["day_of_week"] / 7)

    engineered["price_zscore_30d"] = (
        (engineered["price_idr"] - engineered["rolling_mean_30d"])
        / (engineered["rolling_std_30d"] + EPSILON)
    )
    engineered["ma_gap_7_30_pct"] = (
        engineered["rolling_mean_7d"] / (engineered["rolling_mean_30d"] + EPSILON) - 1
    ) * 100
    engineered["lag_gap_1_7_pct"] = (
        engineered["price_lag_1d"] / (engineered["price_lag_7d"] + EPSILON) - 1
    ) * 100
    engineered["lag_gap_7_30_pct"] = (
        engineered["price_lag_7d"] / (engineered["price_lag_30d"] + EPSILON) - 1
    ) * 100
    engineered["momentum_gap_7_30_pct"] = (
        engineered["price_change_7d_pct"] - engineered["price_change_30d_pct"]
    )
    engineered["volatility_x_momentum"] = (
        engineered["volatility_30d_pct"] * engineered["price_change_7d_pct"]
    )

    q75 = engineered.groupby("commodity")["volatility_30d_pct"].transform(lambda series: series.quantile(0.75))
    engineered["high_vol_regime"] = (engineered["volatility_30d_pct"] >= q75).astype(int)

    LOGGER.info("Base feature engineering complete with %s rows.", len(engineered))
    return engineered


def attach_training_targets(df: pd.DataFrame, stable_band: float = 1.0) -> pd.DataFrame:
    """Attach training-only target classes for all supported horizons."""
    target_df = df.copy()
    for horizon, target_value_col in TARGET_VALUE_COLUMNS.items():
        if target_value_col not in target_df.columns:
            raise ValueError(
                f"Training input missing required target value column '{target_value_col}' for {horizon}d horizon"
            )
        target_df[TARGET_CLASS_COLUMNS[horizon]] = target_df[target_value_col].apply(
            lambda v: _make_inflation_class(v, stable_band=stable_band)
        )
    return target_df
