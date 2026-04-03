from __future__ import annotations

import pandas as pd
import pytest

from src.training.feature_engineering import FEATURE_COLUMNS, attach_training_targets, build_base_features
from src.training.train import _assert_no_target_leakage


def test_base_features_and_inference_share_contract():
    df = pd.read_csv("data/raw/pihps_food_inflation_indonesia.csv").head(100)
    df["date"] = pd.to_datetime(df["date"])

    base_df = build_base_features(df)

    assert "target_7d_class" not in base_df.columns
    assert "target_30d_class" not in base_df.columns


def test_attach_training_targets_adds_horizon_targets_only_for_training():
    df = pd.read_csv("data/raw/pihps_food_inflation_indonesia.csv").head(100)
    df["date"] = pd.to_datetime(df["date"])

    base_df = build_base_features(df)
    training_df = attach_training_targets(base_df, stable_band=1.0)

    assert "target_7d_class" in training_df.columns
    assert "target_30d_class" in training_df.columns


def test_missing_raw_required_columns_raises_error():
    df = pd.read_csv("data/raw/pihps_food_inflation_indonesia.csv").head(5)
    df = df.drop(columns=["rolling_mean_30d"])
    df["date"] = pd.to_datetime(df["date"])

    with pytest.raises(ValueError, match="Missing required raw columns"):
        build_base_features(df)


def test_target_leakage_prevention_guard_raises_on_target_columns():
    leaked_features = pd.DataFrame(
        {
            "commodity": ["Rice"],
            "target_7d_class": ["Stable"],
        }
    )

    with pytest.raises(ValueError, match="Target leakage detected"):
        _assert_no_target_leakage(leaked_features)


def test_feature_set_no_longer_contains_batch_relative_high_vol_regime():
    assert "high_vol_regime" not in FEATURE_COLUMNS


def test_base_features_remain_consistent_when_input_row_repeated():
    df = pd.read_csv("data/raw/pihps_food_inflation_indonesia.csv").head(1)
    df["date"] = pd.to_datetime(df["date"])

    single_features = build_base_features(df)[FEATURE_COLUMNS].iloc[0]
    duplicated = pd.concat([df, df], ignore_index=True)
    duplicated_features = build_base_features(duplicated)[FEATURE_COLUMNS].iloc[0]

    pd.testing.assert_series_equal(single_features, duplicated_features, check_names=False)