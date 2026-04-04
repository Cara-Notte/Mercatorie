from __future__ import annotations

import pandas as pd
import pytest

from src.dataset_builder.build_feature_ready_dataset import build_feature_ready_dataset
from src.dataset_builder.build_training_targets import build_training_targets
from src.dataset_builder.filter_top_level_commodities import filter_top_level_commodities
from src.dataset_builder.normalize_prices_and_names import normalize_prices_and_names
from src.dataset_builder.parse_wide_pihps import parse_wide_pihps
from src.dataset_builder.time_horizon_config import (
    LagHorizonConfig,
    TargetHorizonConfig,
)
from src.dataset_builder.validate_canonical_dataset import validate_canonical_dataset
from src.training.feature_engineering import RAW_REQUIRED_COLUMNS


def _wide_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "No": ["I", "1", "II", "1", "III", "1", "IV", "V", "VI", "VII"],
            "Komoditas (Rp)": [
                "Beras",
                "Beras Kualitas Medium I",
                "Daging Ayam",
                "Daging Ayam Ras Segar",
                "Telur Ayam",
                "Telur Ayam Ras Segar",
                "Cabai Merah",
                "Bawang Merah",
                "Minyak Goreng",
                "Gula Pasir",
            ],
            "31/ 03/ 2026": [
                "16,100",
                "15,900",
                "45,000",
                "45,000",
                "28,000",
                "28,000",
                "70,100",
                "38,500",
                "18,000",
                "17,000",
            ],
            "01/ 04/ 2026": [
                "16,150",
                "15,900",
                "44,900",
                "44,900",
                "28,100",
                "28,100",
                "70,200",
                "38,900",
                "17,900",
                "17,200",
            ],
        }
    )


def test_parse_wide_pihps_parses_messy_date_headers():
    parsed = parse_wide_pihps(_wide_sample_df())
    assert set(parsed.metadata_columns) == {"No", "Komoditas (Rp)"}
    assert parsed.date_columns == ["31/ 03/ 2026", "01/ 04/ 2026"]
    assert parsed.date_column_mapping["31/ 03/ 2026"] == pd.Timestamp("2026-03-31")


def test_parse_wide_pihps_detects_duplicate_logical_dates():
    raw = pd.DataFrame(
        {
            "No": ["I"],
            "Komoditas (Rp)": ["Beras"],
            "01/ 04/ 2026": ["16,100"],
            "01/04/2026": ["16,200"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate logical dates"):
        parse_wide_pihps(raw)


def test_filter_top_level_commodities_excludes_subtypes():
    parsed = parse_wide_pihps(_wide_sample_df())
    top = filter_top_level_commodities(parsed)
    assert "Beras Kualitas Medium I" not in set(top["Komoditas (Rp)"])
    assert set(top["Komoditas (Rp)"]) == {
        "Beras",
        "Daging Ayam",
        "Telur Ayam",
        "Cabai Merah",
        "Bawang Merah",
        "Minyak Goreng",
        "Gula Pasir",
    }


def test_normalize_prices_and_names_converts_price_strings():
    parsed = parse_wide_pihps(_wide_sample_df())
    top = filter_top_level_commodities(parsed)
    canonical = normalize_prices_and_names(top, parsed)

    assert canonical.columns.tolist() == ["date", "commodity", "price_idr"]
    value = canonical.loc[
        (canonical["commodity"] == "beras")
        & (canonical["date"] == pd.Timestamp("2026-03-31")),
        "price_idr",
    ].iloc[0]
    assert value == 16100.0


def test_validate_canonical_dataset_detects_duplicates():
    canonical = pd.DataFrame(
        {
            "date": ["2026-04-01", "2026-04-01"],
            "commodity": ["beras", "beras"],
            "price_idr": [16100.0, 16100.0],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_canonical_dataset(canonical, require_all_configured_commodities=False)


def test_validate_canonical_dataset_passes_for_expected_schema():
    parsed = parse_wide_pihps(_wide_sample_df())
    top = filter_top_level_commodities(parsed)
    canonical = normalize_prices_and_names(top, parsed)
    validate_canonical_dataset(canonical)


def test_feature_ready_lag_is_calendar_based_not_row_based():
    canonical = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-08",  # missing 2026-01-07
                ]
            ),
            "commodity": ["beras"] * 7,
            "price_idr": [100, 101, 102, 103, 104, 105, 108],
        }
    )
    feat = build_feature_ready_dataset(
        canonical,
        lag_configs={
            "price_lag_1d": LagHorizonConfig(1, 0),
            "price_lag_7d": LagHorizonConfig(7, 0),
            "price_lag_30d": LagHorizonConfig(30, 0),
        },
    )

    row = feat.loc[feat["date"] == pd.Timestamp("2026-01-08")].iloc[0]
    assert row["price_lag_1d"] != 105  # would be row-count shift(1)
    assert pd.isna(row["price_lag_1d"])  # exact 1-day prior (2026-01-07) missing
    assert row["price_lag_7d"] == 100  # exact calendar 7-day lag from 2026-01-01


def test_feature_ready_rolling_uses_calendar_window():
    canonical = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-10"]),
            "commodity": ["beras", "beras", "beras"],
            "price_idr": [100.0, 110.0, 300.0],
        }
    )
    feat = build_feature_ready_dataset(canonical)
    row = feat.loc[feat["date"] == pd.Timestamp("2026-01-10")].iloc[0]
    assert row["rolling_mean_7d"] == pytest.approx(300.0)


def test_feature_ready_missing_date_handling_keeps_nans_without_fabrication():
    canonical = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-10"]),
            "commodity": ["beras", "beras", "beras"],
            "price_idr": [100.0, 110.0, 120.0],
        }
    )
    feat = build_feature_ready_dataset(
        canonical,
        lag_configs={
            "price_lag_1d": LagHorizonConfig(1, 0),
            "price_lag_7d": LagHorizonConfig(7, 0),
            "price_lag_30d": LagHorizonConfig(30, 0),
        },
    )
    assert len(feat) == 3
    row_jan10 = feat.loc[feat["date"] == pd.Timestamp("2026-01-10")].iloc[0]
    assert row_jan10["price_lag_7d"] == 110.0
    assert feat.loc[feat["date"] == pd.Timestamp("2026-01-03"), "price_lag_7d"].isna().all()


def test_training_targets_with_missing_exact_date_uses_forward_tolerance():
    canonical = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-09", "2026-01-31", "2026-02-02"]),
            "commodity": ["beras", "beras", "beras", "beras"],
            "price_idr": [100.0, 112.0, 130.0, 132.0],
        }
    )
    feat = build_feature_ready_dataset(canonical)
    training = build_training_targets(
        feat,
        target_configs={
            "target_7d_inflation_pct": TargetHorizonConfig(7, 3),
            "target_30d_inflation_pct": TargetHorizonConfig(30, 3),
        },
    )

    first_row = training.loc[training["date"] == pd.Timestamp("2026-01-01")].iloc[0]
    assert first_row["target_7d_inflation_pct"] == pytest.approx(12.0)
    assert first_row["target_30d_inflation_pct"] == pytest.approx(30.0)

    second_row = training.loc[training["date"] == pd.Timestamp("2026-01-09")].iloc[0]
    assert pd.isna(second_row["target_7d_inflation_pct"])


def test_feature_ready_output_contains_raw_required_columns():
    canonical = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "commodity": ["beras"] * 40,
            "price_idr": [15000 + i for i in range(40)],
        }
    )
    feat = build_feature_ready_dataset(canonical)
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in feat.columns]
    assert missing == []


def test_filter_top_level_fails_loudly_if_roman_rule_breaks():
    bad_df = _wide_sample_df().copy()
    bad_df["No"] = ["1"] * len(bad_df)
    parsed = parse_wide_pihps(bad_df)
    with pytest.raises(ValueError, match="Roman numeral"):
        filter_top_level_commodities(parsed)
