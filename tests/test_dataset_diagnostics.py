from __future__ import annotations

import pandas as pd
import pytest

from src.dataset_diagnostics.config import DiagnosticsConfig, DiagnosticsThresholds
from src.dataset_diagnostics.diagnostics import ReadinessStatus, _gap_diagnostics, run_dataset_diagnostics


FEATURE_COLS = [
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


def _canonical_sample() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    rows = []
    for commodity, base in [("beras", 100), ("cabai_merah", 220)]:
        for idx, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "commodity": commodity,
                    "price_idr": base + idx,
                }
            )
    return pd.DataFrame(rows)



def _training_ready_sample() -> pd.DataFrame:
    df = _canonical_sample().sort_values(["commodity", "date"]).reset_index(drop=True)

    for col in FEATURE_COLS:
        df[col] = 1.0

    # introduce sparsity patterns for tier and strict/model distinction
    df.loc[df["date"] <= pd.Timestamp("2026-01-10"), "price_lag_7d"] = None
    df.loc[df["date"] <= pd.Timestamp("2026-01-15"), "price_lag_30d"] = None
    df.loc[df["date"] <= pd.Timestamp("2026-01-05"), "rolling_std_30d"] = None
    df.loc[df["date"] <= pd.Timestamp("2026-01-20"), "price_change_1d_pct"] = None

    # 7d target cycles through classes; 30d starts later and narrower
    class_cycle_7d = [2.5, 0.1, -2.5] * 100
    class_cycle_30d = [1.8, -1.8, 0.0] * 100
    df["target_7d_inflation_pct"] = class_cycle_7d[: len(df)]
    df["target_30d_inflation_pct"] = class_cycle_30d[: len(df)]
    df.loc[df["date"] <= pd.Timestamp("2026-01-20"), "target_30d_inflation_pct"] = None

    return df


def test_duplicate_detection_in_canonical_diagnostics():
    canonical = _canonical_sample()
    dup_row = canonical.iloc[[0]].copy()
    canonical = pd.concat([canonical, dup_row], ignore_index=True)

    result = run_dataset_diagnostics(canonical, _training_ready_sample())
    assert result.canonical["duplicate_date_commodity_count"] == 1


def test_gap_statistics_include_required_quantiles_and_buckets():
    canonical = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07"]),
            "commodity": ["beras"] * 5,
            "price_idr": [100, 101, 102, 103, 104],
        }
    )
    gaps = _gap_diagnostics(canonical)["by_commodity"]
    beras = next(row for row in gaps if row["commodity"] == "beras")

    assert beras["span_days"] == 7
    assert beras["observation_count"] == 5
    assert beras["gap_median_days"] == 1.0
    assert beras["gap_p90_days"] == pytest.approx(2.4)
    assert beras["gap_max_days"] == 3
    assert beras["gap_gt_1d_count"] == 1


def test_feature_tier_and_strict_vs_model_usable_rows_are_reported():
    result = run_dataset_diagnostics(_canonical_sample(), _training_ready_sample())

    rows = result.feature_coverage["feature_non_null_rates_by_commodity"]
    beras = next(row for row in rows if row["commodity"] == "beras")
    assert beras["strict_feature_usable_rows"] < beras["model_feature_usable_rows"]
    assert beras["critical_all_non_null_rate"] <= 1.0
    assert "CRITICAL" in result.feature_coverage["feature_tiers"]


def test_training_window_and_class_balance_are_computed():
    result = run_dataset_diagnostics(_canonical_sample(), _training_ready_sample())

    windows = result.target_coverage["training_windows"]
    beras_7d = next(r for r in windows if r["commodity"] == "beras" and r["horizon"] == "7d")
    assert beras_7d["usable_training_row_count"] > 0
    assert beras_7d["usable_training_span_days"] > 30

    class_balance_overall_7d = result.class_balance["overall"]["7d"]
    assert class_balance_overall_7d["class_counts"]["Inflation"] > 0
    assert class_balance_overall_7d["class_counts"]["Stable"] > 0
    assert class_balance_overall_7d["class_counts"]["Deflation"] > 0


def test_match_quality_summary_contains_exact_tolerance_and_missing():
    result = run_dataset_diagnostics(_canonical_sample(), _training_ready_sample())

    overall = result.lag_target_match_quality["overall"]
    lag_7 = next(r for r in overall if r["field"] == "price_lag_7d")
    assert lag_7["exact_count"] >= 0
    assert lag_7["tolerance_count"] >= 0
    assert lag_7["no_match_count"] >= 0
    assert pytest.approx(lag_7["exact_ratio"] + lag_7["tolerance_ratio"] + lag_7["no_match_ratio"], 1e-6) == 1.0


def test_readiness_gate_logic_fails_for_per_commodity_shortfalls_and_class_imbalance():
    imbalanced = _training_ready_sample()
    imbalanced.loc[imbalanced["commodity"] == "cabai_merah", "target_7d_inflation_pct"] = 5.0
    imbalanced.loc[imbalanced["commodity"] == "cabai_merah", "target_30d_inflation_pct"] = 5.0

    strict = DiagnosticsConfig(
        thresholds=DiagnosticsThresholds(
            min_trainable_rows_7d=50,
            min_trainable_rows_30d=40,
            min_trainable_rows_7d_per_commodity=35,
            min_trainable_rows_30d_per_commodity=30,
            min_training_span_days_7d_per_commodity=80,
            min_training_span_days_30d_per_commodity=80,
            min_class_ratio_warn=0.1,
            min_class_ratio_fail=0.05,
            max_tolerance_match_ratio_warn=0.2,
            max_tolerance_match_ratio_fail=0.6,
        )
    )
    result = run_dataset_diagnostics(_canonical_sample(), imbalanced, strict)

    assert result.readiness_status == ReadinessStatus.FAIL
    assert any("class support too narrow" in msg for msg in result.failures)
