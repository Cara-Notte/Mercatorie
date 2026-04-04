from __future__ import annotations

import pandas as pd
import pytest

from src.dataset_diagnostics import DiagnosticsConfig, DiagnosticsThresholds, ReadinessStatus
from src.dataset_diagnostics.diagnostics import _gap_diagnostics, run_dataset_diagnostics


def _canonical_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-06",
                ]
            ),
            "commodity": ["beras"] * 5 + ["cabai_merah"] * 5,
            "price_idr": [100, 101, 102, 103, 104, 200, 198, 205, 210, 212],
        }
    )


def _training_ready_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-06",
                ]
            ),
            "commodity": ["beras"] * 5 + ["cabai_merah"] * 5,
            "price_idr": [100, 101, 102, 103, 104, 200, 198, 205, 210, 212],
            "price_lag_1d": [None, 100, 101, None, 103, None, 200, 198, 205, None],
            "price_lag_7d": [None] * 10,
            "price_lag_30d": [None] * 10,
            "price_change_1d_pct": [None, 1.0, 0.99, None, 0.97, None, -1.0, 3.53, 2.43, None],
            "price_change_7d_pct": [None] * 10,
            "price_change_30d_pct": [None] * 10,
            "rolling_mean_7d": [100, 100.5, 101, 101.5, 102, 200, 199, 201, 203, 205],
            "rolling_mean_30d": [100, 100.5, 101, 101.5, 102, 200, 199, 201, 203, 205],
            "rolling_std_30d": [None, 0.7, 1.0, 1.2, 1.3, None, 1.4, 2.0, 2.2, 2.4],
            "price_vs_ma30_pct": [0, 0.49, 0.99, 1.48, 1.96, 0, -0.5, 1.99, 3.45, 3.41],
            "volatility_30d_pct": [None, 0.7, 0.99, 1.18, 1.27, None, 0.7, 0.99, 1.08, 1.17],
            "target_7d_inflation_pct": [1.5, 1.4, 1.3, None, None, 2.0, 1.9, 1.8, None, None],
            "target_30d_inflation_pct": [4.5, 4.3, None, None, None, 6.0, 5.9, None, None, None],
        }
    )


def test_duplicate_detection_in_canonical_diagnostics():
    canonical = _canonical_sample()
    dup_row = canonical.iloc[[0]].copy()
    canonical = pd.concat([canonical, dup_row], ignore_index=True)

    result = run_dataset_diagnostics(canonical, _training_ready_sample())
    assert result.canonical["duplicate_date_commodity_count"] == 1


def test_gap_statistics_include_required_quantiles_and_buckets():
    gaps = _gap_diagnostics(_canonical_sample())["by_commodity"]
    beras = next(row for row in gaps if row["commodity"] == "beras")

    assert beras["span_days"] == 7
    assert beras["observation_count"] == 5
    assert beras["gap_median_days"] == 1.0
    assert beras["gap_p90_days"] == pytest.approx(2.4)
    assert beras["gap_max_days"] == 3
    assert beras["gap_gt_1d_count"] == 1


def test_feature_coverage_summary_tracks_non_null_rates_and_first_valid_date():
    result = run_dataset_diagnostics(_canonical_sample(), _training_ready_sample())

    rows = result.feature_coverage["feature_non_null_rates_by_commodity"]
    beras = next(row for row in rows if row["commodity"] == "beras")
    assert beras["price_lag_1d_non_null_rate"] == 0.6
    assert beras["fully_usable_feature_rows"] == 0

    first_valid = result.feature_coverage["feature_first_valid_date_by_commodity"]
    rec = next(
        row
        for row in first_valid
        if row["commodity"] == "beras" and row["feature"] == "price_lag_1d"
    )
    assert rec["first_valid_date"] == "2026-01-02"


def test_readiness_gate_logic_emits_fail_on_hard_threshold_breach():
    strict = DiagnosticsConfig(
        thresholds=DiagnosticsThresholds(
            min_trainable_rows_7d=100,
            min_trainable_rows_30d=100,
        )
    )
    result = run_dataset_diagnostics(_canonical_sample(), _training_ready_sample(), strict)

    assert result.readiness_status == ReadinessStatus.FAIL
    assert any("Insufficient total trainable rows for 7d horizon." in msg for msg in result.failures)
