from __future__ import annotations

from dataclasses import dataclass, field


FEATURE_COLUMNS: tuple[str, ...] = (
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
)

TARGET_COLUMNS: tuple[str, ...] = (
    "target_7d_inflation_pct",
    "target_30d_inflation_pct",
)


@dataclass(frozen=True)
class DiagnosticsThresholds:
    """Deterministic thresholds used by the readiness gate."""

    max_duplicate_pairs: int = 0
    max_null_price_ratio: float = 0.01
    max_non_positive_price_ratio: float = 0.0
    max_missing_day_ratio_warn: float = 0.35
    max_missing_day_ratio_fail: float = 0.6
    min_trainable_rows_7d: int = 50
    min_trainable_rows_30d: int = 30
    max_abs_price_change_pct_warn: float = 50.0
    max_abs_price_change_pct_fail: float = 150.0
    max_abs_target_pct_warn: float = 50.0
    max_abs_target_pct_fail: float = 150.0
    min_feature_non_null_rate_warn: float = 0.4
    min_target_non_null_rate_warn: float = 0.25


@dataclass(frozen=True)
class DiagnosticsConfig:
    thresholds: DiagnosticsThresholds = field(default_factory=DiagnosticsThresholds)
    allowed_commodities: tuple[str, ...] | None = None
    enable_outlier_flags: bool = True
