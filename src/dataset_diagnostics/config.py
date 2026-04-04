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

FEATURE_TIERS: dict[str, tuple[str, ...]] = {
    "CRITICAL": (
        "price_lag_7d",
        "price_lag_30d",
        "rolling_mean_30d",
        "rolling_std_30d",
    ),
    "IMPORTANT": (
        "price_change_7d_pct",
        "price_change_30d_pct",
        "price_vs_ma30_pct",
        "volatility_30d_pct",
    ),
    "AUXILIARY": tuple(
        col
        for col in FEATURE_COLUMNS
        if col
        not in {
            "price_lag_7d",
            "price_lag_30d",
            "rolling_mean_30d",
            "rolling_std_30d",
            "price_change_7d_pct",
            "price_change_30d_pct",
            "price_vs_ma30_pct",
            "volatility_30d_pct",
        }
    ),
}

CLASS_LABELS: tuple[str, ...] = ("Deflation", "Stable", "Inflation")

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
    min_trainable_rows_7d_per_commodity: int = 20
    min_trainable_rows_30d_per_commodity: int = 12

    min_training_span_days_7d_per_commodity: int = 45
    min_training_span_days_30d_per_commodity: int = 60

    min_critical_feature_non_null_rate_warn: float = 0.55
    min_critical_feature_non_null_rate_fail: float = 0.35
    min_model_feature_usable_rate_warn: float = 0.4

    min_target_non_null_rate_warn: float = 0.25
    min_target_non_null_rate_per_commodity: float = 0.2

    min_class_ratio_warn: float = 0.05
    min_class_ratio_fail: float = 0.01
    min_classes_present_per_commodity: int = 2

    max_tolerance_match_ratio_warn: float = 0.5
    max_tolerance_match_ratio_fail: float = 0.8

    max_abs_price_change_pct_warn: float = 50.0
    max_abs_price_change_pct_fail: float = 150.0

    max_change_to_median_ratio_warn: float = 20.0
    max_change_to_median_ratio_fail: float = 50.0
    max_flatline_segment_days_warn: int = 21
    max_flatline_segment_days_fail: int = 45
    structural_jump_ratio_warn: float = 8.0
    structural_jump_ratio_fail: float = 15.0


@dataclass(frozen=True)
class DiagnosticsConfig:
    thresholds: DiagnosticsThresholds = field(default_factory=DiagnosticsThresholds)
    allowed_commodities: tuple[str, ...] | None = None
    enable_outlier_flags: bool = True
    stable_band_pct: float = 1.0