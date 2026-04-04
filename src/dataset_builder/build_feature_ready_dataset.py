from __future__ import annotations

import pandas as pd

from src.dataset_builder.time_horizon_config import (
    DEFAULT_LAG_CONFIGS,
    DEFAULT_ROLLING_CONFIGS,
    LagHorizonConfig,
    RollingWindowConfig,
)


def _assert_no_duplicate_date_commodity(df: pd.DataFrame) -> None:
    dup_mask = df.duplicated(subset=["date", "commodity"], keep=False)
    if dup_mask.any():
        preview = df.loc[dup_mask, ["date", "commodity"]].head(10).to_dict(orient="records")
        raise ValueError(
            "Duplicate (date, commodity) pairs found before feature generation. "
            f"Examples: {preview}"
        )


def _apply_calendar_lag(
    commodity_df: pd.DataFrame,
    lag_config: LagHorizonConfig,
) -> pd.Series:
    current = commodity_df[["date", "price_idr"]].copy()
    current = current.sort_values("date")
    lookup = commodity_df[["date", "price_idr"]].copy().sort_values("date")

    current["lag_target_date"] = current["date"] - pd.Timedelta(days=lag_config.horizon_days)

    tolerance = pd.Timedelta(days=lag_config.max_lookback_days)
    merged = pd.merge_asof(
        current,
        lookup.rename(columns={"date": "lookup_date", "price_idr": "lag_price"}),
        left_on="lag_target_date",
        right_on="lookup_date",
        direction="backward",
        tolerance=tolerance,
    )

    # exact-match-only when max_lookback_days = 0 (merge_asof tolerance=0 handles this).
    return merged["lag_price"]


def _apply_calendar_rolling(
    commodity_df: pd.DataFrame,
    rolling_config: RollingWindowConfig,
    agg: str,
) -> pd.Series:
    temp = commodity_df[["date", "price_idr"]].copy().sort_values("date")
    temp = temp.set_index("date")
    rolled = temp["price_idr"].rolling(
        window=f"{rolling_config.window_days}D",
        min_periods=rolling_config.min_observations,
        closed="both",
    )
    if agg == "mean":
        out = rolled.mean()
    elif agg == "std":
        out = rolled.std(ddof=1)
    else:
        raise ValueError(f"Unsupported rolling aggregation: {agg}")
    return out.reset_index(drop=True)


def build_feature_ready_dataset(
    canonical_df: pd.DataFrame,
    lag_configs: dict[str, LagHorizonConfig] | None = None,
    rolling_configs: dict[str, RollingWindowConfig] | None = None,
) -> pd.DataFrame:
    """Build feature-ready tabular data from canonical long prices.

    - Lags are calendar-aware and matched per commodity using horizon/tolerance config.
    - Rolling stats are trailing calendar windows (7D/30D), not fixed row-count windows.
    - Missing dates are not fabricated; unavailable lag/rolling values remain NaN.
    """
    lag_configs = lag_configs or DEFAULT_LAG_CONFIGS
    rolling_configs = rolling_configs or DEFAULT_ROLLING_CONFIGS

    df = canonical_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

    _assert_no_duplicate_date_commodity(df)

    df["is_observed_source"] = 1
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek

    output_parts: list[pd.DataFrame] = []
    for _, commodity_df in df.groupby("commodity", sort=False):
        part = commodity_df.copy().sort_values("date").reset_index(drop=True)

        part["price_lag_1d"] = _apply_calendar_lag(part, lag_configs["price_lag_1d"])
        part["price_lag_7d"] = _apply_calendar_lag(part, lag_configs["price_lag_7d"])
        part["price_lag_30d"] = _apply_calendar_lag(part, lag_configs["price_lag_30d"])

        part["price_change_1d_pct"] = (part["price_idr"] / part["price_lag_1d"] - 1.0) * 100.0
        part["price_change_7d_pct"] = (part["price_idr"] / part["price_lag_7d"] - 1.0) * 100.0
        part["price_change_30d_pct"] = (part["price_idr"] / part["price_lag_30d"] - 1.0) * 100.0

        part["rolling_mean_7d"] = _apply_calendar_rolling(
            part, rolling_configs["rolling_mean_7d"], "mean"
        )
        part["rolling_mean_30d"] = _apply_calendar_rolling(
            part, rolling_configs["rolling_mean_30d"], "mean"
        )
        part["rolling_std_30d"] = _apply_calendar_rolling(
            part, rolling_configs["rolling_std_30d"], "std"
        )

        part["price_vs_ma30_pct"] = (part["price_idr"] / part["rolling_mean_30d"] - 1.0) * 100.0
        part["volatility_30d_pct"] = (part["rolling_std_30d"] / part["rolling_mean_30d"]) * 100.0

        output_parts.append(part)

    return pd.concat(output_parts, ignore_index=True).sort_values(["commodity", "date"]).reset_index(
        drop=True
    )
