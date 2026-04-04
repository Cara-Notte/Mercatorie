from __future__ import annotations

import pandas as pd

from src.dataset_builder.time_horizon_config import DEFAULT_TARGET_CONFIGS, TargetHorizonConfig


def _compute_forward_target_for_commodity(
    commodity_df: pd.DataFrame,
    target_config: TargetHorizonConfig,
) -> pd.Series:
    base = commodity_df[["date", "price_idr"]].copy().sort_values("date")
    lookup = commodity_df[["date", "price_idr"]].copy().sort_values("date")

    base["target_date"] = base["date"] + pd.Timedelta(days=target_config.horizon_days)

    matched = pd.merge_asof(
        base,
        lookup.rename(columns={"date": "future_date", "price_idr": "future_price_idr"}),
        left_on="target_date",
        right_on="future_date",
        direction="forward",
        tolerance=pd.Timedelta(days=target_config.max_lookahead_days),
    )

    return ((matched["future_price_idr"] / matched["price_idr"]) - 1.0) * 100.0


def build_training_targets(
    dataset_df: pd.DataFrame,
    target_configs: dict[str, TargetHorizonConfig] | None = None,
) -> pd.DataFrame:
    """Attach calendar-based forward inflation targets per commodity.

    Rule:
    - For horizon H, seek first observed future price on or after `date + H`.
    - Accept match only within configured lookahead tolerance.
    - If no match exists in tolerance window, target remains NaN.
    """
    target_configs = target_configs or DEFAULT_TARGET_CONFIGS

    df = dataset_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["commodity", "date"]).reset_index(drop=True)

    output_parts: list[pd.DataFrame] = []
    for _, commodity_df in df.groupby("commodity", sort=False):
        part = commodity_df.copy().sort_values("date").reset_index(drop=True)
        for target_col, cfg in target_configs.items():
            part[target_col] = _compute_forward_target_for_commodity(part, cfg)
        output_parts.append(part)

    return pd.concat(output_parts, ignore_index=True).sort_values(["commodity", "date"]).reset_index(
        drop=True
    )
