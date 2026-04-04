from __future__ import annotations

import pandas as pd

from src.dataset_builder.commodity_config import ALLOWED_CANONICAL_COMMODITIES

REQUIRED_CANONICAL_COLUMNS = ["date", "commodity", "price_idr"]


def validate_canonical_dataset(
    canonical_df: pd.DataFrame,
    require_all_configured_commodities: bool = True,
) -> None:
    """Validate canonical PIHPS long dataset and fail loudly on schema issues."""
    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in canonical_df.columns]
    if missing:
        raise ValueError(f"Canonical dataset missing required columns: {missing}")

    parsed_dates = pd.to_datetime(canonical_df["date"], errors="coerce")
    if parsed_dates.isna().any():
        bad_count = int(parsed_dates.isna().sum())
        raise ValueError(f"Canonical dataset has {bad_count} rows with unparsable 'date'")

    commodity_values = set(canonical_df["commodity"].dropna().astype(str).unique())
    unexpected = commodity_values - ALLOWED_CANONICAL_COMMODITIES
    if unexpected:
        raise ValueError(f"Unexpected commodities in canonical dataset: {sorted(unexpected)}")

    if require_all_configured_commodities:
        missing_commodities = ALLOWED_CANONICAL_COMMODITIES - commodity_values
        if missing_commodities:
            raise ValueError(
                f"Configured commodities missing from canonical dataset: {sorted(missing_commodities)}"
            )

    numeric_price = pd.to_numeric(canonical_df["price_idr"], errors="coerce")
    non_null_original = canonical_df["price_idr"].notna()
    invalid_numeric = non_null_original & numeric_price.isna()
    if invalid_numeric.any():
        raise ValueError(
            f"Canonical dataset has non-numeric price_idr values at rows: "
            f"{canonical_df.index[invalid_numeric].tolist()[:10]}"
        )

    dup_mask = canonical_df.duplicated(subset=["date", "commodity"], keep=False)
    if dup_mask.any():
        dup_preview = canonical_df.loc[dup_mask, ["date", "commodity"]].head(10)
        raise ValueError(
            "Duplicate (date, commodity) pairs detected in canonical dataset. "
            f"Examples: {dup_preview.to_dict(orient='records')}"
        )
