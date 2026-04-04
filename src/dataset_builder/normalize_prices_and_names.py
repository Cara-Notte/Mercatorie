from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.dataset_builder.commodity_config import RAW_TO_CANONICAL
from src.dataset_builder.parse_wide_pihps import ParsedWidePIHPS


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _parse_price_to_float(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = _normalize_text(value)
    if text in {"", "-", "na", "n/a", "None"}:
        return np.nan
    digits_only = re.sub(r"[^0-9]", "", text)
    if digits_only == "":
        return np.nan
    return float(digits_only)


def normalize_prices_and_names(top_level_df: pd.DataFrame, parsed: ParsedWidePIHPS) -> pd.DataFrame:
    """Normalize commodity names and prices into canonical long-format dataset.

    Output columns are exactly: date, commodity, price_idr.
    """
    working = top_level_df.copy()
    working["commodity_raw"] = working["Komoditas (Rp)"].apply(_normalize_text)

    mapped = working["commodity_raw"].map(RAW_TO_CANONICAL)
    working = working.loc[mapped.notna()].copy()
    working["commodity"] = mapped[mapped.notna()]

    if working.empty:
        raise ValueError("No configured top-level commodities found after normalization")

    long_df = working.melt(
        id_vars=["commodity"],
        value_vars=parsed.date_columns,
        var_name="date_col_raw",
        value_name="price_raw",
    )
    long_df["date"] = long_df["date_col_raw"].map(parsed.date_column_mapping)
    long_df["price_idr"] = long_df["price_raw"].apply(_parse_price_to_float)

    canonical = long_df[["date", "commodity", "price_idr"]].copy()
    canonical = canonical.sort_values(["commodity", "date"]).reset_index(drop=True)
    return canonical
