from __future__ import annotations

import re

import pandas as pd

from src.dataset_builder.parse_wide_pihps import ParsedWidePIHPS

_ROMAN_NUMERAL_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)


def _is_top_level_no_value(value: object) -> bool:
    if value is None:
        return False
    token = str(value).strip()
    return bool(_ROMAN_NUMERAL_RE.fullmatch(token))


def filter_top_level_commodities(parsed: ParsedWidePIHPS) -> pd.DataFrame:
    """Keep only top-level commodity rows based on Roman numeral `No` values.

    Fails loudly if assumption is violated, to catch source schema drift early.
    """
    df = parsed.raw.copy()
    top_level_mask = df["No"].apply(_is_top_level_no_value)

    if not top_level_mask.any():
        raise ValueError(
            "No top-level rows found using Roman numeral rule on column 'No'. "
            "Raw row-structure assumption may have changed."
        )

    top_level_df = df.loc[top_level_mask, [*parsed.metadata_columns, *parsed.date_columns]].copy()
    top_level_df = top_level_df.reset_index(drop=True)
    return top_level_df
