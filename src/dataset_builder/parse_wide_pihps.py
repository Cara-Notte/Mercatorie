from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

REQUIRED_METADATA_COLUMNS = ("No", "Komoditas (Rp)")
_DATE_HEADER_RE = re.compile(r"^\s*(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})\s*$")


@dataclass(frozen=True)
class ParsedWidePIHPS:
    """Parsed view of a wide-format raw PIHPS dataframe."""

    raw: pd.DataFrame
    metadata_columns: list[str]
    date_columns: list[str]
    date_column_mapping: dict[str, pd.Timestamp]


def normalize_date_header(header: object) -> str:
    """Normalize PIHPS date headers like '31/ 03/ 2026' -> '31/03/2026'."""
    if header is None:
        return ""
    text = str(header).strip()
    m = _DATE_HEADER_RE.match(text)
    if not m:
        return text
    day, month, year = m.groups()
    return f"{int(day):02d}/{int(month):02d}/{year}"


def parse_wide_pihps(raw_df: pd.DataFrame) -> ParsedWidePIHPS:
    """Identify metadata/date columns and parse date headers safely.

    This function only handles wide-format source structure parsing.
    It intentionally does not compute model features.
    """
    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Raw PIHPS schema drift: missing metadata columns {missing}")

    metadata_columns = [c for c in REQUIRED_METADATA_COLUMNS]
    date_columns: list[str] = []
    mapping: dict[str, pd.Timestamp] = {}

    for col in raw_df.columns:
        if col in metadata_columns:
            continue
        normalized = normalize_date_header(col)
        ts = pd.to_datetime(normalized, format="%d/%m/%Y", errors="coerce")
        if pd.isna(ts):
            continue
        date_columns.append(col)
        mapping[col] = ts

    if not date_columns:
        raise ValueError("No parseable date columns were found in raw PIHPS wide table")

    parsed_dates = pd.Series(list(mapping.values()), dtype="datetime64[ns]")
    dup_mask = parsed_dates.duplicated(keep=False)
    if dup_mask.any():
        duplicate_columns = [
            col for col, dt in mapping.items() if dt in set(parsed_dates.loc[dup_mask].tolist())
        ]
        raise ValueError(
            "Duplicate logical dates detected after date-header normalization/parsing. "
            f"Columns involved: {duplicate_columns}"
        )

    return ParsedWidePIHPS(
        raw=raw_df,
        metadata_columns=metadata_columns,
        date_columns=date_columns,
        date_column_mapping=mapping,
    )
