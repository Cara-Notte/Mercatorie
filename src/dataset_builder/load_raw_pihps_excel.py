from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_pihps_excel(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Load raw PIHPS Excel without coercing all columns to inferred numeric/datetime.

    Notes:
    - Preserves source headers and cell values as object dtype where possible.
    - Assumes a single sheet by default (sheet index 0), but allows explicit override.
    """
    excel_path = Path(path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Raw PIHPS Excel file not found: {excel_path}")

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=object)
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx requires an Excel engine (e.g., openpyxl). "
            "Install dependencies before running the dataset pipeline."
        ) from exc

    if df.empty:
        raise ValueError(f"Loaded Excel is empty: {excel_path}")
    return df
