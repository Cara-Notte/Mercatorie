from __future__ import annotations

import pandas as pd

from src.training.feature_engineering import FEATURE_COLUMNS, build_base_features


def build_inference_features(raw_input: pd.DataFrame) -> pd.DataFrame:
    """Build inference features from feature-ready tabular input.

    Current contract does not accept raw time-series-only payloads; upstream processing
    must already provide lag/rolling/pct-change columns required by build_base_features.
    """
    features_df = build_base_features(raw_input)
    missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"Inference feature build failed. Missing columns after engineering: {missing}")
    return features_df[FEATURE_COLUMNS].copy()