from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.dataset_builder.build_feature_ready_dataset import build_feature_ready_dataset
from src.dataset_builder.build_training_targets import build_training_targets
from src.dataset_builder.filter_top_level_commodities import filter_top_level_commodities
from src.dataset_builder.load_raw_pihps_excel import load_raw_pihps_excel
from src.dataset_builder.normalize_prices_and_names import normalize_prices_and_names
from src.dataset_builder.parse_wide_pihps import parse_wide_pihps
from src.dataset_builder.validate_canonical_dataset import validate_canonical_dataset


@dataclass(frozen=True)
class DatasetPipelineOutputs:
    canonical_long: pd.DataFrame
    feature_ready: pd.DataFrame
    training_ready: pd.DataFrame


def run_dataset_pipeline(raw_excel_path: str | Path) -> DatasetPipelineOutputs:
    """Run full offline dataset build from raw PIHPS Excel export."""
    raw_df = load_raw_pihps_excel(raw_excel_path)
    parsed = parse_wide_pihps(raw_df)
    top_level_df = filter_top_level_commodities(parsed)
    canonical = normalize_prices_and_names(top_level_df, parsed)
    validate_canonical_dataset(canonical)

    feature_ready = build_feature_ready_dataset(canonical)
    training_ready = build_training_targets(feature_ready)

    return DatasetPipelineOutputs(
        canonical_long=canonical,
        feature_ready=feature_ready,
        training_ready=training_ready,
    )


def save_pipeline_outputs(outputs: DatasetPipelineOutputs, output_dir: str | Path) -> None:
    """Persist deterministic CSV outputs for reproducible downstream training."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs.canonical_long.sort_values(["commodity", "date"]).to_csv(
        target_dir / "canonical_long.csv", index=False
    )
    outputs.feature_ready.sort_values(["commodity", "date"]).to_csv(
        target_dir / "feature_ready_tabular.csv", index=False
    )
    outputs.training_ready.sort_values(["commodity", "date"]).to_csv(
        target_dir / "training_ready_tabular.csv", index=False
    )
