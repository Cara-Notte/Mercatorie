from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.common.config import (
    INPUT_CONTRACT,
    MODEL_VERSION,
    RAW_DATA_PATH,
    SUPPORTED_HORIZONS,
    ensure_artifact_dirs,
    metadata_artifact_path,
    model_artifact_path,
)
from src.training.evaluate import evaluate_classifier
from src.training.feature_engineering import (
    FEATURE_COLUMNS,
    RAW_REQUIRED_COLUMNS,
    TARGET_CLASS_COLUMNS,
    attach_training_targets,
    build_base_features,
)
from src.training.preprocess import build_tree_preprocessor, time_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def _assert_no_target_leakage(feature_df: pd.DataFrame) -> None:
    forbidden_prefixes = ("target_",)
    leaked = [col for col in feature_df.columns if col.startswith(forbidden_prefixes)]
    if leaked:
        raise ValueError(f"Target leakage detected in feature matrix columns: {leaked}")


def _date_range(series: pd.Series) -> tuple[str, str]:
    return series.min().isoformat(), series.max().isoformat()


def train_and_save(raw_data_path=RAW_DATA_PATH, horizons: tuple[int, ...] = SUPPORTED_HORIZONS, stable_band: float = 1.0) -> None:
    ensure_artifact_dirs()

    raw_df = pd.read_csv(raw_data_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    base_features_df = build_base_features(raw_df)
    training_df = attach_training_targets(base_features_df, stable_band=stable_band)

    for horizon in horizons:
        if horizon not in TARGET_CLASS_COLUMNS:
            raise ValueError(f"Unsupported horizon '{horizon}'. Expected one of {sorted(TARGET_CLASS_COLUMNS)}")

        target_column = TARGET_CLASS_COLUMNS[horizon]
        split = time_split(training_df, target_column=target_column)
        _assert_no_target_leakage(split.X_train)

        train_start, train_end = _date_range(split.train_dates)
        valid_start, valid_end = _date_range(split.valid_dates)

        LOGGER.info("Training %sd model | rows=%d | range=%s to %s", horizon, len(split.X_train), train_start, train_end)
        LOGGER.info("Validation %sd model | rows=%d | range=%s to %s", horizon, len(split.X_valid), valid_start, valid_end)
        LOGGER.info("Class distribution (%sd): %s", horizon, split.y_train.value_counts().to_dict())

        preprocessor = build_tree_preprocessor()
        model = GradientBoostingClassifier(random_state=42)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(split.X_train, split.y_train)

        metrics = evaluate_classifier(pipeline, split.X_valid, split.y_valid)

        artifact_path = model_artifact_path(horizon)
        metadata_path = metadata_artifact_path(horizon)

        joblib.dump(pipeline, artifact_path)

        metadata = {
            "input_contract": INPUT_CONTRACT,
            "model_version": MODEL_VERSION,
            "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "class_names": list(pipeline.named_steps["model"].classes_),
            "stable_band": stable_band,
            "raw_required_columns": RAW_REQUIRED_COLUMNS,
            "engineered_feature_columns": FEATURE_COLUMNS,
            "target_column": target_column,
            "horizon": horizon,
            "training_start_date": train_start,
            "training_end_date": train_end,
            "model_type": pipeline.named_steps["model"].__class__.__name__,
            "artifact_name": artifact_path.name,
            "metrics": metrics,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        LOGGER.info("Saved %sd pipeline artifact: %s", horizon, artifact_path)
        LOGGER.info("Saved %sd metadata artifact: %s", horizon, metadata_path)


if __name__ == "__main__":
    train_and_save()