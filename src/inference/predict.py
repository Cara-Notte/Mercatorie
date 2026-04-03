from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.common.config import CLASS_ORDER, SUPPORTED_HORIZONS, metadata_artifact_path, model_artifact_path
from src.inference.feature_builder import build_inference_features

LOGGER = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, horizon: int = 7, model_path=None, metadata_path=None):
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon '{horizon}'. Expected one of {SUPPORTED_HORIZONS}")

        self.horizon = horizon
        self.model_path = model_path or model_artifact_path(horizon)
        self.metadata_path = metadata_path or metadata_artifact_path(horizon)

        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.expected_columns = self.metadata["engineered_feature_columns"]
        self.class_names = self.metadata["class_names"]
        self._validate_model_metadata_contract()

    def _validate_model_metadata_contract(self) -> None:
        model_classes = list(self.model.named_steps["model"].classes_)
        if model_classes != self.class_names:
            raise ValueError(
                "Metadata/model class mismatch. "
                f"Metadata classes: {self.class_names}; model classes: {model_classes}"
            )

        if self.metadata["horizon"] != self.horizon:
            raise ValueError(
                f"Horizon mismatch: metadata has {self.metadata['horizon']}d, service initialized with {self.horizon}d"
            )

    def _validate_column_order(self, feature_df: pd.DataFrame) -> None:
        incoming = list(feature_df.columns)
        if incoming != self.expected_columns:
            raise ValueError(
                "Column order mismatch for inference. "
                f"Expected {self.expected_columns}, got {incoming}"
            )

    def predict(self, raw_input: pd.DataFrame) -> list[dict]:
        feature_df = build_inference_features(raw_input)

        missing = [col for col in self.expected_columns if col not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing required features for inference: {missing}")

        incoming_expected = [col for col in feature_df.columns if col in self.expected_columns]
        if incoming_expected != self.expected_columns:
            raise ValueError(
                "Engineered feature order mismatch before model scoring. "
                f"Expected {self.expected_columns}, got {incoming_expected}"
            )

        ordered = feature_df[self.expected_columns].copy()
        self._validate_column_order(ordered)

        probabilities = self.model.predict_proba(ordered)
        label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}

        results: list[dict] = []
        for row_probs in probabilities:
            structured = {
                "deflation": float(row_probs[label_to_idx["Deflation"]]),
                "stable": float(row_probs[label_to_idx["Stable"]]),
                "inflation": float(row_probs[label_to_idx["Inflation"]]),
            }
            prediction = max(CLASS_ORDER, key=lambda cls: structured[cls.lower()])
            results.append(
                {
                    "prediction": prediction,
                    "probabilities": structured,
                    "horizon": self.horizon,
                    "model_version": self.metadata["model_version"],
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

        LOGGER.info("Generated %d inference predictions for %sd horizon.", len(results), self.horizon)
        return results
