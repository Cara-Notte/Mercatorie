from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.inference.predict import InferenceService


class _FakeModelStep:
    classes_ = ["Deflation", "Stable", "Inflation"]


class _FakePipeline:
    named_steps = {"model": _FakeModelStep()}

    @staticmethod
    def predict_proba(_):
        return [[0.2, 0.5, 0.3]]


def _build_service(monkeypatch, tmp_path: Path, feature_columns: list[str]) -> InferenceService:
    metadata = {
        "engineered_feature_columns": feature_columns,
        "class_names": ["Deflation", "Stable", "Inflation"],
        "horizon": 7,
        "model_version": "1.1.0",
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr("src.inference.predict.joblib.load", lambda _: _FakePipeline())
    return InferenceService(horizon=7, model_path=tmp_path / "model.joblib", metadata_path=metadata_path)


def test_prediction_output_schema(monkeypatch, tmp_path: Path):
    service = _build_service(monkeypatch, tmp_path, ["commodity", "year"])
    monkeypatch.setattr(
        "src.inference.predict.build_inference_features",
        lambda _: pd.DataFrame([{"commodity": "Rice", "year": 2024}]),
    )

    result = service.predict(pd.DataFrame([{"commodity": "Rice", "year": 2024}]))[0]

    assert set(result.keys()) == {
        "prediction",
        "probabilities",
        "horizon",
        "model_version",
        "generated_at_utc",
    }
    assert set(result["probabilities"].keys()) == {"deflation", "stable", "inflation"}


def test_probabilities_sum_to_one(monkeypatch, tmp_path: Path):
    service = _build_service(monkeypatch, tmp_path, ["commodity", "year"])
    monkeypatch.setattr(
        "src.inference.predict.build_inference_features",
        lambda _: pd.DataFrame([{"commodity": "Rice", "year": 2024}]),
    )

    result = service.predict(pd.DataFrame([{"commodity": "Rice", "year": 2024}]))[0]
    assert pytest.approx(sum(result["probabilities"].values())) == 1.0


def test_feature_column_order_validation(monkeypatch, tmp_path: Path):
    service = _build_service(monkeypatch, tmp_path, ["commodity", "year"])
    monkeypatch.setattr(
        "src.inference.predict.build_inference_features",
        lambda _: pd.DataFrame([{"year": 2024, "commodity": "Rice"}]),
    )

    with pytest.raises(ValueError, match="Engineered feature order mismatch"):
        service.predict(pd.DataFrame([{"commodity": "Rice", "year": 2024}]))
