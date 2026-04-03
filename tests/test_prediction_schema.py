from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.config import INPUT_CONTRACT
from src.inference.predict import InferenceService


class _FakeModelStep:
    classes_ = ["Deflation", "Stable", "Inflation"]


class _FakePipeline:
    named_steps = {"model": _FakeModelStep()}

    @staticmethod
    def predict_proba(X):
        return [[0.2, 0.5, 0.3] for _ in range(len(X))]


def _build_service(monkeypatch, tmp_path: Path, feature_columns: list[str]) -> InferenceService:
    metadata = {
        "engineered_feature_columns": feature_columns,
        "class_names": ["Deflation", "Stable", "Inflation"],
        "horizon": 7,
        "model_version": "1.2.0",
        "input_contract": INPUT_CONTRACT,
        "artifact_name": "inflation_classifier_7d.joblib",
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


def test_single_row_inference(monkeypatch, tmp_path: Path):
    service = _build_service(monkeypatch, tmp_path, ["commodity", "year"])
    monkeypatch.setattr(
        "src.inference.predict.build_inference_features",
        lambda _: pd.DataFrame([{"commodity": "Rice", "year": 2024}]),
    )

    results = service.predict(pd.DataFrame([{"commodity": "Rice", "year": 2024}]))
    assert len(results) == 1


def test_batch_inference(monkeypatch, tmp_path: Path):
    service = _build_service(monkeypatch, tmp_path, ["commodity", "year"])
    monkeypatch.setattr(
        "src.inference.predict.build_inference_features",
        lambda _: pd.DataFrame(
            [
                {"commodity": "Rice", "year": 2024},
                {"commodity": "Sugar", "year": 2024},
            ]
        ),
    )

    results = service.predict(pd.DataFrame([{"x": 1}, {"x": 2}]))
    assert len(results) == 2
    assert all(pytest.approx(sum(row["probabilities"].values())) == 1.0 for row in results)