from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.config import metadata_artifact_path, model_artifact_path
from src.inference.predict import InferenceService


class _FakeModelStep:
    classes_ = ["Deflation", "Stable", "Inflation"]


class _FakePipeline:
    named_steps = {"model": _FakeModelStep()}


def _write_metadata(path: Path, horizon: int, class_names: list[str] | None = None) -> None:
    payload = {
        "engineered_feature_columns": ["commodity"],
        "class_names": class_names or ["Deflation", "Stable", "Inflation"],
        "horizon": horizon,
        "model_version": "1.1.0",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_horizon_specific_artifact_loading(monkeypatch, tmp_path: Path):
    metadata_7d = tmp_path / "inflation_classifier_7d_metadata.json"
    metadata_30d = tmp_path / "inflation_classifier_30d_metadata.json"
    _write_metadata(metadata_7d, horizon=7)
    _write_metadata(metadata_30d, horizon=30)

    monkeypatch.setattr("src.inference.predict.joblib.load", lambda _: _FakePipeline())

    service_7d = InferenceService(horizon=7, model_path=tmp_path / "m7.joblib", metadata_path=metadata_7d)
    service_30d = InferenceService(horizon=30, model_path=tmp_path / "m30.joblib", metadata_path=metadata_30d)

    assert service_7d.horizon == 7
    assert service_30d.horizon == 30
    assert model_artifact_path(7).name == "inflation_classifier_7d.joblib"
    assert metadata_artifact_path(30).name == "inflation_classifier_30d_metadata.json"


def test_metadata_model_class_alignment_validation(monkeypatch, tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    _write_metadata(metadata, horizon=7, class_names=["Deflation", "Inflation", "Stable"])

    monkeypatch.setattr("src.inference.predict.joblib.load", lambda _: _FakePipeline())

    with pytest.raises(ValueError, match="Metadata/model class mismatch"):
        InferenceService(horizon=7, model_path=tmp_path / "model.joblib", metadata_path=metadata)
