from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.config import INPUT_CONTRACT, metadata_artifact_path, model_artifact_path
from src.inference.predict import InferenceService


class _FakeModelStep:
    classes_ = ["Deflation", "Stable", "Inflation"]


class _FakePipeline:
    named_steps = {"model": _FakeModelStep()}


def _write_metadata(path: Path, horizon: int, class_names: list[str] | None = None, input_contract: str = INPUT_CONTRACT) -> None:
    payload = {
        "engineered_feature_columns": ["commodity"],
        "class_names": class_names or ["Deflation", "Stable", "Inflation"],
        "horizon": horizon,
        "model_version": "1.2.0",
        "input_contract": input_contract,
        "artifact_name": f"inflation_classifier_{horizon}d.joblib",
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


def test_metadata_contract_consistency(monkeypatch, tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    _write_metadata(metadata, horizon=7, input_contract=INPUT_CONTRACT)

    monkeypatch.setattr("src.inference.predict.joblib.load", lambda _: _FakePipeline())
    service = InferenceService(horizon=7, model_path=tmp_path / "model.joblib", metadata_path=metadata)

    assert service.metadata["input_contract"] == "feature_ready_tabular"


def test_absence_of_stale_artifact_metadata(monkeypatch, tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    _write_metadata(metadata, horizon=7)

    monkeypatch.setattr("src.inference.predict.joblib.load", lambda _: _FakePipeline())
    service = InferenceService(horizon=7, model_path=tmp_path / "model.joblib", metadata_path=metadata)

    assert "preprocessor_artifact_name" not in service.metadata


def test_invalid_input_contract_raises(monkeypatch, tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    _write_metadata(metadata, horizon=7, input_contract="raw_timeseries")

    monkeypatch.setattr("src.inference.predict.joblib.load", lambda _: _FakePipeline())

    with pytest.raises(ValueError, match="Unsupported metadata input contract"):
        InferenceService(horizon=7, model_path=tmp_path / "model.joblib", metadata_path=metadata)