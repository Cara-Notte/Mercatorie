from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_ROOT / "models"
PREPROCESSORS_DIR = ARTIFACTS_ROOT / "preprocessors"
METADATA_DIR = ARTIFACTS_ROOT / "metadata"

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pihps_food_inflation_indonesia.csv"

MODEL_VERSION = "1.2.0"
CLASS_ORDER = ["Deflation", "Stable", "Inflation"]
SUPPORTED_HORIZONS = (7, 30)
INPUT_CONTRACT = "feature_ready_tabular"


def model_artifact_path(horizon: int) -> Path:
    return MODELS_DIR / f"inflation_classifier_{horizon}d.joblib"


def metadata_artifact_path(horizon: int) -> Path:
    return METADATA_DIR / f"inflation_classifier_{horizon}d_metadata.json"


def ensure_artifact_dirs() -> None:
    for path in (MODELS_DIR, PREPROCESSORS_DIR, METADATA_DIR):
        path.mkdir(parents=True, exist_ok=True)