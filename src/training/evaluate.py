from __future__ import annotations

import logging
from typing import Any

from sklearn.metrics import classification_report

LOGGER = logging.getLogger(__name__)


def evaluate_classifier(model: Any, X_valid, y_valid) -> dict:
    predictions = model.predict(X_valid)
    report = classification_report(y_valid, predictions, output_dict=True)
    LOGGER.info("Validation accuracy: %.4f", report.get("accuracy", 0.0))
    return report
