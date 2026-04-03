from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.inference.predict import InferenceService

st.set_page_config(page_title="Inflation Classifier", layout="centered")
st.title("Food Inflation Classifier")
st.caption("Inference-only app. No training logic is executed here.")

st.markdown("Provide one JSON object with feature-ready raw fields required by shared feature engineering.")

example_payload = {
    "date": "2024-10-01",
    "commodity": "Rice",
    "is_observed_source": 1,
    "is_month_start": 1,
    "is_month_end": 0,
    "year": 2024,
    "month": 10,
    "week_of_year": 40,
    "day_of_week": 2,
    "price_idr": 14000,
    "price_lag_1d": 13950,
    "price_lag_7d": 13800,
    "price_lag_30d": 13700,
    "price_change_1d_pct": 0.35,
    "price_change_7d_pct": 1.45,
    "price_change_30d_pct": 2.20,
    "rolling_mean_7d": 13900,
    "rolling_mean_30d": 13820,
    "rolling_std_30d": 175.0,
    "price_vs_ma30_pct": 1.30,
    "volatility_30d_pct": 2.15,
}

horizon = st.selectbox("Prediction horizon", options=[7, 30], index=0)
payload_text = st.text_area("Input JSON", value=json.dumps(example_payload, indent=2), height=340)

if st.button("Predict"):
    try:
        payload = json.loads(payload_text)
        df = pd.DataFrame([payload])
        service = InferenceService(horizon=horizon)
        output = service.predict(df)[0]
        st.success(f"Prediction: {output['prediction']}")
        st.json(output)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
