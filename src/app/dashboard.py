from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Mercatorie Inflation Monitor", layout="wide")

# -----------------------------------
# Paths
# -----------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

MODEL_7D_PATH = PROJECT_ROOT / "models" / "model_7d.joblib"
MODEL_30D_PATH = PROJECT_ROOT / "models" / "model_30d_v2.joblib"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_cols.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pihps_feature_engineered_v2.csv"

# -----------------------------------
# Load artifacts
# -----------------------------------
@st.cache_resource
def load_models():
    model_7d = joblib.load(MODEL_7D_PATH)
    model_30d = joblib.load(MODEL_30D_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model_7d, model_30d, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

model_7d, model_30d, feature_cols = load_models()
df = load_data()

# -----------------------------------
# Helpers
# -----------------------------------
def get_status_style(predicted_class: str):
    if predicted_class == "Inflation":
        return "warning", "📈 Inflation risk"
    elif predicted_class == "Deflation":
        return "error", "📉 Deflation risk"
    return "success", "➡️ Stable"

def predict_row(row_df: pd.DataFrame, horizon: str):
    if horizon == "7d":
        model = model_7d
        model_name = "Gradient Boosting"
    else:
        model = model_30d
        model_name = "Logistic Regression"

    X = row_df[feature_cols].copy()
    pred_class = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    class_labels = list(model.named_steps["model"].classes_)

    result = {
        "horizon": horizon,
        "model_name": model_name,
        "predicted_class": pred_class,
        "prediction_confidence": float(pred_proba.max())
    }

    for label, prob in zip(class_labels, pred_proba):
        result[f"prob_{label}"] = float(prob)

    return result

# -----------------------------------
# UI
# -----------------------------------
st.title("Mercatorie")
st.caption("Inflation signal dashboard for Indonesian food commodities")

with st.sidebar:
    st.header("Controls")

    horizon = st.radio(
        "Prediction horizon",
        options=["7d", "30d"],
        format_func=lambda x: "7 Days Ahead" if x == "7d" else "30 Days Ahead"
    )

    commodity_options = sorted(df["commodity"].unique().tolist())
    selected_commodity = st.selectbox("Commodity", commodity_options)

    commodity_df = df[df["commodity"] == selected_commodity].sort_values("date")
    available_dates = commodity_df["date"].dt.strftime("%Y-%m-%d").tolist()

    selected_date_str = st.selectbox(
        "Date",
        options=available_dates,
        index=len(available_dates) - 1
    )

selected_row = commodity_df[
    commodity_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str
].iloc[[0]].copy()

prediction = predict_row(selected_row, horizon=horizon)

status_type, status_text = get_status_style(prediction["predicted_class"])

# -----------------------------------
# Main status cards
# -----------------------------------
left, mid, right = st.columns(3)

with left:
    st.subheader("Prediction")
    if status_type == "success":
        st.success(status_text)
    elif status_type == "warning":
        st.warning(status_text)
    else:
        st.error(status_text)

with mid:
    st.subheader("Confidence")
    confidence_pct = prediction["prediction_confidence"] * 100
    st.metric(label="Model confidence", value=f"{confidence_pct:.1f}%")
    st.progress(float(prediction["prediction_confidence"]))

with right:
    st.subheader("Context")
    st.metric("Commodity", selected_commodity)
    st.metric("Date", selected_date_str)
    st.metric("Model", prediction["model_name"])

# -----------------------------------
# Plain-language summary
# -----------------------------------
if prediction["predicted_class"] == "Inflation":
    summary_text = f"The model sees upward price pressure in the selected {horizon} horizon."
elif prediction["predicted_class"] == "Deflation":
    summary_text = f"The model sees downward price pressure in the selected {horizon} horizon."
else:
    summary_text = f"The model sees relatively stable conditions in the selected {horizon} horizon."

st.info(summary_text)

# -----------------------------------
# Optional details
# -----------------------------------
with st.expander("Show details"):
    proba_table = pd.DataFrame({
        "Class": ["Deflation", "Stable", "Inflation"],
        "Probability": [
            prediction.get("prob_Deflation", 0.0),
            prediction.get("prob_Stable", 0.0),
            prediction.get("prob_Inflation", 0.0),
        ]
    })
    st.dataframe(proba_table, use_container_width=True)

    st.markdown("**Recent price trend**")
    trend_df = commodity_df.sort_values("date").tail(90)
    st.line_chart(
        trend_df.set_index("date")["price_idr"],
        use_container_width=True
    )

    st.markdown("**Recent engineered signals**")
    detail_cols = [
        "price_idr",
        "price_change_1d_pct",
        "price_change_7d_pct",
        "price_change_30d_pct",
        "price_zscore_30d",
        "volatility_30d_pct"
    ]
    available_detail_cols = [c for c in detail_cols if c in selected_row.columns]
    st.dataframe(selected_row[available_detail_cols].T.rename(columns={selected_row.index[0]: "value"}))

st.markdown("---")
st.caption("Main screen is simplified for decision support. Details are optional.")