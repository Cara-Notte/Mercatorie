from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.config import PROJECT_ROOT
from src.inference.predict import InferenceService

PAGE_TITLE = "Mercatorie Price Intelligence for Tani Padu"
CLASS_ORDER = ["Deflation", "Stable", "Inflation"]
CLASS_COLORS = {
    "Deflation": "#2f80ed",
    "Stable": "#18a77a",
    "Inflation": "#e45b3c",
}
CLASS_COPY = {
    "Deflation": "Cooling pressure",
    "Stable": "Balanced market",
    "Inflation": "Rising pressure",
}
DEFAULT_PAYLOAD = {
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

st.set_page_config(page_title=PAGE_TITLE, page_icon="M", layout="wide", initial_sidebar_state="expanded")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #d7deea;
            --panel: #ffffff;
            --wash: #f4f7fb;
            --navy: #111827;
            --teal: #1f7a8c;
            --red: #e45b3c;
        }

        .stApp {
            background: var(--wash);
            color: var(--ink);
        }

        .block-container {
            max-width: 1240px;
            padding: 1.6rem 2rem 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #f8fafc !important;
        }

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--ink);
        }

        [data-testid="stWidgetLabel"] p,
        label,
        .stSlider label {
            color: #29384d !important;
            font-weight: 700 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(23, 32, 51, 0.05);
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 112px;
            box-shadow: 0 8px 24px rgba(23, 32, 51, 0.05);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-size: 0.78rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
        }

        div[data-baseweb="input"],
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] > div,
        textarea {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: #cbd5e1 !important;
            border-radius: 8px !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        .stButton button,
        .stFormSubmitButton button {
            border-radius: 8px !important;
            font-weight: 800 !important;
            min-height: 44px;
        }

        .stButton button[kind="primary"],
        .stFormSubmitButton button[kind="primary"] {
            background: var(--teal) !important;
            border-color: var(--teal) !important;
            color: #ffffff !important;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
        }

        button[data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--ink);
            padding: 10px 14px;
        }

        [data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_service(horizon: int) -> InferenceService:
    return InferenceService(horizon=horizon)


@st.cache_data(show_spinner=False)
def load_latest_market_rows() -> pd.DataFrame:
    data_path = PROJECT_ROOT / "data" / "processed" / "pihps_feature_engineered.csv"
    required_columns = list(DEFAULT_PAYLOAD)
    if not data_path.exists():
        return pd.DataFrame([DEFAULT_PAYLOAD])

    market_df = pd.read_csv(data_path, usecols=lambda column: column in required_columns)
    market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")
    market_df = market_df.dropna(subset=["date", "commodity"])
    latest = (
        market_df.sort_values(["commodity", "date"])
        .groupby("commodity", as_index=False)
        .tail(1)
        .sort_values("commodity")
        .reset_index(drop=True)
    )
    return latest if not latest.empty else pd.DataFrame([DEFAULT_PAYLOAD])


def percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def pct_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def format_idr(value: Any) -> str:
    return f"IDR {float(value):,.0f}"


def display_commodity(value: str) -> str:
    return str(value).replace("_", " ").title()


def parse_date(value: Any) -> date:
    return pd.to_datetime(value).date()


def calendar_fields(selected_date: date) -> dict[str, int]:
    return {
        "year": selected_date.year,
        "month": selected_date.month,
        "week_of_year": selected_date.isocalendar().week,
        "day_of_week": selected_date.weekday(),
        "is_month_start": int(selected_date.day == 1),
        "is_month_end": int((pd.Timestamp(selected_date) + pd.offsets.MonthEnd(0)).date() == selected_date),
    }


def calculate_pct_change(current: float, previous: float) -> float:
    if abs(previous) < 1e-9:
        return 0.0
    return (current / previous - 1) * 100


def calculate_vs_average(current: float, average: float) -> float:
    if abs(average) < 1e-9:
        return 0.0
    return (current / average - 1) * 100


def metadata_metric(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get("metrics", {}).get(key)
    return float(value) if isinstance(value, (float, int)) else None


def macro_f1(metadata: dict[str, Any]) -> float | None:
    value = metadata.get("metrics", {}).get("macro avg", {}).get("f1-score")
    return float(value) if isinstance(value, (float, int)) else None


def training_window(metadata: dict[str, Any]) -> str:
    start = pd.to_datetime(metadata.get("training_start_date")).strftime("%b %Y")
    end = pd.to_datetime(metadata.get("training_end_date")).strftime("%b %Y")
    return f"{start} to {end}"


def row_to_payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    payload = {key: row.get(key, DEFAULT_PAYLOAD[key]) for key in DEFAULT_PAYLOAD}
    payload["date"] = parse_date(payload["date"]).isoformat()
    for key in (
        "is_observed_source",
        "is_month_start",
        "is_month_end",
        "year",
        "month",
        "week_of_year",
        "day_of_week",
    ):
        payload[key] = int(payload[key])
    for key in (
        "price_idr",
        "price_lag_1d",
        "price_lag_7d",
        "price_lag_30d",
        "price_change_1d_pct",
        "price_change_7d_pct",
        "price_change_30d_pct",
        "rolling_mean_7d",
        "rolling_mean_30d",
        "rolling_std_30d",
        "price_vs_ma30_pct",
        "volatility_30d_pct",
    ):
        payload[key] = float(payload[key])
    return payload


def score_payload(payload: dict[str, Any], horizon: int, service: InferenceService) -> dict[str, Any]:
    output = service.predict(pd.DataFrame([payload]))[0]
    st.session_state["active_payload"] = payload
    st.session_state["active_result"] = output
    st.session_state["active_horizon"] = horizon
    return output


def build_probability_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome": CLASS_ORDER,
            "probability": [result["probabilities"][label.lower()] for label in CLASS_ORDER],
            "signal": [CLASS_COPY[label] for label in CLASS_ORDER],
            "color": [CLASS_COLORS[label] for label in CLASS_ORDER],
        }
    )


def build_price_frame(payload: dict[str, Any]) -> pd.DataFrame:
    current_date = pd.to_datetime(payload["date"])
    observations = [
        (current_date - pd.Timedelta(days=30), "Observed price", payload["price_lag_30d"]),
        (current_date - pd.Timedelta(days=7), "Observed price", payload["price_lag_7d"]),
        (current_date - pd.Timedelta(days=1), "Observed price", payload["price_lag_1d"]),
        (current_date, "Observed price", payload["price_idr"]),
    ]
    references = []
    for point_date, _, _ in observations:
        references.append((point_date, "30-day average", payload["rolling_mean_30d"]))
        references.append((point_date, "7-day average", payload["rolling_mean_7d"]))
    return pd.DataFrame(observations + references, columns=["date", "series", "price"])


def build_momentum_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("1 day", payload["price_change_1d_pct"]),
        ("7 days", payload["price_change_7d_pct"]),
        ("30 days", payload["price_change_30d_pct"]),
    ]
    return pd.DataFrame(
        {
            "period": [row[0] for row in rows],
            "change": [row[1] for row in rows],
            "direction": ["Increase" if row[1] >= 0 else "Decrease" for row in rows],
        }
    )


def build_model_metric_frame(metadata: dict[str, Any]) -> pd.DataFrame:
    metrics = metadata.get("metrics", {})
    rows = []
    for label in CLASS_ORDER:
        rows.append(
            {
                "outcome": label,
                "precision": metrics.get(label, {}).get("precision", 0),
                "recall": metrics.get(label, {}).get("recall", 0),
                "f1": metrics.get(label, {}).get("f1-score", 0),
            }
        )
    return pd.DataFrame(rows)


def probability_chart(result: dict[str, Any]) -> None:
    data = build_probability_frame(result)
    spec = {
        "height": 230,
        "mark": {"type": "bar", "cornerRadiusEnd": 6},
        "encoding": {
            "x": {
                "field": "probability",
                "type": "quantitative",
                "axis": {"format": "%", "title": None},
                "scale": {"domain": [0, 1]},
            },
            "y": {"field": "outcome", "type": "nominal", "sort": CLASS_ORDER, "axis": {"title": None}},
            "color": {
                "field": "outcome",
                "type": "nominal",
                "scale": {"domain": CLASS_ORDER, "range": [CLASS_COLORS[label] for label in CLASS_ORDER]},
                "legend": None,
            },
            "tooltip": [
                {"field": "outcome", "type": "nominal", "title": "Outcome"},
                {"field": "signal", "type": "nominal", "title": "Signal"},
                {"field": "probability", "type": "quantitative", "title": "Probability", "format": ".1%"},
            ],
        },
    }
    st.vega_lite_chart(data, spec, width="stretch", theme=None)


def price_chart(payload: dict[str, Any]) -> None:
    data = build_price_frame(payload)
    spec = {
        "height": 260,
        "mark": {"type": "line", "point": {"filled": True, "size": 80}, "strokeWidth": 3},
        "encoding": {
            "x": {"field": "date", "type": "temporal", "axis": {"title": None, "format": "%d %b"}},
            "y": {
                "field": "price",
                "type": "quantitative",
                "axis": {"title": "IDR"},
                "scale": {"zero": False},
            },
            "color": {
                "field": "series",
                "type": "nominal",
                "scale": {"range": ["#172033", "#1f7a8c", "#c98a1f"]},
                "legend": {"orient": "bottom", "title": None},
            },
            "tooltip": [
                {"field": "date", "type": "temporal", "title": "Date", "format": "%d %b %Y"},
                {"field": "series", "type": "nominal", "title": "Series"},
                {"field": "price", "type": "quantitative", "title": "Price", "format": ",.0f"},
            ],
        },
    }
    st.vega_lite_chart(data, spec, width="stretch", theme=None)


def momentum_chart(payload: dict[str, Any]) -> None:
    data = build_momentum_frame(payload)
    spec = {
        "height": 220,
        "mark": {"type": "bar", "cornerRadiusEnd": 6},
        "encoding": {
            "x": {"field": "period", "type": "nominal", "axis": {"title": None}},
            "y": {"field": "change", "type": "quantitative", "axis": {"title": "Percent change"}},
            "color": {
                "field": "direction",
                "type": "nominal",
                "scale": {"domain": ["Increase", "Decrease"], "range": ["#e45b3c", "#2f80ed"]},
                "legend": None,
            },
            "tooltip": [
                {"field": "period", "type": "nominal", "title": "Period"},
                {"field": "change", "type": "quantitative", "title": "Change", "format": "+.2f"},
            ],
        },
    }
    st.vega_lite_chart(data, spec, width="stretch", theme=None)


def model_quality_chart(metadata: dict[str, Any]) -> None:
    data = build_model_metric_frame(metadata).melt(id_vars="outcome", var_name="metric", value_name="score")
    spec = {
        "height": 260,
        "mark": {"type": "bar", "cornerRadiusEnd": 4},
        "encoding": {
            "x": {"field": "score", "type": "quantitative", "axis": {"format": "%", "title": None}, "scale": {"domain": [0, 1]}},
            "y": {"field": "outcome", "type": "nominal", "sort": CLASS_ORDER, "axis": {"title": None}},
            "color": {
                "field": "metric",
                "type": "nominal",
                "scale": {"range": ["#1f7a8c", "#c98a1f", "#172033"]},
                "legend": {"orient": "bottom", "title": None},
            },
            "tooltip": [
                {"field": "outcome", "type": "nominal", "title": "Outcome"},
                {"field": "metric", "type": "nominal", "title": "Metric"},
                {"field": "score", "type": "quantitative", "title": "Score", "format": ".1%"},
            ],
        },
    }
    st.vega_lite_chart(data, spec, width="stretch", theme=None)


def render_header(horizon: int, metadata: dict[str, Any]) -> None:
    st.title("Mercatorie Price Intelligence")
    st.caption("Forecast likely food-price movement from recent market prices, momentum, and volatility.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast horizon", f"{horizon} days")
    c2.metric("Model version", f"v{metadata.get('model_version', '-')}")
    c3.metric("Accuracy", percent(metadata_metric(metadata, "accuracy")))
    c4.metric("Macro F1", percent(macro_f1(metadata)))


def render_scenario_form(horizon: int, service: InferenceService, market_rows: pd.DataFrame) -> dict[str, Any]:
    st.subheader("Market Scenario")
    st.caption("Adjust the market snapshot and run a new forecast. The dashboard starts with the latest saved sample.")

    options = market_rows["commodity"].astype(str).tolist()
    default_index = 0
    selected_commodity = st.selectbox(
        "Commodity",
        options=options,
        index=default_index,
        format_func=display_commodity,
        key="selected_commodity",
    )
    seed_row = market_rows.loc[market_rows["commodity"].astype(str) == selected_commodity].iloc[0]
    seed = row_to_payload(seed_row)
    prefix = f"{selected_commodity}_{horizon}".replace(" ", "_").replace("/", "_")

    with st.form("market_scenario_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_date = st.date_input("Market date", value=parse_date(seed["date"]), key=f"date_{prefix}")
        with c2:
            observed_source = st.toggle("Observed market source", value=bool(seed["is_observed_source"]), key=f"source_{prefix}")
        with c3:
            volatility_30d_pct = st.number_input(
                "30-day volatility",
                value=float(seed["volatility_30d_pct"]),
                min_value=0.0,
                step=0.1,
                format="%.2f",
                key=f"vol_{prefix}",
            )

        st.markdown("##### Recent prices")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            price_idr = st.number_input("Current", value=float(seed["price_idr"]), min_value=0.0, step=100.0, key=f"current_{prefix}")
        with p2:
            price_lag_1d = st.number_input("Yesterday", value=float(seed["price_lag_1d"]), min_value=0.0, step=100.0, key=f"lag1_{prefix}")
        with p3:
            price_lag_7d = st.number_input("7 days ago", value=float(seed["price_lag_7d"]), min_value=0.0, step=100.0, key=f"lag7_{prefix}")
        with p4:
            price_lag_30d = st.number_input("30 days ago", value=float(seed["price_lag_30d"]), min_value=0.0, step=100.0, key=f"lag30_{prefix}")

        st.markdown("##### Market averages")
        a1, a2, a3 = st.columns(3)
        with a1:
            rolling_mean_7d = st.number_input("7-day average", value=float(seed["rolling_mean_7d"]), min_value=0.0, step=100.0, key=f"mean7_{prefix}")
        with a2:
            rolling_mean_30d = st.number_input("30-day average", value=float(seed["rolling_mean_30d"]), min_value=0.0, step=100.0, key=f"mean30_{prefix}")
        with a3:
            rolling_std_30d = st.number_input("30-day spread", value=float(seed["rolling_std_30d"]), min_value=0.0, step=10.0, key=f"std30_{prefix}")

        submitted = st.form_submit_button(f"Update {horizon}-day forecast", type="primary", width="stretch")

    cal = calendar_fields(selected_date)
    payload = {
        "date": selected_date.isoformat(),
        "commodity": selected_commodity,
        "is_observed_source": int(observed_source),
        "is_month_start": cal["is_month_start"],
        "is_month_end": cal["is_month_end"],
        "year": cal["year"],
        "month": cal["month"],
        "week_of_year": cal["week_of_year"],
        "day_of_week": cal["day_of_week"],
        "price_idr": float(price_idr),
        "price_lag_1d": float(price_lag_1d),
        "price_lag_7d": float(price_lag_7d),
        "price_lag_30d": float(price_lag_30d),
        "price_change_1d_pct": calculate_pct_change(float(price_idr), float(price_lag_1d)),
        "price_change_7d_pct": calculate_pct_change(float(price_idr), float(price_lag_7d)),
        "price_change_30d_pct": calculate_pct_change(float(price_idr), float(price_lag_30d)),
        "rolling_mean_7d": float(rolling_mean_7d),
        "rolling_mean_30d": float(rolling_mean_30d),
        "rolling_std_30d": float(rolling_std_30d),
        "price_vs_ma30_pct": calculate_vs_average(float(price_idr), float(rolling_mean_30d)),
        "volatility_30d_pct": float(volatility_30d_pct),
    }

    if submitted:
        try:
            score_payload(payload, horizon, service)
            st.success("Forecast updated.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Forecast could not be generated: {exc}")

    return payload


def render_forecast_panel(result: dict[str, Any], payload: dict[str, Any]) -> None:
    prediction = result["prediction"]
    confidence = max(result["probabilities"].values())
    st.subheader("Forecast")
    with st.container(border=True):
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            st.metric("Likely movement", prediction, CLASS_COPY.get(prediction, "Market signal"))
            st.caption(f"{result['horizon']}-day view with {percent(confidence)} model confidence.")
        with c2:
            st.metric("Current price", format_idr(payload["price_idr"]), pct_value(payload["price_vs_ma30_pct"]) + " vs 30-day average")

    with st.container(border=True):
        st.markdown("##### Outcome probability")
        probability_chart(result)


def render_market_story(payload: dict[str, Any]) -> None:
    st.subheader("Market View")
    m1, m2, m3 = st.columns(3)
    m1.metric("1-day movement", pct_value(payload["price_change_1d_pct"]))
    m2.metric("7-day movement", pct_value(payload["price_change_7d_pct"]))
    m3.metric("30-day movement", pct_value(payload["price_change_30d_pct"]))

    c1, c2 = st.columns([1.25, 0.75], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("##### Price path and averages")
            price_chart(payload)
    with c2:
        with st.container(border=True):
            st.markdown("##### Momentum profile")
            momentum_chart(payload)


def render_model_evidence(service: InferenceService) -> None:
    metadata = service.metadata
    st.subheader("Model Evidence")
    e1, e2, e3 = st.columns(3)
    e1.metric("Validation window", training_window(metadata))
    e2.metric("Decision classes", f"{len(metadata.get('class_names', []))}")
    e3.metric("Feature signals", f"{len(metadata.get('engineered_feature_columns', []))}")

    with st.container(border=True):
        st.markdown("##### Validation performance by outcome")
        model_quality_chart(metadata)


def main() -> None:
    inject_theme()

    with st.sidebar:
        st.markdown("## Mercatorie")
        st.caption("Reviewer dashboard")
        horizon = st.radio("Forecast horizon", options=[7, 30], horizontal=True, format_func=lambda value: f"{value}d")
        st.markdown("---")
        st.caption("Built for reviewer-ready scenario walkthroughs.")

    try:
        service = load_service(horizon)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the forecasting model: {exc}")
        st.stop()

    market_rows = load_latest_market_rows()
    default_payload = row_to_payload(market_rows.iloc[0])
    if st.session_state.get("active_horizon") != horizon or "active_result" not in st.session_state:
        score_payload(default_payload, horizon, service)

    render_header(horizon, service.metadata)

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        edited_payload = render_scenario_form(horizon, service, market_rows)
    with right:
        active_result = st.session_state.get("active_result")
        active_payload = st.session_state.get("active_payload", edited_payload)
        if active_result:
            render_forecast_panel(active_result, active_payload)

    active_payload = st.session_state.get("active_payload", edited_payload)
    render_market_story(active_payload)
    render_model_evidence(service)


if __name__ == "__main__":
    main()
