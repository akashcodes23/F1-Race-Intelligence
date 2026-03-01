import os
import sys
import fastf1

# Enable FastF1 cache (important for Streamlit Cloud)
if not os.path.exists("cache"):
    os.makedirs("cache")

fastf1.Cache.enable_cache("cache")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from data_loader import load_race_data
from metrics import (
    train_lap_models,
    optimize_pit_window
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="F1 Race Intelligence", layout="wide")

st.title("🏎️ F1 Race Intelligence Platform")
st.caption("Predictive Modeling · Strategy Optimization · Race Intelligence")

# =====================================================
# CLEAN SECTION COMPONENT
# =====================================================

def comparison_section(title: str, emoji: str):
    st.markdown(
        f"""
        <div style="margin-top:50px; margin-bottom:25px;">
            <h2 style="
                border-bottom: 2px solid #2E2E2E;
                padding-bottom: 10px;
                font-weight: 600;">
                {emoji} {title}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# CACHED LOADERS
# =====================================================

@st.cache_data(show_spinner=False)
def get_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    return schedule["EventName"].tolist()

@st.cache_data(show_spinner=False)
def get_drivers(year, race):
    session = fastf1.get_session(year, race, "R")
    session.load()
    return sorted(session.drivers)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🎛 Control Panel")

season = st.sidebar.selectbox("Season", [2021, 2022, 2023])
gp_list = get_schedule(season)
grand_prix = st.sidebar.selectbox("Grand Prix", gp_list)

pit_penalty = st.sidebar.slider("Pit Stop Loss (sec)", 15, 35, 22)

driver_list = get_drivers(season, grand_prix)
drivers = st.sidebar.multiselect("Select exactly 2 drivers", driver_list)

if len(drivers) != 2:
    st.warning("Select exactly 2 drivers.")
    st.stop()

driver1, driver2 = drivers

# =====================================================
# LOAD DATA
# =====================================================

laps1 = load_race_data(season, grand_prix, driver1)
laps2 = load_race_data(season, grand_prix, driver2)

if len(laps1) < 15 or len(laps2) < 15:
    st.error("Not enough lap data for reliable modeling.")
    st.stop()

# =====================================================
# TRAIN MODELS
# =====================================================

with st.spinner("Training predictive models..."):
    models1 = train_lap_models(laps1)
    models2 = train_lap_models(laps2)

rf1 = models1["random_forest"]
rf2 = models2["random_forest"]

# =====================================================
# 🧠 MODEL INTELLIGENCE
# =====================================================

comparison_section("Model Intelligence Overview", "🧠")

col1, col2 = st.columns(2)

with col1:
    st.subheader(driver1)
    st.metric("R²", f"{rf1['test_r2']:.3f}")
    st.metric("RMSE", f"{rf1['test_rmse']:.3f}")

with col2:
    st.subheader(driver2)
    st.metric("R²", f"{rf2['test_r2']:.3f}")
    st.metric("RMSE", f"{rf2['test_rmse']:.3f}")

# =====================================================
# 📊 DRIVER DELTA PACE
# =====================================================

comparison_section("Driver Delta Pace", "📊")

def plot_driver_delta(l1, l2):

    df1 = l1.copy()
    df2 = l2.copy()

    df1["LapTimeSec"] = df1["LapTime"].dt.total_seconds()
    df2["LapTimeSec"] = df2["LapTime"].dt.total_seconds()

    merged = pd.merge(
        df1[["LapNumber", "LapTimeSec"]],
        df2[["LapNumber", "LapTimeSec"]],
        on="LapNumber",
        suffixes=(f"_{driver1}", f"_{driver2}")
    )

    merged["Delta"] = (
        merged[f"LapTimeSec_{driver1}"]
        - merged[f"LapTimeSec_{driver2}"]
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged["LapNumber"],
        y=merged["Delta"],
        mode="lines",
        name="Delta"
    ))

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Lap",
        yaxis_title="Delta (sec)"
    )

    return fig

st.plotly_chart(plot_driver_delta(laps1, laps2), width="stretch")

# =====================================================
# 📉 TIRE DEGRADATION
# =====================================================

comparison_section("Tire Degradation", "📉")

def plot_degradation(laps, driver):

    df = laps.copy()
    df["LapTimeSec"] = df["LapTime"].dt.total_seconds()
    df = df.dropna(subset=["LapNumber", "LapTimeSec"])

    if len(df) < 5:
        st.warning(f"Not enough clean data for {driver}")
        return go.Figure()

    x = df["LapNumber"].values
    y = df["LapTimeSec"].values

    slope, intercept = np.polyfit(x, y, 1)
    trend = intercept + slope * x

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Lap Time"
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=trend,
        mode="lines",
        name="Trend"
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"{driver} Degradation: {slope:.4f} sec/lap",
        xaxis_title="Lap",
        yaxis_title="Lap Time (sec)"
    )

    return fig

c1, c2 = st.columns(2)
c1.plotly_chart(plot_degradation(laps1, driver1), width="stretch")
c2.plotly_chart(plot_degradation(laps2, driver2), width="stretch")

# =====================================================
# 🏁 STRATEGY PROJECTION (SAFE FEATURE MATCH)
# =====================================================

comparison_section("Strategy Projection", "🏁")

def plot_projection(model, laps_df, future=10):

    current_lap = int(laps_df["LapNumber"].max())
    tire_age = 5

    feature_order = model.feature_names_in_

    cumulative = 0
    laps = []
    preds = []

    for i in range(1, future + 1):

        X = pd.DataFrame({
            col: 0 for col in feature_order
        }, index=[0])

        if "LapNumber" in feature_order:
            X["LapNumber"] = current_lap + i
        if "TireAge" in feature_order:
            X["TireAge"] = tire_age + i

        pred = model.predict(X)[0]
        cumulative += pred

        laps.append(current_lap + i)
        preds.append(cumulative)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=laps,
        y=preds,
        mode="lines",
        name="Projection"
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Lap",
        yaxis_title="Cumulative Time"
    )

    return fig

st.plotly_chart(
    plot_projection(rf1["model"], laps1),
    width="stretch"
)

# =====================================================
# 📈 PIT SENSITIVITY CURVE
# =====================================================

comparison_section("Pit Sensitivity Curve", "📈")

def plot_pit_sensitivity(model, laps_df):

    current_lap = int(laps_df["LapNumber"].max())
    penalties = list(range(18, 31))
    optimal_laps = []

    for p in penalties:
        best_lap, _ = optimize_pit_window(
            model,
            current_lap,
            current_lap + 20,
            5,
            pit_penalty=p
        )
        optimal_laps.append(best_lap)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=penalties,
        y=optimal_laps,
        mode="lines+markers"
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Pit Loss (sec)",
        yaxis_title="Optimal Pit Lap"
    )

    return fig

st.plotly_chart(
    plot_pit_sensitivity(rf1["model"], laps1),
    width="stretch"
)