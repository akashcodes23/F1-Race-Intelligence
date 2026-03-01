import os
import sys
import streamlit as st
import fastf1
import pandas as pd
import numpy as np

# --------------------------------
# Enable FastF1 Disk Cache
# --------------------------------
CACHE_DIR = "cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)

# --------------------------------
# Add src to path
# --------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_race_data
from metrics import train_lap_models, evaluate_models
from visualization import (
    plot_driver_delta,
    plot_tire_degradation,
    plot_strategy_projection,
    plot_pit_sensitivity
)

# --------------------------------
# Streamlit Page Config
# --------------------------------
st.set_page_config(
    page_title="F1 Race Intelligence",
    layout="wide"
)

st.title("🏎️ F1 Race Intelligence Platform")

# --------------------------------
# Sidebar Controls
# --------------------------------
st.sidebar.header("Race Configuration")

season = st.sidebar.selectbox("Season", [2022, 2023])
grand_prix = st.sidebar.text_input("Grand Prix Name", "Monaco")
driver1 = st.sidebar.text_input("Driver 1", "VER")
driver2 = st.sidebar.text_input("Driver 2", "HAM")

run_analysis = st.sidebar.button("Run Analysis")

# --------------------------------
# Cached Session Loader
# --------------------------------
@st.cache_data(show_spinner=False)
def get_session(year, gp):
    session = fastf1.get_session(year, gp, "R")
    session.load()
    return session

# --------------------------------
# Cached Driver Laps
# --------------------------------
@st.cache_data(show_spinner=False)
def get_driver_laps(year, gp, driver):
    session = get_session(year, gp)
    laps = session.laps.pick_driver(driver)
    return laps

# --------------------------------
# Cached Model Training
# --------------------------------
@st.cache_resource
def train_models_cached(laps):
    return train_lap_models(laps)

# --------------------------------
# Main Execution Block
# --------------------------------
if run_analysis:

    with st.spinner("Loading telemetry & training models..."):

        # Load driver data
        laps1 = get_driver_laps(season, grand_prix, driver1)
        laps2 = get_driver_laps(season, grand_prix, driver2)

        if laps1.empty or laps2.empty:
            st.error("No lap data found. Check driver names or race.")
            st.stop()

        # Train models
        model1 = train_models_cached(laps1)
        model2 = train_models_cached(laps2)

        # Evaluate
        metrics1 = evaluate_models(model1, laps1)
        metrics2 = evaluate_models(model2, laps2)

    # --------------------------------
    # 🧠 Model Intelligence Overview
    # --------------------------------
    st.markdown("## 🧠 Model Intelligence Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Driver 1 R²", round(metrics1["r2"], 3))
        st.metric("Driver 1 RMSE", round(metrics1["rmse"], 3))

    with col2:
        st.metric("Driver 2 R²", round(metrics2["r2"], 3))
        st.metric("Driver 2 RMSE", round(metrics2["rmse"], 3))

    # --------------------------------
    # 📊 Driver Delta Pace
    # --------------------------------
    st.markdown("## 📊 Driver Delta Pace")
    fig_delta = plot_driver_delta(laps1, laps2)
    st.plotly_chart(fig_delta, use_container_width=True)

    # --------------------------------
    # 📉 Tire Degradation
    # --------------------------------
    st.markdown("## 📉 Tire Degradation")
    fig_tire = plot_tire_degradation(laps1, laps2)
    st.plotly_chart(fig_tire, use_container_width=True)

    # --------------------------------
    # 🏁 Strategy Projection
    # --------------------------------
    st.markdown("## 🏁 Strategy Projection")
    fig_strategy = plot_strategy_projection(model1, model2)
    st.plotly_chart(fig_strategy, use_container_width=True)

    # --------------------------------
    # 📈 Pit Sensitivity Curve
    # --------------------------------
    st.markdown("## 📈 Pit Sensitivity Curve")
    fig_pit = plot_pit_sensitivity(model1, model2)
    st.plotly_chart(fig_pit, use_container_width=True)

else:
    st.info("Configure race parameters and click 'Run Analysis' to begin.")