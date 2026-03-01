import os
import sys
import streamlit as st
import fastf1

# --------------------------------------
# Enable FastF1 Cache
# --------------------------------------
CACHE_DIR = "cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)

# --------------------------------------
# Add src to path
# --------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_race_data
from metrics import train_lap_models
from visualization import compare_drivers_plot, degradation_curve

# --------------------------------------
# Streamlit Config
# --------------------------------------
st.set_page_config(page_title="F1 Race Intelligence", layout="wide")
st.title("🏎️ F1 Race Intelligence Platform")

# --------------------------------------
# Sidebar
# --------------------------------------
st.sidebar.header("Race Configuration")

season = st.sidebar.selectbox("Season", [2022, 2023])
grand_prix = st.sidebar.text_input("Grand Prix", "Monaco")
driver1 = st.sidebar.text_input("Driver 1", "VER")
driver2 = st.sidebar.text_input("Driver 2", "HAM")

run_analysis = st.sidebar.button("Run Analysis")

# --------------------------------------
# Cached Race Loader
# --------------------------------------
@st.cache_data(show_spinner=False)
def get_race(year, gp):
    return load_race_data(year, gp)

model = train_model_cached(laps)

# --------------------------------------
# Main Execution
# --------------------------------------
if run_analysis:

    with st.spinner("Loading race data & training models..."):

        laps = get_race(season, grand_prix)

        if laps.empty:
            st.error("No lap data found.")
            st.stop()

        # Train model once
        model = train_model_cached(laps)

    # =====================================================
    # 🧠 Model Intelligence Overview
    # =====================================================
    st.markdown("## 🧠 Model Intelligence Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Laps", len(laps))

    with col2:
        st.metric("Drivers in Session", laps["Driver"].nunique())

    # =====================================================
    # 📊 Driver Delta Pace
    # =====================================================
    st.markdown("## 📊 Driver Delta Pace")

    fig_delta = compare_drivers_plot(laps, driver1, driver2)
    st.plotly_chart(fig_delta, use_container_width=True)

    # =====================================================
    # 📉 Tire Degradation
    # =====================================================
    st.markdown("## 📉 Tire Degradation")

    from metrics import clean_laps

    laps1_clean = clean_laps(laps.pick_drivers(driver1))
    laps2_clean = clean_laps(laps.pick_drivers(driver2))

    col1, col2 = st.columns(2)

    with col1:
        fig_deg1 = degradation_curve(laps1_clean, driver1)
        st.plotly_chart(fig_deg1, use_container_width=True)

    with col2:
        fig_deg2 = degradation_curve(laps2_clean, driver2)
        st.plotly_chart(fig_deg2, use_container_width=True)

else:
    st.info("Configure parameters and click 'Run Analysis' to begin.")