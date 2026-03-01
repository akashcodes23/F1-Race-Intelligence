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
from visualization import compare_drivers_plot, degradation_curve
from metrics import clean_laps

# --------------------------------------
# Streamlit Config
# --------------------------------------
st.set_page_config(page_title="F1 Race Intelligence", layout="wide")
st.markdown("""
# 🏎️ F1 Race Intelligence Platform  
### AI-Driven Motorsport Analytics Dashboard
""")
st.divider()
# --------------------------------------
# Sidebar
# --------------------------------------
st.sidebar.header("Race Configuration")

season = st.sidebar.selectbox("Season", [2022, 2023])
grand_prix = st.sidebar.text_input("Grand Prix", "Monaco")
driver1 = st.sidebar.text_input("Driver 1 (3-letter code)", "VER")
driver2 = st.sidebar.text_input("Driver 2 (3-letter code)", "HAM")

run_analysis = st.sidebar.button("Run Analysis")
# --------------------------------------
# Cached Race Loader
# --------------------------------------
@st.cache_data(show_spinner=False)
def get_race(year, gp):
    return load_race_data(year, gp)


# --------------------------------------
# Main Execution
# --------------------------------------
if run_analysis:

    with st.spinner("Loading race data..."):


        laps = get_race(season, grand_prix)

    if laps is None or len(laps) == 0:
        st.error("No lap data found.")
        st.stop()

    st.success("Data loaded successfully. Analysis ready.")

    # --------------------------------------
    # Defensive Driver Validation
    # --------------------------------------
    available_drivers = laps["Driver"].unique()

    if driver1 not in available_drivers:
        st.error(f"Driver {driver1} not found in this session.")
        st.stop()

    if driver2 not in available_drivers:
        st.error(f"Driver {driver2} not found in this session.")
        st.stop()
    # =====================================================
    # 🧠 Model Intelligence Overview
    # =====================================================
    st.markdown("## 🧠 Model Intelligence Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Laps", len(laps))

    with col2:
        st.metric("Drivers in Session", laps["Driver"].nunique())
    st.divider()
    # =====================================================
    # 📊 Driver Delta Pace
    # =====================================================
    st.markdown("## 📊 Driver Delta Pace")

    fig_delta = compare_drivers_plot(laps, driver1, driver2)
    st.plotly_chart(fig_delta, use_container_width=True)
    st.divider()
    # =====================================================
    # 📉 Tire Degradation
    # =====================================================
    st.markdown("## 📉 Tire Degradation")

    laps1_clean = clean_laps(laps.pick_driver(driver1))
    laps2_clean = clean_laps(laps.pick_driver(driver2))

    col1, col2 = st.columns(2)

    with col1:
        fig_deg1 = degradation_curve(laps1_clean, driver1)
        st.plotly_chart(fig_deg1, use_container_width=True)

    with col2:
        fig_deg2 = degradation_curve(laps2_clean, driver2)
        st.plotly_chart(fig_deg2, use_container_width=True)

else:
    st.info("Configure parameters and click 'Run Analysis' to begin.")

st.divider()