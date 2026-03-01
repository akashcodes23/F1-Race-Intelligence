import streamlit as st
import pandas as pd

from data_loader import load_race_data
from visualization import (
    compare_drivers_plot,
    degradation_curve
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="F1 Race Intelligence",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Race Intelligence Platform")

# =====================================================
# CACHED DATA LOADER
# =====================================================

@st.cache_data(show_spinner=False)
def get_session_laps(year, grand_prix, session_type):
    return load_race_data(year, grand_prix, session_type)

# =====================================================
# SIDEBAR CONFIGURATION
# =====================================================

st.sidebar.header("Race Configuration")

season = st.sidebar.selectbox(
    "Season",
    [2021, 2022, 2023, 2024]
)

grand_prix = st.sidebar.text_input(
    "Grand Prix",
    "Monaco"
)

session_type = st.sidebar.selectbox(
    "Session Type",
    ["R", "Q", "FP1", "FP2", "FP3"]
)

driver1 = st.sidebar.text_input("Driver 1 (3-letter code)", "VER")
driver2 = st.sidebar.text_input("Driver 2 (3-letter code)", "HAM")

run_analysis = st.sidebar.button("Run Analysis")

# =====================================================
# MAIN EXECUTION
# =====================================================

if run_analysis:

    with st.spinner("Loading race data..."):
        laps = get_session_laps(season, grand_prix, session_type)

    if laps.empty:
        st.error("No lap data found. Please check inputs.")
        st.stop()

    st.success("Data Loaded Successfully")

    # =====================================================
    # DRIVER COMPARISON
    # =====================================================

    st.markdown("## 📊 Driver Delta Comparison")

    try:
        fig_delta = compare_drivers_plot(laps, driver1, driver2)
        st.plotly_chart(fig_delta, use_container_width=True)
    except Exception as e:
        st.error(f"Driver comparison failed: {e}")

    # =====================================================
    # TIRE DEGRADATION
    # =====================================================

    st.markdown("## 📉 Tire Degradation Analysis")

    try:
        driver1_laps = laps.pick_driver(driver1)
        driver2_laps = laps.pick_driver(driver2)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{driver1} Degradation")
            fig_deg1 = degradation_curve(driver1_laps, driver1)
            st.plotly_chart(fig_deg1, use_container_width=True)

        with col2:
            st.subheader(f"{driver2} Degradation")
            fig_deg2 = degradation_curve(driver2_laps, driver2)
            st.plotly_chart(fig_deg2, use_container_width=True)

    except Exception as e:
        st.error(f"Degradation analysis failed: {e}")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption("Built with FastF1 + Streamlit | AI Motorsport Intelligence")