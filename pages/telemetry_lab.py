"""
F1 Race Intelligence — Telemetry Lab UI
"""

import pandas as pd
import streamlit as st

from src.telemetry import (
    align_telemetry_by_distance,
    extract_lap_telemetry,
    get_circuit_coordinates,
)
from src.visualization import circuit_map_plot, telemetry_overlay_plot


def render_telemetry_lab(session, laps: pd.DataFrame, default_d1: str = "VER", default_d2: str = "HAM"):
    st.markdown("### ⚡ Telemetry Lab & Circuit Analysis")

    if session is None or laps is None or laps.empty:
        st.warning("⚠️ No session data loaded.")
        return

    available_drivers = sorted(laps["Driver"].dropna().unique())
    if len(available_drivers) < 2:
        st.warning("⚠️ At least two drivers are required for telemetry overlay.")
        return

    col1, col2 = st.columns(2)
    with col1:
        idx1 = available_drivers.index(default_d1) if default_d1 in available_drivers else 0
        driver1 = st.selectbox("Driver 1 Telemetry", available_drivers, index=idx1, key="tel_d1")
    with col2:
        idx2 = available_drivers.index(default_d2) if default_d2 in available_drivers else (1 if len(available_drivers) > 1 else 0)
        driver2 = st.selectbox("Driver 2 Telemetry", available_drivers, index=idx2, key="tel_d2")

    with st.spinner("Extracting high-frequency telemetry channels..."):
        tel1 = extract_lap_telemetry(session, driver1)
        tel2 = extract_lap_telemetry(session, driver2)

    if tel1 is None or tel2 is None or tel1.empty or tel2.empty:
        st.warning("⚠️ High-frequency telemetry is not available for this session or was not loaded with the session data.")
        st.info("💡 Practice and historical sessions from early seasons may not have full continuous telemetry streams available via FastF1.")
        return

    aligned = align_telemetry_by_distance(tel1, tel2)
    fig_tel = telemetry_overlay_plot(aligned, driver1, driver2)
    st.plotly_chart(fig_tel, use_container_width=True)

    # ── Circuit Map ──
    st.markdown("---")
    st.markdown("#### 🗺️ Circuit Layout & Velocity Profile")
    track_df = get_circuit_coordinates(session)
    if track_df is not None and not track_df.empty:
        event_name = getattr(session, "event", {}).get("EventName", "") if hasattr(session, "event") else ""
        fig_map = circuit_map_plot(track_df, event_name)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Circuit coordinates unavailable.")
