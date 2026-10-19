"""
F1 Race Intelligence 3.0 — Driver Telemetry Lab & Multi-Car Comparison

High-frequency telemetry overlays, corner-by-corner braking points,
minimum apex speed matrix, micro-sector delta ribbons (50m bins),
and aerodynamic speed-trap telemetry.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.telemetry import (
    align_telemetry_by_distance,
    extract_lap_telemetry,
)
from src.visualization.charts import telemetry_overlay_plot
from src.visualization.track_3d import Track3DEngine


def reconstruct_offline_telemetry(
    circuit_name: str,
    driver: str,
    driver_laps: pd.DataFrame,
    num_points: int = 600,
) -> pd.DataFrame:
    """
    Reconstructs continuous distance-aligned telemetry for offline demo data
    using the authentic circuit apex geometry and driver speed-trap metrics.
    """
    engine = Track3DEngine(circuit_name=circuit_name, num_spline_points=num_points)
    base_track = engine.track_df.copy()

    # Get driver fastest lap in session
    fastest_lap = driver_laps.loc[driver_laps["LapTimeSec"].idxmin()] if not driver_laps.empty else None

    # Base track distance (meters)
    approx_length_m = 5400.0
    distance = base_track["progress"].values * approx_length_m

    # Base speed curve from 3D track model
    spd = base_track["speed"].values.copy()

    # Scale with actual speed trap if present
    if fastest_lap is not None and pd.notna(fastest_lap.get("SpeedST")):
        scale_st = float(fastest_lap["SpeedST"]) / (spd.max() + 1e-6)
        spd = spd * (0.3 + 0.7 * scale_st)

    # Derive Throttle & Brake from speed gradient
    accel = np.gradient(spd, distance + 1.0)
    throttle = np.clip(accel * 45.0 + 50.0, 0.0, 100.0)
    brake = np.clip(-accel * 75.0, 0.0, 100.0)

    # Clean throttle / brake overlap
    throttle[brake > 10.0] = 0.0

    # Gear estimation from speed
    gear = np.clip(np.digitize(spd, [0, 80, 120, 160, 200, 240, 280, 320]), 1, 8)
    rpm = np.clip(9000 + (spd % 40) * 120, 9500, 12500)

    return pd.DataFrame({
        "Distance": distance,
        "Speed": spd,
        "Throttle": throttle,
        "Brake": brake,
        "nGear": gear,
        "RPM": rpm,
        "Driver": driver,
    })


def compute_corner_apex_speeds(tel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies local speed minima (corner apexes) and braking onset distance.
    """
    if tel_df is None or tel_df.empty or "Speed" not in tel_df.columns:
        return pd.DataFrame()

    spd = tel_df["Speed"].values
    dist = tel_df["Distance"].values

    # Find local minima with minimum prominence
    apexes = []
    for i in range(10, len(spd) - 10):
        if spd[i] < spd[i - 5:i].min() and spd[i] < spd[i + 1:i + 6].min() and spd[i] < 210.0:
            # Find preceding braking point
            lookback = max(0, i - 40)
            brake_idx = i
            for j in range(i, lookback, -1):
                if tel_df["Brake"].iloc[j] > 10.0 or tel_df["Throttle"].iloc[j] < 15.0:
                    brake_idx = j
            brake_dist = dist[brake_idx]
            apex_dist = dist[i]

            apexes.append({
                "Corner": f"Turn {len(apexes) + 1}",
                "Apex Distance (m)": int(apex_dist),
                "Apex Speed (km/h)": round(float(spd[i]), 1),
                "Braking Point (m)": int(brake_dist),
                "Braking Distance (m)": max(0, int(apex_dist - brake_dist)),
            })

    return pd.DataFrame(apexes).drop_duplicates(subset=["Apex Distance (m)"]).head(15)


def compute_micro_sector_ribbon(aligned_df: pd.DataFrame, bin_size_m: float = 50.0) -> pd.DataFrame:
    """
    Computes time delta across 50m micro-sectors.
    Positive delta = Driver 1 is faster.
    """
    if aligned_df is None or aligned_df.empty or "Distance" not in aligned_df.columns:
        return pd.DataFrame()

    max_dist = aligned_df["Distance"].max()
    bins = np.arange(0, max_dist, bin_size_m)
    labels = bins[:-1] + (bin_size_m / 2.0)

    aligned_df["SectorBin"] = pd.cut(aligned_df["Distance"], bins=bins, labels=labels)
    grouped = aligned_df.groupby("SectorBin", observed=False).agg({
        "Speed_1": "mean",
        "Speed_2": "mean",
    }).reset_index()

    grouped["Distance"] = grouped["SectorBin"].astype(float)
    # Delta time in seconds: dt = distance / v
    v1_ms = np.maximum(grouped["Speed_1"] / 3.6, 10.0)
    v2_ms = np.maximum(grouped["Speed_2"] / 3.6, 10.0)
    # Time delta over 50m bin: positive means driver 1 faster (takes less time)
    grouped["TimeDeltaSec"] = (bin_size_m / v2_ms) - (bin_size_m / v1_ms)
    grouped["CumulativeDeltaSec"] = grouped["TimeDeltaSec"].cumsum()

    return grouped


def render_telemetry_lab(
    session: Any,
    laps: Optional[pd.DataFrame],
    default_d1: str = "VER",
    default_d2: str = "HAM",
):
    st.markdown("""
    <div style="border-bottom: 2px solid #E10600; padding-bottom: 8px; margin-bottom: 18px;">
        <h2 style="margin: 0; font-size: 24px; font-weight: 800; color: #FFFFFF;">DRIVER TELEMETRY LAB & MULTI-CAR COMPARISON</h2>
        <div style="font-size: 11px; color: #9E9E9E; letter-spacing: 0.5px;">HIGH-FREQUENCY TRACES • 50M MICRO-SECTOR DELTAS • CORNER APEX DYNAMICS</div>
    </div>
    """, unsafe_allow_html=True)

    if laps is None or laps.empty:
        st.warning("⚠️ No lap data loaded. Please select a session from the sidebar.")
        return

    available_drivers = sorted(laps["Driver"].dropna().unique())
    if len(available_drivers) < 2:
        st.warning("⚠️ At least two drivers are required for telemetry comparison.")
        return

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        idx1 = available_drivers.index(default_d1) if default_d1 in available_drivers else 0
        driver1 = st.selectbox("Driver 1 Telemetry (Reference)", available_drivers, index=idx1, key="tel_d1")
    with col_sel2:
        idx2 = available_drivers.index(default_d2) if default_d2 in available_drivers else (1 if len(available_drivers) > 1 else 0)
        driver2 = st.selectbox("Driver 2 Telemetry (Challenger)", available_drivers, index=idx2, key="tel_d2")

    circuit_name = laps["GrandPrix"].iloc[0] if "GrandPrix" in laps.columns else "Bahrain Grand Prix"

    # Extract or reconstruct telemetry
    d1_laps = laps[laps["Driver"] == driver1]
    d2_laps = laps[laps["Driver"] == driver2]
    is_live_telemetry = session is not None and hasattr(session, "laps")

    if is_live_telemetry:
        with st.spinner("Extracting continuous high-frequency channels from FastF1..."):
            tel1 = extract_lap_telemetry(session, driver1)
            tel2 = extract_lap_telemetry(session, driver2)
        st.success(
            "📡 **[OBSERVED] Live Onboard Telemetry Loaded:** Continuous high-frequency time-series data "
            "directly extracted from FastF1 session channels."
        )
    else:
        tel1 = reconstruct_offline_telemetry(circuit_name, driver1, d1_laps)
        tel2 = reconstruct_offline_telemetry(circuit_name, driver2, d2_laps)
        st.warning(
            "⚠️ **[SIMULATED] Reconstructed Telemetry (Offline Educational Mode):** "
            "High-frequency time-series telemetry is not stored in the lightweight offline demo dataset. "
            "Speed traces are reconstructed from verified circuit apex geometry and driver speed-trap observations (`SpeedST`). "
            "Throttle, Brake, RPM, and Gear traces are simulated from longitudinal acceleration gradients ($\nabla v$) "
            "and MUST NOT be treated as recorded onboard physical measurements."
        )

    if tel1 is None or tel2 is None or tel1.empty or tel2.empty:
        st.error("⚠️ Telemetry traces could not be extracted.")
        return

    aligned = align_telemetry_by_distance(tel1, tel2)
    if aligned.empty:
        st.error("⚠️ Failed to align telemetry by distance.")
        return

    # ─── High-Frequency Overlay Plot ──────────────────────────────
    plot_badge = "[OBSERVED]" if is_live_telemetry else "[SIMULATED]"
    st.markdown(f"#### 📊 High-Frequency Overlay Traces {plot_badge}")
    fig_tel = telemetry_overlay_plot(aligned, driver1, driver2)
    st.plotly_chart(fig_tel, use_container_width=True)

    st.divider()

    # ─── 50m Micro-Sector Delta Ribbon ────────────────────────────
    ribbon_badge = "[DERIVED]" if is_live_telemetry else "[SIMULATED]"
    st.markdown(f"#### ⚡ Micro-Sector Delta Ribbon (50m Bins) {ribbon_badge}")
    st.caption(
        "Shows meter-by-meter where lap time is won or lost. "
        "**Polarity Convention:** Positive Cumulative Delta = Driver 1 is faster (ahead on elapsed micro-sector time)."
    )

    ribbon_df = compute_micro_sector_ribbon(aligned, bin_size_m=50.0)
    if not ribbon_df.empty:
        fig_ribbon = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)

        # Micro-sector delta bar
        colors = ["#00F5D4" if dt >= 0 else "#FF1801" for dt in ribbon_df["TimeDeltaSec"]]
        fig_ribbon.add_trace(
            go.Bar(
                x=ribbon_df["Distance"],
                y=ribbon_df["TimeDeltaSec"],
                marker=dict(color=colors),
                name="Per-Sector Gain (s)",
                hoverinfo="x+y",
            ),
            row=1, col=1,
        )

        # Cumulative time delta
        fig_ribbon.add_trace(
            go.Scatter(
                x=ribbon_df["Distance"],
                y=ribbon_df["CumulativeDeltaSec"],
                mode="lines",
                line=dict(color="#FFD700", width=2.5),
                name=f"Cumulative Delta ({driver1} vs {driver2})",
            ),
            row=2, col=1,
        )

        fig_ribbon.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0C10",
            plot_bgcolor="#0A0C10",
            height=340,
            margin=dict(l=30, r=20, t=20, b=30),
            xaxis2=dict(title="Lap Distance (meters)", gridcolor="#1F2238"),
            yaxis1=dict(title="Δt / 50m (s)", gridcolor="#1F2238"),
            yaxis2=dict(title="Net Delta (s)", gridcolor="#1F2238"),
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(10,12,16,0.8)", font=dict(color="#e0e0e0", size=10)),
        )
        st.plotly_chart(fig_ribbon, use_container_width=True)

    # ─── Corner Apex Speeds & Braking Points Matrix ───────────────
    apex_badge = "[DERIVED]" if is_live_telemetry else "[SIMULATED]"
    st.markdown(f"#### 🎯 Corner Apex Speeds & Braking Points Matrix {apex_badge}")
    st.caption("Identifies local speed minima (corner apexes) and preceding longitudinal braking onset distance.")
    col_ap1, col_ap2 = st.columns(2)

    with col_ap1:
        st.markdown(f"**{driver1} Corner Apex Dynamics**")
        apexes_d1 = compute_corner_apex_speeds(tel1)
        if not apexes_d1.empty:
            st.dataframe(apexes_d1, use_container_width=True, hide_index=True)
        else:
            st.caption("Apex telemetry not available.")

    with col_ap2:
        st.markdown(f"**{driver2} Corner Apex Dynamics**")
        apexes_d2 = compute_corner_apex_speeds(tel2)
        if not apexes_d2.empty:
            st.dataframe(apexes_d2, use_container_width=True, hide_index=True)
        else:
            st.caption("Apex telemetry not available.")

    # ─── Aerodynamic Speed Trap Comparison ────────────────────────
    st.markdown("---")
    st.markdown("#### 🏎️ Aerodynamic & Speed Trap Comparison [OBSERVED]")
    st.caption("Recorded maximum velocities at official FIA circuit timing traps across Intermediate 1, Intermediate 2, Finish Line, and Speed Trap.")

    speed_cols = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
    present_cols = [c for c in speed_cols if c in laps.columns]

    if present_cols:
        st_summary = []
        for d in [driver1, driver2]:
            sub_d = laps[laps["Driver"] == d]
            row = {"Driver": d}
            for col in present_cols:
                row[f"Max {col} (km/h)"] = round(sub_d[col].max(), 1) if not sub_d.empty and pd.notna(sub_d[col].max()) else "N/A"
            st_summary.append(row)
        st.dataframe(pd.DataFrame(st_summary), use_container_width=True, hide_index=True)
    else:
        st.caption("Speed trap channels not present in active session.")
