"""
F1 Race Intelligence — Tyre Intelligence & Degradation UI (Phase 1)
"""

import pandas as pd
import streamlit as st

from src.features.fuel import estimate_fuel_effect
from src.tyre_analysis import (
    analyze_all_stints,
    compute_driver_consistency_index,
    get_compound_summary,
)
from src.visualization import degradation_curve


def render_tyre_intelligence(laps: pd.DataFrame, default_driver: str = "VER"):
    st.markdown("### 🛞 Tyre Intelligence & Stint Degradation Engine")
    st.caption("Robust Huber degradation regression with fuel-burn mass correction, 95% Confidence Intervals, and tyre cliff change-point detection.")

    if laps is None or laps.empty or "Driver" not in laps.columns:
        st.warning("⚠️ No valid lap data found for tyre analysis.")
        return

    available_drivers = sorted(laps["Driver"].dropna().unique())

    # ── 1. Fuel Burn Effect Header ──
    fuel_info = estimate_fuel_effect(laps)
    fuel_val = fuel_info["fuel_effect"]
    source_label = "Empirical (Estimated)" if fuel_info["source"] == "estimated" else "Default (Physics Clamped)"

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.metric("Fuel Burn Effect", f"{fuel_val:.3f} s/lap", help="Pace gained per lap as fuel mass burns off")
    with fc2:
        st.metric("Estimation Source", source_label, help=fuel_info.get("reason", ""))
    with fc3:
        st.metric("Dry Matched Stints", fuel_info["transitions_count"])
    with fc4:
        use_fuel_corr = st.toggle("Enable Fuel Correction", value=True, help="Normalizes laps to race-start fuel mass to isolate true tyre wear")

    if not fuel_info.get("is_estimated", False):
        st.info(f"ℹ️ **Fuel Estimation Note**: Using default {fuel_val:.3f} s/lap — {fuel_info.get('reason', '')}")

    st.markdown("---")

    # ── 2. Driver Selection & Per-Stint Chart ──
    col1, col2 = st.columns([1, 2])
    with col1:
        idx = available_drivers.index(default_driver) if default_driver in available_drivers else 0
        selected_driver = st.selectbox("Select Driver to Inspect", available_drivers, index=idx)

    drv_laps = laps[laps["Driver"] == selected_driver].copy()

    st.markdown(f"#### Robust Stint Degradation & Tyre Cliffs — {selected_driver}")
    fig_deg = degradation_curve(
        drv_laps,
        selected_driver,
        fuel_effect=fuel_val,
        use_fuel_correction=use_fuel_corr,
    )
    st.plotly_chart(fig_deg, use_container_width=True)

    # ── 3. Tabs: Stint Breakdown, Consistency Index, Compound Benchmarks ──
    tab1, tab2, tab3 = st.tabs([
        "🔬 Per-Stint Robust Huber Regressions",
        "🎯 Driver Consistency & Pace Rankings",
        "🏁 Compound Performance Summary",
    ])

    with tab1:
        st.markdown(f"**Detailed Stint Huber Regressions & Cliff Diagnostics — {selected_driver}**")
        all_stints = analyze_all_stints(drv_laps, fuel_effect=fuel_val, use_fuel_correction=use_fuel_corr)
        if not all_stints.empty:
            st.dataframe(all_stints, use_container_width=True, hide_index=True)
            if (all_stints["LapsAnalyzed"] < 8).any():
                st.caption("⚠️ **Low Sample Stints (N < 8 laps)**: Stints with fewer than 8 clean laps have wider uncertainty bands; slopes should be interpreted with caution.")
        else:
            st.info("Insufficient clean laps within individual stints for robust regression fitting (minimum 5 clean laps required).")

    with tab2:
        st.markdown("**Driver Consistency Index (Robust MAD) & Session Pace Ranking**")
        st.caption("Consistency Index is computed as Median Absolute Deviation (MAD × 1.4826) of fuel-corrected clean laps. Lower is more consistent.")
        consistency_df = compute_driver_consistency_index(laps, fuel_effect=fuel_val)
        if not consistency_df.empty:
            st.dataframe(consistency_df, use_container_width=True, hide_index=True)
        else:
            st.info("Insufficient clean laps across field to compute consistency rankings.")

    with tab3:
        st.markdown("**Field-Wide Compound Benchmarks & Confidence Bounds**")
        st.caption(
            "⚠️ **Statistical Guidance**: Compound comparisons are subject to overlapping confidence intervals and sample size variance. "
            "Where 95% CIs overlap, differences in estimated degradation are statistically indistinguishable."
        )
        compound_df = get_compound_summary(laps, fuel_effect=fuel_val)
        if not compound_df.empty:
            st.dataframe(compound_df, use_container_width=True, hide_index=True)
        else:
            st.info("Compound summary not available for this session.")
