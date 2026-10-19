"""
F1 Race Intelligence — Driver Performance Lab UI
"""

import pandas as pd
import streamlit as st

from src.driver_comparison import (
    compare_driver_consistency,
    compare_driver_sectors,
    compute_driver_delta,
)
from src.visualization import compare_drivers_plot


def render_driver_comparison(laps: pd.DataFrame, default_d1: str = "VER", default_d2: str = "HAM"):
    st.markdown("### ⚔️ Driver Performance Lab: Head-to-Head Comparison")

    if laps is None or laps.empty or "Driver" not in laps.columns:
        st.warning("⚠️ No valid lap data found for driver comparison.")
        return

    available_drivers = sorted(laps["Driver"].dropna().unique())
    if len(available_drivers) < 2:
        st.warning("⚠️ At least 2 drivers are required for comparison.")
        return

    # Driver Selectors
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        idx1 = available_drivers.index(default_d1) if default_d1 in available_drivers else 0
        driver1 = st.selectbox("Select Driver 1", available_drivers, index=idx1, key="comp_d1")
    with col_d2:
        idx2 = available_drivers.index(default_d2) if default_d2 in available_drivers else (1 if len(available_drivers) > 1 else 0)
        driver2 = st.selectbox("Select Driver 2", available_drivers, index=idx2, key="comp_d2")

    if driver1 == driver2:
        st.info("ℹ️ Select two different drivers to view head-to-head comparison analytics.")
        return

    # Compute deltas
    merged_delta, summary = compute_driver_delta(laps, driver1, driver2)

    if merged_delta.empty:
        st.warning(f"⚠️ No matching laps found between {driver1} and {driver2}.")
        return

    # ── Summary KPI Cards [DERIVED / OBSERVED] ──
    st.caption("ℹ️ **Scientific Integrity Note:** Metric Polarity: Positive (+) Delta indicates Driver 1 was slower than Driver 2 on matched laps.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Matched Laps [OBSERVED]", summary["matched_laps"], help="Count of clean matched laps completed by both drivers")
    with c2:
        st.metric(
            f"Faster Laps ({driver1} vs {driver2}) [DERIVED]",
            f"{summary['driver1_faster_count']} - {summary['driver2_faster_count']}",
            help="Count of individual laps where each driver set the lower lap time",
        )
    with c3:
        st.metric(
            "Median Lap Delta [DERIVED]",
            f"{summary['median_delta']:+.3f}s",
            help="Median of (Driver 1 Lap Time - Driver 2 Lap Time). Positive means Driver 1 slower.",
        )
    with c4:
        st.metric(
            "Cumulative Gap [DERIVED]",
            f"{summary['final_cumulative_delta']:+.3f}s",
            delta=f"{summary['faster_driver']} ahead",
            delta_color="inverse",
            help="Sum of lap deltas across all matched laps",
        )

    st.markdown("---")

    # ── Delta Chart ──
    fig_delta = compare_drivers_plot(laps, driver1, driver2)
    st.plotly_chart(fig_delta, use_container_width=True)

    # ── Consistency & Distribution Comparison ──
    st.markdown("---")
    st.markdown("#### 🎯 Consistency & Pace Dispersion [DERIVED]")
    st.caption(
        "Dispersion metrics computed across verified clean laps. "
        "**Context & Confounders:** Consistency measures lap-to-lap pace repeatability; "
        "it does not equate directly to driver skill and is confounded by traffic, fuel burn, tyre degradation, and car machinery."
    )
    consistency = compare_driver_consistency(laps, driver1, driver2)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"**{driver1} Pace Profile**")
        p1 = consistency.get(driver1, {})
        st.write(f"- **Best Clean Lap [OBSERVED]:** `{p1.get('best_formatted', 'N/A')}`")
        st.write(f"- **Median Pace [DERIVED]:** `{p1.get('median_formatted', 'N/A')}`")
        st.write(f"- **Standard Deviation (Variability):** `{p1.get('std', 0):.3f}s`")
        st.write(f"- **Interquartile Range (IQR):** `{p1.get('iqr', 0):.3f}s`")
        st.write(f"- **Valid Clean Laps Analyzed:** `{p1.get('n_laps', 0)}`")

    with cc2:
        st.markdown(f"**{driver2} Pace Profile**")
        p2 = consistency.get(driver2, {})
        st.write(f"- **Best Clean Lap [OBSERVED]:** `{p2.get('best_formatted', 'N/A')}`")
        st.write(f"- **Median Pace [DERIVED]:** `{p2.get('median_formatted', 'N/A')}`")
        st.write(f"- **Standard Deviation (Variability):** `{p2.get('std', 0):.3f}s`")
        st.write(f"- **Interquartile Range (IQR):** `{p2.get('iqr', 0):.3f}s`")
        st.write(f"- **Valid Clean Laps Analyzed:** `{p2.get('n_laps', 0)}`")

    # ── Sector-by-Sector Comparison ──
    sector_df, sector_stats = compare_driver_sectors(laps, driver1, driver2)
    if not sector_df.empty:
        st.markdown("---")
        st.markdown("#### ⚡ Sector Performance & Hypothetical Optimal Lap")
        st.caption(
            "⚠️ **[SIMULATED] Theoretical Construction:** The hypothetical optimal lap combines individual best sector times "
            "($S1_{best} + S2_{best} + S3_{best}$) which may originate from non-contiguous laps, different tyre compounds, "
            "fuel masses, or evolving track conditions. It does not represent an achievable single-lap physical run."
        )
        st.dataframe(sector_df, use_container_width=True, hide_index=True)
