"""
F1 Race Intelligence — Race Overview UI
"""

import pandas as pd
import streamlit as st

from src.race_analytics import (
    get_position_progression_data,
    get_session_leaderboard,
    get_session_summary,
    get_stint_summary,
)
from src.visualization import (
    lap_time_distribution_plot,
    position_progression_plot,
    stint_strategy_chart,
)


def render_race_overview(session, laps: pd.DataFrame):
    st.markdown("### 🏁 Session Overview & Race Intelligence")

    if laps is None or laps.empty:
        st.warning("⚠️ No valid lap data found for this session.")
        return

    summary = get_session_summary(session, laps)

    # ── Top KPI Banner ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Laps Recorded", f"{summary['total_laps']:,}")
    with col2:
        st.metric("Drivers in Session", summary["total_drivers"])
    with col3:
        fastest_txt = summary.get("fastest_lap_formatted", "N/A")
        drv = summary.get("fastest_driver", "N/A")
        st.metric("Fastest Lap", fastest_txt, delta=f"{drv} (L{summary.get('fastest_lap_num', '-')})")
    with col4:
        comp = summary.get("fastest_compound", "N/A")
        st.metric("Fastest Compound", comp)

    st.markdown("---")

    # ── Leaderboard & Position Progression Tabs ──
    tab1, tab2, tab3 = st.tabs(["📊 Session Leaderboard", "📈 Position Progression", "⏱️ Pace Distribution"])

    with tab1:
        st.markdown("#### Official Session Leaderboard")
        leaderboard = get_session_leaderboard(laps)
        if not leaderboard.empty:
            st.dataframe(
                leaderboard,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Leaderboard times not available.")

    with tab2:
        st.markdown("#### Position Evolution Across Laps")
        pos_df = get_position_progression_data(laps)
        if not pos_df.empty:
            all_drivers = sorted(pos_df["Driver"].unique())
            focus_drivers = st.multiselect(
                "Highlight Drivers (leave empty to show all):",
                options=all_drivers,
                default=all_drivers[:4] if len(all_drivers) >= 4 else all_drivers,
            )
            fig_pos = position_progression_plot(pos_df, focus_drivers if focus_drivers else None)
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("Position tracking data is not available for this session format.")

    with tab3:
        st.markdown("#### Clean Lap-Time Distribution (Top Drivers)")
        fig_dist = lap_time_distribution_plot(laps)
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Stint Strategy ──
    st.markdown("---")
    st.markdown("#### 🛞 Tyre Stints & Pit Stop Breakdown")
    stint_df = get_stint_summary(laps)
    if not stint_df.empty:
        fig_stint = stint_strategy_chart(stint_df)
        st.plotly_chart(fig_stint, use_container_width=True)

        with st.expander("📋 View Detailed Stint Telemetry Table"):
            st.dataframe(stint_df, use_container_width=True, hide_index=True)
    else:
        st.info("Tyre stint breakdown is only available for full race or timed sessions.")
