"""
F1 Race Intelligence — Pit Wall Strategy Simulator UI
"""

import pandas as pd
import streamlit as st

from src.strategy_simulator import (
    evaluate_pit_window,
    pit_loss_sensitivity_analysis,
    simulate_strategy_horizon,
)
from src.visualization import (
    pit_window_optimizer_plot,
    strategy_horizon_plot,
)


def render_strategy_simulator(laps: pd.DataFrame, default_driver: str = "VER"):
    st.markdown("### ⏱️ Pit Wall Strategy Simulator")
    st.markdown("Model race-distance scenarios, calculate mathematically optimal pit windows, and run sensitivity checks against pit loss.")

    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        current_lap = st.number_input("Current Lap Number", min_value=1, max_value=70, value=15, step=1)
        total_laps = st.number_input("Total Race Laps", min_value=10, max_value=78, value=50, step=1)
    with col2:
        current_tire_age = st.number_input("Current Tyre Age (laps)", min_value=1, max_value=40, value=15, step=1)
        current_deg = st.slider("Current Compound Degradation (s/lap)", min_value=0.01, max_value=0.30, value=0.10, step=0.01)
    with col3:
        pit_loss = st.slider("Pit Loss Penalty (seconds)", min_value=14.0, max_value=30.0, value=22.0, step=0.5)
        new_compound_delta = st.slider("New Tyre Initial Pace Advantage (s/lap)", min_value=-2.0, max_value=0.5, value=-0.6, step=0.1)

    pit_lap = st.slider("Scheduled Pit Lap for Plan B", min_value=int(current_lap), max_value=int(total_laps) - 1, value=min(int(current_lap) + 5, int(total_laps) - 1))

    # Run Simulation
    sim_res = simulate_strategy_horizon(
        current_lap=int(current_lap),
        total_race_laps=int(total_laps),
        current_tire_age=int(current_tire_age),
        base_pace=80.0,
        current_compound_deg=float(current_deg),
        new_compound_base_offset=float(new_compound_delta),
        new_compound_deg=0.06,
        pit_loss=float(pit_loss),
        pit_lap=int(pit_lap),
    )

    # Summary KPI
    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Total Projected Time (Stay Out)", f"{sim_res['total_stay_time']:.1f}s")
    with sc2:
        st.metric(f"Total Projected Time (Pit Lap {pit_lap})", f"{sim_res['total_pit_time']:.1f}s")
    with sc3:
        adv = sim_res["net_advantage"]
        st.metric("Net Advantage / Time Saved", f"{adv:+.2f}s", delta=sim_res["optimal_strategy"])

    # Strategy Horizon Plot
    fig_sim = strategy_horizon_plot(sim_res["laps"], int(pit_lap))
    st.plotly_chart(fig_sim, use_container_width=True)

    # ── Pit Window Optimizer ──
    st.markdown("---")
    st.markdown("#### 🎯 Mathematical Pit Window Optimization")
    st.caption("Evaluates every candidate pit lap within a ±6 lap window around current lap.")

    start_cand = max(int(current_lap), 2)
    end_cand = min(int(total_laps) - 2, int(current_lap) + 12)

    eval_df = evaluate_pit_window(
        start_candidate_lap=start_cand,
        end_candidate_lap=end_cand,
        total_race_laps=int(total_laps),
        current_lap=int(current_lap),
        current_tire_age=int(current_tire_age),
        base_pace=80.0,
        current_deg=float(current_deg),
        new_deg=0.06,
        new_compound_delta=float(new_compound_delta),
        pit_loss=float(pit_loss),
    )

    fig_opt = pit_window_optimizer_plot(eval_df)
    st.plotly_chart(fig_opt, use_container_width=True)

    # ── Sensitivity Analysis ──
    st.markdown("---")
    st.markdown("#### 🔬 Pit Loss Sensitivity Matrix")
    sens_df = pit_loss_sensitivity_analysis(
        current_lap=int(current_lap),
        total_race_laps=int(total_laps),
        current_tire_age=int(current_tire_age),
        base_pace=80.0,
        current_deg=float(current_deg),
        new_deg=0.06,
        new_compound_delta=float(new_compound_delta),
        pit_lap=int(pit_lap),
    )
    st.dataframe(sens_df, use_container_width=True, hide_index=True)
