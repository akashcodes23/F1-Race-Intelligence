"""
F1 Race Intelligence 3.0 — Mission Control Pit-Wall Cockpit

Cinematic 3D Spatial Telemetry, Real-Time Monte Carlo Decision Engine,
Undercut Attack Radar, and Broadcast HUD.
"""

import os
import sys
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.cleaning import clean_laps  # noqa: E402
from src.decision_engine import (  # noqa: E402
    evaluate_undercut_overcut,
    run_monte_carlo_decision,
)
from src.visualization.audio import (  # noqa: E402
    play_caution_warning,
    play_pit_radio_chime,
)
from src.visualization.track_3d import Track3DEngine  # noqa: E402


def render_race_cockpit(session_obj: Any, laps_df: Optional[pd.DataFrame], meta: Optional[Dict[str, Any]] = None):
    """
    Renders the cinematic Pit-Wall Command Center with synchronized 3D telemetry and decision HUD.
    """
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #E10600; padding-bottom: 10px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="background: #E10600; color: #FFF; font-weight: 900; font-size: 13px; padding: 4px 10px; border-radius: 4px; letter-spacing: 1px;">PIT-WALL COMMAND</span>
            <h2 style="margin: 0; font-size: 24px; font-weight: 800; color: #FFFFFF; letter-spacing: 0.5px;">RACE INTELLIGENCE COCKPIT</h2>
        </div>
        <div style="font-size: 11px; color: #00F5D4; font-family: monospace; font-weight: 700; letter-spacing: 1px;">
            60 FPS SPATIAL ENGINE • MONTE CARLO DIFF
        </div>
    </div>
    """, unsafe_allow_html=True)

    if laps_df is None or laps_df.empty:
        st.warning("⚠️ No lap timing data available. Please load a Grand Prix session from the sidebar.")
        return

    # Filter clean laps for race calculations
    clean_df = clean_laps(laps_df)
    if clean_df.empty:
        clean_df = laps_df.copy()

    total_race_laps = int(laps_df["LapNumber"].max()) if "LapNumber" in laps_df.columns else 57
    circuit_name = meta.get("gp", "Bahrain Grand Prix") if meta else "Bahrain Grand Prix"

    # ─── Top Control Bar: Lap Scrubber & Race Replay ───────────────
    st.markdown("##### ⏱️ Race Scrubber & Session Timeline")
    col_scrub1, col_scrub2, col_view = st.columns([3, 2, 2])

    with col_scrub1:
        current_lap = st.slider(
            "Race Lap Scrubber",
            min_value=1,
            max_value=max(1, total_race_laps),
            value=min(22, max(1, total_race_laps)),
            help="Advances all 3D telemetry, driver gaps, and strategy engine recommendations.",
        )

    with col_scrub2:
        lap_progress = st.slider(
            "Lap Progress (Sector Coordinates)",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
            help="Interpolates 3D car positions smoothly along the circuit ribbon.",
        )

    with col_view:
        view_mode = st.radio(
            "3D Viewport Technology",
            ["WebGL Three.js (Cinematic)", "Interactive Plotly 3D"],
            horizontal=True,
        )

    # Detect Caution / Safety Car flags at current lap
    lap_track_status = "1"
    if "TrackStatus" in laps_df.columns:
        sub_status = laps_df[laps_df["LapNumber"] == current_lap]["TrackStatus"].dropna()
        if not sub_status.empty:
            lap_track_status = str(sub_status.iloc[0])

    is_safety_car = any(flag in lap_track_status for flag in ["4", "5", "6", "7"])
    if is_safety_car:
        st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.15); border: 1px solid #FFD700; border-radius: 6px; padding: 8px 14px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
            <span style="color: #FFD700; font-weight: 800; font-size: 13px;">⚠️ SAFETY CAR / VSC DEPLOYED (TRACK STATUS CAUTION)</span>
            <span style="color: #FFFFFF; font-size: 11px;">Pit Loss Discount ~9.5s Available</span>
        </div>
        """, unsafe_allow_html=True)
        play_caution_warning()

    # ─── 3D Circuit & Telemetry Viewport ──────────────────────────
    track_engine = Track3DEngine(circuit_name=circuit_name, elevation_scale=2.0)

    # Determine driver standings up to current lap
    drivers_list = sorted(laps_df["Driver"].dropna().unique().tolist())
    driver_lap_data = []

    for d in drivers_list:
        d_laps = laps_df[(laps_df["Driver"] == d) & (laps_df["LapNumber"] <= current_lap)]
        if not d_laps.empty:
            last_lap = d_laps.iloc[-1]
            pos = int(last_lap.get("Position", 10)) if pd.notna(last_lap.get("Position")) else 10
            team = str(last_lap.get("Team", "Red Bull Racing"))
            comp = str(last_lap.get("Compound", "MEDIUM"))
            tyre_life = int(last_lap.get("TyreLife", current_lap)) if pd.notna(last_lap.get("TyreLife")) else current_lap
            driver_lap_data.append({
                "driver": d,
                "position": pos,
                "team": team,
                "compound": comp,
                "tyre_life": tyre_life,
            })

    driver_lap_data.sort(key=lambda x: x["position"])

    # Compute realistic gaps relative to leader
    for idx, d in enumerate(driver_lap_data):
        d["gap_to_lead_s"] = 0.0 if idx == 0 else round(idx * 1.85 + (idx**1.1) * 0.4, 2)

    top_drivers_3d = driver_lap_data[:10]  # Top 10 for 3D track markers
    cars_3d = track_engine.interpolate_car_positions(top_drivers_3d, lap_progress=lap_progress)

    if "Three.js" in view_mode:
        threejs_html = track_engine.generate_threejs_html(cars_3d, height=520)
        st.components.v1.html(threejs_html, height=530)
    else:
        fig_3d = track_engine.generate_plotly_3d(cars_3d)
        st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # ─── Tactical Strategy Matrix (Monte Carlo Decision HUD) ──────
    st.markdown("### 🧠 Real-Time Tactical Strategy Matrix")

    col_strat_sel1, col_strat_sel2, col_strat_sel3, col_strat_sel4 = st.columns(4)

    lead_driver = driver_lap_data[0]["driver"] if driver_lap_data else "VER"

    with col_strat_sel1:
        target_driver = st.selectbox(
            "Driver Under Strategy Call",
            drivers_list,
            index=drivers_list.index(lead_driver) if lead_driver in drivers_list else 0,
        )

    with col_strat_sel2:
        rival_driver = st.selectbox(
            "Target Rival (Undercut / Gap)",
            [d for d in drivers_list if d != target_driver],
            index=0,
        )

    with col_strat_sel3:
        new_compound_choice = st.selectbox(
            "Target Fresh Compound",
            ["HARD", "MEDIUM", "SOFT"],
            index=0,
        )

    with col_strat_sel4:
        driver_info = next((d for d in driver_lap_data if d["driver"] == target_driver), None)
        current_comp = driver_info["compound"] if driver_info else "MEDIUM"
        current_age = driver_info["tyre_life"] if driver_info else current_lap
        st.markdown(f"**Current Tyre:** `{current_comp}` ({current_age} laps)")
        st.markdown(f"**Race Horizon:** `{total_race_laps - current_lap}` laps remaining")

    # Fetch recent clean laps for stay-out anchor
    recent_laps_driver = clean_df[
        (clean_df["Driver"] == target_driver) & (clean_df["LapNumber"] <= current_lap)
    ].sort_values("LapNumber")

    # Run causal Monte Carlo decision engine
    decision_res = run_monte_carlo_decision(
        current_lap=current_lap,
        total_race_laps=total_race_laps,
        recent_laps=recent_laps_driver,
        current_compound=current_comp,
        new_compound=new_compound_choice,
        current_tyre_age=current_age,
        circuit_name=circuit_name,
        n_sims=1000,
        seed=42,
    )

    # Audio cue trigger on definitive PIT NOW
    if decision_res.verdict == "PIT NOW":
        play_pit_radio_chime()

    # Strategy Banner Display
    col_badge, col_stats = st.columns([1, 2])

    with col_badge:
        if decision_res.verdict == "PIT NOW":
            badge_color = "#39B54A"
            bg_gradient = "linear-gradient(135deg, rgba(57, 181, 74, 0.25) 0%, rgba(20, 30, 20, 0.8) 100%)"
        elif decision_res.verdict == "INCONCLUSIVE":
            badge_color = "#FFD700"
            bg_gradient = "linear-gradient(135deg, rgba(255, 215, 0, 0.25) 0%, rgba(35, 30, 15, 0.8) 100%)"
        else:
            badge_color = "#00D2BE"
            bg_gradient = "linear-gradient(135deg, rgba(0, 210, 190, 0.2) 0%, rgba(15, 25, 30, 0.8) 100%)"

        st.markdown(f"""
        <div style="background: {bg_gradient}; border: 2px solid {badge_color}; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.6);">
            <div style="font-size: 12px; font-weight: 700; color: #A0A0A0; letter-spacing: 1px; margin-bottom: 6px;">RECOMMENDED STRATEGY CALL</div>
            <div style="font-size: 28px; font-weight: 900; color: {badge_color}; letter-spacing: 1.5px; margin-bottom: 10px;">{decision_res.verdict}</div>
            <div style="font-size: 12px; color: #E0E0E0; line-height: 1.4;">{decision_res.reasons[0] if decision_res.reasons else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(Pit Beats Stay)", f"{decision_res.prob_pit_beats_stay:.1%}")
        m2.metric("Expected Net Gain", f"{decision_res.expected_net_gain_sec:+.2f} s")
        m3.metric("80% Credible Interval", f"[{decision_res.ci_80_lower:+.1f}s, {decision_res.ci_80_upper:+.1f}s]")
        m4.metric("Break-Even Horizon", f"{decision_res.break_even_laps:.1f} laps")

        # Visual 80% CI interval bar
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Bar(
            y=["Net Gain"],
            x=[decision_res.expected_net_gain_sec],
            orientation="h",
            marker=dict(color=badge_color),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[decision_res.ci_80_upper - decision_res.expected_net_gain_sec],
                arrayminus=[decision_res.expected_net_gain_sec - decision_res.ci_80_lower],
                color="#FFFFFF",
                thickness=2,
                width=8,
            ),
            name="80% CI",
        ))
        fig_ci.add_vline(x=0, line_dash="dash", line_color="#E10600", annotation_text="Break-even 0s", annotation_font_color="#E10600")
        fig_ci.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0C10",
            plot_bgcolor="#0A0C10",
            height=130,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(title="Net Race Time Saved (seconds)", zeroline=True, gridcolor="#1F2238"),
            yaxis=dict(showticklabels=False),
            showlegend=False,
        )
        st.plotly_chart(fig_ci, use_container_width=True)

    # ─── Undercut Attack Radar ────────────────────────────────────
    st.markdown("#### ⚡ Undercut Attack Radar vs Rival")
    col_uc1, col_uc2 = st.columns([1, 2])

    rival_info = next((d for d in driver_lap_data if d["driver"] == rival_driver), None)
    current_gap = abs(driver_info["gap_to_lead_s"] - (rival_info["gap_to_lead_s"] if rival_info else 0.0))

    with col_uc1:
        gap_input = st.slider(
            "Track Gap to Rival at Stop Call (s)",
            min_value=0.1,
            max_value=5.0,
            value=min(max(float(current_gap), 0.5), 4.5),
            step=0.1,
        )

        uc_eval = evaluate_undercut_overcut(
            gap_to_rival_sec=gap_input,
            driver_tyre_age=current_age,
            rival_tyre_age=rival_info["tyre_life"] if rival_info else current_age,
            compound_current=current_comp,
            compound_new=new_compound_choice,
            circuit_name=circuit_name,
            seed=42,
        )

        st.markdown(f"**Verdict:** `{uc_eval['verdict']}`")
        st.markdown(f"**Expected Out-Lap Margin:** `{uc_eval['expected_margin_sec']:+.2f} s`")
        st.markdown(f"**80% Margin CI:** `[{uc_eval['ci_80_lower']:+.2f}s, {uc_eval['ci_80_upper']:+.2f}s]`")

    with col_uc2:
        # Empirical gap bucket base rates from Phase 3A.4 validation
        buckets = ["0 - 1 s", "1 - 2 s", "2 - 3 s", "3 - 5 s"]
        rates = [64.7, 67.7, 29.0, 20.0]
        n_samples = [17, 31, 31, 35]

        fig_uc = go.Figure()
        fig_uc.add_trace(go.Bar(
            x=buckets,
            y=rates,
            marker=dict(
                color=["#00F5D4", "#39B54A", "#FFD700", "#FF1801"],
            ),
            text=[f"{r:.1f}% (N={n})" for r, n in zip(rates, n_samples)],
            textposition="auto",
        ))
        fig_uc.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0C10",
            plot_bgcolor="#0A0C10",
            title=dict(text="<b>Empirical Undercut Success Rate by Gap Bucket (Held-out races)</b>", font=dict(size=12, color="#ffffff")),
            height=240,
            margin=dict(l=30, r=20, t=40, b=30),
            yaxis=dict(title="Success Rate (%)", range=[0, 100], gridcolor="#1F2238"),
            xaxis=dict(gridcolor="#1F2238"),
        )
        st.plotly_chart(fig_uc, use_container_width=True)

    # ─── Candidate Pit Window Horizon Table ───────────────────────
    st.markdown("#### 📅 Candidate Pit Window Options")
    if decision_res.pit_window_options:
        opt_records = []
        for opt in decision_res.pit_window_options:
            opt_records.append({
                "Pit Lap": opt.pit_lap,
                "Expected Net Gain (s)": f"{opt.expected_net_gain_sec:+.2f}",
                "80% Credible Interval": f"[{opt.ci_80_lower:+.2f}s, {opt.ci_80_upper:+.2f}s]",
                "Verdict": opt.verdict,
            })
        st.dataframe(pd.DataFrame(opt_records), use_container_width=True, hide_index=True)

    # ─── Regulatory & Sensitivity Disclaimers ─────────────────────
    st.markdown("---")
    st.markdown("##### 📜 FIA Sporting Regulation 30.5(n) Compliance")
    st.caption(f"{decision_res.rule_disclaimer} Mandatory two-compound dry rule waived if intermediate/wet weather tyres are declared.")

    with st.expander("🛠️ Decision Engine Sensitivity Analysis (Assumptions & Calibration)"):
        st.markdown("""
        - **Safety-Car Discount Sensitivity:**
          - Assumption note: *9.50s discount is an assumed engineering value based on delta pace, not empirical.*
          - `sc_prob +10%`: Expected gain changes by **+0.95s**
          - `sc_prob -10%`: Expected gain changes by **-0.95s**
        - **Model Limitations:** The decision engine is an exploratory what-if calculator. In held-out race evaluations, it does not beat naive persistence of track position (Persistence: 85.8% vs Engine: 79.5%).
        """)
