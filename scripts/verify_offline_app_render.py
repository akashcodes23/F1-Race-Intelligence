"""
Offline Application Verification Script
Validates that with network calls blocked at the OS socket level,
all 13 demo races load successfully and the main analytics pages render.
"""

import os
import socket
import sys


# Block network calls at socket connection level
def blocked_connect(self, *args, **kwargs):
    raise socket.error("Network access is strictly blocked during offline verification test.")
socket.socket.connect = blocked_connect

# Set paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_demo_races, load_session
from src.decision_engine import run_monte_carlo_decision
from src.strategy_simulator import evaluate_pit_window
from src.tyre_analysis import analyze_all_stints


def verify_offline_app():
    print("======================================================================")
    print("RUNNING OFFLINE APPLICATION VERIFICATION (NETWORK STRICTLY BLOCKED)")
    print("======================================================================")

    # 1. Verify demo data exists and is recognized
    demo_races = get_demo_races()
    print(f"Total offline demo races discovered: {len(demo_races)}")
    assert len(demo_races) == 13, f"Expected 13 demo races, found {len(demo_races)}"

    # 2. Iterate and load all 13 demo races
    for idx, (year, gp) in enumerate(demo_races, 1):
        session_obj, laps_df = load_session(year, gp, "R", load_telemetry=False)
        assert laps_df is not None and len(laps_df) > 0, f"No laps found for {year} {gp}"

        n_laps = len(laps_df)
        drivers = laps_df["Driver"].nunique()
        print(f"  [{idx:02d}/13] LOADED: {year} {gp:<28} | Laps: {n_laps:4d} | Drivers: {drivers:2d}")

    print("\n--- Testing Main Analytics and Decision Logic Offline ---")

    # Test on Monaco 2023
    _, laps_df = load_session(2023, "Monaco Grand Prix", "R", load_telemetry=False)

    # Page 1 logic: Tyre Intelligence
    stints_summary = analyze_all_stints(laps_df)
    print(f"  Tyre Intelligence: {len(stints_summary)} stints analyzed offline.")
    assert len(stints_summary) > 0, "Tyre analysis failed"

    # Page 2 logic: Strategy Simulator
    sim_res = evaluate_pit_window(
        start_candidate_lap=28,
        end_candidate_lap=34,
        total_race_laps=78,
        current_lap=25,
        current_tire_age=25,
        base_pace=76.5,
        current_deg=0.038,
        new_deg=0.039,
        new_compound_delta=-0.10,
        pit_loss=19.5,
    )
    print(f"  Strategy Simulator: Evaluated pit window across {len(sim_res)} candidate laps.")
    assert len(sim_res) > 0, "Strategy simulation failed"

    # Page 3 logic: Decision Engine Stop Timing
    ver_laps = laps_df[laps_df["Driver"] == "VER"].sort_values("LapNumber").iloc[:25]
    mc_res = run_monte_carlo_decision(
        current_lap=25,
        total_race_laps=78,
        recent_laps=ver_laps,
        current_compound="MEDIUM",
        new_compound="HARD",
        current_tyre_age=25,
        used_compounds=["MEDIUM"],
        candidate_window_radius=3,
        n_sims=200,
        seed=42,
    )
    print(f"  Decision Engine: Evaluated VER Lap 25 offline: Verdict={mc_res.verdict}, BreakEven={mc_res.break_even_laps} laps, {len(mc_res.pit_window_options)} candidate options.")
    assert len(mc_res.pit_window_options) > 0, "Stop timing evaluation failed"
    print(f"  Regulation Check: Legal={mc_res.is_legal_strategy} ({mc_res.rule_disclaimer})")

    print("\nSUCCESS: All 13 races loaded and main analytics pages render with network strictly blocked!")

if __name__ == "__main__":
    verify_offline_app()
