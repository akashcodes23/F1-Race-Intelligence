"""
F1 Race Intelligence — Strategy Simulator Engine

Provides race-engineering pit wall strategy modeling:
- Projected race time: Stay-out vs Pit-now over equal lap horizons
- Dynamic pit window optimization (evaluating all candidate pit laps)
- Sensitivity analysis over pit lane loss time and degradation uncertainty
- Projections labeled explicitly as model-based estimates
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_PIT_LOSS_SECONDS

logger = logging.getLogger(__name__)


def simulate_strategy_horizon(
    current_lap: int,
    total_race_laps: int,
    current_tire_age: int,
    base_pace: float,
    current_compound_deg: float,
    new_compound_base_offset: float = -0.5,
    new_compound_deg: float = 0.08,
    pit_loss: float = DEFAULT_PIT_LOSS_SECONDS,
    pit_lap: int = 15,
) -> Dict[str, Any]:
    """
    Simulate lap times and cumulative time to the end of the race for two strategies:
    1. Plan A (Stay out on current set)
    2. Plan B (Pit at pit_lap for fresh tyres)

    Parameters:
        current_lap: Starting lap for simulation
        total_race_laps: Finish lap (horizon)
        current_tire_age: Age of tyre at current_lap
        base_pace: Base lap time (seconds) on fresh current tyre
        current_compound_deg: Degradation rate (sec/lap) on current compound
        new_compound_base_offset: Pace delta of new tyre compared to base (e.g. -0.5s for softer tyre)
        new_compound_deg: Degradation rate of new tyre
        pit_loss: Stationary + pit lane transit time penalty
        pit_lap: Scheduled lap to pit for Plan B
    """
    laps_to_sim = list(range(current_lap, total_race_laps + 1))
    if not laps_to_sim:
        return {"laps": pd.DataFrame(), "total_stay": 0.0, "total_pit": 0.0, "delta": 0.0}

    stay_laps = []
    pit_laps = []

    cum_stay = 0.0
    cum_pit = 0.0

    # Stay out
    for i, lap in enumerate(laps_to_sim):
        age = current_tire_age + i
        pace = base_pace + (age * current_compound_deg)
        cum_stay += pace
        stay_laps.append({"Lap": lap, "Pace": pace, "CumTime": cum_stay, "TireAge": age})

    # Pit strategy
    current_plan_b_age = current_tire_age
    for i, lap in enumerate(laps_to_sim):
        if lap < pit_lap:
            age = current_plan_b_age + i
            pace = base_pace + (age * current_compound_deg)
            cum_pit += pace
            pit_laps.append({"Lap": lap, "Pace": pace, "CumTime": cum_pit, "TireAge": age, "Pit": False})
        elif lap == pit_lap:
            # Pit entry / stop lap includes pit loss
            age = 0
            pace = base_pace + new_compound_base_offset + pit_loss
            cum_pit += pace
            pit_laps.append({"Lap": lap, "Pace": pace, "CumTime": cum_pit, "TireAge": age, "Pit": True})
        else:
            # Laps on new set
            laps_on_new = lap - pit_lap
            pace = (base_pace + new_compound_base_offset) + (laps_on_new * new_compound_deg)
            cum_pit += pace
            pit_laps.append({"Lap": lap, "Pace": pace, "CumTime": cum_pit, "TireAge": laps_on_new, "Pit": False})

    sim_df = pd.DataFrame({
        "LapNumber": laps_to_sim,
        "StayOut_LapTime": [x["Pace"] for x in stay_laps],
        "StayOut_CumTime": [x["CumTime"] for x in stay_laps],
        "Pit_LapTime": [x["Pace"] for x in pit_laps],
        "Pit_CumTime": [x["CumTime"] for x in pit_laps],
        "Delta_Seconds": [p["CumTime"] - s["CumTime"] for p, s in zip(pit_laps, stay_laps)],
    })

    total_stay = cum_stay
    total_pit = cum_pit
    net_advantage = total_stay - total_pit  # Positive means pitting saved race time

    return {
        "laps": sim_df,
        "total_stay_time": round(total_stay, 2),
        "total_pit_time": round(total_pit, 2),
        "net_advantage": round(net_advantage, 2),
        "optimal_strategy": "PIT" if net_advantage > 0 else "STAY OUT",
    }


def evaluate_pit_window(
    start_candidate_lap: int,
    end_candidate_lap: int,
    total_race_laps: int,
    current_lap: int,
    current_tire_age: int,
    base_pace: float,
    current_deg: float,
    new_deg: float,
    new_compound_delta: float,
    pit_loss: float = DEFAULT_PIT_LOSS_SECONDS,
) -> pd.DataFrame:
    """
    Evaluates every candidate pit stop lap within a window to identify the mathematically optimal lap.
    """
    eval_records = []
    for cand_lap in range(start_candidate_lap, end_candidate_lap + 1):
        res = simulate_strategy_horizon(
            current_lap=current_lap,
            total_race_laps=total_race_laps,
            current_tire_age=current_tire_age,
            base_pace=base_pace,
            current_compound_deg=current_deg,
            new_compound_base_offset=new_compound_delta,
            new_compound_deg=new_deg,
            pit_loss=pit_loss,
            pit_lap=cand_lap,
        )
        eval_records.append({
            "PitLap": cand_lap,
            "TotalRaceTime": res["total_pit_time"],
            "AdvantageVsStayOut": res["net_advantage"],
        })

    eval_df = pd.DataFrame(eval_records)
    if not eval_df.empty:
        best_idx = eval_df["TotalRaceTime"].idxmin()
        eval_df["IsOptimal"] = eval_df.index == best_idx
    return eval_df


def pit_loss_sensitivity_analysis(
    current_lap: int,
    total_race_laps: int,
    current_tire_age: int,
    base_pace: float,
    current_deg: float,
    new_deg: float,
    new_compound_delta: float,
    pit_lap: int,
    loss_range: Tuple[float, float, float] = (16.0, 26.0, 1.0),
) -> pd.DataFrame:
    """
    Computes sensitivity of strategy advantage across different pit loss penalties
    (e.g., standard pit stop vs slow pit stop or VSC / Safety Car pit stop).
    """
    start, end, step = loss_range
    sens_rows = []
    for loss in np.arange(start, end + 0.1, step):
        res = simulate_strategy_horizon(
            current_lap=current_lap,
            total_race_laps=total_race_laps,
            current_tire_age=current_tire_age,
            base_pace=base_pace,
            current_compound_deg=current_deg,
            new_compound_base_offset=new_compound_delta,
            new_compound_deg=new_deg,
            pit_loss=float(loss),
            pit_lap=pit_lap,
        )
        scenario = "Normal Pit Stop"
        if loss < 18:
            scenario = "Safety Car / VSC Pit"
        elif loss > 24:
            scenario = "Slow Pit Stop / Traffic"

        sens_rows.append({
            "PitLossSeconds": round(float(loss), 1),
            "Scenario": scenario,
            "NetAdvantage": res["net_advantage"],
            "Recommendation": "PIT" if res["net_advantage"] > 0 else "STAY OUT",
        })

    return pd.DataFrame(sens_rows)
