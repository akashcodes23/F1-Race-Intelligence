"""
F1 Race Intelligence — Fuel Load Effect Estimation & Correction

Estimates per-lap fuel burn pace advantage (seconds gained per lap as fuel mass burns off)
and computes fuel-corrected lap times to isolate pure tyre degradation.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import (
    DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    MAX_FUEL_EFFECT_BOUND,
    MIN_FUEL_EFFECT_BOUND,
)

logger = logging.getLogger(__name__)


def estimate_fuel_effect(
    laps: pd.DataFrame,
    default_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    min_bound: float = MIN_FUEL_EFFECT_BOUND,
    max_bound: float = MAX_FUEL_EFFECT_BOUND,
) -> Dict[str, Any]:
    """
    Empirically estimate the fuel burn pace gain (s/lap) from session stint transitions.

    Methodology:
      Examines multi-stint drivers on dry compounds. For each driver transition between
      Stint i and Stint i+1, measures the pace differential at early tyre life (laps 2-5)
      normalized over the lap distance between stints.

      If sufficient consistent transitions exist (>= 3) and the median falls within
      [min_bound, max_bound] (nominally 0.015 to 0.075 s/lap), adopts the empirical estimate.
      Otherwise, falls back to the configured physics default (~0.033 s/lap).

    Returns:
      dict with keys:
        - fuel_effect: float (seconds per lap)
        - is_estimated: bool
        - method: str
        - transitions_count: int
        - default_effect: float
    """
    fallback_res: Dict[str, Any] = {
        "fuel_effect": float(default_effect),
        "is_estimated": False,
        "method": "config_default",
        "transitions_count": 0,
        "default_effect": float(default_effect),
    }

    if laps is None or laps.empty:
        return fallback_res

    req_cols = ["Driver", "Stint", "LapNumber", "LapTimeSec"]
    if not all(c in laps.columns for c in req_cols):
        return fallback_res

    # Group by driver and stint to find early-stint baseline pace
    stint_baselines = []
    for (driver, stint), grp in laps.groupby(["Driver", "Stint"]):
        if pd.isna(stint) or len(grp) < 3:
            continue

        # Look at laps with tyre life 2 to 6 if TyreLife exists, otherwise relative stint order
        if "TyreLife" in grp.columns:
            early_laps = grp[(grp["TyreLife"] >= 2) & (grp["TyreLife"] <= 6)]
        else:
            early_laps = grp.iloc[1:min(5, len(grp))]

        if early_laps.empty or len(early_laps) < 2:
            continue

        median_pace = early_laps["LapTimeSec"].median()
        median_lap = early_laps["LapNumber"].median()
        compound = grp["Compound"].iloc[0] if "Compound" in grp.columns else "UNKNOWN"

        stint_baselines.append({
            "Driver": driver,
            "Stint": int(stint),
            "Compound": compound,
            "MedianPace": median_pace,
            "MedianLap": median_lap,
        })

    if len(stint_baselines) < 4:
        return fallback_res

    base_df = pd.DataFrame(stint_baselines).sort_values(["Driver", "Stint"])
    rates = []

    # Calculate transition rates between consecutive stints for each driver
    for _driver, drv_stints in base_df.groupby("Driver"):
        if len(drv_stints) < 2:
            continue
        drv_stints = drv_stints.sort_values("Stint").reset_index(drop=True)
        for i in range(len(drv_stints) - 1):
            s1 = drv_stints.iloc[i]
            s2 = drv_stints.iloc[i + 1]

            lap_delta = s2["MedianLap"] - s1["MedianLap"]
            pace_gain = s1["MedianPace"] - s2["MedianPace"]  # Positive means later stint was faster

            # Require at least 10 laps between stint measurement points
            if lap_delta >= 10:
                rate = pace_gain / lap_delta
                # Only include physically plausible positive pace gains
                if min_bound <= rate <= max_bound:
                    rates.append(rate)

    if len(rates) >= 3:
        empirical_rate = float(np.median(rates))
        logger.info(
            f"Empirical fuel burn pace effect estimated from {len(rates)} transitions: "
            f"{empirical_rate:.4f} s/lap (bounds: {min_bound} to {max_bound})"
        )
        return {
            "fuel_effect": round(empirical_rate, 4),
            "is_estimated": True,
            "method": "empirical_stint_transition_median",
            "transitions_count": len(rates),
            "default_effect": float(default_effect),
        }

    return fallback_res


def apply_fuel_correction(
    laps: pd.DataFrame,
    fuel_effect_sec_per_lap: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    reference_lap: int = 1,
) -> pd.DataFrame:
    """
    Apply fuel-mass correction to lap times.

    Formula:
      FuelCorrectedLapTimeSec = LapTimeSec + fuel_effect * (LapNumber - reference_lap)

    Normalizes all laps to the fuel mass of reference_lap (start of race).
    As race progresses and car gets lighter, adding back the fuel burn advantage
    isolates the true tyre degradation.

    Returns:
      pd.DataFrame with FuelCorrectedLapTimeSec and FuelEffectAddedSec columns.
    """
    if laps is None or laps.empty:
        return pd.DataFrame()

    df = laps.copy()
    if "LapTimeSec" not in df.columns:
        if "LapTime" in df.columns:
            df["LapTimeSec"] = df["LapTime"].dt.total_seconds()
        else:
            return df

    lap_num = df["LapNumber"] if "LapNumber" in df.columns else pd.Series(1, index=df.index)
    laps_from_ref = (lap_num - reference_lap).clip(lower=0)

    fuel_added = laps_from_ref * fuel_effect_sec_per_lap
    df["FuelEffectAddedSec"] = fuel_added
    df["FuelCorrectedLapTimeSec"] = df["LapTimeSec"] + fuel_added

    return df
