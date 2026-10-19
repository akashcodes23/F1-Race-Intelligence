"""
F1 Race Intelligence — Fuel Load Effect Estimation & Correction

Estimates per-lap fuel burn pace advantage (seconds gained per lap as fuel mass burns off)
and computes fuel-corrected lap times to isolate pure tyre degradation.

Phase 1.5 Constraints:
- Estimates only from dry, compound-matched, caution-free stint pairs.
- Excludes wet compounds (INTERMEDIATE, WET) and track-transition laps.
- Clamps to configurable plausible range (default 0.02 - 0.05 s/lap).
- Falls back to config default and logs WHY when evidence is insufficient or estimate hits bounds.
- Exposes fuel_effect and source ("estimated" vs "default") in results.
"""

import logging
from typing import Any, Dict, Set

import numpy as np
import pandas as pd

from config import (
    DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    MAX_FUEL_EFFECT_BOUND,
    MIN_FUEL_EFFECT_BOUND,
)
from src.data.cleaning import clean_laps

logger = logging.getLogger(__name__)

WET_COMPOUNDS: Set[str] = {"INTERMEDIATE", "WET"}


def estimate_fuel_effect(
    laps: pd.DataFrame,
    default_effect: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    min_bound: float = MIN_FUEL_EFFECT_BOUND,
    max_bound: float = MAX_FUEL_EFFECT_BOUND,
) -> Dict[str, Any]:
    """
    Empirically estimate the fuel burn pace gain (s/lap) from session stint transitions.

    Methodology:
      1. Excludes wet compounds (INTERMEDIATE, WET) and track-transition laps.
      2. Cleans laps to remove pit stops, caution periods (SC/VSC), and yellow flags.
      3. Identifies multi-stint drivers having compound-matched dry stints (Stint i and Stint i+1
         on the exact same tyre compound).
      4. Measures early tyre-life baseline pace (laps 2-5) normalized over the lap delta.
      5. Validates against [min_bound, max_bound] (0.02 to 0.05 s/lap).
      6. Falls back to config default and logs explicit reason if insufficient evidence
         or if estimate hits bounds.

    Returns:
      dict with keys:
        - fuel_effect: float (seconds per lap)
        - source: str ("estimated" | "default")
        - is_estimated: bool
        - method: str
        - reason: str
        - transitions_count: int
        - default_effect: float
        - bounds: [float, float]
    """
    fallback_res: Dict[str, Any] = {
        "fuel_effect": float(default_effect),
        "source": "default",
        "is_estimated": False,
        "method": "config_default",
        "reason": "Insufficient evidence; using physics default",
        "transitions_count": 0,
        "default_effect": float(default_effect),
        "bounds": [float(min_bound), float(max_bound)],
    }

    if laps is None or laps.empty:
        fallback_res["reason"] = "Input laps DataFrame is empty"
        logger.info(f"Fuel estimation fallback: {fallback_res['reason']}")
        return fallback_res

    df_in = laps.copy()
    if "LapTimeSec" not in df_in.columns and "LapTime" in df_in.columns:
        df_in["LapTimeSec"] = df_in["LapTime"].dt.total_seconds()

    req_cols = ["Driver", "Stint", "LapNumber", "LapTimeSec"]
    if not all(c in df_in.columns for c in req_cols):
        fallback_res["reason"] = f"Missing required columns (required: {req_cols})"
        logger.info(f"Fuel estimation fallback: {fallback_res['reason']}")
        return fallback_res

    # Clean laps to caution-free, pit-free laps
    clean_df = clean_laps(df_in, drop_pit_laps=True, drop_caution_laps=True, drop_yellow_laps=True)
    if clean_df.empty:
        fallback_res["reason"] = "No clean caution-free laps available"
        logger.info(f"Fuel estimation fallback: {fallback_res['reason']}")
        return fallback_res

    # Exclude wet compounds
    if "Compound" in clean_df.columns:
        dry_df = clean_df[~clean_df["Compound"].astype(str).str.upper().isin(WET_COMPOUNDS)].copy()
    else:
        dry_df = clean_df.copy()

    if dry_df.empty:
        fallback_res["reason"] = "No dry compound laps available (all INTERMEDIATE/WET)"
        logger.info(f"Fuel estimation fallback: {fallback_res['reason']}")
        return fallback_res

    # Group by driver and stint to find early-stint baseline pace
    stint_baselines = []
    for (driver, stint), grp in dry_df.groupby(["Driver", "Stint"]):
        if pd.isna(stint) or len(grp) < 3:
            continue

        compound = str(grp["Compound"].iloc[0]).upper() if "Compound" in grp.columns else "UNKNOWN"
        if compound in WET_COMPOUNDS:
            continue

        # Look at early laps (tyre life 2 to 6 if available, else first 2-5 stint laps)
        if "TyreLife" in grp.columns:
            early_laps = grp[(grp["TyreLife"] >= 2) & (grp["TyreLife"] <= 6)]
        else:
            early_laps = grp.iloc[1:min(5, len(grp))]

        if early_laps.empty or len(early_laps) < 2:
            continue

        median_pace = float(early_laps["LapTimeSec"].median())
        median_lap = float(early_laps["LapNumber"].median())

        stint_baselines.append({
            "Driver": str(driver),
            "Stint": int(stint),
            "Compound": compound,
            "MedianPace": median_pace,
            "MedianLap": median_lap,
        })

    if len(stint_baselines) < 2:
        fallback_res["reason"] = f"Insufficient clean dry stints found ({len(stint_baselines)} stints)"
        logger.info(f"Fuel estimation fallback: {fallback_res['reason']}")
        return fallback_res

    base_df = pd.DataFrame(stint_baselines).sort_values(["Driver", "Stint"])
    rates = []

    # Calculate transition rates between consecutive compound-matched stints for each driver
    for _driver, drv_stints in base_df.groupby("Driver"):
        if len(drv_stints) < 2:
            continue
        drv_stints = drv_stints.sort_values("Stint").reset_index(drop=True)
        for i in range(len(drv_stints) - 1):
            s1 = drv_stints.iloc[i]
            s2 = drv_stints.iloc[i + 1]

            # Compound-matched requirement: only compare identical dry compounds
            if s1["Compound"] != s2["Compound"]:
                continue

            lap_delta = s2["MedianLap"] - s1["MedianLap"]
            pace_gain = s1["MedianPace"] - s2["MedianPace"]  # Positive means later stint was faster

            # Require at least 8 laps between stint measurement points
            if lap_delta >= 8:
                rate = pace_gain / lap_delta
                rates.append(float(rate))

    if len(rates) < 2:
        fallback_res["reason"] = (
            f"Insufficient dry, compound-matched stint pairs found ({len(rates)} pairs, need >= 2)"
        )
        fallback_res["transitions_count"] = len(rates)
        logger.info(f"Fuel estimation fallback: {fallback_res['reason']}")
        return fallback_res

    empirical_rate = float(np.median(rates))

    # Clamp validation to configurable plausible bounds
    if min_bound <= empirical_rate <= max_bound:
        logger.info(
            f"Empirical fuel burn effect successfully estimated from {len(rates)} compound-matched "
            f"transitions: {empirical_rate:.4f} s/lap (bounds: [{min_bound}, {max_bound}])"
        )
        return {
            "fuel_effect": round(empirical_rate, 4),
            "source": "estimated",
            "is_estimated": True,
            "method": "empirical_compound_matched_stints",
            "reason": f"Estimated from {len(rates)} dry, compound-matched stint transitions",
            "transitions_count": len(rates),
            "default_effect": float(default_effect),
            "bounds": [float(min_bound), float(max_bound)],
        }
    else:
        reason = (
            f"Empirical rate {empirical_rate:.4f} s/lap outside plausible bounds "
            f"[{min_bound}, {max_bound}]; excluded from pooling"
        )
        logger.warning(f"Fuel estimation out-of-bounds: {reason}")
        return {
            "fuel_effect": float(default_effect),
            "empirical_rate": round(empirical_rate, 4),
            "source": "out_of_bounds",
            "is_estimated": False,
            "is_out_of_bounds": True,
            "method": "out_of_bounds_excluded",
            "reason": reason,
            "transitions_count": len(rates),
            "default_effect": float(default_effect),
            "bounds": [float(min_bound), float(max_bound)],
        }


def normalize_lap_fuel(
    lap_time_sec: float | np.ndarray | pd.Series,
    lap_number: float | np.ndarray | pd.Series,
    reference_lap: float,
    fuel_effect_sec_per_lap: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> float | np.ndarray | pd.Series:
    """
    Normalize lap time(s) to the fuel mass of reference_lap.

    Documented Sign Convention:
      Lap time falls by `fuel_effect` per lap as fuel burns off (car gets lighter and faster).
      Therefore:
        - For an earlier lap (lap_number < reference_lap), the car was heavier (slower).
          Normalizing to reference_lap fuel mass makes it faster:
          T_norm = T - fuel_effect * (reference_lap - lap_number)
        - For a later lap (lap_number > reference_lap), the car was lighter (faster).
          Normalizing to reference_lap fuel mass makes it slower:
          T_norm = T + fuel_effect * (lap_number - reference_lap)

      Both cases are unified by the single canonical formula:
        T_norm = T + fuel_effect * (lap_number - reference_lap)

    Parameters:
      lap_time_sec: Raw lap time(s) in seconds.
      lap_number: Lap number(s) corresponding to lap_time_sec.
      reference_lap: The reference lap number whose fuel mass to normalize to.
      fuel_effect_sec_per_lap: Seconds gained per lap from fuel burn (positive, default ~0.058 s/lap).

    Returns:
      Fuel-normalized lap time(s) in seconds.
    """
    return lap_time_sec + fuel_effect_sec_per_lap * (lap_number - reference_lap)


def predict_fuel_pace(
    anchor_pace_sec: float | np.ndarray,
    horizon_laps: float | np.ndarray,
    fuel_effect_sec_per_lap: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> float | np.ndarray:
    """
    Predict pace change due to fuel burn over horizon_laps.

    As fuel burns, the car gets lighter and lap times decrease by fuel_effect per lap:
      Predicted Pace = anchor_pace - fuel_effect * horizon_laps
    """
    return anchor_pace_sec - fuel_effect_sec_per_lap * horizon_laps


def predict_s1a_lap_pace(
    anchor_pace_sec: float,
    horizon_laps: int,
    compound_deg_slope: float,
    fuel_effect_sec_per_lap: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
) -> np.ndarray:
    """
    Predict future lap times using S1a anchored physics model with canonical fuel sign.

    For step h in 1..H:
      Tyre wear increases lap time by: +compound_deg_slope * h
      Fuel burn decreases lap time by: -fuel_effect * h
      Net slope = compound_deg_slope - fuel_effect
      Pace(h) = anchor_pace + (compound_deg_slope - fuel_effect) * h
    """
    steps = np.arange(1, horizon_laps + 1)
    net_slope = compound_deg_slope - fuel_effect_sec_per_lap
    return anchor_pace_sec + net_slope * steps


def apply_fuel_correction(
    laps: pd.DataFrame,
    fuel_effect_sec_per_lap: float = DEFAULT_FUEL_EFFECT_SEC_PER_LAP,
    reference_lap: int = 1,
) -> pd.DataFrame:
    """
    Apply fuel-mass correction to lap times using normalize_lap_fuel.

    Normalizes all laps to the fuel mass of reference_lap (start of race).
    As race progresses and car gets lighter, adding back the fuel burn advantage
    isolates the pure tyre degradation.

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
    df["FuelCorrectedLapTimeSec"] = normalize_lap_fuel(
        df["LapTimeSec"], lap_num, reference_lap=reference_lap, fuel_effect_sec_per_lap=fuel_effect_sec_per_lap
    )
    df["FuelEffectAddedSec"] = df["FuelCorrectedLapTimeSec"] - df["LapTimeSec"]

    return df

