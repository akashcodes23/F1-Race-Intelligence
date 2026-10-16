"""
F1 Race Intelligence — Telemetry Lab & Circuit Visuals

Loads high-frequency car telemetry (Speed, Throttle, Brake, Gear, RPM),
aligns data onto a shared distance coordinate, and generates circuit coordinates.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_lap_telemetry(session, driver: str, lap_number: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Extract high-frequency telemetry for a specific driver lap (defaults to their fastest lap).
    """
    if session is None or not hasattr(session, "laps") or session.laps is None:
        return None

    try:
        if hasattr(session.laps, "pick_drivers"):
            drv_laps = session.laps.pick_drivers(driver)
        elif hasattr(session.laps, "pick_driver"):
            drv_laps = session.laps.pick_driver(driver)
        else:
            drv_laps = session.laps[session.laps["Driver"] == driver]
        if drv_laps.empty:
            return None

        if lap_number is not None:
            lap = drv_laps[drv_laps["LapNumber"] == lap_number]
            if lap.empty:
                return None
            target_lap = lap.iloc[0]
        else:
            # Pick fastest lap
            target_lap = drv_laps.pick_fastest()

        if target_lap is None:
            return None

        tel = target_lap.get_telemetry()
        if tel is None or tel.empty:
            return None

        # Add lap reference
        df = tel.copy()
        df["Driver"] = driver
        df["LapNumber"] = target_lap["LapNumber"]
        return df

    except Exception as e:
        logger.warning(f"Could not extract telemetry for driver {driver}: {e}")
        return None


def align_telemetry_by_distance(
    tel1: pd.DataFrame,
    tel2: pd.DataFrame,
    num_points: int = 500,
) -> pd.DataFrame:
    """
    Interpolates telemetry traces onto a normalized shared distance axis
    so drivers can be compared meter-by-meter.
    """
    if tel1 is None or tel2 is None or tel1.empty or tel2.empty:
        return pd.DataFrame()

    if "Distance" not in tel1.columns or "Distance" not in tel2.columns:
        return pd.DataFrame()

    max_dist = min(tel1["Distance"].max(), tel2["Distance"].max())
    if max_dist <= 0:
        return pd.DataFrame()

    distance_grid = np.linspace(0, max_dist, num_points)

    cols_to_interp = ["Speed", "Throttle", "Brake", "nGear", "RPM"]

    aligned_dict = {"Distance": distance_grid}

    for col in cols_to_interp:
        if col in tel1.columns:
            aligned_dict[f"{col}_1"] = np.interp(distance_grid, tel1["Distance"], tel1[col])
        if col in tel2.columns:
            aligned_dict[f"{col}_2"] = np.interp(distance_grid, tel2["Distance"], tel2[col])

    df = pd.DataFrame(aligned_dict)
    if "Speed_1" in df.columns and "Speed_2" in df.columns:
        df["SpeedDelta"] = df["Speed_1"] - df["Speed_2"]

    return df


def get_circuit_coordinates(session) -> Optional[pd.DataFrame]:
    """
    Extracts (X, Y) circuit map coordinates from the fastest lap in the session.
    """
    if session is None or not hasattr(session, "laps") or session.laps is None:
        return None

    try:
        fastest_lap = session.laps.pick_fastest()
        if fastest_lap is None:
            return None

        tel = fastest_lap.get_telemetry()
        if tel is None or "X" not in tel.columns or "Y" not in tel.columns:
            return None

        # Downsample to ~500 points for smooth track visualization
        step = max(1, len(tel) // 500)
        track = tel.iloc[::step][["X", "Y", "Speed", "Distance"]].copy()
        return track
    except Exception as e:
        logger.warning(f"Could not extract circuit coordinates: {e}")
        return None
