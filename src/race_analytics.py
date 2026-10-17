"""
F1 Race Intelligence — Race Overview & Session Analytics

Provides session-wide analytics, position progression, stint summaries,
lap-time distributions, and session leaderboards without Streamlit dependencies.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data_validator import clean_laps
from src.utils import format_seconds

logger = logging.getLogger(__name__)


def get_session_summary(session, laps: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute high-level session summary metrics.
    """
    summary: Dict[str, Any] = {
        "event_name": getattr(session, "event", {}).get("EventName", "Unknown Grand Prix") if hasattr(session, "event") else "Unknown Grand Prix",
        "session_name": getattr(session, "name", "Session"),
        "total_laps": 0,
        "total_drivers": 0,
        "fastest_lap": None,
        "fastest_driver": None,
        "fastest_lap_num": None,
        "fastest_compound": None,
    }

    if laps is None or laps.empty:
        return summary

    summary["total_laps"] = int(len(laps))
    summary["total_drivers"] = int(laps["Driver"].nunique()) if "Driver" in laps.columns else 0

    # Ensure LapTimeSec
    valid_laps = laps.dropna(subset=["LapTime"]) if "LapTime" in laps.columns else laps
    if not valid_laps.empty:
        if "LapTimeSec" not in valid_laps.columns:
            valid_laps = valid_laps.copy()
            valid_laps["LapTimeSec"] = valid_laps["LapTime"].dt.total_seconds()

        valid_laps = valid_laps[valid_laps["LapTimeSec"] > 0]
        if not valid_laps.empty:
            fastest_idx = valid_laps["LapTimeSec"].idxmin()
            best_lap = valid_laps.loc[fastest_idx]
            summary["fastest_lap"] = float(best_lap["LapTimeSec"])
            summary["fastest_lap_formatted"] = format_seconds(best_lap["LapTimeSec"])
            summary["fastest_driver"] = str(best_lap.get("Driver", "N/A"))
            summary["fastest_lap_num"] = int(best_lap.get("LapNumber", 0))
            summary["fastest_compound"] = str(best_lap.get("Compound", "UNKNOWN"))

    return summary


def get_position_progression_data(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Extract lap-by-lap position data for all drivers in the session.
    """
    if laps is None or laps.empty or "Position" not in laps.columns or "LapNumber" not in laps.columns:
        return pd.DataFrame()

    cols = ["LapNumber", "Driver", "Position"]
    if "Team" in laps.columns:
        cols.append("Team")

    pos_df = laps[cols].dropna(subset=["Position"]).copy()
    pos_df["Position"] = pos_df["Position"].astype(int)
    pos_df["LapNumber"] = pos_df["LapNumber"].astype(int)
    return pos_df.sort_values(by=["LapNumber", "Position"])


def get_stint_summary(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stint-level summaries for each driver:
    stint number, compound, start lap, end lap, lap count, average pace, fastest lap.
    """
    if laps is None or laps.empty or "Stint" not in laps.columns or "Driver" not in laps.columns:
        return pd.DataFrame()

    df = laps.copy()
    if "LapTimeSec" not in df.columns and "LapTime" in df.columns:
        df["LapTimeSec"] = df["LapTime"].dt.total_seconds()

    stints_list = []
    for (driver, stint), group in df.groupby(["Driver", "Stint"]):
        if pd.isna(stint):
            continue
        compound = group["Compound"].dropna().iloc[0] if "Compound" in group.columns and not group["Compound"].dropna().empty else "UNKNOWN"
        start_lap = group["LapNumber"].min()
        end_lap = group["LapNumber"].max()
        lap_count = len(group)

        valid_times = group["LapTimeSec"].dropna()
        valid_times = valid_times[valid_times > 0]
        avg_pace = valid_times.mean() if not valid_times.empty else np.nan
        best_pace = valid_times.min() if not valid_times.empty else np.nan

        stints_list.append({
            "Driver": driver,
            "Stint": int(stint),
            "Compound": compound,
            "StartLap": int(start_lap),
            "EndLap": int(end_lap),
            "TotalLaps": int(lap_count),
            "AvgLapTime": avg_pace,
            "BestLapTime": best_pace,
            "AvgLapTimeFormatted": format_seconds(avg_pace) if pd.notna(avg_pace) else "N/A",
            "BestLapTimeFormatted": format_seconds(best_pace) if pd.notna(best_pace) else "N/A",
        })

    if not stints_list:
        return pd.DataFrame()

    return pd.DataFrame(stints_list).sort_values(by=["Driver", "Stint"])


def get_driver_lap_distributions(laps: pd.DataFrame, min_laps: int = 5) -> pd.DataFrame:
    """
    Returns cleaned lap times with statistical quantiles per driver for boxplots/violin distributions.
    """
    if laps is None or laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    cleaned = clean_laps(laps, min_laps=min_laps)
    if cleaned.empty:
        return pd.DataFrame()

    # Filter drivers who meet the minimum valid laps threshold
    counts = cleaned["Driver"].value_counts()
    eligible_drivers = counts[counts >= min_laps].index
    return cleaned[cleaned["Driver"].isin(eligible_drivers)].copy()


def get_session_leaderboard(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Computes session leaderboard sorted by fastest lap time.
    """
    if laps is None or laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    df = laps.copy()
    if "LapTimeSec" not in df.columns and "LapTime" in df.columns:
        df["LapTimeSec"] = df["LapTime"].dt.total_seconds()

    valid = df.dropna(subset=["LapTimeSec"])
    valid = valid[valid["LapTimeSec"] > 0]
    if valid.empty:
        return pd.DataFrame()

    best_laps = valid.loc[valid.groupby("Driver")["LapTimeSec"].idxmin()]
    best_laps = best_laps.sort_values(by="LapTimeSec").reset_index(drop=True)

    fastest_time = best_laps["LapTimeSec"].iloc[0]
    best_laps["Gap"] = best_laps["LapTimeSec"] - fastest_time
    best_laps["Position"] = range(1, len(best_laps) + 1)
    best_laps["BestLapFormatted"] = best_laps["LapTimeSec"].apply(format_seconds)
    best_laps["GapFormatted"] = best_laps["Gap"].apply(lambda g: f"+{g:.3f}s" if g > 0 else "FASTEST")

    cols = ["Position", "Driver", "BestLapFormatted", "GapFormatted", "LapNumber", "Compound"]
    if "Team" in best_laps.columns:
        cols.insert(2, "Team")
    avail = [c for c in cols if c in best_laps.columns]
    return best_laps[avail]
