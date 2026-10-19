"""
F1 Race Intelligence — Driver Performance Lab & Comparison

Computes detailed driver head-to-head metrics:
- Lap-by-lap delta and cumulative delta
- Consistency (std, IQR, median pace)
- Sector times (S1, S2, S3) and hypothetical optimal lap comparison
- Stint-by-stint pace differentials
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.data_validator import clean_laps
from src.utils import format_seconds

logger = logging.getLogger(__name__)


def compute_driver_delta(
    laps: pd.DataFrame,
    driver1: str,
    driver2: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute lap-by-lap delta between driver1 and driver2.

    Sign convention: delta = driver1 - driver2.
    Positive delta (+) indicates driver1 was slower.
    Negative delta (-) indicates driver1 was faster.

    Returns:
        (merged_dataframe, summary_stats_dict)
    """
    if laps is None or laps.empty:
        return pd.DataFrame(), {}

    d1_laps = clean_laps(laps[laps["Driver"] == driver1])
    d2_laps = clean_laps(laps[laps["Driver"] == driver2])

    if d1_laps.empty or d2_laps.empty:
        return pd.DataFrame(), {}

    req_cols = ["LapNumber", "LapTimeSec"]
    for extra in ["Compound", "Stint", "TireAge"]:
        if extra in d1_laps.columns:
            req_cols.append(extra)

    d1_sub = d1_laps[[c for c in req_cols if c in d1_laps.columns]]
    d2_sub = d2_laps[[c for c in req_cols if c in d2_laps.columns]]

    merged = pd.merge(
        d1_sub,
        d2_sub,
        on="LapNumber",
        suffixes=(f"_{driver1}", f"_{driver2}"),
    ).sort_values("LapNumber").reset_index(drop=True)

    if merged.empty:
        return pd.DataFrame(), {}

    merged["Delta"] = merged[f"LapTimeSec_{driver1}"] - merged[f"LapTimeSec_{driver2}"]
    merged["CumulativeDelta"] = merged["Delta"].cumsum()
    merged["RollingDelta"] = merged["Delta"].rolling(window=3, min_periods=1).mean()

    # Summary statistics
    d1_faster = int((merged["Delta"] < 0).sum())
    d2_faster = int((merged["Delta"] > 0).sum())
    equal = int((merged["Delta"] == 0).sum())
    median_delta = float(merged["Delta"].median())
    mean_delta = float(merged["Delta"].mean())
    cum_final = float(merged["CumulativeDelta"].iloc[-1])

    summary = {
        "matched_laps": len(merged),
        "driver1_faster_count": d1_faster,
        "driver2_faster_count": d2_faster,
        "equal_count": equal,
        "median_delta": median_delta,
        "mean_delta": mean_delta,
        "final_cumulative_delta": cum_final,
        "faster_driver": driver1 if cum_final < 0 else (driver2 if cum_final > 0 else "TIED"),
    }

    return merged, summary


def compare_driver_consistency(
    laps: pd.DataFrame,
    driver1: str,
    driver2: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute pace & consistency metrics for both drivers:
    Mean, Std, Median, 25th percentile, 75th percentile, IQR, and Best Lap.
    """
    res = {}
    for drv in (driver1, driver2):
        drv_clean = clean_laps(laps[laps["Driver"] == drv])
        if drv_clean.empty or len(drv_clean) < 3:
            res[drv] = {
                "n_laps": len(drv_clean),
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "iqr": np.nan,
                "best": np.nan,
                "best_formatted": "N/A",
                "median_formatted": "N/A",
            }
            continue

        times = drv_clean["LapTimeSec"]
        q25 = times.quantile(0.25)
        q75 = times.quantile(0.75)
        best = times.min()
        median = times.median()

        res[drv] = {
            "n_laps": len(times),
            "mean": float(times.mean()),
            "std": float(times.std()),
            "median": float(median),
            "iqr": float(q75 - q25),
            "best": float(best),
            "best_formatted": format_seconds(best),
            "median_formatted": format_seconds(median),
        }

    return res


def compare_driver_sectors(
    laps: pd.DataFrame,
    driver1: str,
    driver2: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compare sector times (Sector 1, 2, and 3) between drivers.
    Computes best sectors and hypothetical optimal lap time.
    """
    if laps is None or laps.empty:
        return pd.DataFrame(), {}

    sector_cols = ["Sector1Time", "Sector2Time", "Sector3Time"]
    has_sec_cols = all(f"{col}Sec" in laps.columns for col in sector_cols)
    has_raw_cols = all(col in laps.columns for col in sector_cols)

    if not (has_sec_cols or has_raw_cols):
        return pd.DataFrame(), {}

    df = laps.copy()
    if not has_sec_cols:
        for col in sector_cols:
            sec_name = f"{col}Sec"
            if sec_name not in df.columns and col in df.columns:
                df[sec_name] = df[col].dt.total_seconds()

    d1 = df[df["Driver"] == driver1].dropna(subset=["Sector1TimeSec", "Sector2TimeSec", "Sector3TimeSec"])
    d2 = df[df["Driver"] == driver2].dropna(subset=["Sector1TimeSec", "Sector2TimeSec", "Sector3TimeSec"])

    if d1.empty or d2.empty:
        return pd.DataFrame(), {}

    sectors_summary = {}

    for drv, sub in [(driver1, d1), (driver2, d2)]:
        s1_best = float(sub["Sector1TimeSec"].min())
        s2_best = float(sub["Sector2TimeSec"].min())
        s3_best = float(sub["Sector3TimeSec"].min())
        s1_median = float(sub["Sector1TimeSec"].median())
        s2_median = float(sub["Sector2TimeSec"].median())
        s3_median = float(sub["Sector3TimeSec"].median())

        sectors_summary[drv] = {
            "s1_best": s1_best,
            "s2_best": s2_best,
            "s3_best": s3_best,
            "s1_median": s1_median,
            "s2_median": s2_median,
            "s3_median": s3_median,
            "optimal_lap": s1_best + s2_best + s3_best,
            "optimal_lap_formatted": format_seconds(s1_best + s2_best + s3_best),
        }

    # Prepare comparison table
    comparison_rows = [
        {
            "Metric": "Sector 1 Best",
            driver1: f"{sectors_summary[driver1]['s1_best']:.3f}s",
            driver2: f"{sectors_summary[driver2]['s1_best']:.3f}s",
            "Advantage": driver1 if sectors_summary[driver1]["s1_best"] < sectors_summary[driver2]["s1_best"] else driver2,
        },
        {
            "Metric": "Sector 2 Best",
            driver1: f"{sectors_summary[driver1]['s2_best']:.3f}s",
            driver2: f"{sectors_summary[driver2]['s2_best']:.3f}s",
            "Advantage": driver1 if sectors_summary[driver1]["s2_best"] < sectors_summary[driver2]["s2_best"] else driver2,
        },
        {
            "Metric": "Sector 3 Best",
            driver1: f"{sectors_summary[driver1]['s3_best']:.3f}s",
            driver2: f"{sectors_summary[driver2]['s3_best']:.3f}s",
            "Advantage": driver1 if sectors_summary[driver1]["s3_best"] < sectors_summary[driver2]["s3_best"] else driver2,
        },
        {
            "Metric": "Hypothetical Optimal Lap",
            driver1: sectors_summary[driver1]["optimal_lap_formatted"],
            driver2: sectors_summary[driver2]["optimal_lap_formatted"],
            "Advantage": driver1 if sectors_summary[driver1]["optimal_lap"] < sectors_summary[driver2]["optimal_lap"] else driver2,
        },
    ]

    return pd.DataFrame(comparison_rows), sectors_summary


def compute_leader_gap_proxy(
    laps: pd.DataFrame,
    current_lap: int,
    reference_lap_time: float = 90.0,
) -> pd.DataFrame:
    """
    Computes deterministic lap elapsed timing proxy relative to the race leader at current_lap.

    Scientific & Timing Distinction:
    - This calculates the cumulative elapsed lap timing proxy across completed laps (Lap 1..current_lap).
    - It is NOT the official Formula 1 live telemetry timing-tower gap (which measures spatial transponder deltas
      at the instant the leader crosses a sector line).
    - For drivers on the lead lap who completed current_lap, gap is:
        delta_sec = driver_cumulative_time - leader_cumulative_time
    - For lapped drivers (completed_laps < current_lap), official timing displays lap deficit:
        gap_display = '+1 LAP' or '+N LAPS'
    - For cars with missing/invalid lap times or non-positive records:
        status is 'INCOMPLETE_DATA' and gap is labeled 'N/A'.
    - For drivers with no records up to current_lap:
        status is 'OUT'.

    Returns:
        pd.DataFrame sorted by track position with columns:
        ['driver', 'position', 'team', 'compound', 'tyre_life', 'completed_laps',
         'cum_time', 'gap_to_lead_s', 'gap_display', 'status', 'is_lapped', 'metric_type']
    """
    if laps is None or laps.empty:
        return pd.DataFrame()

    ref_time = max(40.0, float(reference_lap_time))
    all_drivers = sorted(laps["Driver"].dropna().unique().tolist())
    records = []

    # Filter laps up to current_lap
    up_to_cur = laps[laps["LapNumber"] <= current_lap].copy()
    if "LapTimeSec" not in up_to_cur.columns and "LapTime" in up_to_cur.columns:
        up_to_cur["LapTimeSec"] = up_to_cur["LapTime"].dt.total_seconds()

    for drv in all_drivers:
        d_laps = up_to_cur[up_to_cur["Driver"] == drv]
        if d_laps.empty:
            records.append({
                "driver": drv,
                "position": 99,
                "team": "Unknown",
                "compound": "UNKNOWN",
                "tyre_life": 0,
                "completed_laps": 0,
                "cum_time": 0.0,
                "gap_to_lead_s": None,
                "gap_display": "OUT",
                "status": "OUT",
                "is_lapped": False,
                "metric_type": "UNAVAILABLE",
            })
            continue

        last_lap = d_laps.iloc[-1]
        pos = int(last_lap.get("Position", 99)) if pd.notna(last_lap.get("Position")) else 99
        team = str(last_lap.get("Team", "Unknown"))
        compound = str(last_lap.get("Compound", "MEDIUM"))
        completed_laps = int(d_laps["LapNumber"].max())
        tyre_life = int(last_lap.get("TyreLife", completed_laps)) if pd.notna(last_lap.get("TyreLife")) else completed_laps

        # Validate lap times
        valid_times = d_laps["LapTimeSec"].dropna()
        valid_times = valid_times[valid_times > 0]

        has_missing = len(valid_times) < len(d_laps)
        cum_time = float(valid_times.sum()) if not valid_times.empty else 0.0

        records.append({
            "driver": drv,
            "position": pos,
            "team": team,
            "compound": compound,
            "tyre_life": tyre_life,
            "completed_laps": completed_laps,
            "cum_time": cum_time,
            "has_missing": has_missing,
        })

    # Sort records by completed laps (most laps first), then track position, then cumulative time
    records.sort(key=lambda r: (-r.get("completed_laps", 0), r.get("position", 99), r.get("cum_time", 0.0)))

    if not records:
        return pd.DataFrame()

    # Identify leader
    leader = records[0]
    leader_cum_time = leader["cum_time"]
    leader_laps = leader["completed_laps"]

    results = []
    for idx, r in enumerate(records):
        drv = r["driver"]
        pos = r["position"]
        comp_laps = r["completed_laps"]
        cum_t = r["cum_time"]
        has_miss = r.get("has_missing", False)

        if idx == 0:
            results.append({
                "driver": drv,
                "position": pos,
                "team": r["team"],
                "compound": r["compound"],
                "tyre_life": r["tyre_life"],
                "completed_laps": comp_laps,
                "cum_time": cum_t,
                "gap_to_lead_s": 0.0,
                "gap_display": "LEADER",
                "status": "LEADER",
                "is_lapped": False,
                "metric_type": "LEADER",
            })
            continue

        if comp_laps == 0 or cum_t <= 0.0:
            results.append({
                "driver": drv,
                "position": pos,
                "team": r["team"],
                "compound": r["compound"],
                "tyre_life": r["tyre_life"],
                "completed_laps": comp_laps,
                "cum_time": cum_t,
                "gap_to_lead_s": None,
                "gap_display": "OUT",
                "status": "OUT",
                "is_lapped": False,
                "metric_type": "UNAVAILABLE",
            })
            continue

        if has_miss:
            results.append({
                "driver": drv,
                "position": pos,
                "team": r["team"],
                "compound": r["compound"],
                "tyre_life": r["tyre_life"],
                "completed_laps": comp_laps,
                "cum_time": cum_t,
                "gap_to_lead_s": None,
                "gap_display": "N/A",
                "status": "INCOMPLETE_DATA",
                "is_lapped": False,
                "metric_type": "INCOMPLETE_DATA",
            })
            continue

        # Check for lapped car
        if comp_laps < leader_laps:
            lap_deficit = leader_laps - comp_laps
            gap_disp = f"+{lap_deficit} LAP" if lap_deficit == 1 else f"+{lap_deficit} LAPS"
            # Spatial proxy gap for 3D track ribbon calculation
            spatial_proxy = round(lap_deficit * ref_time + max(0.0, cum_t - (leader_cum_time * (comp_laps / max(1, leader_laps)))), 2)
            results.append({
                "driver": drv,
                "position": pos,
                "team": r["team"],
                "compound": r["compound"],
                "tyre_life": r["tyre_life"],
                "completed_laps": comp_laps,
                "cum_time": cum_t,
                "gap_to_lead_s": spatial_proxy,
                "gap_display": gap_disp,
                "status": "LAPPED",
                "is_lapped": True,
                "metric_type": "LAP_DEFICIT",
            })
        else:
            # On lead lap with identical completed laps
            delta_sec = round(max(0.0, cum_t - leader_cum_time), 2)
            results.append({
                "driver": drv,
                "position": pos,
                "team": r["team"],
                "compound": r["compound"],
                "tyre_life": r["tyre_life"],
                "completed_laps": comp_laps,
                "cum_time": cum_t,
                "gap_to_lead_s": delta_sec,
                "gap_display": f"+{delta_sec:.2f}s",
                "status": "ON_LEAD_LAP",
                "is_lapped": False,
                "metric_type": "CUMULATIVE_ELAPSED_PROXY",
            })

    return pd.DataFrame(results)

