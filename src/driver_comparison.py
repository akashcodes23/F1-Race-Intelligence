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
