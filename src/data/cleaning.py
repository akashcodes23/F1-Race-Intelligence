"""
F1 Race Intelligence — Data Cleaning & Validation Layer

Provides robust lap data cleaning that filters:
- Pit entry and exit laps (PitInTime / PitOutTime)
- Safety Car, VSC, and Red Flag caution laps (TrackStatus containing '4', '5', '6', '7')
- Yellow flag laps (TrackStatus containing '2', configurable)
- Inaccurate laps (IsAccurate == False)
- Standing start Lap 1
- Quantile-based statistical pace outliers

Logs and exposes dropped lap telemetry globally and per driver.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config import (
    DROP_YELLOW_LAPS_DEFAULT,
    MIN_LAPS_FOR_ANALYSIS,
    OUTLIER_QUANTILE_HIGH,
    OUTLIER_QUANTILE_LOW,
)
from src.data.schema import validate_laps_df

logger = logging.getLogger(__name__)


def is_caution_track_status(status: Any, drop_yellow: bool = False) -> bool:
    """
    Check if a FastF1 TrackStatus string contains caution or flag codes.

    FastF1 TrackStatus Codes:
      '1': Track Clear
      '2': Yellow Flag
      '4': Safety Car
      '5': Red Flag
      '6': Virtual Safety Car (VSC)
      '7': VSC Ending

    Treats TrackStatus as a string that may contain concatenated transition codes (e.g. '12', '124').
    Laps containing '4', '5', '6', or '7' are always treated as caution.
    Laps containing '2' are caution only if drop_yellow is True.
    """
    if pd.isna(status):
        return False
    s = str(status).strip()
    if not s or s in ("1", "1.0"):
        return False

    # Check for SC (4), Red Flag (5), VSC (6), VSC Ending (7)
    if any(code in s for code in ("4", "5", "6", "7")):
        return True

    # Check for Yellow Flag (2)
    if drop_yellow and "2" in s:
        return True

    return False


def clean_laps(
    laps: pd.DataFrame,
    drop_pit_laps: bool = True,
    drop_caution_laps: bool = True,
    drop_yellow_laps: bool = DROP_YELLOW_LAPS_DEFAULT,
    drop_inaccurate: bool = True,
    drop_lap1: bool = True,
    min_laps: int = MIN_LAPS_FOR_ANALYSIS,
    outlier_low: float = OUTLIER_QUANTILE_LOW,
    outlier_high: float = OUTLIER_QUANTILE_HIGH,
    remove_pit_laps: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Clean and validate lap data according to motorsport engineering standards.
    """
    cleaned_df, _ = clean_laps_with_stats(
        laps=laps,
        drop_pit_laps=drop_pit_laps,
        drop_caution_laps=drop_caution_laps,
        drop_yellow_laps=drop_yellow_laps,
        drop_inaccurate=drop_inaccurate,
        drop_lap1=drop_lap1,
        min_laps=min_laps,
        outlier_low=outlier_low,
        outlier_high=outlier_high,
        remove_pit_laps=remove_pit_laps,
    )
    return cleaned_df


def clean_laps_with_stats(
    laps: pd.DataFrame,
    drop_pit_laps: bool = True,
    drop_caution_laps: bool = True,
    drop_yellow_laps: bool = DROP_YELLOW_LAPS_DEFAULT,
    drop_inaccurate: bool = True,
    drop_lap1: bool = True,
    min_laps: int = MIN_LAPS_FOR_ANALYSIS,
    outlier_low: float = OUTLIER_QUANTILE_LOW,
    outlier_high: float = OUTLIER_QUANTILE_HIGH,
    remove_pit_laps: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean laps and return quantitative audit statistics detailing dropped records globally and per driver.

    Returns:
      (pd.DataFrame, dict): Cleaned data and audit dictionary with keys:
        - original_count: int
        - final_count: int
        - removed_count: int
        - filter_breakdown: Dict[str, int]
        - driver_breakdown: Dict[str, Dict[str, int]]
        - sufficient: bool
    """
    if remove_pit_laps is not None:
        drop_pit_laps = remove_pit_laps

    original_count = len(laps) if laps is not None else 0
    empty_stats: Dict[str, Any] = {
        "original_count": original_count,
        "final_count": 0,
        "removed_count": original_count,
        "filter_breakdown": {
            "invalid_time": 0,
            "lap1": 0,
            "pit_laps": 0,
            "caution_laps": 0,
            "yellow_laps": 0,
            "inaccurate_laps": 0,
            "outliers": 0,
        },
        "driver_breakdown": {},
        "sufficient": False,
    }

    if laps is None or laps.empty:
        return pd.DataFrame(), empty_stats

    df = laps.copy()

    # Track removals per driver
    driver_col = "Driver" if "Driver" in df.columns else None
    filter_counts: Dict[str, int] = defaultdict(int)
    driver_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_dropped(dropped_mask: pd.Series, filter_name: str) -> None:
        count = int(dropped_mask.sum())
        if count == 0:
            return
        filter_counts[filter_name] += count
        if driver_col:
            for drv, grp in df.loc[dropped_mask].groupby(driver_col):
                driver_counts[str(drv)][filter_name] += len(grp)

    # 1. Ensure LapTimeSec column
    if "LapTime" in df.columns and "LapTimeSec" not in df.columns:
        df["LapTimeSec"] = df["LapTime"].dt.total_seconds()

    if "LapTimeSec" not in df.columns:
        logger.warning("No LapTimeSec column found or convertible")
        return pd.DataFrame(), empty_stats

    # 2. Filter missing / non-positive lap times
    invalid_time_mask = df["LapTimeSec"].isna() | (df["LapTimeSec"] <= 0)
    record_dropped(invalid_time_mask, "invalid_time")
    df = df[~invalid_time_mask]

    # 3. Filter Lap 1 (standing start pace distortion)
    if drop_lap1 and "LapNumber" in df.columns and len(df) > min_laps:
        lap1_mask = df["LapNumber"] == 1
        record_dropped(lap1_mask, "lap1")
        df = df[~lap1_mask]

    # 4. Filter Pit In-laps and Out-laps
    if drop_pit_laps:
        pit_mask = pd.Series(False, index=df.index)
        for col in ("PitInTime", "PitOutTime"):
            if col in df.columns:
                pit_mask = pit_mask | df[col].notna()
        record_dropped(pit_mask, "pit_laps")
        df = df[~pit_mask]

    # 5. Filter Safety Car, VSC, and Red Flag caution laps
    if drop_caution_laps and "TrackStatus" in df.columns:
        sc_mask = df["TrackStatus"].apply(lambda s: is_caution_track_status(s, drop_yellow=False))
        record_dropped(sc_mask, "caution_laps")
        df = df[~sc_mask]

    # 6. Filter Yellow Flag laps if requested
    if drop_yellow_laps and "TrackStatus" in df.columns:
        yellow_mask = df["TrackStatus"].astype(str).str.contains("2")
        record_dropped(yellow_mask, "yellow_laps")
        df = df[~yellow_mask]

    # 7. Filter Inaccurate laps
    if drop_inaccurate and "IsAccurate" in df.columns:
        inaccurate_mask = ~df["IsAccurate"].fillna(True).astype(bool)
        # Only filter if sufficient accurate laps remain
        if (len(df) - inaccurate_mask.sum()) >= min_laps:
            record_dropped(inaccurate_mask, "inaccurate_laps")
            df = df[~inaccurate_mask]

    # 8. Quantile-based outlier trimming
    if len(df) >= min_laps and outlier_low > 0.0 and outlier_high < 1.0:
        q_low = df["LapTimeSec"].quantile(outlier_low)
        q_high = df["LapTimeSec"].quantile(outlier_high)
        outlier_mask = (df["LapTimeSec"] < q_low) | (df["LapTimeSec"] > q_high)
        record_dropped(outlier_mask, "outliers")
        df = df[~outlier_mask]

    # 9. Schema validation and normalization
    cleaned_df = validate_laps_df(df)

    final_count = len(cleaned_df)
    removed_count = original_count - final_count

    # Convert defaultdicts to regular dicts
    clean_driver_breakdown = {k: dict(v) for k, v in driver_counts.items()}
    clean_filter_breakdown = dict(filter_counts)

    stats: Dict[str, Any] = {
        "original_count": original_count,
        "final_count": final_count,
        "removed_count": removed_count,
        "filter_breakdown": clean_filter_breakdown,
        "driver_breakdown": clean_driver_breakdown,
        "sufficient": final_count >= min_laps,
    }

    logger.info(
        f"Lap cleaning complete: {original_count} -> {final_count} laps ({removed_count} removed). "
        f"Filters: {clean_filter_breakdown}"
    )

    return cleaned_df, stats
