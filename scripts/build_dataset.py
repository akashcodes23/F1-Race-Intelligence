"""
F1 Race Intelligence — Multi-Race Lap Pace Dataset Builder

Builds a consolidated multi-race parquet dataset across 2021-2024 races:
1. Cleans laps according to Phase 1.5 engineering rules (dropping standing-start lap 1,
   pit in/out laps, safety car/VSC caution laps, and rain-transition laps).
2. Excludes wet compound laps (INTERMEDIATE, WET).
3. Establishes race-level reference pace per driver:
   DriverReferencePaceSec = median of driver's first 5 clean laps in the race (Laps 2-6).
4. Targets relative delta pace:
   RelativeLapTimeSec = LapTimeSec - DriverReferencePaceSec.
5. Engineers causal features strictly without future leakage.
6. Saves to data/race_laps_dataset.parquet (gitignored).
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastf1
import pandas as pd

from config import CACHE_DIR, PROJECT_ROOT
from src.data.cleaning import clean_laps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

# Default target list of 16 diverse dry/mixed races across 2021-2024
DEFAULT_RACES: List[Tuple[int, str]] = [
    (2021, "Bahrain Grand Prix"),
    (2021, "Spanish Grand Prix"),
    (2021, "Austrian Grand Prix"),
    (2021, "British Grand Prix"),
    (2021, "Italian Grand Prix"),
    (2021, "United States Grand Prix"),
    (2022, "Bahrain Grand Prix"),
    (2022, "Spanish Grand Prix"),
    (2022, "British Grand Prix"),
    (2022, "Austrian Grand Prix"),
    (2022, "French Grand Prix"),
    (2022, "Italian Grand Prix"),
    (2023, "Bahrain Grand Prix"),
    (2023, "Monaco Grand Prix"),
    (2023, "Spanish Grand Prix"),
    (2023, "British Grand Prix"),
]


def extract_race_features(laps_raw: pd.DataFrame, year: int, gp_name: str) -> pd.DataFrame:
    """
    Process raw laps into feature-engineered relative-pace observations.
    """
    if laps_raw is None or laps_raw.empty:
        return pd.DataFrame()

    # Clean laps using robust motorsport pipeline
    cleaned = clean_laps(laps_raw)
    if cleaned.empty or len(cleaned) < 20:
        return pd.DataFrame()

    df = cleaned.copy()

    # Filter out wet compounds
    if "Compound" in df.columns:
        df = df[~df["Compound"].astype(str).str.upper().isin(WET_COMPOUNDS)]
    if df.empty:
        return pd.DataFrame()

    # Sort chronologically per driver
    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    # 1. Driver Reference Pace: median of driver's first 5 clean laps
    ref_paces = {}
    for drv, grp in df.groupby("Driver"):
        early_laps = grp.iloc[:min(5, len(grp))]
        ref_paces[drv] = float(early_laps["LapTimeSec"].median())

    df["DriverReferencePaceSec"] = df["Driver"].map(ref_paces)
    df["RelativeLapTimeSec"] = df["LapTimeSec"] - df["DriverReferencePaceSec"]

    # 2. Tyre Life (TireAge within stint)
    if "TyreLife" not in df.columns or df["TyreLife"].isna().all():
        if "Stint" in df.columns:
            df["TyreLife"] = df.groupby(["Driver", "Stint"]).cumcount() + 1
        else:
            df["TyreLife"] = df.groupby("Driver").cumcount() + 1

    # 3. Laps Remaining / Fuel Proxy
    total_race_laps = int(df["LapNumber"].max())
    df["TotalRaceLaps"] = total_race_laps
    df["LapsRemaining"] = (total_race_laps - df["LapNumber"]).clip(lower=0)

    # 4. Previous-lap relative pace within stint (strictly lagged, no lookahead)
    stint_col = "Stint" if "Stint" in df.columns else "Driver"
    df["PrevLapRelativePace"] = df.groupby(["Driver", stint_col])["RelativeLapTimeSec"].shift(1)
    # Fill first lap of stint with 0.0 (baseline pace delta)
    df["PrevLapRelativePace"] = df["PrevLapRelativePace"].fillna(0.0)

    # 5. Metadata columns
    df["RaceYear"] = int(year)
    df["EventName"] = str(gp_name)
    df["RaceID"] = f"{year}_{gp_name.replace(' ', '_')}"

    # Select standard modeling columns
    keep_cols = [
        "RaceID",
        "RaceYear",
        "EventName",
        "Driver",
        "Team",
        "Compound",
        "Stint",
        "LapNumber",
        "TyreLife",
        "LapsRemaining",
        "PrevLapRelativePace",
        "LapTimeSec",
        "DriverReferencePaceSec",
        "RelativeLapTimeSec",
    ]
    present_cols = [c for c in keep_cols if c in df.columns]
    return df[present_cols]


def build_dataset(
    races: Optional[List[Tuple[int, str]]] = None,
    output_path: Optional[str] = None,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """
    Ingest, clean, and consolidate races into a parquet dataset.
    """
    if races is None:
        races = DEFAULT_RACES

    if output_path is None:
        output_dir = os.path.join(PROJECT_ROOT, "data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "race_laps_dataset.parquet")

    fastf1.Cache.enable_cache(cache_dir)

    all_dfs = []
    loaded_races = 0

    for year, gp in races:
        try:
            logger.info(f"Loading {year} {gp}...")
            session = fastf1.get_session(year, gp, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)

            race_df = extract_race_features(session.laps, year, gp)
            if not race_df.empty:
                all_dfs.append(race_df)
                loaded_races += 1
                logger.info(f"  Successfully extracted {len(race_df)} clean dry laps from {year} {gp}")
            else:
                logger.warning(f"  No clean laps extracted for {year} {gp}")
        except Exception as e:
            logger.warning(f"  Skipping {year} {gp}: {e}")

    if not all_dfs:
        logger.error("No races could be loaded into dataset.")
        return pd.DataFrame()

    full_dataset = pd.concat(all_dfs, ignore_index=True)
    full_dataset.to_parquet(output_path, index=False)
    logger.info(
        f"Consolidated dataset saved to {output_path}: "
        f"{loaded_races} races, {len(full_dataset)} total clean dry laps."
    )
    return full_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build F1 Lap Pace Dataset")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    build_dataset(output_path=args.output)
