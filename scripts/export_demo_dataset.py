"""
F1 Race Intelligence — Demo Dataset Export & Rebuild Script

Rebuilds the minimal demo dataset (data/demo/demo_laps.parquet) from raw
cached FastF1 race sessions for the 13 supported offline Grand Prix events.

Usage:
    PYTHONPATH=. python scripts/export_demo_dataset.py [--output data/demo/demo_laps.parquet]
"""

import argparse
import logging
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import CACHE_DIR
from src.data.loader import get_demo_races, init_cache, load_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rebuild_demo_dataset(output_path: str = "data/demo/demo_laps.parquet"):
    """
    Consolidates clean race laps for the 13 verified demo Grand Prix events
    and saves a compressed parquet archive.
    """
    init_cache(CACHE_DIR)
    demo_races = get_demo_races()
    logger.info(f"Rebuilding demo dataset for {len(demo_races)} Grand Prix events...")

    all_laps = []
    for year, gp in demo_races:
        logger.info(f"Loading {year} {gp}...")
        _, laps = load_session(year, gp, "R", load_telemetry=False, fallback_to_demo=True)
        if laps is not None and not laps.empty:
            laps = laps.copy()
            if "RaceID" not in laps.columns:
                laps["RaceID"] = f"{year}_{gp.replace(' ', '_')}"
            all_laps.append(laps)
        else:
            logger.warning(f"Could not load laps for {year} {gp}")

    if not all_laps:
        logger.error("No lap data loaded. Aborting export.")
        sys.exit(1)

    combined_df = pd.concat(all_laps, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_parquet(output_path, compression="snappy", index=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Successfully rebuilt demo dataset: {output_path} ({size_mb:.2f} MB, {len(combined_df)} laps across {combined_df['RaceID'].nunique()} races).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and rebuild offline demo dataset")
    parser.add_argument("--output", default="data/demo/demo_laps.parquet", help="Path to output parquet file")
    args = parser.parse_args()
    rebuild_demo_dataset(args.output)
