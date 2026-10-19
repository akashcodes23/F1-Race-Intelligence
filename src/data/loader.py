"""
F1 Race Intelligence — FastF1 Session Ingestion & Discovery

Handles dynamic season event discovery, session type mapping,
cache initialization, and robust session data loading.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)


def init_cache(cache_dir: str) -> None:
    """Initialise the FastF1 HTTP cache. Call once at application startup."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    logger.info(f"FastF1 cache enabled: {cache_dir}")


DEMO_PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "demo",
    "demo_laps.parquet",
)


def get_demo_races() -> List[Tuple[int, str]]:
    """Return list of (year, grand_prix) available in the offline demo dataset."""
    if os.path.exists(DEMO_PARQUET_PATH):
        try:
            df = pd.read_parquet(DEMO_PARQUET_PATH, columns=["Year", "GrandPrix"])
            pairs = df.drop_duplicates().sort_values(["Year", "GrandPrix"])
            return list(zip(pairs["Year"].astype(int), pairs["GrandPrix"].astype(str)))
        except Exception as e:
            logger.error(f"Failed to read demo dataset: {e}")
    return []


def load_demo_laps(year: int, gp: str) -> Optional[pd.DataFrame]:
    """Load cleaned laps from the offline demo dataset for year and gp."""
    if not os.path.exists(DEMO_PARQUET_PATH):
        return None
    try:
        df = pd.read_parquet(DEMO_PARQUET_PATH)
        gp_clean = gp.lower().replace("grand prix", "").strip()
        matched = df[(df["Year"] == year) & (df["GrandPrix"].str.lower().str.contains(gp_clean))]
        if not matched.empty:
            logger.info(f"Loaded {len(matched)} laps from offline demo dataset for {year} {gp}")
            return matched.copy()
    except Exception as e:
        logger.error(f"Failed to load demo laps for {year} {gp}: {e}")
    return None


def get_event_schedule(year: int) -> pd.DataFrame:
    """
    Return the official event schedule for *year*, excluding pre-season testing.
    Returns an empty DataFrame on failure.
    """
    try:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except TypeError:
            schedule = fastf1.get_event_schedule(year)
            if "EventFormat" in schedule.columns:
                schedule = schedule[schedule["EventFormat"] != "testing"]
        return schedule
    except Exception as e:
        logger.error(f"Failed to load event schedule for {year}: {e}")
        return pd.DataFrame()


def get_event_names(year: int) -> List[str]:
    """Return a list of Grand Prix names for *year*."""
    schedule = get_event_schedule(year)
    if not schedule.empty and "EventName" in schedule.columns:
        return schedule["EventName"].tolist()

    # Fallback to demo races if live schedule discovery fails or is offline
    demo_races = get_demo_races()
    matched = [gp for (y, gp) in demo_races if y == year]
    if matched:
        return matched
    return []


def get_available_sessions(year: int, gp: str) -> Dict[str, int]:
    """
    Discover available sessions (Practice, Qualifying, Sprint, Race) for a Grand Prix.
    Returns mapping of Session Name -> Session Number (1-5).
    """
    try:
        event = fastf1.get_event(year, gp)
        sessions: Dict[str, int] = {}
        for i in range(1, 6):
            name = event.get(f"Session{i}", None)
            if name is not None and str(name) not in ("None", "", "nan") and not pd.isna(name):
                sessions[str(name)] = i
        return sessions if sessions else {"Race": 5}
    except Exception as e:
        logger.error(f"Failed to discover sessions for {year} {gp}: {e}")
        return {"Race": 5}


def load_session(
    year: int,
    gp: str,
    session_identifier,
    load_telemetry: bool = False,
    fallback_to_demo: bool = True,
) -> Tuple[Optional[object], Optional[pd.DataFrame]]:
    """
    Load a FastF1 session with laps and optional high-frequency telemetry.
    If live loading fails and fallback_to_demo is True, attempts to load from offline demo dataset.
    """
    try:
        session = fastf1.get_session(year, gp, session_identifier)
        session.load(
            laps=True,
            telemetry=load_telemetry,
            weather=False,
            messages=False,
        )

        laps = session.laps
        if laps is None or len(laps) == 0:
            logger.warning(f"No lap data: {year} {gp} session {session_identifier}")
            if fallback_to_demo:
                demo_laps = load_demo_laps(year, gp)
                if demo_laps is not None and not demo_laps.empty:
                    return session, demo_laps
            return session, pd.DataFrame()

        # Guarantee LapTimeSec float column
        laps = laps.copy()
        if "LapTime" in laps.columns and "LapTimeSec" not in laps.columns:
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

        return session, laps

    except Exception as e:
        logger.error(f"Session load failed ({year} {gp} {session_identifier}): {e}")
        if fallback_to_demo:
            demo_laps = load_demo_laps(year, gp)
            if demo_laps is not None and not demo_laps.empty:
                return None, demo_laps
        return None, None


def get_session_drivers(laps: pd.DataFrame) -> List[str]:
    """Sorted list of unique driver codes present in *laps*."""
    if laps is None or laps.empty or "Driver" not in laps.columns:
        return []
    return sorted(laps["Driver"].dropna().unique().tolist())


def get_driver_laps(laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    """Filter *laps* to a single driver code."""
    if hasattr(laps, "pick_drivers"):
        try:
            return laps.pick_drivers(driver)
        except Exception:
            pass
    if hasattr(laps, "pick_driver"):
        try:
            return laps.pick_driver(driver)
        except Exception:
            pass
    return laps[laps["Driver"] == driver]
