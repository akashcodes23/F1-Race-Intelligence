import fastf1
import os

# -------------------------------------------------
# Enable FastF1 Cache
# -------------------------------------------------

CACHE_DIR = "cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)


def load_race_data(year, race, session_type="R"):
    """
    Load full session laps.
    """
    session = fastf1.get_session(year, race, session_type)
    session.load()

    return session.laps