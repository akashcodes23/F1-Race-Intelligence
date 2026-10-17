"""
F1 Race Intelligence — Data Loader (Compatibility Shim)

Re-exports from src.data.loader.
"""

from src.data.loader import (
    get_available_sessions,
    get_driver_laps,
    get_event_names,
    get_event_schedule,
    get_session_drivers,
    init_cache,
    load_session,
)

__all__ = [
    "init_cache",
    "get_event_schedule",
    "get_event_names",
    "get_available_sessions",
    "load_session",
    "get_session_drivers",
    "get_driver_laps",
]
