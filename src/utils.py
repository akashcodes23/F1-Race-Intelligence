"""
F1 Race Intelligence — Shared Utilities

Formatting helpers, time conversions, and common constants.
"""

from typing import Optional

import pandas as pd


def lap_time_to_seconds(lap_time) -> Optional[float]:
    """Convert a timedelta or numeric lap time to float seconds."""
    if pd.isna(lap_time):
        return None
    if hasattr(lap_time, "total_seconds"):
        return lap_time.total_seconds()
    try:
        return float(lap_time)
    except (ValueError, TypeError):
        return None


def format_lap_time(seconds: float) -> str:
    """
    Format seconds as M:SS.mmm.

    >>> format_lap_time(83.456)
    '1:23.456'
    """
    if pd.isna(seconds) or seconds <= 0:
        return "—"
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}:{remaining:06.3f}"


format_seconds = format_lap_time


def format_delta(seconds: float) -> str:
    """
    Format a time delta with explicit +/- sign.

    >>> format_delta(0.342)
    '+0.342s'
    >>> format_delta(-1.2)
    '-1.200s'
    """
    if pd.isna(seconds):
        return "—"
    sign = "+" if seconds >= 0 else ""
    return f"{sign}{seconds:.3f}s"


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    try:
        result = float(value)
        if pd.isna(result):
            return default
        return result
    except (ValueError, TypeError):
        return default
