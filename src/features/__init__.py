"""
F1 Race Intelligence — Feature Engineering Package
"""

from src.features.fuel import (
    apply_fuel_correction,
    compute_s1a_anchor,
    estimate_fuel_effect,
    normalize_lap_fuel,
    predict_fuel_pace,
    predict_s1a_lap_pace,
    predict_s1a_pit_pace_gain,
)

__all__ = [
    "estimate_fuel_effect",
    "apply_fuel_correction",
    "normalize_lap_fuel",
    "predict_fuel_pace",
    "predict_s1a_lap_pace",
    "compute_s1a_anchor",
    "predict_s1a_pit_pace_gain",
]


