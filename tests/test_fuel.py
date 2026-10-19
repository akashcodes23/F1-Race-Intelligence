import unittest

import numpy as np
import pandas as pd

from config import DEFAULT_FUEL_EFFECT_SEC_PER_LAP
from src.features.fuel import (
    apply_fuel_correction,
    estimate_fuel_effect,
    normalize_lap_fuel,
    predict_s1a_lap_pace,
)
from src.tyre_analysis import compute_stint_degradation


class TestFuelEffect(unittest.TestCase):
    def setUp(self):
        # 15 laps: clean stint with true degradation ~0.08s/lap
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, 15)
        self.stint_laps = pd.DataFrame({
            "Driver": ["LEC"] * 15,
            "Stint": [1] * 15,
            "Compound": ["SOFT"] * 15,
            "LapNumber": list(range(2, 17)),
            "LapTimeSec": [78.0 + (i * 0.08) + noise[i] for i in range(15)],
            "TyreLife": list(range(1, 16)),
        })

    def test_apply_fuel_correction(self):
        df = pd.DataFrame({
            "Driver": ["VER"] * 5,
            "LapNumber": [1, 2, 3, 4, 5],
            "LapTimeSec": [80.0, 80.0, 80.0, 80.0, 80.0],
        })
        corrected = apply_fuel_correction(df, fuel_effect_sec_per_lap=0.033, reference_lap=1)
        self.assertIn("FuelCorrectedLapTimeSec", corrected.columns)
        self.assertIn("FuelEffectAddedSec", corrected.columns)
        # Lap 1 added: 0.0
        self.assertAlmostEqual(corrected["FuelEffectAddedSec"].iloc[0], 0.0)
        # Lap 5 added: 4 * 0.033 = 0.132
        self.assertAlmostEqual(corrected["FuelEffectAddedSec"].iloc[4], 0.132, places=3)
        self.assertAlmostEqual(corrected["FuelCorrectedLapTimeSec"].iloc[4], 80.132, places=3)

    def test_fuel_correction_steeper_slope(self):
        # With fuel correction (+0.033 s/lap), degradation slope must be steeper (higher)
        res_raw = compute_stint_degradation(self.stint_laps, min_laps=5, use_fuel_correction=False)
        res_fuel = compute_stint_degradation(self.stint_laps, min_laps=5, fuel_effect=0.033, use_fuel_correction=True)
        self.assertTrue(res_fuel["valid"])
        self.assertGreater(res_fuel["slope"], res_raw["slope"])

    def test_estimate_fuel_effect_fallback_insufficient_pairs(self):
        # Single stint -> insufficient transitions -> returns config default
        res = estimate_fuel_effect(self.stint_laps)
        self.assertFalse(res["is_estimated"])
        self.assertEqual(res["source"], "default")
        self.assertEqual(res["method"], "config_default")
        self.assertAlmostEqual(res["fuel_effect"], DEFAULT_FUEL_EFFECT_SEC_PER_LAP, places=3)
        self.assertIn("Insufficient", res["reason"])

    def test_estimate_fuel_effect_excludes_wet_compounds(self):
        # Transition between two INTERMEDIATE stints must NOT be used
        stint1 = pd.DataFrame({
            "Driver": ["VER"] * 10,
            "Stint": [1] * 10,
            "Compound": ["INTERMEDIATE"] * 10,
            "LapNumber": list(range(2, 12)),
            "LapTimeSec": [90.0 - i * 0.03 for i in range(10)],
            "TyreLife": list(range(1, 11)),
        })
        stint2 = pd.DataFrame({
            "Driver": ["VER"] * 10,
            "Stint": [2] * 10,
            "Compound": ["INTERMEDIATE"] * 10,
            "LapNumber": list(range(20, 30)),
            "LapTimeSec": [89.0 - i * 0.03 for i in range(10)],
            "TyreLife": list(range(1, 11)),
        })
        laps = pd.concat([stint1, stint2], ignore_index=True)
        res = estimate_fuel_effect(laps)
        self.assertEqual(res["source"], "default")
        self.assertIn("No dry compound", res["reason"])

    def test_estimate_fuel_effect_compound_matched_only(self):
        # Transition between SOFT and HARD must be skipped (not compound-matched)
        stint1 = pd.DataFrame({
            "Driver": ["VER"] * 10,
            "Stint": [1] * 10,
            "Compound": ["SOFT"] * 10,
            "LapNumber": list(range(2, 12)),
            "LapTimeSec": [80.0 for _ in range(10)],
            "TyreLife": list(range(1, 11)),
        })
        stint2 = pd.DataFrame({
            "Driver": ["VER"] * 10,
            "Stint": [2] * 10,
            "Compound": ["HARD"] * 10,
            "LapNumber": list(range(20, 30)),
            "LapTimeSec": [79.5 for _ in range(10)],
            "TyreLife": list(range(1, 11)),
        })
        laps = pd.concat([stint1, stint2], ignore_index=True)
        res = estimate_fuel_effect(laps)
        self.assertEqual(res["source"], "default")
        self.assertIn("Insufficient dry, compound-matched", res["reason"])

    def test_estimate_fuel_effect_clamping_and_bounds(self):
        # 2 drivers with 2 matched dry stints where pace gain is abnormally high (e.g. 0.15 s/lap)
        # Should be clamped / fall back to default
        drivers_data = []
        for d in ["VER", "PER"]:
            s1 = pd.DataFrame({
                "Driver": [d] * 10,
                "Stint": [1] * 10,
                "Compound": ["MEDIUM"] * 10,
                "LapNumber": list(range(2, 12)),
                "LapTimeSec": [82.0 for _ in range(10)],
                "TyreLife": list(range(1, 11)),
            })
            s2 = pd.DataFrame({
                "Driver": [d] * 10,
                "Stint": [2] * 10,
                "Compound": ["MEDIUM"] * 10,
                "LapNumber": list(range(22, 32)),
                "LapTimeSec": [78.0 for _ in range(10)],  # 4s over 20 laps = 0.20 s/lap (out of bounds)
                "TyreLife": list(range(1, 11)),
            })
            drivers_data.extend([s1, s2])
        laps = pd.concat(drivers_data, ignore_index=True)
        res = estimate_fuel_effect(laps, min_bound=0.02, max_bound=0.05)
        self.assertEqual(res["source"], "out_of_bounds")
        self.assertTrue(res["is_out_of_bounds"])
        self.assertIn("outside plausible bounds", res["reason"])

    def test_fuel_out_of_bounds_exclusion(self):
        # Explicit test for out-of-bounds exclusion
        # When fuel effect calculation yields negative or abnormally high values,
        # it must be marked out_of_bounds and excluded from adoption (falling back to config default).
        drivers_data = []
        for d in ["HAM", "RUS"]:
            s1 = pd.DataFrame({
                "Driver": [d] * 8,
                "Stint": [1] * 8,
                "Compound": ["HARD"] * 8,
                "LapNumber": list(range(2, 10)),
                "LapTimeSec": [85.0 for _ in range(8)],
                "TyreLife": list(range(1, 9)),
            })
            s2 = pd.DataFrame({
                "Driver": [d] * 8,
                "Stint": [2] * 8,
                "Compound": ["HARD"] * 8,
                "LapNumber": list(range(25, 33)),
                # Negative pace change: laps get SLOWER by 2s (fuel effect would be negative)
                "LapTimeSec": [87.0 for _ in range(8)],
                "TyreLife": list(range(1, 9)),
            })
            drivers_data.extend([s1, s2])
        laps = pd.concat(drivers_data, ignore_index=True)
        res = estimate_fuel_effect(laps, min_bound=0.02, max_bound=0.08)
        self.assertEqual(res["source"], "out_of_bounds")
        self.assertTrue(res["is_out_of_bounds"])
        self.assertFalse(res["is_estimated"])
        self.assertEqual(res["fuel_effect"], DEFAULT_FUEL_EFFECT_SEC_PER_LAP)

    def test_estimate_fuel_effect_successful_estimation(self):
        # 2 drivers with 2 matched dry stints where pace gain is realistic (~0.035 s/lap)
        drivers_data = []
        for d in ["VER", "PER"]:
            s1 = pd.DataFrame({
                "Driver": [d] * 10,
                "Stint": [1] * 10,
                "Compound": ["MEDIUM"] * 10,
                "LapNumber": list(range(2, 12)),
                "LapTimeSec": [81.0 for _ in range(10)],
                "TyreLife": list(range(1, 11)),
            })
            s2 = pd.DataFrame({
                "Driver": [d] * 10,
                "Stint": [2] * 10,
                "Compound": ["MEDIUM"] * 10,
                "LapNumber": list(range(22, 32)),
                "LapTimeSec": [81.0 - (20 * 0.035) for _ in range(10)],  # 0.035 s/lap
                "TyreLife": list(range(1, 11)),
            })
            drivers_data.extend([s1, s2])
        laps = pd.concat(drivers_data, ignore_index=True)
        res = estimate_fuel_effect(laps, min_bound=0.02, max_bound=0.05)
        self.assertEqual(res["source"], "estimated")
        self.assertTrue(res["is_estimated"])
    def test_normalize_lap_fuel_constant_under_zero_tyre_effect(self):
        # Known-answer test: lap times fall by exactly 0.058 s/lap (pure fuel burn, zero tyre wear)
        # T(L) = 90.0 - 0.058 * L
        laps = np.arange(1, 31)
        raw_times = 90.0 - 0.058 * laps
        # Normalize to reference lap 1
        normalized = normalize_lap_fuel(raw_times, laps, reference_lap=1, fuel_effect_sec_per_lap=0.058)
        # Expected value is 90.0 - 0.058 * 1 = 89.942 across all laps
        np.testing.assert_allclose(normalized, 89.942, atol=1e-6)
        self.assertAlmostEqual(float(np.std(normalized)), 0.0, places=6)

    def test_stop_no_tyre_effect_gives_zero_pace_gain(self):
        # Known-answer test: Pit stop at lap 20 with identical compound and zero tyre wear
        # Laps 16-20 (pre-stop) and Laps 22-26 (post-stop) follow pure fuel burn T(L) = 85.0 - 0.058 * L
        pit_lap = 20
        pre_laps = np.array([16, 17, 18, 19, 20])
        post_laps = np.array([22, 23, 24, 25, 26])
        pre_raw = 85.0 - 0.058 * pre_laps
        post_raw = 85.0 - 0.058 * post_laps

        # Normalize both to fuel load at pit_lap using shared helper
        pre_norm = normalize_lap_fuel(pre_raw, pre_laps, reference_lap=pit_lap, fuel_effect_sec_per_lap=0.058)
        post_norm = normalize_lap_fuel(post_raw, post_laps, reference_lap=pit_lap, fuel_effect_sec_per_lap=0.058)

        # Pre and post normalized lap times must both equal 85.0 - 0.058 * 20 = 83.840
        np.testing.assert_allclose(pre_norm, 83.840, atol=1e-6)
        np.testing.assert_allclose(post_norm, 83.840, atol=1e-6)
        delta_gain = float(np.mean(post_norm) - np.mean(pre_norm))
        self.assertAlmostEqual(delta_gain, 0.0, places=6)

    def test_s1a_recovers_fuel_coefficient_right_sign(self):
        # Known-answer test: S1a trajectory with zero tyre degradation
        # As fuel burns, pace improves (lap time falls) by fuel_effect per lap
        anchor = 80.0
        h = 10
        traj = predict_s1a_lap_pace(anchor, horizon_laps=h, compound_deg_slope=0.0, fuel_effect_sec_per_lap=0.058)
        # Lap h=1: 80.0 - 0.058 = 79.942
        # Lap h=10: 80.0 - 0.580 = 79.420
        self.assertAlmostEqual(traj[0], 79.942, places=3)
        self.assertAlmostEqual(traj[9], 79.420, places=3)
        # Slope across steps must be exactly -0.058 s/lap (negative)
        steps = np.arange(1, h + 1)
        fitted_slope = float(np.polyfit(steps, traj, deg=1)[0])
        self.assertAlmostEqual(fitted_slope, -0.058, places=6)


if __name__ == "__main__":
    unittest.main()

