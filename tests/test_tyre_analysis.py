import unittest

import numpy as np
import pandas as pd

from src.features.fuel import estimate_fuel_effect
from src.tyre_analysis import (
    analyze_all_stints,
    compute_driver_consistency_index,
    compute_stint_degradation,
    detect_tyre_cliff,
    get_compound_summary,
)


class TestTyreAnalysis(unittest.TestCase):
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

    def test_compute_stint_degradation_robust(self):
        # Test without fuel correction first to verify pure Huber slope
        res = compute_stint_degradation(self.stint_laps, min_laps=5, use_fuel_correction=False)
        self.assertTrue(res["valid"])
        self.assertAlmostEqual(res["slope"], 0.08, places=1)
        self.assertIn("slope_ci_lower", res)
        self.assertIn("slope_ci_upper", res)
        self.assertTrue(res["slope_ci_lower"] <= res["slope"] <= res["slope_ci_upper"])
        self.assertEqual(res["compound"], "SOFT")

    def test_fuel_correction_effect(self):
        # With fuel correction (+0.033 s/lap), degradation slope should be higher
        res_raw = compute_stint_degradation(self.stint_laps, min_laps=5, use_fuel_correction=False)
        res_fuel = compute_stint_degradation(self.stint_laps, min_laps=5, fuel_effect=0.033, use_fuel_correction=True)
        self.assertTrue(res_fuel["valid"])
        self.assertGreater(res_fuel["slope"], res_raw["slope"])

    def test_detect_tyre_cliff(self):
        # Synthetic stint with clear cliff at lap 12
        x = np.arange(1, 20)
        y = 80.0 + np.where(x <= 11, 0.04 * x, 0.04 * 11 + 0.25 * (x - 11))
        cliff = detect_tyre_cliff(x, y, min_cliff_lap=8, slope_multiplier=1.5)
        self.assertTrue(cliff["cliff_detected"])
        self.assertIsNotNone(cliff["cliff_lap"])
        self.assertGreater(cliff["post_cliff_slope"], cliff["pre_cliff_slope"])

    def test_analyze_all_stints(self):
        stint2 = pd.DataFrame({
            "Driver": ["LEC"] * 15,
            "Stint": [2] * 15,
            "Compound": ["HARD"] * 15,
            "LapNumber": list(range(18, 33)),
            "LapTimeSec": [79.0 + (i * 0.05) for i in range(15)],
            "TyreLife": list(range(1, 16)),
        })
        all_laps = pd.concat([self.stint_laps, stint2], ignore_index=True)
        summary = analyze_all_stints(all_laps)
        self.assertEqual(len(summary), 2)
        self.assertIn("DegradationRate", summary.columns)
        self.assertIn("SlopeCI", summary.columns)
        self.assertIn("TyreCliff", summary.columns)

    def test_get_compound_summary(self):
        summary = get_compound_summary(self.stint_laps)
        self.assertFalse(summary.empty)
        self.assertEqual(summary["Compound"].iloc[0], "SOFT")
        self.assertIn("CliffFrequency", summary.columns)

    def test_compute_driver_consistency_index(self):
        ham_laps = pd.DataFrame({
            "Driver": ["HAM"] * 15,
            "Stint": [1] * 15,
            "Compound": ["MEDIUM"] * 15,
            "LapNumber": list(range(2, 17)),
            "LapTimeSec": [78.5 + (i * 0.02) for i in range(15)],
            "TyreLife": list(range(1, 16)),
        })
        field_laps = pd.concat([self.stint_laps, ham_laps], ignore_index=True)
        consistency = compute_driver_consistency_index(field_laps)
        self.assertEqual(len(consistency), 2)
        self.assertIn("ConsistencyIndex_MAD", consistency.columns)
        self.assertIn("PaceRank", consistency.columns)
        self.assertIn("ConsistencyRank", consistency.columns)

    def test_estimate_fuel_effect_fallback(self):
        # Insufficient stint transitions -> returns config default
        res = estimate_fuel_effect(self.stint_laps)
        self.assertFalse(res["is_estimated"])
        self.assertEqual(res["method"], "config_default")
        self.assertAlmostEqual(res["fuel_effect"], 0.033, places=3)


if __name__ == "__main__":
    unittest.main()
