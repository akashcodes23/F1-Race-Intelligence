import unittest

import numpy as np
import pandas as pd

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
        self.assertTrue(res["is_comparable"])
        self.assertAlmostEqual(res["slope"], 0.08, places=1)

        # Check both i.i.d. and HAC Confidence Intervals
        self.assertIn("slope_ci_lower_iid", res)
        self.assertIn("slope_ci_upper_iid", res)
        self.assertIn("slope_ci_lower_hac", res)
        self.assertIn("slope_ci_upper_hac", res)
        self.assertTrue(res["slope_ci_lower_iid"] <= res["slope"] <= res["slope_ci_upper_iid"])
        self.assertTrue(res["slope_ci_lower_hac"] <= res["slope"] <= res["slope_ci_upper_hac"])
        self.assertEqual(res["compound"], "SOFT")

    def test_fuel_correction_effect(self):
        # With fuel correction (+0.033 s/lap), degradation slope should be higher (steeper)
        res_raw = compute_stint_degradation(self.stint_laps, min_laps=5, use_fuel_correction=False)
        res_fuel = compute_stint_degradation(self.stint_laps, min_laps=5, fuel_effect=0.033, use_fuel_correction=True)
        self.assertTrue(res_fuel["valid"])
        self.assertGreater(res_fuel["slope"], res_raw["slope"])

    def test_wet_compound_not_comparable(self):
        # Wet compound (INTERMEDIATE or WET) must return slope=None and status 'not comparable'
        wet_laps = pd.DataFrame({
            "Driver": ["VER"] * 15,
            "Stint": [2] * 15,
            "Compound": ["INTERMEDIATE"] * 15,
            "LapNumber": list(range(2, 17)),
            "LapTimeSec": [85.0 + (i * 0.05) for i in range(15)],
            "TyreLife": list(range(1, 16)),
        })
        res = compute_stint_degradation(wet_laps, min_laps=5)
        self.assertTrue(res["valid"])
        self.assertFalse(res["is_comparable"])
        self.assertIsNone(res["slope"])
        self.assertIn("not comparable", res["status"])
        self.assertEqual(res["degradation_formatted"], "Not comparable (wet compound)")

    def test_detect_tyre_cliff(self):
        # Synthetic stint with clear cliff at lap 12 (satisfying min_cliff_lap=10, min_pre=5, min_post=4)
        x = np.arange(1, 20)
        y = 80.0 + np.where(x <= 11, 0.04 * x, 0.04 * 11 + 0.30 * (x - 11))
        cliff = detect_tyre_cliff(x, y, min_cliff_lap=10, min_laps_pre=5, min_laps_post=4)
        self.assertTrue(cliff["cliff_detected"])
        self.assertIsNotNone(cliff["cliff_lap"])
        self.assertGreaterEqual(cliff["cliff_lap"], 10)
        self.assertGreater(cliff["post_cliff_slope"], cliff["pre_cliff_slope"])
    @staticmethod
    def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        return center - half, center + half

    def test_cliff_detector_false_positive_rate_bounded(self):
        # Monte Carlo test on 1000 noisy linear stints with NO cliff (i.i.d. noise)
        np.random.seed(42)
        n_simulations = 1000
        false_positives = 0

        for _ in range(n_simulations):
            x = np.arange(1, 26)
            noise = np.random.normal(0, 0.05, len(x))
            y = 80.0 + 0.05 * x + noise

            res = detect_tyre_cliff(x, y, min_cliff_lap=10, min_laps_pre=5, min_laps_post=4)
            if res["cliff_detected"]:
                false_positives += 1

        fp_rate = false_positives / n_simulations
        ci_lower, ci_upper = self._wilson_interval(false_positives, n_simulations)
        print(f"\n[CLIFF DETECTOR i.i.d.] FP Rate: {fp_rate:.1%} ({false_positives}/{n_simulations}), 95% Wilson: [{ci_lower:.3%}, {ci_upper:.3%}]")
        self.assertLessEqual(fp_rate, 0.05, f"False positive rate {fp_rate:.1%} exceeds 5% threshold")
        self.assertLessEqual(ci_upper, 0.06)

    def test_cliff_detector_ar1_autocorrelated_noise(self):
        # Test cliff detector against autocorrelated AR(1) lap time noise (rho = 0.5) with NO cliff (1000 sims)
        np.random.seed(42)
        n_sims = 1000
        false_positives = 0
        rho = 0.5
        sigma = 0.05

        for _ in range(n_sims):
            x = np.arange(1, 26)
            n = len(x)
            eps = np.random.normal(0, sigma, n)
            e = np.zeros(n)
            e[0] = eps[0]
            for t in range(1, n):
                e[t] = rho * e[t - 1] + np.sqrt(1 - rho**2) * eps[t]
            y = 80.0 + 0.05 * x + e

            res = detect_tyre_cliff(x, y, min_cliff_lap=10, min_laps_pre=5, min_laps_post=4)
            if res["cliff_detected"]:
                false_positives += 1

        ar1_fp_rate = false_positives / n_sims
        ci_lower, ci_upper = self._wilson_interval(false_positives, n_sims)
        print(f"\n[CLIFF DETECTOR AR(1) NOISE] FP Rate: {ar1_fp_rate:.1%} ({false_positives}/{n_sims}), 95% Wilson: [{ci_lower:.3%}, {ci_upper:.3%}]")
        self.assertLessEqual(ar1_fp_rate, 0.05, f"AR(1) false positive rate {ar1_fp_rate:.1%} exceeds threshold")
        self.assertLessEqual(ci_upper, 0.065)

    def test_cliff_detector_power_curve(self):
        # Power tests: Inject cliffs of delta_s in [+0.05, +0.10, +0.20] s/lap at Lap 15 under AR(1) noise (1000 sims)
        np.random.seed(42)
        n_sims = 1000
        rates = {}
        rho = 0.5
        sigma = 0.05

        for delta_s in [0.05, 0.10, 0.20]:
            detected = 0
            for _ in range(n_sims):
                x = np.arange(1, 26)
                n = len(x)
                eps = np.random.normal(0, sigma, n)
                e = np.zeros(n)
                e[0] = eps[0]
                for t in range(1, n):
                    e[t] = rho * e[t - 1] + np.sqrt(1 - rho**2) * eps[t]
                y = 80.0 + np.where(x <= 15, 0.04 * x, 0.04 * 15 + (0.04 + delta_s) * (x - 15)) + e
                res = detect_tyre_cliff(x, y, min_cliff_lap=10, min_laps_pre=5, min_laps_post=4)
                if res["cliff_detected"] and 13 <= res["cliff_lap"] <= 17:
                    detected += 1
            rates[delta_s] = detected / n_sims

        print(f"\n[CLIFF DETECTOR POWER TESTS under AR(1)] Detection rates: {rates}")
        # Verify sensitivity increases monotonically with cliff severity
        self.assertGreaterEqual(rates[0.05], 0.70)
        self.assertGreaterEqual(rates[0.10], 0.95)
        self.assertGreaterEqual(rates[0.20], 0.99)

    def test_analyze_all_stints(self):
        stint2 = pd.DataFrame({
            "Driver": ["LEC"] * 15,
            "Stint": [2] * 15,
            "Compound": ["HARD"] * 15,
            "LapNumber": list(range(18, 33)),
            "LapTimeSec": [79.0 + (i * 0.05) for i in range(15)],
            "TyreLife": list(range(1, 16)),
        })
        wet_stint = pd.DataFrame({
            "Driver": ["LEC"] * 10,
            "Stint": [3] * 10,
            "Compound": ["INTERMEDIATE"] * 10,
            "LapNumber": list(range(35, 45)),
            "LapTimeSec": [90.0 for _ in range(10)],
            "TyreLife": list(range(1, 11)),
        })
        all_laps = pd.concat([self.stint_laps, stint2, wet_stint], ignore_index=True)
        summary = analyze_all_stints(all_laps)
        self.assertEqual(len(summary), 3)
        self.assertIn("DegradationRate", summary.columns)
        self.assertIn("SlopeCI", summary.columns)
        self.assertIn("SlopeCI_HAC", summary.columns)
        self.assertIn("TyreCliff", summary.columns)

        # Check wet stint row
        wet_row = summary[summary["Compound"] == "INTERMEDIATE"].iloc[0]
        self.assertTrue(pd.isna(wet_row["DegradationRate"]))
        self.assertEqual(wet_row["DegradationFormatted"], "Not comparable (wet)")
        self.assertEqual(wet_row["SlopeCI_HAC"], "N/A")

    def test_get_compound_summary(self):
        summary = get_compound_summary(self.stint_laps)
        self.assertFalse(summary.empty)
        self.assertEqual(summary["Compound"].iloc[0], "SOFT")
        self.assertIn("CliffFrequency", summary.columns)
        self.assertIn("Slope95CI_HAC", summary.columns)
        self.assertIn("CIWarning", summary.columns)

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


if __name__ == "__main__":
    unittest.main()
