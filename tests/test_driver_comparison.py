import unittest

import pandas as pd

from src.driver_comparison import (
    compare_driver_consistency,
    compare_driver_sectors,
    compute_driver_delta,
)


class TestDriverComparison(unittest.TestCase):
    def setUp(self):
        # 20 laps for VER and HAM
        ver_times = [80.0 + i * 0.05 for i in range(20)]
        ham_times = [80.2 + i * 0.04 for i in range(20)]

        laps_ver = pd.DataFrame({
            "Driver": ["VER"] * 20,
            "LapNumber": list(range(1, 21)),
            "LapTimeSec": ver_times,
            "Sector1TimeSec": [25.0] * 20,
            "Sector2TimeSec": [28.0] * 20,
            "Sector3TimeSec": [27.0] * 20,
        })
        laps_ham = pd.DataFrame({
            "Driver": ["HAM"] * 20,
            "LapNumber": list(range(1, 21)),
            "LapTimeSec": ham_times,
            "Sector1TimeSec": [25.1] * 20,
            "Sector2TimeSec": [27.9] * 20,
            "Sector3TimeSec": [27.2] * 20,
        })
        self.laps = pd.concat([laps_ver, laps_ham], ignore_index=True)

    def test_compute_driver_delta(self):
        merged, summary = compute_driver_delta(self.laps, "VER", "HAM")
        self.assertEqual(len(merged), 19)
        self.assertIn("Delta", merged.columns)
        self.assertIn("CumulativeDelta", merged.columns)
        self.assertIn("faster_driver", summary)
        # Verify delta calculation
        self.assertTrue(len(merged["Delta"]) > 0)

    def test_compare_driver_consistency(self):
        res = compare_driver_consistency(self.laps, "VER", "HAM")
        self.assertIn("VER", res)
        self.assertIn("HAM", res)
        self.assertTrue(res["VER"]["std"] > 0)
        self.assertEqual(res["VER"]["n_laps"], 19)

    def test_compare_driver_sectors(self):
        df, summary = compare_driver_sectors(self.laps, "VER", "HAM")
        self.assertFalse(df.empty)
        self.assertIn("optimal_lap", summary["VER"])
        self.assertAlmostEqual(summary["VER"]["optimal_lap"], 80.0)


if __name__ == "__main__":
    unittest.main()
