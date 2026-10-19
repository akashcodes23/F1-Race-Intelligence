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

    def test_compute_leader_gap_proxy_lead_lap(self):
        from src.driver_comparison import compute_leader_gap_proxy
        df = compute_leader_gap_proxy(self.laps, current_lap=10, reference_lap_time=80.0)
        self.assertFalse(df.empty)
        leader_row = df.iloc[0]
        self.assertEqual(leader_row["driver"], "VER")
        self.assertEqual(leader_row["gap_to_lead_s"], 0.0)
        self.assertEqual(leader_row["gap_display"], "LEADER")
        self.assertEqual(leader_row["status"], "LEADER")

        ham_row = df[df["driver"] == "HAM"].iloc[0]
        self.assertEqual(ham_row["status"], "ON_LEAD_LAP")
        self.assertFalse(ham_row["is_lapped"])
        self.assertTrue(ham_row["gap_to_lead_s"] > 0)
        self.assertTrue(ham_row["gap_display"].startswith("+"))

    def test_compute_leader_gap_proxy_lapped_and_out(self):
        from src.driver_comparison import compute_leader_gap_proxy
        # Add a lapped car (NOR with 8 laps) and a retired car (ALO with 0 laps)
        laps_nor = pd.DataFrame({
            "Driver": ["NOR"] * 8,
            "LapNumber": list(range(1, 9)),
            "LapTimeSec": [82.0] * 8,
            "Position": [3] * 8,
        })
        laps_alo = pd.DataFrame({
            "Driver": ["ALO"] * 1,
            "LapNumber": [1],
            "LapTimeSec": pd.Series([float("nan")], dtype="float64"),
            "Position": [20],
        })
        extended_laps = pd.concat([self.laps, laps_nor, laps_alo], ignore_index=True)

        df = compute_leader_gap_proxy(extended_laps, current_lap=10, reference_lap_time=80.0)
        nor_row = df[df["driver"] == "NOR"].iloc[0]
        self.assertTrue(nor_row["is_lapped"])
        self.assertEqual(nor_row["status"], "LAPPED")
        self.assertEqual(nor_row["gap_display"], "+2 LAPS")

        alo_row = df[df["driver"] == "ALO"].iloc[0]
        self.assertIn(alo_row["status"], ["INCOMPLETE_DATA", "OUT"])


if __name__ == "__main__":
    unittest.main()
