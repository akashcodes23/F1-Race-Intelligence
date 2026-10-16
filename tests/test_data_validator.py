import unittest

import pandas as pd

from src.data_validator import clean_laps, clean_laps_with_stats


class TestDataValidator(unittest.TestCase):
    def setUp(self):
        # Create a mock laps dataframe
        self.mock_laps = pd.DataFrame({
            "LapNumber": list(range(1, 31)),
            "LapTimeSec": [80.0 + i * 0.1 for i in range(30)],
            "PitInTime": [pd.NaT] * 29 + [pd.Timestamp("2023-01-01 12:00:00")],
            "PitOutTime": [pd.NaT] * 30,
            "Compound": ["MEDIUM"] * 30,
        })

    def test_clean_laps_empty(self):
        empty_df = clean_laps(pd.DataFrame())
        self.assertTrue(empty_df.empty)

    def test_clean_laps_removes_pit_laps(self):
        cleaned = clean_laps(self.mock_laps, remove_pit_laps=True)
        # Lap 30 has PitInTime, so it should be dropped
        self.assertNotIn(30, cleaned["LapNumber"].values)

    def test_clean_laps_with_stats(self):
        cleaned, stats = clean_laps_with_stats(self.mock_laps)
        self.assertIn("original_count", stats)
        self.assertIn("final_count", stats)
        self.assertEqual(stats["original_count"], 30)
        self.assertTrue(stats["sufficient"])

    def test_clean_laps_handles_missing_laptimesec_with_laptime(self):
        df = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": [pd.Timedelta(seconds=85.2), pd.Timedelta(seconds=84.8), pd.Timedelta(seconds=85.0)],
        })
        cleaned = clean_laps(df, outlier_low=0.0, outlier_high=1.0)
        self.assertIn("LapTimeSec", cleaned.columns)
        self.assertEqual(len(cleaned), 3)
        self.assertAlmostEqual(cleaned["LapTimeSec"].iloc[0], 85.2)


if __name__ == "__main__":
    unittest.main()
