"""
Unit tests for Telemetry Lab, distance alignment, and offline telemetry reconstruction.
"""

import unittest

import numpy as np
import pandas as pd

from pages.telemetry_lab import (
    compute_corner_apex_speeds,
    compute_micro_sector_ribbon,
    reconstruct_offline_telemetry,
)
from src.telemetry import align_telemetry_by_distance


class TestTelemetry(unittest.TestCase):
    def setUp(self):
        # 100 sample distance points from 0 to 5000m
        dist = np.linspace(0, 5000, 100)
        self.tel1 = pd.DataFrame({
            "Distance": dist,
            "Speed": 200 + 50 * np.sin(dist / 400),
            "Throttle": np.clip(50 + 50 * np.sin(dist / 400), 0, 100),
            "Brake": np.clip(-50 * np.sin(dist / 400), 0, 100),
            "nGear": np.clip(np.digitize(200 + 50 * np.sin(dist / 400), [0, 100, 150, 200, 250, 300]), 1, 8),
            "RPM": 11000 + 500 * np.sin(dist / 400),
        })
        self.tel2 = pd.DataFrame({
            "Distance": dist,
            "Speed": 195 + 48 * np.sin(dist / 400),
            "Throttle": np.clip(48 + 48 * np.sin(dist / 400), 0, 100),
            "Brake": np.clip(-48 * np.sin(dist / 400), 0, 100),
            "nGear": np.clip(np.digitize(195 + 48 * np.sin(dist / 400), [0, 100, 150, 200, 250, 300]), 1, 8),
            "RPM": 10800 + 500 * np.sin(dist / 400),
        })

    def test_align_telemetry_by_distance(self):
        aligned = align_telemetry_by_distance(self.tel1, self.tel2, num_points=200)
        self.assertEqual(len(aligned), 200)
        self.assertIn("Distance", aligned.columns)
        self.assertIn("Speed_1", aligned.columns)
        self.assertIn("Speed_2", aligned.columns)
        self.assertIn("SpeedDelta", aligned.columns)

        # Polarity: SpeedDelta = Speed_1 - Speed_2
        expected_delta = aligned["Speed_1"] - aligned["Speed_2"]
        np.testing.assert_allclose(aligned["SpeedDelta"].values, expected_delta.values, rtol=1e-5)

    def test_missing_channels_not_invented_or_zeroed(self):
        # tel_missing lacks RPM and nGear
        tel_missing = pd.DataFrame({
            "Distance": np.linspace(0, 5000, 50),
            "Speed": [250.0] * 50,
        })
        aligned = align_telemetry_by_distance(self.tel1, tel_missing, num_points=100)
        self.assertIn("Speed_1", aligned.columns)
        self.assertIn("Speed_2", aligned.columns)
        # RPM_2 and nGear_2 should not be present or zero-filled
        self.assertNotIn("RPM_2", aligned.columns)
        self.assertNotIn("nGear_2", aligned.columns)

    def test_align_telemetry_empty_and_none(self):
        self.assertTrue(align_telemetry_by_distance(None, self.tel2).empty)
        self.assertTrue(align_telemetry_by_distance(self.tel1, pd.DataFrame()).empty)
        self.assertTrue(align_telemetry_by_distance(pd.DataFrame(), pd.DataFrame()).empty)

    def test_reconstruct_offline_telemetry(self):
        dummy_laps = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTimeSec": [80.5, 79.8, 80.1],
            "SpeedST": [325.0, 328.0, 326.0],
        })
        recon = reconstruct_offline_telemetry("Bahrain Grand Prix", "VER", dummy_laps, num_points=300)
        self.assertFalse(recon.empty)
        self.assertEqual(len(recon), 300)
        self.assertIn("Distance", recon.columns)
        self.assertIn("Speed", recon.columns)
        self.assertIn("Throttle", recon.columns)
        self.assertIn("Brake", recon.columns)
        self.assertIn("nGear", recon.columns)
        self.assertIn("RPM", recon.columns)

        # Physical range checks
        self.assertTrue((recon["Speed"] >= 50.0).all())
        self.assertTrue((recon["Throttle"] >= 0.0).all() and (recon["Throttle"] <= 100.0).all())
        self.assertTrue((recon["Brake"] >= 0.0).all() and (recon["Brake"] <= 100.0).all())
        self.assertTrue((recon["nGear"] >= 1).all() and (recon["nGear"] <= 8).all())

    def test_compute_corner_apex_speeds(self):
        apexes = compute_corner_apex_speeds(self.tel1)
        self.assertIsInstance(apexes, pd.DataFrame)
        if not apexes.empty:
            self.assertIn("Corner", apexes.columns)
            self.assertIn("Apex Distance (m)", apexes.columns)
            self.assertIn("Apex Speed (km/h)", apexes.columns)
            self.assertIn("Braking Point (m)", apexes.columns)

    def test_compute_micro_sector_ribbon(self):
        aligned = align_telemetry_by_distance(self.tel1, self.tel2, num_points=200)
        ribbon = compute_micro_sector_ribbon(aligned, bin_size_m=50.0)
        self.assertFalse(ribbon.empty)
        self.assertIn("TimeDeltaSec", ribbon.columns)
        self.assertIn("CumulativeDeltaSec", ribbon.columns)


if __name__ == "__main__":
    unittest.main()
