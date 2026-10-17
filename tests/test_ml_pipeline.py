import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.ml_pipeline import load_trained_model, prepare_ml_features, train_and_evaluate_models


class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a small synthetic dataset of 40 laps for a single driver
        laps_list = []
        for i in range(1, 41):
            stint = 1 if i <= 20 else 2
            compound = "MEDIUM" if stint == 1 else "HARD"
            tyre_life = i if stint == 1 else (i - 20)
            base_time = 82.0 if compound == "MEDIUM" else 82.8
            lap_time = base_time + (tyre_life * 0.08) + np.random.normal(0, 0.05)
            laps_list.append({
                "RaceID": "2023_TEST_GP",
                "Driver": "VER",
                "LapNumber": i,
                "LapTimeSec": lap_time,
                "FuelCorrectedLapTimeSec": lap_time - (0.033 * (40 - i)),
                "Compound": compound,
                "Stint": stint,
                "TyreLife": tyre_life,
            })
        self.mock_laps = pd.DataFrame(laps_list)

    def test_prepare_ml_features_structure(self):
        X, y, features = prepare_ml_features(self.mock_laps)
        self.assertFalse(X.empty)
        self.assertEqual(len(y), len(X))
        self.assertIn("LapNumber", features)
        self.assertIn("TyreLife", features)
        self.assertIn("LapsRemaining", features)
        self.assertIn("PrevLapRelativePace", features)
        self.assertIn("Compound_SOFT", features)
        self.assertIn("Compound_MEDIUM", features)
        self.assertIn("Compound_HARD", features)

        # Target should be relative pace (centered around 0.0 for early laps)
        self.assertAlmostEqual(y.iloc[:5].median(), 0.0, places=1)

    def test_feature_causality_and_no_leakage(self):
        """
        Critical validation: verify that feature vector X at lap t does NOT
        leak target y[t] or any future information y[t+k] (k >= 0).
        """
        # Baseline feature extraction
        X1, y1, _ = prepare_ml_features(self.mock_laps)

        # Inject an extreme anomaly into Lap 15 target pace
        laps_perturbed = self.mock_laps.copy()
        target_lap_idx = 14  # Lap 15 (0-indexed 14)
        laps_perturbed.loc[target_lap_idx, "LapTimeSec"] += 15.0  # +15 second pit mistake / lockup

        X2, y2, _ = prepare_ml_features(laps_perturbed)

        # 1. Feature row at Lap 15 MUST NOT be affected by the +15s anomaly on Lap 15
        # (PrevLapRelativePace for Lap 15 comes strictly from Lap 14)
        lap_15_features_1 = X1.loc[target_lap_idx, ["TyreLife", "LapsRemaining", "LapNumber", "PrevLapRelativePace"]]
        lap_15_features_2 = X2.loc[target_lap_idx, ["TyreLife", "LapsRemaining", "LapNumber", "PrevLapRelativePace"]]
        pd.testing.assert_series_equal(lap_15_features_1, lap_15_features_2)

        # 2. Features for all earlier laps (1 to 14) MUST be identical
        pd.testing.assert_frame_equal(
            X1.loc[:target_lap_idx - 1, ["TyreLife", "LapsRemaining", "LapNumber", "PrevLapRelativePace"]],
            X2.loc[:target_lap_idx - 1, ["TyreLife", "LapsRemaining", "LapNumber", "PrevLapRelativePace"]],
        )

        # 3. Only the SUBSEQUENT lap (Lap 16) should observe the Lap 15 pace via PrevLapRelativePace
        self.assertNotEqual(
            X1.loc[target_lap_idx + 1, "PrevLapRelativePace"],
            X2.loc[target_lap_idx + 1, "PrevLapRelativePace"],
        )

    def test_multi_race_loro_cross_validation(self):
        """Test Leave-One-Race-Out cross validation across two synthetic races."""
        np.random.seed(99)
        multi_race_laps = []
        for race in ["2023_RACE_A", "2023_RACE_B"]:
            for driver in ["VER", "HAM"]:
                for lap in range(1, 31):
                    stint = 1 if lap <= 15 else 2
                    compound = "MEDIUM" if stint == 1 else "HARD"
                    tyre_life = lap if stint == 1 else (lap - 15)
                    lap_time = 85.0 + (tyre_life * 0.07) + np.random.normal(0, 0.05)
                    multi_race_laps.append({
                        "RaceID": race,
                        "Driver": driver,
                        "LapNumber": lap,
                        "LapTimeSec": lap_time,
                        "FuelCorrectedLapTimeSec": lap_time - (0.033 * (30 - lap)),
                        "Compound": compound,
                        "Stint": stint,
                        "TyreLife": tyre_life,
                    })
        df_multi = pd.DataFrame(multi_race_laps)

        results = train_and_evaluate_models(df_multi, save_artifact_path=None)
        self.assertTrue(results["success"])
        self.assertTrue(results["is_multi_race"])
        self.assertEqual(results["n_races"], 2)

        # Check LORO cross validation metrics
        self.assertIn("naive_baseline", results)
        self.assertIn("linear_model", results)
        self.assertIn("gradient_boosting", results)
        self.assertGreater(results["naive_baseline"]["mean_mae"], 0.0)
        self.assertGreater(results["linear_model"]["mean_mae"], 0.0)
        self.assertGreater(results["gradient_boosting"]["mean_mae"], 0.0)

        # Check per-race fold evaluations
        self.assertEqual(len(results["per_race_eval"]), 2)
        for fold in results["per_race_eval"]:
            self.assertIn("RaceID", fold)
            self.assertIn("Naive_MAE", fold)
            self.assertIn("Linear_MAE", fold)
            self.assertIn("HGBR_MAE", fold)
            self.assertIn("HGBR_vs_Linear_Gain", fold)

        # Check residual breakdowns
        self.assertIn("by_tyre_age", results["residual_breakdown"])
        self.assertIn("by_compound", results["residual_breakdown"])

    def test_single_race_fallback(self):
        """Test chronological split when only a single race is provided."""
        results = train_and_evaluate_models(self.mock_laps, save_artifact_path=None)
        self.assertTrue(results["success"])
        self.assertFalse(results["is_multi_race"])
        self.assertIn("gradient_boosting", results)
        self.assertGreater(results["gradient_boosting"]["mean_mae"], 0.0)

    def test_model_artifact_save_and_load(self):
        """Test saving and loading the model artifact with joblib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "test_model.joblib")
            results = train_and_evaluate_models(self.mock_laps, save_artifact_path=artifact_path)
            self.assertTrue(results["artifact_saved"])
            self.assertTrue(os.path.exists(artifact_path))

            # Load artifact
            loaded = load_trained_model(artifact_path)
            self.assertIsNotNone(loaded)
            self.assertIn("model", loaded)
            self.assertIn("feature_names", loaded)
            self.assertIn("metadata", loaded)
            self.assertIn("cv_results", loaded["metadata"])


if __name__ == "__main__":
    unittest.main()
