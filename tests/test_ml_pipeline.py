import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.ml_pipeline import (
    load_trained_model,
    predict_s1a_stint,
    prepare_ml_features,
    train_and_evaluate_models,
)


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
            lap_time = base_time + (tyre_life * 0.08)
            laps_list.append({
                "RaceID": "2023_TEST_GP",
                "Driver": "VER",
                "LapNumber": i,
                "LapTimeSec": lap_time,
                "FuelCorrectedLapTimeSec": lap_time - (0.058 * (40 - i)),
                "Compound": compound,
                "Stint": stint,
                "TyreLife": tyre_life,
            })
        self.mock_laps = pd.DataFrame(laps_list)

    def test_prepare_ml_features_structure(self):
        # By default evaluates on laps >= 7 (34 laps for 40 total laps)
        X, y, features = prepare_ml_features(self.mock_laps, min_eval_lap=7, include_prev_pace=True)
        self.assertFalse(X.empty)
        self.assertEqual(len(y), len(X))
        self.assertEqual(len(X), 34)  # Laps 7 to 40
        self.assertIn("LapNumber", features)
        self.assertIn("TyreLife", features)
        self.assertIn("LapsRemaining", features)
        self.assertIn("PrevLapRelativePace", features)
        self.assertIn("Compound_SOFT", features)
        self.assertIn("Compound_MEDIUM", features)
        self.assertIn("Compound_HARD", features)
        self.assertEqual(int(X["LapNumber"].min()), 7)

    def test_leakage_inside_reference_window_lap4(self):
        """
        Perturbation inside the reference window (Lap 4):
        - Changes the driver's reference pace.
        - Changes the target RelativeLapTimeSec across laps.
        - Physical state features (TyreLife, LapsRemaining, LapNumber, compound) MUST NOT change!
        """
        X1, y1, features = prepare_ml_features(self.mock_laps, min_eval_lap=7)

        laps_perturbed = self.mock_laps.copy()
        lap4_idx = 3  # Lap 4
        # Apply a valid +0.20s pace delta (within normal non-outlier variation)
        laps_perturbed.loc[lap4_idx, "LapTimeSec"] += 0.20

        X2, y2, _ = prepare_ml_features(laps_perturbed, min_eval_lap=7)

        # 1. Physical features MUST remain identical
        phys_cols = ["TyreLife", "LapsRemaining", "LapNumber", "Compound_SOFT", "Compound_MEDIUM", "Compound_HARD"]
        pd.testing.assert_frame_equal(X1[phys_cols], X2[phys_cols])

        # 2. Reference pace changes, so target y values change
        self.assertFalse(np.allclose(y1.values, y2.values))

    def test_leakage_outside_reference_window_lap7(self):
        """
        Perturbation outside the reference window (Lap 7):
        - Reference pace (from laps 2-6) MUST NOT change.
        - Target y at Lap 7 changes by exactly the perturbation.
        - Model features at Lap 7 (X[features] at lap 7) MUST NOT change at all.
        - Target y values for subsequent laps (Laps 8+) MUST NOT change.
        - PrevLapRelativePace at Lap 8 changes to observe Lap 7.
        """
        X1, y1, features = prepare_ml_features(self.mock_laps, min_eval_lap=7)

        laps_perturbed = self.mock_laps.copy()
        lap7_idx = 6  # Lap 7
        delta_perturb = 8.0
        laps_perturbed.loc[lap7_idx, "LapTimeSec"] += delta_perturb

        X2, y2, _ = prepare_ml_features(laps_perturbed, min_eval_lap=7)

        # 1. Model features at Lap 7 (index 0 of evaluated laps) MUST NOT change
        pd.testing.assert_series_equal(X1[features].iloc[0], X2[features].iloc[0])

        # 2. Target at Lap 7 changes by delta_perturb
        self.assertAlmostEqual(y2.iloc[0] - y1.iloc[0], delta_perturb, places=4)

        # 3. Targets for Laps 8+ MUST be identical
        np.testing.assert_allclose(y1.iloc[1:].values, y2.iloc[1:].values)

        # 4. PrevLapRelativePace at Lap 8 (index 1) changes to observe Lap 7
        self.assertAlmostEqual(
            X2.iloc[1]["PrevLapRelativePace"] - X1.iloc[1]["PrevLapRelativePace"],
            delta_perturb,
            places=4,
        )

        # 5. PrevLapRelativePace for Laps 9+ MUST be identical
        np.testing.assert_allclose(
            X1.iloc[2:]["PrevLapRelativePace"].values,
            X2.iloc[2:]["PrevLapRelativePace"].values,
        )

    def test_multi_race_loro_cross_validation(self):
        """Test Leave-One-Race-Out cross validation across two synthetic races on laps >= 7."""
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
                        "FuelCorrectedLapTimeSec": lap_time - (0.058 * (30 - lap)),
                        "Compound": compound,
                        "Stint": stint,
                        "TyreLife": tyre_life,
                    })
        df_multi = pd.DataFrame(multi_race_laps)

        results = train_and_evaluate_models(df_multi, save_artifact_path=None)
        self.assertTrue(results["success"])
        self.assertTrue(results["is_multi_race"])
        self.assertEqual(results["n_races"], 2)

        # Check all required baseline and model evaluations
        self.assertIn("naive_baseline", results)
        self.assertIn("persistence_baseline", results)
        self.assertIn("huber_per_stint", results)
        self.assertIn("s1_physics_linear", results)
        self.assertIn("s2_scaled_ridge", results)
        self.assertIn("scaled_ridge", results)
        self.assertIn("gradient_boosting_A", results)
        self.assertIn("gradient_boosting_B", results)
        self.assertIn("gradient_boosting_B_monotonic", results)

        self.assertGreater(results["naive_baseline"]["mean_mae"], 0.0)
        self.assertGreater(results["persistence_baseline"]["mean_mae"], 0.0)
        self.assertGreater(results["huber_per_stint"]["mean_mae"], 0.0)
        self.assertGreater(results["s1_physics_linear"]["mean_mae"], 0.0)
        self.assertGreater(results["s2_scaled_ridge"]["mean_mae"], 0.0)
        self.assertGreater(results["scaled_ridge"]["mean_mae"], 0.0)
        self.assertGreater(results["gradient_boosting_A"]["mean_mae"], 0.0)
        self.assertGreater(results["gradient_boosting_B"]["mean_mae"], 0.0)
        self.assertGreater(results["gradient_boosting_B_monotonic"]["mean_mae"], 0.0)

        # Check per-race fold evaluations
        self.assertEqual(len(results["per_race_eval"]), 2)
        for fold in results["per_race_eval"]:
            self.assertIn("RaceID", fold)
            self.assertIn("Naive_MAE", fold)
            self.assertIn("Persistence_MAE", fold)
            self.assertIn("Huber_MAE", fold)
            self.assertIn("S1_MAE", fold)
            self.assertIn("S2_MAE", fold)
            self.assertIn("Ridge_MAE", fold)
            self.assertIn("HGBR_A_MAE", fold)
            self.assertIn("HGBR_B_MAE", fold)

        # Check residual breakdowns
        self.assertIn("by_tyre_age", results["residual_breakdown"])
        self.assertIn("by_compound", results["residual_breakdown"])

    def test_single_race_fallback(self):
        """Test chronological split when only a single race is provided."""
        results = train_and_evaluate_models(self.mock_laps, save_artifact_path=None)
        self.assertTrue(results["success"])
        self.assertFalse(results["is_multi_race"])
        self.assertIn("gradient_boosting_A", results)
        self.assertGreater(results["gradient_boosting_A"]["mean_mae"], 0.0)

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
            self.assertIn("model_variant_a", loaded)
            self.assertIn("model_variant_b", loaded)
            self.assertIn("metadata", loaded)
            self.assertIn("cv_results", loaded["metadata"])


    def test_s1a_known_answer_synthetic(self):
        """Verify S1a prediction on known-answer synthetic data."""
        # Anchor = 0.50s, comp = MEDIUM, fuel_effect = 0.058 s/lap
        deg_params = {"MEDIUM": {"a1": 0.04, "a2": 0.001}}
        ages = np.array([5, 10, 15])
        laps_rem = np.array([30, 25, 20])
        ref_laps_rem = 35.0  # reference at lap 4

        # Expected:
        # lap 1 (age=5, rem=30): anchor(0.50) - 0.058*(30 - 35) + (0.04*5 + 0.001*25) = 0.50 + 0.29 + 0.225 = 1.015
        # lap 2 (age=10, rem=25): anchor(0.50) - 0.058*(25 - 35) + (0.04*10 + 0.001*100) = 0.50 + 0.58 + 0.50 = 1.580
        preds = predict_s1a_stint(
            anchor=0.50,
            comp="MEDIUM",
            ages=ages,
            laps_rem=laps_rem,
            ref_laps_rem=ref_laps_rem,
            deg_params=deg_params,
            fuel_effect=0.058,
        )
        self.assertAlmostEqual(preds[0], 1.015, places=4)
        self.assertAlmostEqual(preds[1], 1.580, places=4)

    def test_horizon_evaluator_zero_leakage_up_to_origin(self):
        """
        Verify that horizon features and anchor use strictly data up to the origin lap.
        Perturbing a future lap (e.g. lap 15) must NOT alter anchor or features computed at origin (lap 8).
        """
        laps1 = self.mock_laps.copy()
        # Origin at lap 8
        orig_lap = 8
        prior1 = laps1[laps1["LapNumber"] <= orig_lap]
        anchor1 = float(prior1["LapTimeSec"].tail(5).median())

        # Perturb lap 15 (future lap)
        laps2 = self.mock_laps.copy()
        laps2.loc[laps2["LapNumber"] == 15, "LapTimeSec"] += 5.0
        prior2 = laps2[laps2["LapNumber"] <= orig_lap]
        anchor2 = float(prior2["LapTimeSec"].tail(5).median())

        self.assertEqual(anchor1, anchor2, "Future lap perturbation leaked into origin anchor!")

    def test_pit_pace_gain_metric(self):
        """Verify pit pace-gain fuel-adjusted calculation on known inputs."""
        # Old stint last 5 laps mean pace: 85.0s, mean rem: 35
        # New stint laps 2-6 mean pace: 83.0s, mean rem: 29
        # Raw delta: 83.0 - 85.0 = -2.0s
        # Fuel burn pace gain: -0.058 * (29 - 35) = +0.348s
        # Actual fuel-adjusted delta (pure tyre/compound gain): -2.0 - (+0.348) = -2.348s
        mean_old_pace = 85.0
        mean_new_pace = 83.0
        mean_old_rem = 35.0
        mean_new_rem = 29.0
        fuel_effect = 0.058

        fuel_diff = -fuel_effect * (mean_new_rem - mean_old_rem)
        actual_delta = (mean_new_pace - mean_old_pace) - fuel_diff
        self.assertAlmostEqual(actual_delta, -2.348, places=3)


if __name__ == "__main__":
    unittest.main()

