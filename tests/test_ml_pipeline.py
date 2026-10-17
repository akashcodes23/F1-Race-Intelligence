import unittest

import numpy as np
import pandas as pd

from src.ml_pipeline import prepare_ml_features, train_and_evaluate_models


class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        # Generate 40 synthetic laps
        np.random.seed(42)
        laps_list = []
        for i in range(1, 41):
            stint = 1 if i <= 20 else 2
            compound = "MEDIUM" if stint == 1 else "HARD"
            tire_age = i if stint == 1 else (i - 20)
            base_time = 82.0 if compound == "MEDIUM" else 82.8
            lap_time = base_time + (tire_age * 0.08) + np.random.normal(0, 0.1)
            laps_list.append({
                "LapNumber": i,
                "LapTimeSec": lap_time,
                "Compound": compound,
                "Stint": stint,
            })
        self.mock_laps = pd.DataFrame(laps_list)

    def test_prepare_ml_features(self):
        X, y, features = prepare_ml_features(self.mock_laps)
        self.assertFalse(X.empty)
        self.assertEqual(len(y), len(X))
        self.assertIn("LapNumber", features)
        self.assertIn("TireAge", features)
        # Check compound dummy columns exist
        comp_cols = [c for c in features if c.startswith("Compound_")]
        self.assertTrue(len(comp_cols) >= 1)

    def test_train_and_evaluate_models(self):
        results = train_and_evaluate_models(self.mock_laps, test_size=0.2, cv_splits=2)
        self.assertTrue(results["success"])
        self.assertIn("baseline", results)
        self.assertIn("Linear Regression", results["models"])
        self.assertIn("Random Forest Regressor", results["models"])

        # Check metrics exist
        rf_res = results["models"]["Random Forest Regressor"]
        self.assertIn("test_mae", rf_res)
        self.assertIn("test_rmse", rf_res)
        self.assertIn("residuals", rf_res)
        self.assertEqual(len(rf_res["residuals"]), results["n_test"])


if __name__ == "__main__":
    unittest.main()
