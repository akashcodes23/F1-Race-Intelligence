"""
Unit tests for S1a known-answer synthetic benchmarks.
Verifies exact mathematical parameter recovery for:
1. Fuel and tyre-wear normalized anchor recovery (zero bias).
2. Future horizon pace predictions at H=1, 5, 10, 20 (zero bias).
3. Pit pace gain prediction from known compound offsets and degradation slopes (zero bias).
"""

import unittest

import numpy as np
import pandas as pd

from src.features.fuel import (
    compute_s1a_anchor,
    predict_s1a_lap_pace,
    predict_s1a_pit_pace_gain,
)


class TestS1aKnownAnswer(unittest.TestCase):
    def setUp(self):
        self.fuel_effect = 0.058
        self.deg_slope = 0.040
        self.base_pace = 80.000

        # Generate 30 laps of clean stint:
        # Time(L) = base_pace + deg_slope * L - fuel_effect * (L - 1)
        laps = np.arange(1, 31)
        times = self.base_pace + self.deg_slope * laps - self.fuel_effect * (laps - 1.0)
        self.synthetic_stint = pd.DataFrame({
            "LapNumber": laps,
            "TyreLife": laps,
            "LapTimeSec": times,
            "RelativeLapTimeSec": times - self.base_pace,
            "Compound": ["MEDIUM"] * 30,
        })

    def test_s1a_anchor_recovery_zero_bias(self):
        # At origin lap 10 (tyre age 10):
        orig_lap = 10
        orig_age = 10
        true_orig_pace = float(self.synthetic_stint.loc[self.synthetic_stint["LapNumber"] == orig_lap, "RelativeLapTimeSec"].iloc[0])

        prior_laps = self.synthetic_stint[self.synthetic_stint["LapNumber"] <= orig_lap]
        recovered_anchor = compute_s1a_anchor(
            recent_laps=prior_laps,
            origin_lap=orig_lap,
            origin_tyre_age=orig_age,
            compound_deg_slope=self.deg_slope,
            fuel_effect_sec_per_lap=self.fuel_effect,
            n_prior_laps=5,
        )

        # Anchor must match true origin pace with zero bias (< 1e-6s)
        self.assertAlmostEqual(recovered_anchor, true_orig_pace, places=5)

    def test_s1a_horizon_predictions_zero_bias_all_horizons(self):
        # Forecast from origin lap 10 across horizons H = 1, 5, 10, 20
        orig_lap = 10
        orig_age = 10
        prior_laps = self.synthetic_stint[self.synthetic_stint["LapNumber"] <= orig_lap]
        anchor = compute_s1a_anchor(
            recent_laps=prior_laps,
            origin_lap=orig_lap,
            origin_tyre_age=orig_age,
            compound_deg_slope=self.deg_slope,
            fuel_effect_sec_per_lap=self.fuel_effect,
            n_prior_laps=5,
        )

        for h in [1, 5, 10, 20]:
            fut_lap = orig_lap + h
            true_future_pace = float(self.synthetic_stint.loc[self.synthetic_stint["LapNumber"] == fut_lap, "RelativeLapTimeSec"].iloc[0])
            preds = predict_s1a_lap_pace(
                anchor_pace_sec=anchor,
                horizon_laps=h,
                compound_deg_slope=self.deg_slope,
                fuel_effect_sec_per_lap=self.fuel_effect,
            )
            pred_at_h = preds[-1]
            # Must predict with zero bias (< 1e-5s)
            self.assertAlmostEqual(pred_at_h, true_future_pace, places=4, msg=f"Failed zero-bias test at H={h}")

    def test_s1a_pit_pace_gain_recovery_zero_bias(self):
        # Stop from Medium (age 20) to Hard (age 2)
        # Ground truth offsets vs Medium:
        offsets = {"SOFT": -0.500, "MEDIUM": 0.000, "HARD": +0.250}
        slopes = {"SOFT": 0.080, "MEDIUM": 0.040, "HARD": 0.035}

        pre_age = 20.0
        post_age = 2.0

        # True pace relative to base:
        # Pace_pre = Offset_med + slope_med * pre_age = 0.0 + 0.040 * 20 = +0.800s
        # Pace_post = Offset_hard + slope_hard * post_age = +0.250 + 0.035 * 2 = +0.320s
        # True gain (pre - post): 0.800 - 0.320 = +0.480s (faster on new tyres)
        true_gain = (0.000 + slopes["MEDIUM"] * pre_age) - (offsets["HARD"] + slopes["HARD"] * post_age)

        pred_gain = predict_s1a_pit_pace_gain(
            compound_pre="MEDIUM",
            compound_post="HARD",
            pre_tyre_age=pre_age,
            post_tyre_age=post_age,
            compound_offsets=offsets,
            compound_slopes=slopes,
        )

        self.assertAlmostEqual(pred_gain, true_gain, places=6)
        self.assertAlmostEqual(pred_gain, +0.480, places=3)


if __name__ == "__main__":
    unittest.main()
