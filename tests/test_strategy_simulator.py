import unittest

from src.strategy_simulator import (
    evaluate_pit_window,
    pit_loss_sensitivity_analysis,
    simulate_strategy_horizon,
)


class TestStrategySimulator(unittest.TestCase):
    def test_simulate_strategy_horizon(self):
        res = simulate_strategy_horizon(
            current_lap=10,
            total_race_laps=30,
            current_tire_age=10,
            base_pace=80.0,
            current_compound_deg=0.15,
            new_compound_base_offset=-0.5,
            new_compound_deg=0.08,
            pit_loss=20.0,
            pit_lap=15,
        )
        self.assertIn("laps", res)
        self.assertEqual(len(res["laps"]), 21)  # Laps 10 to 30 inclusive
        self.assertIn("optimal_strategy", res)
        self.assertTrue(res["total_stay_time"] > 0)
        self.assertTrue(res["total_pit_time"] > 0)

    def test_evaluate_pit_window(self):
        eval_df = evaluate_pit_window(
            start_candidate_lap=12,
            end_candidate_lap=18,
            total_race_laps=40,
            current_lap=10,
            current_tire_age=10,
            base_pace=80.0,
            current_deg=0.15,
            new_deg=0.08,
            new_compound_delta=-0.5,
            pit_loss=20.0,
        )
        self.assertEqual(len(eval_df), 7)
        self.assertIn("IsOptimal", eval_df.columns)
        self.assertEqual(eval_df["IsOptimal"].sum(), 1)

    def test_pit_loss_sensitivity_analysis(self):
        sens_df = pit_loss_sensitivity_analysis(
            current_lap=10,
            total_race_laps=40,
            current_tire_age=10,
            base_pace=80.0,
            current_deg=0.15,
            new_deg=0.08,
            new_compound_delta=-0.5,
            pit_lap=15,
            loss_range=(18.0, 24.0, 2.0),
        )
        self.assertFalse(sens_df.empty)
        self.assertIn("Scenario", sens_df.columns)


if __name__ == "__main__":
    unittest.main()
