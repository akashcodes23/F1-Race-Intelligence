import unittest

import pandas as pd

from src.decision_engine import (
    evaluate_undercut_overcut,
    run_monte_carlo_decision,
)


class TestDecisionEngine(unittest.TestCase):
    def setUp(self):
        # Synthetic clean 10-lap stint
        self.synthetic_laps = pd.DataFrame({
            "LapNumber": list(range(1, 11)),
            "LapTimeSec": [80.0 + i * 0.10 for i in range(10)],
            "Compound": ["MEDIUM"] * 10,
            "TyreLife": list(range(1, 11)),
        })

    def test_determinism_with_seed(self):
        # Monte Carlo must produce identical results with identical seed
        res1 = run_monte_carlo_decision(
            current_lap=15,
            total_race_laps=40,
            recent_laps=self.synthetic_laps,
            current_compound="MEDIUM",
            new_compound="HARD",
            current_tyre_age=15,
            n_sims=500,
            seed=42,
        )
        res2 = run_monte_carlo_decision(
            current_lap=15,
            total_race_laps=40,
            recent_laps=self.synthetic_laps,
            current_compound="MEDIUM",
            new_compound="HARD",
            current_tyre_age=15,
            n_sims=500,
            seed=42,
        )
        self.assertEqual(res1.prob_pit_beats_stay, res2.prob_pit_beats_stay)
        self.assertEqual(res1.expected_net_gain_sec, res2.expected_net_gain_sec)
        self.assertEqual(res1.ci_80_lower, res2.ci_80_lower)
        self.assertEqual(res1.ci_80_upper, res2.ci_80_upper)
        self.assertEqual(res1.verdict, res2.verdict)

    def test_probabilities_bounded_in_0_1(self):
        # All probabilities must be in [0, 1]
        for age in [5, 15, 25]:
            res = run_monte_carlo_decision(
                current_lap=10,
                total_race_laps=35,
                recent_laps=self.synthetic_laps,
                current_compound="MEDIUM",
                new_compound="SOFT",
                current_tyre_age=age,
                n_sims=300,
                seed=123,
            )
            self.assertGreaterEqual(res.prob_pit_beats_stay, 0.0)
            self.assertLessEqual(res.prob_pit_beats_stay, 1.0)
            self.assertGreaterEqual(res.prob_best_window, 0.0)
            self.assertLessEqual(res.prob_best_window, 1.0)

    def test_monotonic_in_pit_loss(self):
        # As pit loss increases, probability of pitting must decrease monotonically
        p_prev = 1.0
        for loss in [16.0, 20.0, 24.0, 28.0, 35.0]:
            res = run_monte_carlo_decision(
                current_lap=15,
                total_race_laps=40,
                recent_laps=self.synthetic_laps,
                current_compound="MEDIUM",
                new_compound="HARD",
                current_tyre_age=15,
                fixed_pit_loss=loss,
                n_sims=500,
                seed=42,
            )
            self.assertLessEqual(res.prob_pit_beats_stay, p_prev + 1e-6)
            p_prev = res.prob_pit_beats_stay

    def test_zero_degradation_never_recommends_pitting(self):
        # When tyres have zero degradation and same compound, paying pit loss should never recommend pitting
        flat_laps = pd.DataFrame({
            "LapNumber": list(range(1, 11)),
            "LapTimeSec": [80.0] * 10,
            "Compound": ["HARD"] * 10,
            "TyreLife": list(range(1, 11)),
        })
        res = run_monte_carlo_decision(
            current_lap=15,
            total_race_laps=30,
            recent_laps=flat_laps,
            current_compound="HARD",
            new_compound="HARD",
            current_tyre_age=2,  # Brand new tyre, zero degradation
            fixed_pit_loss=22.0,
            n_sims=500,
            seed=42,
        )
        self.assertNotEqual(res.verdict, "PIT NOW")
        self.assertIn(res.verdict, ["STAY OUT", "INCONCLUSIVE"])
        self.assertLess(res.prob_pit_beats_stay, 0.25)

    def test_inconclusive_rule_triggers_when_interval_spans_zero(self):
        # When expected net gain is close to zero and uncertainty is high,
        # 80% CI spans zero, forcing verdict to INCONCLUSIVE
        # Hand-craft a marginal scenario: small degradation, moderate pit loss
        marginal_laps = pd.DataFrame({
            "LapNumber": list(range(1, 11)),
            "LapTimeSec": [80.0 + i * 0.03 for i in range(10)],
            "Compound": ["MEDIUM"] * 10,
            "TyreLife": list(range(1, 11)),
        })
        res = run_monte_carlo_decision(
            current_lap=25,
            total_race_laps=40,
            recent_laps=marginal_laps,
            current_compound="MEDIUM",
            new_compound="HARD",
            current_tyre_age=14,
            fixed_pit_loss=21.0,
            n_sims=1000,
            seed=42,
        )
        # Verify that if CI spans zero, verdict is strictly INCONCLUSIVE
        if res.ci_80_lower <= 0.0 <= res.ci_80_upper:
            self.assertEqual(res.verdict, "INCONCLUSIVE")
            self.assertIn("INCONCLUSIVE", res.verdict)

    def test_known_answer_undercut_case(self):
        # Case A: Rival is only 0.4s ahead. Fresh soft tyre pace gain is ~1.03s/lap.
        # Expected margin should be positive
        res_close = evaluate_undercut_overcut(
            gap_to_rival_sec=0.4,
            driver_tyre_age=18,
            rival_tyre_age=18,
            compound_current="MEDIUM",
            compound_new="SOFT",
            seed=42,
        )
        self.assertIn(res_close["verdict"], ["UNDERCUT VIABLE", "INCONCLUSIVE"])
        self.assertGreater(res_close["expected_margin_sec"], 0.20)
        self.assertIsNone(res_close["prob_success"])
        self.assertIn("Disabled", res_close["win_probability_policy"])

        # Case B: Rival is 12.0s ahead. Impossible to bridge via 1 lap tyre gain.
        # Undercut must NOT be viable.
        res_far = evaluate_undercut_overcut(
            gap_to_rival_sec=12.0,
            driver_tyre_age=18,
            rival_tyre_age=18,
            compound_current="MEDIUM",
            compound_new="SOFT",
            seed=42,
        )
        self.assertEqual(res_far["verdict"], "UNDERCUT NOT VIABLE")
        self.assertLess(res_far["expected_margin_sec"], -10.0)
        self.assertIsNone(res_far["prob_success"])



if __name__ == "__main__":
    unittest.main()
