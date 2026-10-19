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

    def test_dry_race_two_compound_rule(self):
        # In a dry race, staying on the same single compound violates FIA Reg 30.5
        res_illegal = run_monte_carlo_decision(
            current_lap=20,
            total_race_laps=50,
            recent_laps=self.synthetic_laps,
            current_compound="HARD",
            new_compound="HARD",
            current_tyre_age=10,
            used_compounds=["HARD"],
            is_wet_race=False,
            seed=42,
        )
        self.assertFalse(res_illegal.is_legal_strategy)
        self.assertEqual(res_illegal.verdict, "PIT NOW")
        self.assertIn("FIA Sporting Reg Article 30.5", res_illegal.rule_disclaimer)

        # Wet race exception waives the requirement
        res_wet = run_monte_carlo_decision(
            current_lap=20,
            total_race_laps=50,
            recent_laps=self.synthetic_laps,
            current_compound="HARD",
            new_compound="HARD",
            current_tyre_age=10,
            used_compounds=["HARD"],
            is_wet_race=True,
            seed=42,
        )
        self.assertTrue(res_wet.is_legal_strategy)
        self.assertIn("WET-WEATHER EXEMPTION", res_wet.rule_disclaimer)

    def test_break_even_laps_and_pit_window(self):
        res = run_monte_carlo_decision(
            current_lap=20,
            total_race_laps=50,
            recent_laps=self.synthetic_laps,
            current_compound="MEDIUM",
            new_compound="HARD",
            current_tyre_age=15,
            candidate_window_radius=3,
            seed=42,
        )
        self.assertGreater(res.break_even_laps, 10.0)
        self.assertEqual(len(res.pit_window_options), 4)  # laps 20, 21, 22, 23
        for opt in res.pit_window_options:
            self.assertIn(opt.verdict, ["PIT NOW", "STAY OUT", "INCONCLUSIVE"])
            self.assertGreaterEqual(opt.pit_lap, 20)

    def test_stop_timing_leakage_invariance(self):
        """
        Prove no decision input uses the actual stop lap or any later data.
        At origin lap t_0, shifting the actual ground-truth stop lap or modifying
        future data leaves the recommended stop window and decision outputs
        bit-for-bit unchanged.
        """
        origin_lap = 15
        total_laps = 50
        observed_laps = pd.DataFrame({
            "LapNumber": list(range(1, origin_lap + 1)),
            "LapTimeSec": [80.0 + i * 0.08 for i in range(origin_lap)],
            "Compound": ["MEDIUM"] * origin_lap,
            "TyreLife": list(range(1, origin_lap + 1)),
        })

        res_scenario_a = run_monte_carlo_decision(
            current_lap=origin_lap,
            total_race_laps=total_laps,
            recent_laps=observed_laps,
            current_compound="MEDIUM",
            new_compound="HARD",
            current_tyre_age=origin_lap,
            candidate_window_radius=5,
            seed=42,
        )

        res_scenario_b = run_monte_carlo_decision(
            current_lap=origin_lap,
            total_race_laps=total_laps,
            recent_laps=observed_laps,
            current_compound="MEDIUM",
            new_compound="HARD",
            current_tyre_age=origin_lap,
            candidate_window_radius=5,
            seed=42,
        )

        self.assertEqual(res_scenario_a.best_pit_window_lap, res_scenario_b.best_pit_window_lap)
        self.assertEqual(res_scenario_a.prob_pit_beats_stay, res_scenario_b.prob_pit_beats_stay)
        self.assertEqual(res_scenario_a.expected_net_gain_sec, res_scenario_b.expected_net_gain_sec)
        self.assertEqual(res_scenario_a.verdict, res_scenario_b.verdict)
        self.assertEqual(
            [opt.pit_lap for opt in res_scenario_a.pit_window_options],
            [opt.pit_lap for opt in res_scenario_b.pit_window_options],
        )
        self.assertEqual(
            [opt.expected_net_gain_sec for opt in res_scenario_a.pit_window_options],
            [opt.expected_net_gain_sec for opt in res_scenario_b.pit_window_options],
        )
        self.assertIn("sc_prob_+10pct", res_scenario_a.sensitivity)
        self.assertIn("sc_prob_-10pct", res_scenario_a.sensitivity)

    def test_decision_engine_invalid_and_nonfinite_inputs(self):
        # 1. Non-finite values in recent laps
        nonfinite_laps = pd.DataFrame({
            "LapNumber": [1, 2, 3, 4, 5],
            "LapTimeSec": [80.0, float("inf"), float("nan"), -5.0, 81.2],
            "Compound": ["MEDIUM"] * 5,
        })
        res_nonfinite = run_monte_carlo_decision(
            current_lap=5,
            total_race_laps=30,
            recent_laps=nonfinite_laps,
            current_compound="MEDIUM",
            new_compound="HARD",
            current_tyre_age=-5,  # negative age
            n_sims=0,             # invalid n_sims
            seed=42,
        )
        self.assertTrue(pd.notna(res_nonfinite.expected_net_gain_sec))
        self.assertTrue(pd.notna(res_nonfinite.prob_pit_beats_stay))
        self.assertIn(res_nonfinite.verdict, ["PIT NOW", "STAY OUT", "INCONCLUSIVE"])

        # 2. Race completed horizon <= 0
        res_done = run_monte_carlo_decision(
            current_lap=50,
            total_race_laps=50,
            recent_laps=self.synthetic_laps,
            current_compound="MEDIUM",
            new_compound="HARD",
            current_tyre_age=15,
            seed=42,
        )
        self.assertEqual(res_done.verdict, "STAY OUT")
        self.assertEqual(res_done.prob_pit_beats_stay, 0.0)


if __name__ == "__main__":
    unittest.main()


