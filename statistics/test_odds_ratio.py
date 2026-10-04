"""
Unit test suite for deterministic Odds Ratio engine.
"""

import math
import unittest
from statistics.odds_ratio import calculate_odds_ratio, OddsRatioInput, OddsRatioResult


class TestOddsRatioEngine(unittest.TestCase):

    def test_normal_2x2_table(self):
        """Test 1: Normal 2x2 table calculation (50/1000 vs 25/1000)."""
        res = calculate_odds_ratio(
            treatment_events=50,
            treatment_n=1000,
            control_events=25,
            control_n=1000
        )
        self.assertEqual(res.a, 50)
        self.assertEqual(res.b, 950)
        self.assertEqual(res.c, 25)
        self.assertEqual(res.d, 975)
        self.assertAlmostEqual(res.treatment_odds, 50 / 950, places=5)
        self.assertAlmostEqual(res.control_odds, 25 / 975, places=5)
        expected_or = (50 * 975) / (950 * 25)  # 2.0526315
        self.assertAlmostEqual(res.odds_ratio, expected_or, places=4)
        self.assertFalse(res.zero_cell_correction_applied)
        self.assertIn("Woolf", res.ci_method)
        self.assertTrue(res.ci_lower < res.odds_ratio < res.ci_upper)

    def test_zero_treatment_events(self):
        """Test 2: Zero treatment events (a = 0)."""
        res = calculate_odds_ratio(
            treatment_events=0,
            treatment_n=1000,
            control_events=20,
            control_n=1000
        )
        self.assertEqual(res.a, 0)
        self.assertEqual(res.b, 1000)
        self.assertTrue(res.zero_cell_correction_applied)
        self.assertIn("Haldane-Anscombe", res.ci_method)
        self.assertTrue(res.odds_ratio > 0.0)
        self.assertTrue(res.ci_lower < res.odds_ratio < res.ci_upper)

    def test_zero_control_events(self):
        """Test 3: Zero control events (c = 0)."""
        res = calculate_odds_ratio(
            treatment_events=20,
            treatment_n=1000,
            control_events=0,
            control_n=1000
        )
        self.assertEqual(res.c, 0)
        self.assertEqual(res.d, 1000)
        self.assertTrue(res.zero_cell_correction_applied)
        self.assertIn("Haldane-Anscombe", res.ci_method)
        self.assertTrue(res.odds_ratio > 1.0)
        self.assertTrue(res.ci_lower < res.odds_ratio < res.ci_upper)

    def test_all_treatment_participants_have_event(self):
        """Test 4: All treatment participants have event (b = 0)."""
        res = calculate_odds_ratio(
            treatment_events=1000,
            treatment_n=1000,
            control_events=500,
            control_n=1000
        )
        self.assertEqual(res.a, 1000)
        self.assertEqual(res.b, 0)
        self.assertTrue(res.zero_cell_correction_applied)
        self.assertIn("Haldane-Anscombe", res.ci_method)
        self.assertTrue(res.odds_ratio > 1.0)
        self.assertTrue(res.ci_lower < res.odds_ratio < res.ci_upper)

    def test_all_control_participants_have_event(self):
        """Test 5: All control participants have event (d = 0)."""
        res = calculate_odds_ratio(
            treatment_events=500,
            treatment_n=1000,
            control_events=1000,
            control_n=1000
        )
        self.assertEqual(res.c, 1000)
        self.assertEqual(res.d, 0)
        self.assertTrue(res.zero_cell_correction_applied)
        self.assertIn("Haldane-Anscombe", res.ci_method)
        self.assertTrue(res.odds_ratio < 1.0)
        self.assertTrue(res.ci_lower < res.odds_ratio < res.ci_upper)

    def test_invalid_events_greater_than_n(self):
        """Test 6: Invalid events > N raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_odds_ratio(
                treatment_events=1050,
                treatment_n=1000,
                control_events=25,
                control_n=1000
            )

    def test_invalid_negative_values(self):
        """Test 7: Invalid negative values raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_odds_ratio(
                treatment_events=-5,
                treatment_n=1000,
                control_events=25,
                control_n=1000
            )

    def test_invalid_n_equals_zero(self):
        """Test 8: Invalid N = 0 raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_odds_ratio(
                treatment_events=10,
                treatment_n=0,
                control_events=25,
                control_n=1000
            )


if __name__ == "__main__":
    unittest.main()
