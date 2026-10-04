"""
Unit tests for deterministic Hypothesis Testing engine.
"""

import unittest
from statistics.hypothesis_test import run_hypothesis_test, HypothesisTestInput, HypothesisTestResult


class TestHypothesisTestingEngine(unittest.TestCase):

    def test_normal_2x2_table(self):
        """Test 1: Normal 2x2 table calculation (50/1000 vs 25/1000)."""
        res = run_hypothesis_test(
            treatment_events=50,
            treatment_n=1000,
            control_events=25,
            control_n=1000
        )
        self.assertEqual(res.a, 50)
        self.assertEqual(res.b, 950)
        self.assertEqual(res.c, 25)
        self.assertEqual(res.d, 975)
        self.assertTrue(res.fisher_p_value < 0.05)
        self.assertTrue(res.chi_square_p_value < 0.05)
        self.assertEqual(res.chi_square_degrees_of_freedom, 1)
        self.assertEqual(len(res.expected_frequencies), 2)
        self.assertEqual(len(res.expected_frequencies[0]), 2)
        self.assertAlmostEqual(res.expected_frequencies[0][0], 37.5, places=2)

    def test_small_count_table(self):
        """Test 2: Small-count table where Fisher is appropriate (4/20 vs 0/20)."""
        res = run_hypothesis_test(
            treatment_events=4,
            treatment_n=20,
            control_events=0,
            control_n=20
        )
        self.assertEqual(res.a, 4)
        self.assertEqual(res.c, 0)
        self.assertTrue(res.fisher_odds_ratio > 1.0)
        self.assertTrue(0.0 <= res.fisher_p_value <= 1.0)

    def test_zero_event_cell(self):
        """Test 3: Zero event cell in one arm."""
        res = run_hypothesis_test(
            treatment_events=0,
            treatment_n=100,
            control_events=10,
            control_n=100
        )
        self.assertEqual(res.a, 0)
        self.assertEqual(res.c, 10)
        self.assertEqual(res.fisher_odds_ratio, 0.0)
        self.assertTrue(res.fisher_p_value < 0.05)

    def test_zero_non_event_cell(self):
        """Test 4: Zero non-event cell (all participants have event)."""
        res = run_hypothesis_test(
            treatment_events=100,
            treatment_n=100,
            control_events=50,
            control_n=100
        )
        self.assertEqual(res.b, 0)
        self.assertEqual(res.fisher_odds_ratio, float("inf"))
        self.assertTrue(res.fisher_p_value < 0.001)

    def test_symmetric_no_association_table(self):
        """Test 5: Symmetric/no-association table (equal proportions)."""
        res = run_hypothesis_test(
            treatment_events=50,
            treatment_n=500,
            control_events=50,
            control_n=500
        )
        self.assertAlmostEqual(res.fisher_odds_ratio, 1.0, places=4)
        self.assertAlmostEqual(res.fisher_p_value, 1.0, places=4)
        self.assertAlmostEqual(res.chi_square_statistic, 0.0, places=4)
        self.assertAlmostEqual(res.chi_square_p_value, 1.0, places=4)

    def test_invalid_events_greater_than_n(self):
        """Test 6: Invalid events > N raises ValueError."""
        with self.assertRaises(ValueError):
            run_hypothesis_test(
                treatment_events=120,
                treatment_n=100,
                control_events=10,
                control_n=100
            )

    def test_invalid_negative_values(self):
        """Test 7: Negative values raise ValueError."""
        with self.assertRaises(ValueError):
            run_hypothesis_test(
                treatment_events=-5,
                treatment_n=100,
                control_events=10,
                control_n=100
            )

    def test_invalid_n_equals_zero(self):
        """Test 8: N = 0 raises ValueError."""
        with self.assertRaises(ValueError):
            run_hypothesis_test(
                treatment_events=10,
                treatment_n=0,
                control_events=10,
                control_n=100
            )


if __name__ == "__main__":
    unittest.main()
