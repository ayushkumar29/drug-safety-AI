import math
import unittest
from statistics.risk_ratio import calculate_risk_ratio, RiskRatioInput, RiskRatioResult


class TestRiskRatioEngine(unittest.TestCase):

    def test_normal_2x2_table(self):
        """Test 1: Normal 2x2 table calculation (50/1000 vs 25/1000 -> RR = 2.0)."""
        res = calculate_risk_ratio(
            treatment_events=50,
            treatment_n=1000,
            control_events=25,
            control_n=1000
        )
        self.assertEqual(res.treatment_risk, 0.05)
        self.assertEqual(res.control_risk, 0.025)
        self.assertAlmostEqual(res.risk_ratio, 2.0, places=4)
        self.assertFalse(res.zero_event_correction_applied)
        self.assertTrue(res.ci_lower < 2.0 < res.ci_upper)
        self.assertIn("Katz", res.ci_method)

    def test_zero_events_in_treatment(self):
        """Test 2: Zero events in treatment group."""
        res = calculate_risk_ratio(
            treatment_events=0,
            treatment_n=1000,
            control_events=20,
            control_n=1000
        )
        self.assertEqual(res.treatment_risk, 0.0)
        self.assertEqual(res.control_risk, 0.02)
        self.assertTrue(res.zero_event_correction_applied)
        self.assertIn("Haldane-Anscombe", res.ci_method)
        self.assertTrue(res.risk_ratio > 0.0)
        self.assertTrue(res.ci_lower < res.risk_ratio < res.ci_upper)

    def test_zero_events_in_control(self):
        """Test 3: Zero events in control group."""
        res = calculate_risk_ratio(
            treatment_events=20,
            treatment_n=1000,
            control_events=0,
            control_n=1000
        )
        self.assertEqual(res.treatment_risk, 0.02)
        self.assertEqual(res.control_risk, 0.0)
        self.assertTrue(res.zero_event_correction_applied)
        self.assertIn("Haldane-Anscombe", res.ci_method)
        self.assertTrue(res.risk_ratio > 1.0)
        self.assertTrue(res.ci_lower < res.risk_ratio < res.ci_upper)

    def test_invalid_events_greater_than_n(self):
        """Test 4: Invalid events > N raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_risk_ratio(
                treatment_events=1050,
                treatment_n=1000,
                control_events=25,
                control_n=1000
            )

    def test_invalid_negative_values(self):
        """Test 5: Invalid negative values raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_risk_ratio(
                treatment_events=-5,
                treatment_n=1000,
                control_events=25,
                control_n=1000
            )

    def test_invalid_n_equals_zero(self):
        """Test 6: Invalid N = 0 raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_risk_ratio(
                treatment_events=10,
                treatment_n=0,
                control_events=25,
                control_n=1000
            )


if __name__ == "__main__":
    unittest.main()

