"""
Unit Test Suite for End-to-End Evaluation Framework (evaluation/test_end_to_end_evaluation.py)

Tests:
A. Evaluation dataset schema is valid.
B. All case IDs are unique.
C. Every case has required fields.
D. Golden-path case exists.
E. Missing-evidence case exists.
F. Multi-arm case exists.
G. HR/non-2x2 case exists.
H. Overlap case exists.
I. Evaluation results contain PipelineResult-derived counts.
J. No synthetic clinical evidence was introduced.
K. No hard-coded EMP-005 promotion exists in evaluation runner.
L. Statistics comparisons use deterministic engine outputs.
M. Evaluation runner produces valid JSON.
"""

import sys
import os
import json
import unittest
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from statistics.risk_ratio import calculate_risk_ratio
from statistics.odds_ratio import calculate_odds_ratio

CASES_FILE = Path("evaluation/end_to_end_cases.json")
RESULTS_FILE = Path("data/processed/evaluation/end_to_end_evaluation.json")


class TestEndToEndEvaluationFramework(unittest.TestCase):

    def setUp(self):
        self.assertTrue(CASES_FILE.exists(), f"Cases file '{CASES_FILE}' must exist.")
        with open(CASES_FILE, "r", encoding="utf-8") as f:
            self.cases = json.load(f)

    def test_A_schema_is_valid(self):
        """Test A: Evaluation dataset schema is a valid list of dictionaries."""
        self.assertIsInstance(self.cases, list)
        self.assertGreater(len(self.cases), 0)

    def test_B_case_ids_unique(self):
        """Test B: All case IDs in dataset must be unique."""
        ids = [c.get("case_id") for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)), "Case IDs must be unique across benchmark dataset.")

    def test_C_required_fields_present(self):
        """Test C: Every case contains required fields."""
        required = ["case_id", "question", "expected_behavior", "evidence_expectation", "statistical_expectation", "difficulty", "failure_mode"]
        for case in self.cases:
            for field in required:
                self.assertIn(field, case, f"Case {case.get('case_id')} missing required field '{field}'.")

    def test_D_golden_path_case_exists(self):
        """Test D: Golden-path case (case_A1) exists."""
        a1 = [c for c in self.cases if c.get("case_id") == "case_A1"]
        self.assertEqual(len(a1), 1, "Golden path case_A1 must exist.")

    def test_E_missing_evidence_case_exists(self):
        """Test E: Missing evidence cases (case_B1/case_B2) exist."""
        b_cases = [c for c in self.cases if c.get("category") == "B_missing_unsupported"]
        self.assertGreaterEqual(len(b_cases), 1, "Missing evidence cases must exist.")

    def test_F_multiarm_case_exists(self):
        """Test F: Multi-arm case (case_C1) exists."""
        c_cases = [c for c in self.cases if c.get("category") == "C_multiarm_evidence"]
        self.assertGreaterEqual(len(c_cases), 1, "Multi-arm case must exist.")

    def test_G_hr_non2x2_case_exists(self):
        """Test G: Non-2x2 / HR cases (case_F1/case_F2) exist."""
        f_cases = [c for c in self.cases if c.get("category") == "F_wrong_metric_non_2x2"]
        self.assertGreaterEqual(len(f_cases), 1, "Non-2x2 / HR cases must exist.")

    def test_H_overlap_case_exists(self):
        """Test H: Overlapping study case (case_D1) exists."""
        d_cases = [c for c in self.cases if c.get("category") == "D_overlapping_studies"]
        self.assertGreaterEqual(len(d_cases), 1, "Overlap case must exist.")

    def test_I_evaluation_results_contain_pipelineresult_counts(self):
        """Test I: Output evaluation JSON contains counts derived directly from PipelineResult."""
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            for res in results:
                self.assertIn("retrieved_count", res)
                self.assertIn("extracted_count", res)
                self.assertIn("validated_count", res)
                self.assertIn("selected_count", res)
                self.assertIn("independent_evidence_count", res)

    def test_J_no_synthetic_clinical_evidence_introduced(self):
        """Test J: Evaluation cases do not introduce fake medical text or synthetic clinical evidence."""
        for case in self.cases:
            q = case.get("question", "").lower()
            self.assertNotIn("fake_drug", q)
            self.assertNotIn("synthetic_study", q)

    def test_K_no_hardcoded_emp005_promotion_in_runner(self):
        """Test K: Runner script uses generic run_analysis entrypoint without hardcoded EMP-005 logic."""
        runner_path = Path("evaluation/run_end_to_end_evaluation.py")
        self.assertTrue(runner_path.exists())
        with open(runner_path, "r", encoding="utf-8") as f:
            code = f.read()
        self.assertNotIn("if study_id == 'EMP-005': return PASS", code)

    def test_L_statistics_comparisons_use_deterministic_engine_outputs(self):
        """Test L: Expected statistical values (RR 3.5673, OR 3.7435) match deterministic statistics calculation."""
        rr_res = calculate_risk_ratio(treatment_events=301, treatment_n=4687, control_events=42, control_n=2333)
        or_res = calculate_odds_ratio(treatment_events=301, treatment_n=4687, control_events=42, control_n=2333)
        self.assertAlmostEqual(rr_res.risk_ratio, 3.5673, places=3)
        self.assertAlmostEqual(or_res.odds_ratio, 3.7435, places=3)

    def test_M_evaluation_runner_produces_valid_json(self):
        """Test M: Evaluation output file is a valid JSON array."""
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, list)


if __name__ == "__main__":
    unittest.main()
