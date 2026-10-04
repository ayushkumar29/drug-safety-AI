"""
Unit Test Suite for Evaluation Summary Framework (evaluation/test_evaluation_summary.py)

Tests:
A. Summary JSON exists after generation.
B. JSON is valid.
C. Required top-level sections exist.
D. Every metric has a source reference.
E. Retrieval metric uses Precision@5 terminology.
F. No 'RAG accuracy = 95%' style claim exists.
G. Golden-path values match the deterministic statistical engine.
H. End-to-end counts match the actual benchmark artifact.
I. No synthetic evidence flag is introduced.
J. Safety boundary flags are consistent with existing tests/artifacts.
K. Summary does not invent unavailable metrics.
"""

import sys
import os
import json
import unittest
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from evaluation.build_evaluation_summary import build_summary, SUMMARY_JSON, SUMMARY_MD
from statistics.risk_ratio import calculate_risk_ratio
from statistics.odds_ratio import calculate_odds_ratio


class TestEvaluationSummaryFramework(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        build_summary()
        with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
            cls.summary = json.load(f)
        with open(SUMMARY_MD, "r", encoding="utf-8") as f:
            cls.summary_md = f.read()

    def test_A_summary_json_exists(self):
        """Test A: Summary JSON file exists on disk."""
        self.assertTrue(SUMMARY_JSON.exists(), f"Summary JSON '{SUMMARY_JSON}' must exist.")

    def test_B_json_is_valid(self):
        """Test B: Summary JSON is valid and parsable dict."""
        self.assertIsInstance(self.summary, dict)

    def test_C_required_top_level_sections_exist(self):
        """Test C: Required top-level sections exist."""
        required = ["project", "evaluation_version", "corpus", "retrieval", "quantitative_evidence", "end_to_end", "safety_boundaries", "golden_path", "limitations"]
        for section in required:
            self.assertIn(section, self.summary, f"Missing required top-level section '{section}'.")

    def test_D_every_metric_has_source_reference(self):
        """Test D: Every metric dictionary contains a 'source' key."""
        sections = ["corpus", "retrieval", "quantitative_evidence", "end_to_end", "safety_boundaries", "golden_path"]
        for sec in sections:
            sec_dict = self.summary[sec]
            for metric_key, metric_val in sec_dict.items():
                self.assertIn("source", metric_val, f"Metric '{sec}.{metric_key}' missing source reference.")

    def test_E_retrieval_metric_uses_precision_at_5_terminology(self):
        """Test E: Retrieval metric uses Precision@5 terminology rather than misleading accuracy claims."""
        ret = self.summary["retrieval"]
        self.assertIn("average_precision_at_5", ret)
        self.assertIn("Precision@5", self.summary_md)

    def test_F_no_rag_accuracy_95_percent_claim(self):
        """Test F: No misleading 'RAG accuracy = 95%' or fake accuracy claims exist in summary JSON or MD."""
        json_str = json.dumps(self.summary).lower()
        self.assertNotIn("rag accuracy = 95%", json_str)
        self.assertNotIn("rag accuracy = 95%", self.summary_md.lower())
        self.assertNotIn("overall model accuracy", json_str)

    def test_G_golden_path_values_match_deterministic_engine(self):
        """Test G: Golden-path statistical values match deterministic engine calculations."""
        gp = self.summary["golden_path"]
        self.assertEqual(gp["study_id"]["value"], "EMP-005")
        self.assertEqual(gp["treatment_events"]["value"], 301)
        self.assertEqual(gp["treatment_n"]["value"], 4687)
        self.assertEqual(gp["control_events"]["value"], 42)
        self.assertEqual(gp["control_n"]["value"], 2333)

        rr_res = calculate_risk_ratio(301, 4687, 42, 2333)
        or_res = calculate_odds_ratio(301, 4687, 42, 2333)

        self.assertAlmostEqual(gp["risk_ratio"]["value"], round(rr_res.risk_ratio, 4), places=3)
        self.assertAlmostEqual(gp["odds_ratio"]["value"], round(or_res.odds_ratio, 4), places=3)

    def test_H_end_to_end_counts_match_benchmark_artifact(self):
        """Test H: End-to-end benchmark counts match data/processed/evaluation/end_to_end_evaluation.json."""
        e2e_path = Path("data/processed/evaluation/end_to_end_evaluation.json")
        if e2e_path.exists():
            with open(e2e_path, "r", encoding="utf-8") as f:
                e2e_data = json.load(f)
            self.assertEqual(self.summary["end_to_end"]["total_cases"]["value"], len(e2e_data))
            self.assertEqual(self.summary["end_to_end"]["passed"]["value"], sum(1 for c in e2e_data if c.get("passed")))

    def test_I_no_synthetic_evidence_flag_introduced(self):
        """Test I: synthetic_evidence_used flag is False."""
        sb = self.summary["safety_boundaries"]
        self.assertFalse(sb["synthetic_evidence_used"]["value"])

    def test_J_safety_boundary_flags_consistent(self):
        """Test J: Safety boundary flags are True for HR, rate, percentage block and overlap protection."""
        sb = self.summary["safety_boundaries"]
        self.assertTrue(sb["hr_blocked_from_2x2"]["value"])
        self.assertTrue(sb["rate_blocked_from_2x2"]["value"])
        self.assertTrue(sb["percentage_blocked_from_2x2"]["value"])
        self.assertTrue(sb["overlap_protection"]["value"])

    def test_K_summary_does_not_invent_unavailable_metrics(self):
        """Test K: Summary does not contain manufactured meta-analysis pooling metrics."""
        self.assertNotIn("meta_analysis_pooled_effect", self.summary)
        self.assertNotIn("cox_proportional_hazards", self.summary)


if __name__ == "__main__":
    unittest.main()
