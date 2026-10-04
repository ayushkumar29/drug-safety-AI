"""
Unit test suite for Evidence-to-Statistics Orchestrator.
"""

import unittest
from llm.evidence import ExtractedEvidence
from statistics.evidence_statistics import analyze_extracted_evidence, EvidenceStatisticalAnalysisResult


class TestEvidenceStatisticsOrchestrator(unittest.TestCase):

    def setUp(self):
        self.valid_base_kwargs = {
            "study_id": "EMP-001",
            "drug": "empagliflozin",
            "comparator": "placebo",
            "outcome": "genital infection",
            "treatment_n": 4221,
            "treatment_events": 259,
            "control_n": 4203,
            "control_events": 67,
            "extracted_metric_type": "event_count",
            "source_file": "EMP-001.pdf",
            "chunk_id": "EMP-001_CHK_004",
            "page_start": 7,
            "page_end": 10,
            "evidence_quote": "Events consistent with genital infection 259 67"
        }

    def test_A_valid_event_count_evidence(self):
        """Test A: Valid event-count evidence executes all three engines."""
        ev = ExtractedEvidence(**self.valid_base_kwargs)
        res = analyze_extracted_evidence(ev)

        self.assertTrue(res.statistical_eligibility)
        self.assertIsNone(res.ineligibility_reason)
        self.assertIsNotNone(res.risk_ratio_result)
        self.assertIsNotNone(res.odds_ratio_result)
        self.assertIsNotNone(res.hypothesis_test_result)
        self.assertAlmostEqual(res.risk_ratio_result.risk_ratio, 3.8492, places=3)
        self.assertAlmostEqual(res.odds_ratio_result.odds_ratio, 4.0354, places=3)
        self.assertLess(res.hypothesis_test_result.fisher_p_value, 1e-10)

    def test_B_incidence_rate_evidence_rejected(self):
        """Test B: Incidence-rate evidence is rejected for 2x2 analysis."""
        kwargs = self.valid_base_kwargs.copy()
        kwargs["extracted_metric_type"] = "incidence_rate"
        ev = ExtractedEvidence(**kwargs)
        res = analyze_extracted_evidence(ev)

        self.assertFalse(res.statistical_eligibility)
        self.assertIn("is not an event-count", res.ineligibility_reason)
        self.assertIsNone(res.risk_ratio_result)

    def test_C_percentage_only_evidence_rejected(self):
        """Test C: Percentage-only evidence is rejected."""
        kwargs = self.valid_base_kwargs.copy()
        kwargs["extracted_metric_type"] = "percentage"
        ev = ExtractedEvidence(**kwargs)
        res = analyze_extracted_evidence(ev)

        self.assertFalse(res.statistical_eligibility)
        self.assertIn("is not an event-count", res.ineligibility_reason)

    def test_D_hazard_ratio_evidence_rejected(self):
        """Test D: Hazard-ratio evidence is rejected."""
        kwargs = self.valid_base_kwargs.copy()
        kwargs["extracted_metric_type"] = "hazard_ratio"
        ev = ExtractedEvidence(**kwargs)
        res = analyze_extracted_evidence(ev)

        self.assertFalse(res.statistical_eligibility)
        self.assertIn("is not an event-count", res.ineligibility_reason)

    def test_E_missing_event_count_rejected(self):
        """Test E: Missing event count is rejected."""
        kwargs = self.valid_base_kwargs.copy()
        kwargs["treatment_events"] = None
        ev = ExtractedEvidence(**kwargs)
        res = analyze_extracted_evidence(ev)

        self.assertFalse(res.statistical_eligibility)
        self.assertIn("Missing one or more required", res.ineligibility_reason)

    def test_F_invalid_events_greater_than_n_rejected(self):
        """Test F: Invalid events > N is rejected by orchestrator."""
        kwargs = self.valid_base_kwargs.copy()
        kwargs["treatment_events"] = 5000
        # ExtractedEvidence model_validator raises Pydantic ValidationError when constructed directly
        # Orchestrator also double-checks bounds explicitly
        try:
            ev = ExtractedEvidence(**kwargs)
            res = analyze_extracted_evidence(ev)
            self.assertFalse(res.statistical_eligibility)
            self.assertIn("exceeds sample size", res.ineligibility_reason)
        except Exception as err:
            self.assertIn("cannot exceed", str(err))


    def test_G_provenance_fields_preserved(self):
        """Test G: Provenance metadata is preserved in result."""
        ev = ExtractedEvidence(**self.valid_base_kwargs)
        res = analyze_extracted_evidence(ev)

        self.assertEqual(res.study_id, "EMP-001")
        self.assertEqual(res.source_file, "EMP-001.pdf")
        self.assertEqual(res.chunk_id, "EMP-001_CHK_004")
        self.assertEqual(res.page_start, 7)
        self.assertEqual(res.page_end, 10)
        self.assertEqual(res.evidence_quote, "Events consistent with genital infection 259 67")

    def test_H_exact_raw_counts_passed_to_all_engines(self):
        """Test H: Verify identical raw counts reach all three engines."""
        ev = ExtractedEvidence(**self.valid_base_kwargs)
        res = analyze_extracted_evidence(ev)

        rr = res.risk_ratio_result
        o_r = res.odds_ratio_result
        hyp = res.hypothesis_test_result

        self.assertEqual(rr.treatment_events, 259)
        self.assertEqual(rr.treatment_n, 4221)
        self.assertEqual(rr.control_events, 67)
        self.assertEqual(rr.control_n, 4203)

        self.assertEqual(o_r.treatment_events, 259)
        self.assertEqual(o_r.treatment_n, 4221)
        self.assertEqual(o_r.control_events, 67)
        self.assertEqual(o_r.control_n, 4203)

        self.assertEqual(hyp.treatment_events, 259)
        self.assertEqual(hyp.treatment_n, 4221)
        self.assertEqual(hyp.control_events, 67)
        self.assertEqual(hyp.control_n, 4203)


if __name__ == "__main__":
    unittest.main()
