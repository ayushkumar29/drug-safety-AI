"""
Unit test suite for Deterministic Evidence Analysis Report Builder (report/report_builder.py).
"""

import json
import unittest
from llm.hypothesis import StructuredHypothesis
from llm.evidence import ExtractedEvidence
from analysis.study_comparison import normalize_study_evidence
from analysis.conflict_analysis import analyze_multi_study_conflicts
from report.report_builder import (
    build_evidence_analysis_report,
    EvidenceAnalysisReport
)


class TestReportBuilder(unittest.TestCase):

    def setUp(self):
        self.hypothesis = StructuredHypothesis(
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            population="patients with type 2 diabetes",
            study_type="randomized controlled trial",
            hypothesis_direction="greater_risk",
            null_hypothesis="Empagliflozin does not increase the risk of genital infection compared with placebo.",
            alternative_hypothesis="Empagliflozin increases the risk of genital infection compared with placebo."
        )

        self.base_evidence_kwargs = {
            "study_id": "EMP-001",
            "drug": "empagliflozin",
            "comparator": "placebo",
            "outcome": "genital infection",
            "treatment_n": 4221,
            "treatment_events": 259,
            "control_n": 4203,
            "control_events": 67,
            "extracted_metric_type": "event_count",
            "dose": "10 mg",
            "source_file": "EMP-001.pdf",
            "chunk_id": "EMP-001_CHK_004",
            "page_start": 7,
            "page_end": 10,
            "evidence_quote": "Events consistent with genital infection 259 67"
        }

    def test_A_single_event_count_study(self):
        """Test A: Single 2x2 study preserves study, counts, RR/OR/p-values, and citation."""
        ev = ExtractedEvidence(**self.base_evidence_kwargs)
        comp = normalize_study_evidence([ev])
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report(
            "Does empagliflozin increase risk?",
            self.hypothesis,
            comp,
            conf
        )

        self.assertEqual(len(report.study_results), 1)
        sr = report.study_results[0]
        self.assertEqual(sr.study_id, "EMP-001")
        self.assertEqual(sr.treatment_n, 4221)
        self.assertEqual(sr.treatment_events, 259)
        self.assertEqual(sr.control_n, 4203)
        self.assertEqual(sr.control_events, 67)
        self.assertAlmostEqual(sr.risk_ratio, 3.8492, places=3)
        self.assertAlmostEqual(sr.odds_ratio, 4.0354, places=3)
        self.assertLess(sr.fisher_p_value, 1e-10)

        self.assertEqual(len(report.citations), 1)
        self.assertEqual(report.citations[0].chunk_id, "EMP-001_CHK_004")

    def test_B_multiple_studies_no_pooling(self):
        """Test B: Multiple studies remain separate in report with correct summary counts."""
        k1 = self.base_evidence_kwargs.copy()
        k2 = self.base_evidence_kwargs.copy()
        k2["study_id"] = "EMP-005"
        k2["chunk_id"] = "EMP-005_CHK_003"
        k2["treatment_n"] = 2000

        evs = [ExtractedEvidence(**k1), ExtractedEvidence(**k2)]
        comp = normalize_study_evidence(evs)
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)

        self.assertEqual(len(report.study_results), 2)
        self.assertEqual(report.evidence_summary.total_studies, 2)
        self.assertEqual(report.study_results[0].treatment_n, 4221)
        self.assertEqual(report.study_results[1].treatment_n, 2000)

    def test_C_mixed_statistical_representations(self):
        """Test C: Mixed representations produce correct summary counts."""
        k1 = self.base_evidence_kwargs.copy()  # event_count
        k2 = self.base_evidence_kwargs.copy()
        k2["study_id"] = "EMP-002"
        k2["extracted_metric_type"] = "incidence_rate"
        k2["treatment_events"] = None

        k3 = self.base_evidence_kwargs.copy()
        k3["study_id"] = "EMP-003"
        k3["extracted_metric_type"] = "hazard_ratio"
        k3["treatment_events"] = None

        evs = [ExtractedEvidence(**k1), ExtractedEvidence(**k2), ExtractedEvidence(**k3)]
        comp = normalize_study_evidence(evs)
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)

        sum_res = report.evidence_summary
        self.assertEqual(sum_res.statistical_event_count_records, 1)
        self.assertEqual(sum_res.incidence_rate_records, 1)
        self.assertEqual(sum_res.hazard_ratio_records, 1)

    def test_D_missing_statistics(self):
        """Test D: Records without usable statistics increment records_without_usable_statistics count."""
        k1 = self.base_evidence_kwargs.copy()
        k1["treatment_events"] = None  # Missing count

        evs = [ExtractedEvidence(**k1)]
        comp = normalize_study_evidence(evs)
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)

        self.assertEqual(report.evidence_summary.records_without_usable_statistics, 1)

    def test_E_conflict_observations_preserved(self):
        """Test E: Conflict observations are preserved without clinical claims."""
        k1 = self.base_evidence_kwargs.copy()  # RR ~ 3.85
        k2 = self.base_evidence_kwargs.copy()  # RR ~ 0.74 (lower_effect)
        k2["study_id"] = "EMP-003"
        k2["treatment_events"] = 50

        evs = [ExtractedEvidence(**k1), ExtractedEvidence(**k2)]
        comp = normalize_study_evidence(evs)
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)

        c_key = list(report.cross_study_observations.keys())[0]
        obs = report.cross_study_observations[c_key]
        self.assertTrue(obs.directional_conflict)

    def test_F_limitations_triggered_only_when_applicable(self):
        """Test F: Methodological limitations are dynamically derived from dataset state."""
        k1 = self.base_evidence_kwargs.copy()
        k1["extracted_metric_type"] = "incidence_rate"
        k1["treatment_events"] = None

        evs = [ExtractedEvidence(**k1)]
        comp = normalize_study_evidence(evs)
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)

        lim_str = " ".join(report.limitations)
        self.assertIn("Incidence-rate evidence was preserved separately", lim_str)
        self.assertIn("did not contain sufficient quantitative data", lim_str)

    def test_G_citation_preservation(self):
        """Test G: Citation fields preserve all metadata."""
        ev = ExtractedEvidence(**self.base_evidence_kwargs)
        comp = normalize_study_evidence([ev])
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)

        cit = report.citations[0]
        self.assertEqual(cit.study_id, "EMP-001")
        self.assertEqual(cit.source_file, "EMP-001.pdf")
        self.assertEqual(cit.chunk_id, "EMP-001_CHK_004")
        self.assertEqual(cit.page_start, 7)
        self.assertEqual(cit.page_end, 10)
        self.assertEqual(cit.evidence_quote, "Events consistent with genital infection 259 67")

    def test_H_missing_provenance_not_fabricated(self):
        """Test H: Missing citation fields remain None and are not fabricated."""
        k1 = self.base_evidence_kwargs.copy()
        k1["source_file"] = None
        k1["chunk_id"] = None

        ev = ExtractedEvidence(**k1)
        comp = normalize_study_evidence([ev])
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)

        cit = report.citations[0]
        self.assertIsNone(cit.source_file)
        self.assertIsNone(cit.chunk_id)

    def test_I_hypothesis_preservation(self):
        """Test I: StructuredHypothesis fields preserved exactly."""
        ev = ExtractedEvidence(**self.base_evidence_kwargs)
        comp = normalize_study_evidence([ev])
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)

        h = report.hypothesis
        self.assertEqual(h.drug, "empagliflozin")
        self.assertEqual(h.comparator, "placebo")
        self.assertEqual(h.outcome, "genital infection")
        self.assertEqual(h.hypothesis_direction, "greater_risk")

    def test_J_no_clinical_conclusions(self):
        """Test J: JSON payload does not contain forbidden subjective/clinical classification keys."""
        ev = ExtractedEvidence(**self.base_evidence_kwargs)
        comp = normalize_study_evidence([ev])
        conf = analyze_multi_study_conflicts(comp)

        report = build_evidence_analysis_report("Question", self.hypothesis, comp, conf)
        report_json = report.model_dump_json()

        # Check structural dictionary keys and report text (ignoring potential verbatim quote text)
        forbidden_keys = ["drug_is_safe", "drug_is_unsafe", "approval_status", "rejection_status"]
        for key in forbidden_keys:
            self.assertNotIn(key, report_json)


if __name__ == "__main__":
    unittest.main()
