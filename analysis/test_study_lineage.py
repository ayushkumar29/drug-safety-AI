"""
Unit tests for Task 15 Study Lineage & Provenance Model.
Verifies lineage metadata resolution, independence checking, double-counting protection, and ExtractedEvidence safety.
"""

import unittest
from analysis.study_lineage import StudyLineage, EvidenceLineage
from analysis.lineage_resolver import (
    get_study_lineage,
    attach_evidence_lineage,
    are_independent,
    validate_study_independence,
    load_lineage_registry
)
from llm.evidence import ExtractedEvidence


class TestStudyLineage(unittest.TestCase):

    def setUp(self):
        # Ensure registry is loaded
        load_lineage_registry()

    def test_A_emp005_is_independent_primary(self):
        lineage = get_study_lineage("EMP-005")
        self.assertEqual(lineage.publication_id, "EMP-005")
        self.assertTrue(lineage.primary_study)
        self.assertEqual(lineage.independence_status, "independent_primary")
        self.assertIn("EMPA-REG_OUTCOME", lineage.underlying_study_ids)

    def test_B_emp001_is_pooled_analysis(self):
        lineage = get_study_lineage("EMP-001")
        self.assertEqual(lineage.publication_id, "EMP-001")
        self.assertFalse(lineage.primary_study)
        self.assertEqual(lineage.independence_status, "pooled_analysis")

    def test_C_emp003_is_pooled_analysis(self):
        lineage = get_study_lineage("EMP-003")
        self.assertEqual(lineage.publication_id, "EMP-003")
        self.assertFalse(lineage.primary_study)
        self.assertEqual(lineage.independence_status, "pooled_analysis")

    def test_D_emp002_is_meta_analysis(self):
        lineage = get_study_lineage("EMP-002")
        self.assertEqual(lineage.publication_id, "EMP-002")
        self.assertFalse(lineage.primary_study)
        self.assertEqual(lineage.independence_status, "meta_analysis")

    def test_E_emp004_is_review(self):
        lineage = get_study_lineage("EMP-004")
        self.assertEqual(lineage.publication_id, "EMP-004")
        self.assertFalse(lineage.primary_study)
        self.assertEqual(lineage.independence_status, "review")

    def test_F_emp001_and_emp005_not_independent(self):
        lineage_1 = get_study_lineage("EMP-001")
        lineage_5 = get_study_lineage("EMP-005")
        res = are_independent(lineage_1, lineage_5)
        self.assertFalse(res["independent"])
        self.assertIn("EMPA-REG_OUTCOME", res["shared_underlying_studies"])

    def test_G_emp003_and_emp005_not_independent(self):
        lineage_3 = get_study_lineage("EMP-003")
        lineage_5 = get_study_lineage("EMP-005")
        res = are_independent(lineage_3, lineage_5)
        self.assertFalse(res["independent"])
        self.assertIn("EMPA-REG_OUTCOME", res["shared_underlying_studies"])

    def test_H_different_publication_ids_alone_do_not_imply_independence(self):
        lineage_1 = get_study_lineage("EMP-001")
        lineage_2 = get_study_lineage("EMP-002")
        res = are_independent(lineage_1, lineage_2)
        self.assertFalse(res["independent"])

    def test_I_unknown_lineage_handled_safely(self):
        lineage_unk = get_study_lineage("UNKNOWN-999")
        lineage_5 = get_study_lineage("EMP-005")
        res = are_independent(lineage_unk, lineage_5)
        self.assertFalse(res["independent"])
        self.assertEqual(lineage_unk.independence_status, "unknown")

    def test_J_evidence_lineage_preserves_ids(self):
        ev = ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=4221,
            treatment_events=259,
            control_n=4203,
            control_events=67,
            extracted_metric_type="event_count",
            evidence_quote="Events consistent with genital infection 67 vs 259"
        )
        ev_lineage = attach_evidence_lineage(ev)
        self.assertEqual(ev_lineage.publication_id, "EMP-001")
        self.assertEqual(ev_lineage.independence_status, "pooled_analysis")
        self.assertIn("EMPA-REG_OUTCOME", ev_lineage.underlying_study_ids)

    def test_K_extracted_evidence_statistical_fields_unchanged(self):
        ev = ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=4221,
            treatment_events=259,
            control_n=4203,
            control_events=67,
            extracted_metric_type="event_count",
            evidence_quote="Events consistent with genital infection 67 vs 259"
        )
        attach_evidence_lineage(ev)
        self.assertEqual(ev.treatment_n, 4221)
        self.assertEqual(ev.treatment_events, 259)
        self.assertEqual(ev.control_n, 4203)
        self.assertEqual(ev.control_events, 67)
        self.assertEqual(ev.extracted_metric_type, "event_count")

    def test_L_validate_study_independence_detects_double_counting(self):
        all_lineages = [get_study_lineage(f"EMP-00{i}") for i in range(1, 6)]
        overlapping = validate_study_independence(all_lineages)
        self.assertTrue(len(overlapping) > 0)
        pair_keys = [tuple(sorted(p["pair"])) for p in overlapping]
        self.assertIn(("EMP-001", "EMP-005"), pair_keys)
        self.assertIn(("EMP-003", "EMP-005"), pair_keys)


if __name__ == "__main__":
    unittest.main()
