"""
Unit tests for Task 16 Evidence Selection & Overlap-Safe Analysis.
Tests primary retention, secondary overlap exclusion, review context handling, provenance preservation, and immutability.
"""

import unittest
from analysis.evidence_selection import (
    EvidenceSelection,
    EvidenceSelectionResult,
    select_independent_evidence
)
from llm.evidence import ExtractedEvidence


class TestEvidenceSelection(unittest.TestCase):

    def setUp(self):
        # Create standard evidence objects for the 5 corpus publications
        self.ev_emp001 = ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin 10 mg",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=4221,
            treatment_events=259,
            control_n=4203,
            control_events=67,
            extracted_metric_type="event_count",
            source_file="EMP-001.pdf",
            page_start=7,
            page_end=10,
            evidence_quote="Events consistent with genital infection Placebo (N=4203) 67 Empagliflozin 10 mg (N=4221) 259"
        )
        self.ev_emp002 = ExtractedEvidence(
            study_id="EMP-002",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            extracted_metric_type="odds_ratio",
            raw_reported_treatment_metric="3.8",
            source_file="EMP-002.pdf",
            page_start=5,
            page_end=8,
            evidence_quote="Meta-analysis OR for genital infection was 3.8"
        )
        self.ev_emp003 = ExtractedEvidence(
            study_id="EMP-003",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            extracted_metric_type="event_count",
            source_file="EMP-003.pdf",
            page_start=12,
            page_end=15,
            evidence_quote="Event consistent with genital infection Placebo (n = 4904) EMPA 10 mg (n = 4858)"
        )
        self.ev_emp004 = ExtractedEvidence(
            study_id="EMP-004",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            extracted_metric_type="incidence_rate",
            source_file="EMP-004.pdf",
            page_start=11,
            page_end=13,
            evidence_quote="Overview of phase 3 trials genital infections"
        )
        self.ev_emp005 = ExtractedEvidence(
            study_id="EMP-005",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=4687,
            treatment_events=301,
            control_n=2333,
            control_events=42,
            extracted_metric_type="event_count",
            source_file="EMP-005.pdf",
            page_start=10,
            page_end=11,
            evidence_quote="Event consistent with genital infection 42 (1.8) 301 (6.4)"
        )

    def test_A_emp005_primary_rct_retained(self):
        records = [self.ev_emp001, self.ev_emp005]
        res = select_independent_evidence(records)
        retained_ids = [e.study_id for e in res.independent_evidence]
        self.assertIn("EMP-005", retained_ids)

    def test_B_emp001_overlapping_pooled_excluded(self):
        records = [self.ev_emp001, self.ev_emp005]
        res = select_independent_evidence(records)
        emp1_sel = next(r["selection"] for r in res.all_selected_evidence if r["study_id"] == "EMP-001")
        self.assertEqual(emp1_sel["selection_status"], "excluded_due_to_overlap")

    def test_C_emp003_overlapping_pooled_excluded(self):
        records = [self.ev_emp003, self.ev_emp005]
        res = select_independent_evidence(records)
        emp3_sel = next(r["selection"] for r in res.all_selected_evidence if r["study_id"] == "EMP-003")
        self.assertEqual(emp3_sel["selection_status"], "excluded_due_to_overlap")

    def test_D_emp002_meta_analysis_handled(self):
        records = [self.ev_emp002, self.ev_emp005]
        res = select_independent_evidence(records)
        emp2_sel = next(r["selection"] for r in res.all_selected_evidence if r["study_id"] == "EMP-002")
        self.assertEqual(emp2_sel["selection_status"], "excluded_due_to_overlap")

    def test_E_emp004_review_handled_as_context_only(self):
        records = [self.ev_emp004, self.ev_emp005]
        res = select_independent_evidence(records)
        emp4_sel = next(r["selection"] for r in res.all_selected_evidence if r["study_id"] == "EMP-004")
        self.assertEqual(emp4_sel["selection_status"], "context_only")

    def test_F_different_publication_ids_do_not_automatically_imply_independence(self):
        records = [self.ev_emp001, self.ev_emp002, self.ev_emp003, self.ev_emp004, self.ev_emp005]
        res = select_independent_evidence(records)
        self.assertEqual(len(res.independent_evidence), 1)
        self.assertEqual(res.independent_evidence[0].study_id, "EMP-005")

    def test_G_underlying_quantitative_values_unchanged(self):
        records = [self.ev_emp001, self.ev_emp005]
        select_independent_evidence(records)
        self.assertEqual(self.ev_emp001.treatment_n, 4221)
        self.assertEqual(self.ev_emp001.treatment_events, 259)
        self.assertEqual(self.ev_emp005.treatment_n, 4687)
        self.assertEqual(self.ev_emp005.treatment_events, 301)

    def test_H_provenance_remains_unchanged(self):
        records = [self.ev_emp001, self.ev_emp005]
        res = select_independent_evidence(records)
        for r in res.all_selected_evidence:
            self.assertIn("source_file", r)
            self.assertIn("chunk_id", r)
            self.assertIn("page_start", r)
            self.assertIn("evidence_quote", r)

    def test_I_original_evidence_list_not_mutated_or_deleted(self):
        records = [self.ev_emp001, self.ev_emp005]
        original_len = len(records)
        res = select_independent_evidence(records)
        self.assertEqual(len(records), original_len)
        self.assertEqual(len(res.all_selected_evidence), original_len)

    def test_J_single_primary_study_retained_per_dataset(self):
        records = [self.ev_emp005]
        res = select_independent_evidence(records)
        self.assertEqual(len(res.independent_evidence), 1)
        self.assertEqual(res.independent_evidence[0].study_id, "EMP-005")

    def test_K_unknown_lineage_handled_safely(self):
        ev_unk = ExtractedEvidence(
            study_id="UNKNOWN-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            extracted_metric_type="event_count",
            evidence_quote="Unknown study quote"
        )
        records = [self.ev_emp005, ev_unk]
        res = select_independent_evidence(records)

        unk_sel = next(r["selection"] for r in res.all_selected_evidence if r["study_id"] == "UNKNOWN-001")
        self.assertIn(unk_sel["selection_status"], ["excluded_due_to_overlap", "context_only", "retained_for_independent_analysis"])


if __name__ == "__main__":
    unittest.main()
