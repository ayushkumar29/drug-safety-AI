"""
Unit Test Suite for Evidence Extractor (llm/test_evidence_extractor.py).
Tests context window preparation, quote verification, rate/percentage disambiguation,
and 2x2 event-count schema validation.
"""

import unittest
import os

try:
    from llm.evidence import ExtractedEvidence
    from llm.evidence_extractor import (
        verify_quote_in_chunk,
        extract_evidence_from_chunk,
        prepare_chunk_context
    )
    from llm.config import LLMConfig
    from validation.evidence_validator import validate_and_sanitize_evidence
except ImportError:
    from evidence import ExtractedEvidence
    from evidence_extractor import (
        verify_quote_in_chunk,
        extract_evidence_from_chunk,
        prepare_chunk_context
    )
    from config import LLMConfig
    from evidence_validator import validate_and_sanitize_evidence


STATIC_MOCK_CHUNK = {
    "chunk_id": "EMP-001_CHK_008",
    "study_id": "EMP-001",
    "source_file": "EMP-001.pdf",
    "page_start": 16,
    "page_end": 17,
    "text": "Among patients receiving empagliflozin, genital infections were reported in 4.2% of patients compared with 1.5% receiving placebo."
}

EMP_001_CHK_004_TEXT = (
    "Placebo (N 5 4203) Empagliflozin 10 mg (N 5 4221) Empagliflozin 25 mg (N 5 4196) "
    "Events consistent with urinary tract infection 629 15.0 9.5 639 15.1 9.2 607 14.5 8.7 "
    + (" " * 900) +
    "Events consistent with genital infection 67 1.6 0.9 259 6.1 3.5 251 6.0 3.4"
)


class TestEvidenceExtractor(unittest.TestCase):

    def test_quote_verification(self):
        """Test deterministic quote verification logic."""
        chunk_text = STATIC_MOCK_CHUNK["text"]
        exact_quote = "genital infections were reported in 4.2% of patients"
        self.assertTrue(verify_quote_in_chunk(exact_quote, chunk_text))

        fuzzy_quote = "GENITAL  INFECTIONS WERE REPORTED   IN 4.2% OF PATIENTS"
        self.assertTrue(verify_quote_in_chunk(fuzzy_quote, chunk_text))

        fabricated_quote = "empagliflozin caused significant severe kidney failure"
        self.assertFalse(verify_quote_in_chunk(fabricated_quote, chunk_text))

    def test_A_context_after_character_800(self):
        """Test A: Evidence occurring past character 800 is preserved in prepared context."""
        hyp = {"drug": "empagliflozin", "comparator": "placebo", "outcome": "genital infection"}
        ctx = prepare_chunk_context(EMP_001_CHK_004_TEXT, hyp, max_chars=4000)

        self.assertIn("genital infection", ctx)
        self.assertIn("259", ctx)
        self.assertIn("67", ctx)

    def test_B_EMP_001_CHK_004_table_exposure(self):
        """Test B: EMP-001_CHK_004 exposes genital infection table to extractor."""
        hyp = {"drug": "empagliflozin", "comparator": "placebo", "outcome": "genital infection"}
        ctx = prepare_chunk_context(EMP_001_CHK_004_TEXT, hyp, max_chars=4000)

        self.assertIn("Placebo (N 5 4203)", ctx)
        self.assertIn("Empagliflozin 10 mg (N 5 4221)", ctx)
        self.assertIn("Events consistent with genital infection", ctx)

    def test_C_rate_and_percentage_not_interpreted_as_counts(self):
        """Test C: Rates and percentages (1.6%, 0.9 per 100 patient-years) are sanitized."""
        quote_rate = "Events consistent with genital infection 67 1.6 0.9 259 6.1 3.5 per 100 patient-years"

        ev = ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            evidence_quote=quote_rate,
            treatment_events=259,
            control_events=67
        )

        sanitized = validate_and_sanitize_evidence(ev)

        self.assertIn(sanitized.extracted_metric_type, ["incidence_rate", "percentage", "hazard_ratio"])
        self.assertIsNone(sanitized.treatment_events)
        self.assertIsNone(sanitized.control_events)

    def test_D_correct_event_count_schema_representation(self):
        """Test D: Valid 2x2 event-count representation for Empagliflozin 10mg vs Placebo."""
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
            evidence_quote="Empagliflozin 10mg (259/4221) vs placebo (67/4203)."
        )

        sanitized = validate_and_sanitize_evidence(ev)

        self.assertEqual(sanitized.treatment_n, 4221)
        self.assertEqual(sanitized.treatment_events, 259)
        self.assertEqual(sanitized.control_n, 4203)
        self.assertEqual(sanitized.control_events, 67)
        self.assertEqual(sanitized.extracted_metric_type, "event_count")

    def test_E_existing_evidence_validation_rules(self):
        """Test E: Existing validation rules (events <= N) remain enforced."""
        invalid_ev = ExtractedEvidence.model_construct(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=100,
            treatment_events=500,
            control_n=100,
            control_events=10,
            evidence_quote="Empagliflozin 500 out of 100",
            extracted_metric_type="event_count"
        )

        with self.assertRaises(ValueError):
            validate_and_sanitize_evidence(invalid_ev)

    def test_candidate_selection_and_max_calls_limit(self):
        """Test Step 5A, D, E: EMP-001_CHK_004 is selected, unrelated chunks filtered, max 3 candidates."""
        from llm.evidence_extractor import select_candidate_chunks

        hyp = {"drug": "empagliflozin", "comparator": "placebo", "outcome": "genital infection"}
        
        chunks = [
            {"chunk_id": "EMP-001_CHK_004", "text": EMP_001_CHK_004_TEXT, "combined_score": 0.7},
            {"chunk_id": "UNRELATED_1", "text": "This chunk discusses blood pressure demographics only.", "combined_score": 0.1},
            {"chunk_id": "EMP-001_CHK_008", "text": "Incidence of genital infection was higher in empagliflozin vs placebo.", "combined_score": 0.8},
            {"chunk_id": "UNRELATED_2", "text": "Pharmacokinetics study in healthy volunteers.", "combined_score": 0.0},
            {"chunk_id": "EMP-003_CHK_001", "text": "Events consistent with genital infection occurred in 3.54 vs 0.95/100 PY.", "combined_score": 0.75}
        ]

        candidates = select_candidate_chunks(chunks, hyp, max_candidates=3)
        candidate_ids = [c["chunk_id"] for c in candidates]

        # A. EMP-001_CHK_004 is selected
        self.assertIn("EMP-001_CHK_004", candidate_ids)

        # D. Unrelated chunks are not selected
        self.assertNotIn("UNRELATED_1", candidate_ids)
        self.assertNotIn("UNRELATED_2", candidate_ids)

        # E. Maximum candidates is capped at 3
        self.assertLessEqual(len(candidates), 3)

    def test_focused_context_preserves_rates_and_counts(self):
        """Test Step 5B, C: Focused context window contains table row and preserves rate/percentage columns."""
        hyp = {"drug": "empagliflozin", "comparator": "placebo", "outcome": "genital infection"}
        ctx = prepare_chunk_context(EMP_001_CHK_004_TEXT, hyp, max_chars=1500)

        # B. Contains table row
        self.assertIn("Events consistent with genital infection", ctx)
        self.assertIn("67", ctx)
        self.assertIn("259", ctx)

        # C. Percentages/rates remain visible for disambiguation
        self.assertIn("1.6", ctx)
        self.assertIn("6.1", ctx)
        self.assertIn("3.5", ctx)


if __name__ == "__main__":
    unittest.main()
