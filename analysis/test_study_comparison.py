"""
Unit test suite for Multi-Study Evidence Normalizer (analysis/study_comparison.py).
"""

import unittest
from llm.evidence import ExtractedEvidence
from analysis.study_comparison import (
    normalize_study_evidence,
    normalize_comparison_key,
    map_statistical_representation,
    StudyComparisonResult,
    NormalizedStudyEvidence
)


class TestMultiStudyEvidenceNormalizer(unittest.TestCase):

    def setUp(self):
        self.base_kwargs = {
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

    def test_A_one_valid_event_count_study(self):
        """Test A: Single valid event-count study attaches RR, OR, and Fisher/Chi-square statistics."""
        ev = ExtractedEvidence(**self.base_kwargs)
        res: StudyComparisonResult = normalize_study_evidence([ev])

        self.assertEqual(res.total_input_records, 1)
        self.assertEqual(res.unique_evidence_records, 1)
        rec = res.study_records[0]
        self.assertEqual(rec.statistical_representation, "2x2_event_count")
        self.assertTrue(rec.statistical_eligibility)
        self.assertIsNotNone(rec.risk_ratio_result)
        self.assertIsNotNone(rec.odds_ratio_result)
        self.assertIsNotNone(rec.hypothesis_test_result)
        self.assertAlmostEqual(rec.risk_ratio_result.risk_ratio, 3.8492, places=3)

    def test_B_two_independent_studies_same_comparison(self):
        """Test B: Two independent studies for same drug/comparator/outcome remain separate records with identical comparison_key."""
        kwargs1 = self.base_kwargs.copy()
        kwargs2 = self.base_kwargs.copy()
        kwargs2["study_id"] = "EMP-003"
        kwargs2["chunk_id"] = "EMP-003_CHK_001"
        kwargs2["treatment_n"] = 2000
        kwargs2["treatment_events"] = 100

        ev1 = ExtractedEvidence(**kwargs1)
        ev2 = ExtractedEvidence(**kwargs2)

        res = normalize_study_evidence([ev1, ev2])

        self.assertEqual(res.total_input_records, 2)
        self.assertEqual(res.unique_evidence_records, 2)
        self.assertEqual(len(res.study_records), 2)
        self.assertEqual(res.study_records[0].comparison_key, res.study_records[1].comparison_key)
        self.assertEqual(len(res.comparison_groups), 1)
        self.assertEqual(res.comparison_groups[res.study_records[0].comparison_key].record_count, 2)

    def test_C_different_outcomes_produce_different_keys(self):
        """Test C: Different outcomes produce distinct comparison_key values."""
        kwargs1 = self.base_kwargs.copy()
        kwargs2 = self.base_kwargs.copy()
        kwargs2["outcome"] = "urinary tract infection"

        ev1 = ExtractedEvidence(**kwargs1)
        ev2 = ExtractedEvidence(**kwargs2)

        res = normalize_study_evidence([ev1, ev2])

        self.assertEqual(len(res.comparison_groups), 2)
        key1 = res.study_records[0].comparison_key
        key2 = res.study_records[1].comparison_key
        self.assertNotEqual(key1, key2)

    def test_D_different_doses_remain_separate_records(self):
        """Test D: Different doses remain distinct records and are not merged."""
        kwargs1 = self.base_kwargs.copy()
        kwargs1["dose"] = "10 mg"
        kwargs2 = self.base_kwargs.copy()
        kwargs2["dose"] = "25 mg"
        kwargs2["treatment_events"] = 251

        ev1 = ExtractedEvidence(**kwargs1)
        ev2 = ExtractedEvidence(**kwargs2)

        res = normalize_study_evidence([ev1, ev2])

        self.assertEqual(res.unique_evidence_records, 2)
        self.assertEqual(res.study_records[0].dose, "10 mg")
        self.assertEqual(res.study_records[1].dose, "25 mg")

    def test_E_incidence_rate_evidence_preserved_no_stats(self):
        """Test E: Incidence-rate evidence is preserved with statistical_representation == incidence_rate and no 2x2 stats."""
        kwargs = self.base_kwargs.copy()
        kwargs["extracted_metric_type"] = "incidence_rate"
        kwargs["raw_reported_treatment_metric"] = "1.6 per 100 PY"
        kwargs["raw_reported_control_metric"] = "0.9 per 100 PY"

        ev = ExtractedEvidence(**kwargs)
        res = normalize_study_evidence([ev])

        rec = res.study_records[0]
        self.assertEqual(rec.statistical_representation, "incidence_rate")
        self.assertFalse(rec.statistical_eligibility)
        self.assertIsNone(rec.risk_ratio_result)
        self.assertIsNone(rec.odds_ratio_result)
        self.assertIsNone(rec.hypothesis_test_result)

    def test_F_hazard_ratio_evidence_preserved_no_stats(self):
        """Test F: Hazard-ratio evidence is preserved with statistical_representation == hazard_ratio and no 2x2 stats."""
        kwargs = self.base_kwargs.copy()
        kwargs["extracted_metric_type"] = "hazard_ratio"
        kwargs["raw_reported_treatment_metric"] = "HR 0.86 (95% CI 0.74-0.99)"

        ev = ExtractedEvidence(**kwargs)
        res = normalize_study_evidence([ev])

        rec = res.study_records[0]
        self.assertEqual(rec.statistical_representation, "hazard_ratio")
        self.assertFalse(rec.statistical_eligibility)
        self.assertIsNone(rec.risk_ratio_result)

    def test_G_missing_quantitative_fields_preserved(self):
        """Test G: Missing quantitative fields are preserved without fabricated statistics."""
        kwargs = self.base_kwargs.copy()
        kwargs["treatment_events"] = None
        kwargs["treatment_n"] = None

        ev = ExtractedEvidence(**kwargs)
        res = normalize_study_evidence([ev])

        rec = res.study_records[0]
        self.assertIsNone(rec.treatment_events)
        self.assertIsNone(rec.treatment_n)
        self.assertFalse(rec.statistical_eligibility)
        self.assertIsNone(rec.risk_ratio_result)

    def test_H_exact_duplicate_evidence_deduplicated(self):
        """Test H: Exact duplicate evidence records are deterministically deduplicated."""
        ev1 = ExtractedEvidence(**self.base_kwargs)
        ev2 = ExtractedEvidence(**self.base_kwargs)

        res = normalize_study_evidence([ev1, ev2])

        self.assertEqual(res.total_input_records, 2)
        self.assertEqual(res.unique_evidence_records, 1)
        self.assertEqual(len(res.study_records), 1)

    def test_I_provenance_preservation(self):
        """Test I: Provenance citation fields are preserved verbatim."""
        ev = ExtractedEvidence(**self.base_kwargs)
        res = normalize_study_evidence([ev])

        rec = res.study_records[0]
        self.assertEqual(rec.source_file, "EMP-001.pdf")
        self.assertEqual(rec.chunk_id, "EMP-001_CHK_004")
        self.assertEqual(rec.page_start, 7)
        self.assertEqual(rec.page_end, 10)
        self.assertEqual(rec.evidence_quote, "Events consistent with genital infection 259 67")

    def test_J_case_whitespace_normalization(self):
        """Test J: Case and whitespace differences produce equivalent comparison_key."""
        key1 = normalize_comparison_key("  Empagliflozin  ", " Placebo ", "Genital Infection")
        key2 = normalize_comparison_key("empagliflozin", "placebo", "genital infection")
        self.assertEqual(key1, key2)

    def test_K_no_pooling(self):
        """Test K: Verify multiple studies are NOT combined or pooled into a single N/events sum."""
        kwargs1 = self.base_kwargs.copy()
        kwargs1["study_id"] = "EMP-001"
        kwargs1["treatment_events"] = 100
        kwargs1["treatment_n"] = 1000

        kwargs2 = self.base_kwargs.copy()
        kwargs2["study_id"] = "EMP-005"
        kwargs2["chunk_id"] = "EMP-005_CHK_003"
        kwargs2["treatment_events"] = 200
        kwargs2["treatment_n"] = 2000

        ev1 = ExtractedEvidence(**kwargs1)
        ev2 = ExtractedEvidence(**kwargs2)

        res = normalize_study_evidence([ev1, ev2])

        self.assertEqual(len(res.study_records), 2)
        # Ensure counts remain un-pooled in each study record
        self.assertEqual(res.study_records[0].treatment_events, 100)
        self.assertEqual(res.study_records[1].treatment_events, 200)


if __name__ == "__main__":
    unittest.main()
