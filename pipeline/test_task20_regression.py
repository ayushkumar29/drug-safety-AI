"""
Task 20 Regression Test Suite (pipeline/test_task20_regression.py)

Regression tests covering:
A. EMP-005 evidence reaches extraction
B. Multi-arm evidence is handled safely
C. No fabricated treatment/control denominator
D. Validated evidence retains provenance
E. EMP-005 survives independence selection
F. EMP-001 pooled evidence cannot reach statistics
G. Statistics receive only independent eligible evidence
H. Artifact counts match PipelineResult
I. No hard-coded EMP-005 path
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from pipeline.pipeline import run_analysis, PipelineResult
from llm.evidence import ExtractedEvidence
from evaluation.artifact_generator import build_artifact
from rag.retriever import retrieve
from llm.evidence_extractor import extract_evidence_from_chunk, score_chunk_relevance


class TestTask20Regression(unittest.TestCase):

    def setUp(self):
        self.question = "Does empagliflozin increase the risk of genital infection compared with placebo?"

    def test_A_emp005_evidence_reaches_extraction(self):
        """Test A: Verify EMP-005 quantitative evidence chunk reaches candidate retrieval set naturally."""
        chunks = retrieve(self.question, top_k=25, outcome="genital infection")
        emp005_chunks = [c for c in chunks if c.get("study_id") == "EMP-005" and ("CHK_007" in c.get("chunk_id", "") or "CHK_006" in c.get("chunk_id", ""))]
        self.assertTrue(len(emp005_chunks) > 0, "EMP-005 genital infection table chunk must be retrieved naturally in top 25.")

    def test_B_multiarm_evidence_handled_safely(self):
        """Test B: Multi-arm evidence from EMP-005 Table 2 is extracted into valid 2x2 comparison without invented arms."""
        chunk = {
            "chunk_id": "EMP-005_CHK_007",
            "study_id": "EMP-005",
            "source_file": "EMP-005.pdf",
            "page_start": 10,
            "page_end": 11,
            "text": "Event consistent with genital infection†† 42 (1.8) 153 (6.5) 148 (6.3) 301 (6.4)†"
        }
        hyp = {"drug": "empagliflozin", "comparator": "placebo", "outcome": "genital infection"}
        ev = extract_evidence_from_chunk(chunk, hyp)
        self.assertIsNotNone(ev, "Extraction must succeed on multi-arm table row.")
        self.assertEqual(ev.control_events, 42)
        self.assertEqual(ev.treatment_events, 301)

    def test_C_no_fabricated_treatment_control_denominator(self):
        """Test C: Sample sizes for treatment/control must match real study data without fabricated denominators."""
        chunk = {
            "chunk_id": "EMP-005_CHK_007",
            "study_id": "EMP-005",
            "source_file": "EMP-005.pdf",
            "page_start": 10,
            "page_end": 11,
            "text": "Event consistent with genital infection†† 42 (1.8) 153 (6.5) 148 (6.3) 301 (6.4)†"
        }
        hyp = {"drug": "empagliflozin", "comparator": "placebo", "outcome": "genital infection"}
        ev = extract_evidence_from_chunk(chunk, hyp)
        self.assertIsNotNone(ev)
        self.assertEqual(ev.control_n, 2333)
        self.assertEqual(ev.treatment_n, 4687)
        self.assertLessEqual(ev.control_events, ev.control_n)
        self.assertLessEqual(ev.treatment_events, ev.treatment_n)

    def test_D_validated_evidence_retains_provenance(self):
        """Test D: Validated evidence preserves source_file, chunk_id, page_start, page_end, and quote."""
        chunk = {
            "chunk_id": "EMP-005_CHK_007",
            "study_id": "EMP-005",
            "source_file": "EMP-005.pdf",
            "page_start": 10,
            "page_end": 11,
            "text": "Event consistent with genital infection†† 42 (1.8) 153 (6.5) 148 (6.3) 301 (6.4)†"
        }
        hyp = {"drug": "empagliflozin", "comparator": "placebo", "outcome": "genital infection"}
        ev = extract_evidence_from_chunk(chunk, hyp)
        self.assertEqual(ev.source_file, "EMP-005.pdf")
        self.assertEqual(ev.chunk_id, "EMP-005_CHK_007")
        self.assertEqual(ev.page_start, 10)
        self.assertEqual(ev.page_end, 11)
        self.assertTrue("genital infection" in ev.evidence_quote.lower())

    def test_E_emp005_survives_independence_selection(self):
        """Test E: Real pipeline execution retains EMP-005 as independent primary RCT evidence."""
        result = run_analysis(self.question, top_k=25)
        retained_studies = [ev.study_id for ev in result.independent_evidence]
        self.assertIn("EMP-005", retained_studies, "EMP-005 primary study must survive independence selection.")

    def test_F_emp001_pooled_evidence_cannot_reach_statistics(self):
        """Test F: Overlapping pooled publications (EMP-001/EMP-003) are excluded from statistics."""
        result = run_analysis(self.question, top_k=25)
        stat_studies = [rec.study_id for rec in result.study_comparison.study_records] if result.study_comparison else []
        self.assertNotIn("EMP-001", stat_studies, "EMP-001 pooled evidence must not reach statistics.")
        self.assertNotIn("EMP-003", stat_studies, "EMP-003 pooled evidence must not reach statistics.")

    def test_G_statistics_receive_only_independent_eligible_evidence(self):
        """Test G: Deterministic statistics engine receives ONLY result.independent_evidence."""
        result = run_analysis(self.question, top_k=25)
        if result.study_comparison:
            self.assertEqual(len(result.study_comparison.study_records), len(result.independent_evidence))

    def test_H_artifact_counts_match_pipeline_result(self):
        """Test H: Build artifact counts strictly match PipelineResult object fields."""
        result = run_analysis(self.question, top_k=25)
        artifact = build_artifact(result)
        self.assertEqual(artifact["retrieved_count"], len(result.retrieved_chunks))
        self.assertEqual(artifact["extracted_count"], len(result.extracted_evidence))
        self.assertEqual(artifact["validated_count"], len(result.validated_evidence))
        self.assertEqual(artifact["selected_count"], len(result.selected_evidence))
        self.assertEqual(artifact["independent_evidence_count"], len(result.independent_evidence))

    def test_I_no_hardcoded_emp005_path(self):
        """Test I: Standard relevance scoring operates without hard-coded string equality for EMP-005."""
        chunk = {
            "chunk_id": "GENERIC_CHK_001",
            "text": "genital infection 10 (1.0) vs placebo 2 (0.2) in randomized trial."
        }
        hyp = {"drug": "empagliflozin", "comparator": "placebo", "outcome": "genital infection"}
        sc = score_chunk_relevance(chunk, hyp)
        self.assertGreaterEqual(sc, 5.0, "Relevance scoring must evaluate generic quantitative chunks properly.")


if __name__ == "__main__":
    unittest.main()
