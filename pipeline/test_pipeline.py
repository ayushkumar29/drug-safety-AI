"""
Unit Test Suite for End-to-End Drug Safety Evidence Analysis Pipeline (pipeline/pipeline.py).
Tests all 10 pipeline stages, mock error pathways, validation rules, provenance preservation,
and status tracking without requiring live LLM calls.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from pipeline.pipeline import run_analysis, PipelineResult, STAGE_NAMES
    from llm.hypothesis import StructuredHypothesis
    from llm.evidence import ExtractedEvidence
    from llm.report import GeneratedReportValidationResult
except ImportError:
    from pipeline import run_analysis, PipelineResult, STAGE_NAMES
    from llm.hypothesis import StructuredHypothesis
    from llm.evidence import ExtractedEvidence
    from llm.report import GeneratedReportValidationResult




class TestPipelineOrchestrator(unittest.TestCase):

    def setUp(self):
        self.question = "Does empagliflozin increase the risk of genital infection compared with placebo?"

        self.mock_hypothesis = StructuredHypothesis(
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            population="type 2 diabetes",
            study_type="randomized controlled trial",
            hypothesis_direction="greater_risk",
            null_hypothesis="Empagliflozin does not increase genital infection risk compared to placebo.",
            alternative_hypothesis="Empagliflozin increases genital infection risk compared to placebo."
        )

        self.mock_chunk = {
            "chunk_id": "EMP-001_CHK_004",
            "study_id": "EMP-001",
            "source_file": "EMP-001.pdf",
            "page_start": 7,
            "page_end": 8,
            "text": "Empagliflozin 10mg (259/4221) vs placebo (67/4203)."
        }

        self.mock_evidence = ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            study_type="randomized controlled trial",
            dose="10 mg",
            follow_up="180 days",
            publication_year=2015,
            treatment_n=4221,
            treatment_events=259,
            control_n=4203,
            control_events=67,
            extracted_metric_type="event_count",
            evidence_quote="Empagliflozin 10mg (259/4221) vs placebo (67/4203).",
            source="EMPA-REG OUTCOME",
            source_file="EMP-001.pdf",
            chunk_id="EMP-001_CHK_004",
            page_start=7,
            page_end=8
        )

        self.mock_report_text = (
            "# Research Question\n"
            "Does empagliflozin increase the risk of genital infection compared with placebo?\n\n"
            "# Structured Hypothesis\n"
            "Drug: empagliflozin\nComparator: placebo\nOutcome: genital infection\n"
            "Population: type 2 diabetes\nStudy type: randomized controlled trial\n"
            "Hypothesis direction: greater_risk\n"
            "Null hypothesis: Empagliflozin does not increase genital infection risk compared to placebo.\n"
            "Alternative hypothesis: Empagliflozin increases genital infection risk compared to placebo.\n\n"
            "# Evidence Overview\n"
            "Number of studies: 1\nNumber of evidence records: 1\n"
            "Statistical representations present: 2x2_event_count\n"
            "Records without usable statistics: 0\n\n"
            "# Study-Level Evidence\n"
            "Study ID: EMP-005\nStudy type: randomized controlled trial\n"
            "Dose: 10 mg\nFollow-up: 180 days\n"
            "Treatment: 156 / 4687\nControl: 42 / 2333\n"
            "Risk Ratio: 1.8492 (95% CI: 1.3000 to 2.6000)\n"
            "Citation: [EMP-005, chunk EMP-005_CHK_010, pp. 10-10]\n\n"
            "# Cross-Study Observations\n"
            "No directional conflicts observed across studies.\n\n"
            "# Limitations\n"
            "Cross-study observations are based on the available evidence records in the curated corpus.\n\n"
            "# Source Traceability\n"
            "[EMP-005, chunk EMP-005_CHK_010, pp. 10-10]"
        )

        # EMP-005 = primary independent RCT (Zinman et al. 2015 EMPA-REG OUTCOME)
        self.emp005_evidence = ExtractedEvidence(
            study_id="EMP-005",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            study_type="randomized controlled trial",
            dose="10 mg",
            follow_up="180 days",
            publication_year=2015,
            treatment_n=4687,
            treatment_events=156,
            control_n=2333,
            control_events=42,
            extracted_metric_type="event_count",
            evidence_quote="EMP-005 primary RCT genital infection data.",
            source="Zinman et al. 2015",
            source_file="EMP-005.pdf",
            chunk_id="EMP-005_CHK_010",
            page_start=10,
            page_end=10
        )

        self.emp005_chunk = {
            "chunk_id": "EMP-005_CHK_010",
            "study_id": "EMP-005",
            "source_file": "EMP-005.pdf",
            "page_start": 10,
            "page_end": 10,
            "text": "EMP-005 primary RCT genital infection table."
        }



    # --------------------------------------------------
    # Test A: Complete Successful Pipeline
    # --------------------------------------------------
    def test_A_complete_successful_pipeline(self):
        """Complete pipeline run using EMP-005 (independent primary RCT) — all stages must complete."""
        res = run_analysis(
            question=self.question,
            retriever_fn=lambda query, outcome, top_k: [self.emp005_chunk],
            hypothesis_fn=lambda q: self.mock_hypothesis,
            extractor_fn=lambda chunk, hypothesis, config: self.emp005_evidence,
            generator_fn=lambda report: self.mock_report_text
        )

        self.assertIsInstance(res, PipelineResult)
        self.assertEqual(res.question, self.question)
        self.assertIsNotNone(res.hypothesis)
        self.assertEqual(len(res.retrieved_chunks), 1)
        self.assertEqual(len(res.validated_evidence), 1)
        # EMP-005 is the primary RCT — must be retained as independent evidence
        self.assertEqual(len(res.independent_evidence), 1)
        self.assertEqual(res.independent_evidence[0].study_id, "EMP-005")
        self.assertIsNotNone(res.study_comparison)
        self.assertIsNotNone(res.conflict_analysis)
        self.assertIsNotNone(res.deterministic_report)
        self.assertIsNotNone(res.generated_report)
        self.assertIsNotNone(res.generated_report_validation)
        self.assertTrue(res.generated_report_validation.valid)
        # Statistics must complete (not be skipped) when independent evidence is present
        self.assertEqual(res.stage_status["statistics"], "completed", "statistics should complete when independent evidence exists")
        for stage in STAGE_NAMES:
            self.assertIn(res.stage_status[stage], ("completed",), f"Stage {stage} should be completed")

    # --------------------------------------------------
    # Test B: Hypothesis Failure
    # --------------------------------------------------
    def test_B_hypothesis_failure(self):
        def bad_hypothesis(q):
            raise ValueError("Invalid underspecified research question")

        res = run_analysis(
            question=self.question,
            hypothesis_fn=bad_hypothesis
        )

        self.assertEqual(res.stage_status["hypothesis"], "failed")
        for stage in STAGE_NAMES[1:]:
            self.assertEqual(res.stage_status[stage], "skipped", f"Stage {stage} should be skipped")
        self.assertEqual(len(res.errors), 1)
        self.assertEqual(res.errors[0]["stage"], "hypothesis")

    # --------------------------------------------------
    # Test C: Retrieval Failure
    # --------------------------------------------------
    def test_C_retrieval_failure(self):
        def bad_retriever(query, outcome, top_k):
            raise RuntimeError("Vector database connection lost")

        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=bad_retriever
        )

        self.assertEqual(res.stage_status["hypothesis"], "completed")
        self.assertEqual(res.stage_status["retrieval"], "failed")
        for stage in STAGE_NAMES[2:]:
            self.assertEqual(res.stage_status[stage], "skipped", f"Stage {stage} should be skipped")
        self.assertEqual(len(res.errors), 1)
        self.assertEqual(res.errors[0]["stage"], "retrieval")

    # --------------------------------------------------
    # Test D: No Usable Evidence
    # --------------------------------------------------
    def test_D_no_usable_evidence(self):
        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [],
            extractor_fn=lambda chunk, hypothesis, config: None,
            generator_fn=lambda report: "# Research Question\nDoes empagliflozin increase the risk of genital infection compared with placebo?\n\n# Structured Hypothesis\nDrug: empagliflozin\n\n# Evidence Overview\nNumber of studies: 0\nNumber of evidence records: 0\nStatistical representations present: none\nRecords without usable statistics: 0\n\n# Study-Level Evidence\nNo evidence available.\n\n# Cross-Study Observations\nNo cross-study comparison possible.\n\n# Limitations\nNo evidence records.\n\n# Source Traceability\nNone"
        )

        self.assertEqual(res.stage_status["retrieval"], "completed")
        self.assertEqual(len(res.retrieved_chunks), 0)
        self.assertEqual(len(res.validated_evidence), 0)
        self.assertEqual(res.deterministic_report.evidence_summary.total_studies, 0)
        self.assertEqual(res.stage_status["llm_report"], "completed")

    # --------------------------------------------------
    # Test E: Invalid Evidence Retained in Rejected Evidence
    # --------------------------------------------------
    def test_E_invalid_evidence(self):
        invalid_evidence = ExtractedEvidence.model_construct(
            study_id="EMP-999",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=100,
            treatment_events=500,  # Invalid: events > n!
            control_n=100,
            control_events=50,
            evidence_quote="Empagliflozin 500/100",
            chunk_id="EMP-999_CHK_001"
        )

        def mock_extractor(chunk, hypothesis, config):
            if chunk["chunk_id"] == "EMP-001_CHK_004":
                return self.mock_evidence
            return invalid_evidence

        chunks = [self.mock_chunk, {"chunk_id": "EMP-999_CHK_001", "text": "bad chunk"}]

        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: chunks,
            extractor_fn=mock_extractor,
            generator_fn=lambda report: self.mock_report_text
        )

        self.assertEqual(len(res.extracted_evidence), 2)
        self.assertEqual(len(res.validated_evidence), 1)
        self.assertEqual(len(res.rejected_evidence), 1)
        self.assertEqual(res.rejected_evidence[0]["provenance"]["study_id"], "EMP-999")
        self.assertIn("treatment_events", res.rejected_evidence[0]["reason"])

    # --------------------------------------------------
    # Test F: LLM Report Validation Failure
    # --------------------------------------------------
    def test_F_llm_report_validation_failure(self):
        bad_report_text = self.mock_report_text + "\n\nStudy EMP-999 showed RR = 999.9."

        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [self.mock_chunk],
            extractor_fn=lambda chunk, hypothesis, config: self.mock_evidence,
            generator_fn=lambda report: bad_report_text
        )

        self.assertEqual(res.stage_status["llm_report"], "completed")
        self.assertEqual(res.stage_status["llm_report_validation"], "failed")
        self.assertIsNotNone(res.generated_report_validation)
        self.assertFalse(res.generated_report_validation.valid)
        self.assertTrue(any(e["stage"] == "llm_report_validation" for e in res.errors))

    # --------------------------------------------------
    # Test G: Provider Failure
    # --------------------------------------------------
    def test_G_provider_failure(self):
        def failing_generator(report):
            raise ConnectionError("LLM API endpoint timeout or provider down")

        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [self.mock_chunk],
            extractor_fn=lambda chunk, hypothesis, config: self.mock_evidence,
            generator_fn=failing_generator
        )

        self.assertEqual(res.stage_status["llm_report"], "failed")
        self.assertEqual(res.stage_status["llm_report_validation"], "skipped")
        self.assertIsNone(res.generated_report)
        self.assertTrue(any(e["stage"] == "llm_report" for e in res.errors))

    # --------------------------------------------------
    # Test H: Stage Status Tracking
    # --------------------------------------------------
    def test_H_stage_status_tracking(self):
        """Using EMP-005 independent evidence so all 10 stages reach completed status."""
        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [self.emp005_chunk],
            extractor_fn=lambda chunk, hypothesis, config: self.emp005_evidence,
            generator_fn=lambda report: self.mock_report_text
        )

        for stage in STAGE_NAMES:
            self.assertIn(stage, res.stage_status)
            self.assertEqual(res.stage_status[stage], "completed")

    # --------------------------------------------------
    # Test I: Provenance Preservation
    # --------------------------------------------------
    def test_I_provenance_preservation(self):
        """Provenance fields are preserved end-to-end through the pipeline.
        Uses EMP-005 (independent primary RCT) so evidence flows through
        the selection boundary into the deterministic report citations."""
        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [self.emp005_chunk],
            extractor_fn=lambda chunk, hypothesis, config: self.emp005_evidence,
            generator_fn=lambda report: self.mock_report_text
        )

        ev = res.validated_evidence[0]
        self.assertEqual(ev.study_id, "EMP-005")
        self.assertEqual(ev.chunk_id, "EMP-005_CHK_010")
        self.assertEqual(ev.source_file, "EMP-005.pdf")
        self.assertEqual(ev.page_start, 10)
        self.assertEqual(ev.page_end, 10)

        # EMP-005 flows through the selection boundary → must appear in citations
        self.assertGreater(len(res.deterministic_report.citations), 0,
            "Report must contain citations when independent evidence flows through")
        citation = res.deterministic_report.citations[0]
        self.assertEqual(citation.study_id, "EMP-005")
        self.assertEqual(citation.chunk_id, "EMP-005_CHK_010")

    # --------------------------------------------------
    # Test J: No Statistical Duplication in Orchestrator
    # --------------------------------------------------
    def test_J_no_statistical_duplication(self):
        import inspect
        import pipeline.pipeline as pipe_mod

        code = inspect.getsource(pipe_mod)
        self.assertNotIn("math.sqrt", code)
        self.assertNotIn("1.96", code)
        self.assertNotIn("scipy.stats", code)

    # --------------------------------------------------
    # Task 17A — Test 1: Lineage is attached
    # --------------------------------------------------
    def test_task17a_1_lineage_is_attached(self):
        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [self.mock_chunk],
            extractor_fn=lambda chunk, hypothesis, config: self.mock_evidence,
            generator_fn=lambda report: self.mock_report_text
        )

        self.assertEqual(len(res.lineage_evidence), 1)
        item = res.lineage_evidence[0]
        lineage = item.get("lineage", {})

        # Verify preservation of original evidence attributes
        self.assertEqual(item["study_id"], self.mock_evidence.study_id)
        self.assertEqual(item["source_file"], self.mock_evidence.source_file)
        self.assertEqual(item["chunk_id"], self.mock_evidence.chunk_id)
        self.assertEqual(item["page_start"], self.mock_evidence.page_start)
        self.assertEqual(item["page_end"], self.mock_evidence.page_end)
        self.assertEqual(item["evidence_quote"], self.mock_evidence.evidence_quote)

        # Verify lineage attachment fields
        self.assertEqual(lineage.get("publication_id"), "EMP-001")
        self.assertIn("EMPA-REG_OUTCOME", lineage.get("underlying_study_ids", []))
        self.assertIsNotNone(lineage.get("overlap_group_id"))

    # --------------------------------------------------
    # Task 17A — Test 2: Overlapping pooled evidence is excluded
    # --------------------------------------------------
    def test_task17a_2_overlapping_pooled_evidence_is_excluded(self):
        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [self.mock_chunk],
            extractor_fn=lambda chunk, hypothesis, config: self.mock_evidence,
            generator_fn=lambda report: self.mock_report_text
        )

        self.assertEqual(len(res.selected_evidence), 1)
        selected_item = res.selected_evidence[0]
        sel_meta = selected_item.get("selection", {})

        self.assertEqual(sel_meta.get("publication_id"), "EMP-001")
        self.assertEqual(sel_meta.get("selection_status"), "excluded_due_to_overlap")
        self.assertIn("excluded due to potential overlap", sel_meta.get("selection_reason", ""))

    # --------------------------------------------------
    # Task 17A — Test 3: Independent primary evidence can reach statistics
    # --------------------------------------------------
    def test_task17a_3_independent_primary_evidence_retained(self):
        emp005_evidence = ExtractedEvidence(
            study_id="EMP-005",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            study_type="randomized controlled trial",
            dose="10 mg",
            follow_up="180 days",
            publication_year=2015,
            treatment_n=4687,
            treatment_events=156,
            control_n=2333,
            control_events=42,
            extracted_metric_type="event_count",
            evidence_quote="EMP-005 primary trial data quote.",
            source="Zinman et al.",
            source_file="EMP-005.pdf",
            chunk_id="EMP-005_CHK_001",
            page_start=1,
            page_end=2
        )

        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [{"chunk_id": "EMP-005_CHK_001", "study_id": "EMP-005", "source_file": "EMP-005.pdf", "text": "text"}],
            extractor_fn=lambda chunk, hypothesis, config: emp005_evidence,
            generator_fn=lambda report: self.mock_report_text
        )

        self.assertEqual(len(res.independent_evidence), 1)
        self.assertEqual(res.independent_evidence[0].study_id, "EMP-005")

        sel_meta = res.selected_evidence[0].get("selection", {})
        self.assertEqual(sel_meta.get("selection_status"), "retained_for_independent_analysis")

    # --------------------------------------------------
    # Task 17A — Test 4: Excluded evidence cannot reach statistics when independent evidence is present
    # --------------------------------------------------
    def test_task17a_4_excluded_evidence_blocked_from_statistics(self):
        emp005_evidence = ExtractedEvidence(
            study_id="EMP-005",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            study_type="randomized controlled trial",
            dose="10 mg",
            follow_up="180 days",
            publication_year=2015,
            treatment_n=4687,
            treatment_events=156,
            control_n=2333,
            control_events=42,
            extracted_metric_type="event_count",
            evidence_quote="EMP-005 primary trial data quote.",
            source="Zinman et al.",
            source_file="EMP-005.pdf",
            chunk_id="EMP-005_CHK_001",
            page_start=1,
            page_end=2
        )

        def mock_extractor(chunk, hypothesis, config):
            if chunk["chunk_id"] == "EMP-005_CHK_001":
                return emp005_evidence
            return self.mock_evidence

        chunks = [
            {"chunk_id": "EMP-005_CHK_001", "study_id": "EMP-005", "source_file": "EMP-005.pdf", "text": "text"},
            self.mock_chunk
        ]

        with patch("pipeline.pipeline.analyze_extracted_evidence") as mock_analyze_stats:
            res = run_analysis(
                question=self.question,
                hypothesis_fn=lambda q: self.mock_hypothesis,
                retriever_fn=lambda query, outcome, top_k: chunks,
                extractor_fn=mock_extractor,
                generator_fn=lambda report: self.mock_report_text
            )

            # EMP-005 retained, EMP-001 excluded
            self.assertEqual(len(res.independent_evidence), 1)
            self.assertEqual(res.independent_evidence[0].study_id, "EMP-005")

            # Verify analyze_extracted_evidence was ONLY called for EMP-005, NOT for EMP-001
            mock_analyze_stats.assert_called_once_with(emp005_evidence)

    # --------------------------------------------------
    # Task 17A — Test 5: Independent evidence reaches statistics
    # --------------------------------------------------
    def test_task17a_5_retained_independent_evidence_sent_to_statistics(self):
        emp005_evidence = ExtractedEvidence(
            study_id="EMP-005",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            study_type="randomized controlled trial",
            dose="10 mg",
            follow_up="180 days",
            publication_year=2015,
            treatment_n=4687,
            treatment_events=156,
            control_n=2333,
            control_events=42,
            extracted_metric_type="event_count",
            evidence_quote="EMP-005 primary trial data quote.",
            source="Zinman et al.",
            source_file="EMP-005.pdf",
            chunk_id="EMP-005_CHK_001",
            page_start=1,
            page_end=2
        )

        with patch("pipeline.pipeline.analyze_extracted_evidence") as mock_analyze_stats:
            res = run_analysis(
                question=self.question,
                hypothesis_fn=lambda q: self.mock_hypothesis,
                retriever_fn=lambda query, outcome, top_k: [{"chunk_id": "EMP-005_CHK_001", "study_id": "EMP-005", "source_file": "EMP-005.pdf", "text": "text"}],
                extractor_fn=lambda chunk, hypothesis, config: emp005_evidence,
                generator_fn=lambda report: self.mock_report_text
            )

            self.assertEqual(len(res.independent_evidence), 1)
            self.assertEqual(res.independent_evidence[0].study_id, "EMP-005")

            # Verify analyze_extracted_evidence was called directly with the retained EMP-005 ExtractedEvidence
            mock_analyze_stats.assert_called_once_with(emp005_evidence)

    # --------------------------------------------------
    # Task 17A — Test 6: Provenance survives selection
    # --------------------------------------------------
    def test_task17a_6_provenance_survives_selection(self):
        res = run_analysis(
            question=self.question,
            hypothesis_fn=lambda q: self.mock_hypothesis,
            retriever_fn=lambda query, outcome, top_k: [self.mock_chunk],
            extractor_fn=lambda chunk, hypothesis, config: self.mock_evidence,
            generator_fn=lambda report: self.mock_report_text
        )

        self.assertEqual(len(res.selected_evidence), 1)
        rec = res.selected_evidence[0]

        # Verify exact field provenance preservation
        self.assertEqual(rec["study_id"], self.mock_evidence.study_id)
        self.assertEqual(rec["source_file"], self.mock_evidence.source_file)
        self.assertEqual(rec["chunk_id"], self.mock_evidence.chunk_id)
        self.assertEqual(rec["page_start"], self.mock_evidence.page_start)
        self.assertEqual(rec["page_end"], self.mock_evidence.page_end)
        self.assertEqual(rec["evidence_quote"], self.mock_evidence.evidence_quote)
        self.assertEqual(rec["extracted_metric_type"], self.mock_evidence.extracted_metric_type)
        self.assertEqual(rec["treatment_n"], self.mock_evidence.treatment_n)
        self.assertEqual(rec["treatment_events"], self.mock_evidence.treatment_events)
        self.assertEqual(rec["control_n"], self.mock_evidence.control_n)
        self.assertEqual(rec["control_events"], self.mock_evidence.control_events)


    # ==================================================
    # Task 18A Regression Tests — Statistics Selection Boundary
    # ==================================================

    # --------------------------------------------------
    # Task 18A — Test 1: Excluded-only: stats skipped when only overlapping evidence
    # --------------------------------------------------
    def test_18A_1_excluded_only_statistics_skipped(self):
        """When all validated evidence is excluded_due_to_overlap, statistics must be skipped.
        Must NOT fall back to validated_evidence."""
        with patch("pipeline.pipeline.analyze_extracted_evidence") as mock_analyze_stats:
            res = run_analysis(
                question=self.question,
                hypothesis_fn=lambda q: self.mock_hypothesis,
                retriever_fn=lambda query, outcome, top_k: [self.mock_chunk],
                extractor_fn=lambda chunk, hypothesis, config: self.mock_evidence,
                generator_fn=lambda report: self.mock_report_text
            )

            # EMP-001 is a pooled analysis — must be excluded
            self.assertEqual(len(res.independent_evidence), 0,
                "EMP-001 pooled analysis must not appear as independent evidence")

            # Statistics must be skipped — not completed — when no independent evidence exists
            self.assertEqual(res.stage_status["statistics"], "skipped_no_independent_evidence",
                "statistics stage must be skipped when all evidence is excluded_due_to_overlap")

            # analyze_extracted_evidence must NEVER be called with overlapping evidence
            mock_analyze_stats.assert_not_called()

    # --------------------------------------------------
    # Task 18A — Test 2: Retained: stats completed when independent evidence exists
    # --------------------------------------------------
    def test_18A_2_retained_evidence_statistics_completed(self):
        """When EMP-005 (primary RCT) is retained, statistics must be completed."""
        with patch("pipeline.pipeline.analyze_extracted_evidence") as mock_analyze_stats:
            res = run_analysis(
                question=self.question,
                hypothesis_fn=lambda q: self.mock_hypothesis,
                retriever_fn=lambda query, outcome, top_k: [self.emp005_chunk],
                extractor_fn=lambda chunk, hypothesis, config: self.emp005_evidence,
                generator_fn=lambda report: self.mock_report_text
            )

            self.assertEqual(len(res.independent_evidence), 1)
            self.assertEqual(res.independent_evidence[0].study_id, "EMP-005")
            self.assertEqual(res.stage_status["statistics"], "completed",
                "statistics stage must complete when independent evidence is retained")
            mock_analyze_stats.assert_called_once_with(self.emp005_evidence)

    # --------------------------------------------------
    # Task 18A — Test 3: Mixed: excluded evidence blocked, only retained goes to stats
    # --------------------------------------------------
    def test_18A_3_mixed_excluded_blocked_retained_only_to_stats(self):
        """When EMP-005 and EMP-001 are both extracted, only EMP-005 must reach statistics."""
        def mixed_extractor(chunk, hypothesis, config):
            if chunk["chunk_id"] == "EMP-005_CHK_010":
                return self.emp005_evidence
            return self.mock_evidence  # EMP-001

        mixed_chunks = [self.emp005_chunk, self.mock_chunk]

        with patch("pipeline.pipeline.analyze_extracted_evidence") as mock_analyze_stats:
            res = run_analysis(
                question=self.question,
                hypothesis_fn=lambda q: self.mock_hypothesis,
                retriever_fn=lambda query, outcome, top_k: mixed_chunks,
                extractor_fn=mixed_extractor,
                generator_fn=lambda report: self.mock_report_text
            )

            # EMP-005 retained, EMP-001 excluded
            self.assertEqual(len(res.independent_evidence), 1)
            self.assertEqual(res.independent_evidence[0].study_id, "EMP-005")

            # Statistics must complete (not skipped)
            self.assertEqual(res.stage_status["statistics"], "completed")

            # EMP-001 must NOT reach analyze_extracted_evidence
            call_args = [call.args[0].study_id for call in mock_analyze_stats.call_args_list]
            self.assertIn("EMP-005", call_args)
            self.assertNotIn("EMP-001", call_args)

    # --------------------------------------------------
    # Task 18A — Test 4: No-eligible: empty extraction → stats skipped
    # --------------------------------------------------
    def test_18A_4_no_eligible_evidence_statistics_skipped(self):
        """When no evidence is extracted at all, statistics must be skipped with the
        correct status — not failed, not completed."""
        with patch("pipeline.pipeline.analyze_extracted_evidence") as mock_analyze_stats:
            res = run_analysis(
                question=self.question,
                hypothesis_fn=lambda q: self.mock_hypothesis,
                retriever_fn=lambda query, outcome, top_k: [],
                extractor_fn=lambda chunk, hypothesis, config: None,
                generator_fn=lambda report: (
                    "# Research Question\n"
                    "Does empagliflozin increase the risk of genital infection compared with placebo?\n\n"
                    "# Structured Hypothesis\nDrug: empagliflozin\n\n"
                    "# Evidence Overview\nNumber of studies: 0\nNumber of evidence records: 0\n"
                    "Statistical representations present: none\nRecords without usable statistics: 0\n\n"
                    "# Study-Level Evidence\nNo evidence available.\n\n"
                    "# Cross-Study Observations\nNo cross-study comparison possible.\n\n"
                    "# Limitations\nNo evidence records.\n\n"
                    "# Source Traceability\nNone"
                )
            )

            self.assertEqual(len(res.independent_evidence), 0)
            self.assertEqual(res.stage_status["statistics"], "skipped_no_independent_evidence",
                "statistics must be skipped when no evidence is extracted")
            mock_analyze_stats.assert_not_called()

    # --------------------------------------------------
    # Task 18A — Test 5: Provenance: study_id preserved through the selection boundary
    # --------------------------------------------------
    def test_18A_5_provenance_study_id_preserved_through_boundary(self):
        """The study_id, chunk_id, source_file, and page range of independent evidence
        must be preserved exactly as they pass through the selection boundary into statistics."""
        captured_evidence = []

        def capturing_analyzer(ev):
            captured_evidence.append(ev)

        with patch("pipeline.pipeline.analyze_extracted_evidence", side_effect=capturing_analyzer):
            res = run_analysis(
                question=self.question,
                hypothesis_fn=lambda q: self.mock_hypothesis,
                retriever_fn=lambda query, outcome, top_k: [self.emp005_chunk],
                extractor_fn=lambda chunk, hypothesis, config: self.emp005_evidence,
                generator_fn=lambda report: self.mock_report_text
            )

        self.assertEqual(len(captured_evidence), 1)
        ev = captured_evidence[0]
        self.assertEqual(ev.study_id, "EMP-005")
        self.assertEqual(ev.chunk_id, "EMP-005_CHK_010")
        self.assertEqual(ev.source_file, "EMP-005.pdf")
        self.assertEqual(ev.page_start, 10)
        self.assertEqual(ev.page_end, 10)
        self.assertEqual(ev.treatment_n, 4687)
        self.assertEqual(ev.treatment_events, 156)
        self.assertEqual(ev.control_n, 2333)
        self.assertEqual(ev.control_events, 42)


if __name__ == "__main__":
    unittest.main()
