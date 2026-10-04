import sys
import os
import unittest
from unittest.mock import patch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from pipeline.pipeline import run_analysis, PipelineResult, STAGE_NAMES
from llm.hypothesis import StructuredHypothesis
from llm.evidence import ExtractedEvidence
from evaluation.artifact_generator import build_artifact


def _hypothesis():
    return StructuredHypothesis(
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        population="type 2 diabetes",
        study_type="randomized controlled trial",
        hypothesis_direction="greater_risk",
        null_hypothesis="No increased risk.",
        alternative_hypothesis="Increased risk.",
    )


def _emp005_evidence():
    return ExtractedEvidence(
        study_id="EMP-005", drug="empagliflozin", comparator="placebo",
        outcome="genital infection", study_type="randomized controlled trial",
        dose="10 mg", follow_up="180 days", publication_year=2015,
        treatment_n=4687, treatment_events=156, control_n=2333, control_events=42,
        extracted_metric_type="event_count",
        evidence_quote="EMP-005 primary RCT genital infection data.",
        source="Zinman et al. 2015", source_file="EMP-005.pdf",
        chunk_id="EMP-005_CHK_010", page_start=10, page_end=10,
    )


def _emp001_evidence():
    return ExtractedEvidence(
        study_id="EMP-001", drug="empagliflozin", comparator="placebo",
        outcome="genital infection", study_type="randomized controlled trial",
        dose="10 mg", follow_up="180 days", publication_year=2017,
        treatment_n=4221, treatment_events=259, control_n=4203, control_events=67,
        extracted_metric_type="event_count",
        evidence_quote="Empagliflozin 10mg (259/4221) vs placebo (67/4203).",
        source="EMPA-REG OUTCOME", source_file="EMP-001.pdf",
        chunk_id="EMP-001_CHK_004", page_start=7, page_end=8,
    )


def _emp005_chunk():
    return {"chunk_id": "EMP-005_CHK_010", "study_id": "EMP-005",
            "source_file": "EMP-005.pdf", "page_start": 10, "page_end": 10,
            "text": "EMP-005 primary RCT."}


def _emp001_chunk():
    return {"chunk_id": "EMP-001_CHK_004", "study_id": "EMP-001",
            "source_file": "EMP-001.pdf", "page_start": 7, "page_end": 8,
            "text": "EMP-001 pooled analysis."}


def _report_for(study_id):
    return (
        "# Research Question\n"
        "Does empagliflozin increase the risk of genital infection compared with placebo?\n\n"
        "# Structured Hypothesis\n"
        "Drug: empagliflozin\nComparator: placebo\nOutcome: genital infection\n"
        "Population: type 2 diabetes\nStudy type: randomized controlled trial\n"
        "Hypothesis direction: greater_risk\n"
        "Null hypothesis: No increased risk.\n"
        "Alternative hypothesis: Increased risk.\n\n"
        "# Evidence Overview\n"
        "Number of studies: 1\nNumber of evidence records: 1\n"
        "Statistical representations present: 2x2_event_count\n"
        "Records without usable statistics: 0\n\n"
        "# Study-Level Evidence\n"
        f"Study ID: {study_id}\nStudy type: randomized controlled trial\n"
        "Dose: 10 mg\nFollow-up: 180 days\n"
        "Treatment: 156 / 4687\nControl: 42 / 2333\n"
        "Risk Ratio: 1.85 (95% CI: 1.30 to 2.60)\n"
        f"Citation: [{study_id}, chunk {study_id}_CHK_010, pp. 10-10]\n\n"
        "# Cross-Study Observations\n"
        "No directional conflicts observed across studies.\n\n"
        "# Limitations\n"
        "Cross-study observations are based on the available evidence records.\n\n"
        "# Source Traceability\n"
        f"[{study_id}, chunk {study_id}_CHK_010, pp. 10-10]"
    )


class TestArtifactConsistency(unittest.TestCase):

    def test_artifact_A_no_independent_evidence(self):
        res = run_analysis(
            question="Does empagliflozin increase genital infection risk?",
            hypothesis_fn=lambda q: _hypothesis(),
            retriever_fn=lambda query, outcome, top_k: [_emp001_chunk()],
            extractor_fn=lambda chunk, hypothesis, config: _emp001_evidence(),
            generator_fn=lambda report: _report_for("EMP-001"),
        )
        art = build_artifact(res, evaluation_name="Test A")
        self.assertEqual(art["statistics_status"], "skipped_no_independent_evidence")
        self.assertFalse(art["statistics_ran"])
        self.assertEqual(art["statistics_input_count"], 0)
        self.assertEqual(art["statistics"], [])
        self.assertEqual(art["evidence"], [])
        self.assertEqual(art["independent_evidence_count"], 0)

    def test_artifact_B_one_retained_evidence(self):
        res = run_analysis(
            question="Does empagliflozin increase genital infection risk?",
            hypothesis_fn=lambda q: _hypothesis(),
            retriever_fn=lambda query, outcome, top_k: [_emp005_chunk()],
            extractor_fn=lambda chunk, hypothesis, config: _emp005_evidence(),
            generator_fn=lambda report: _report_for("EMP-005"),
        )
        art = build_artifact(res, evaluation_name="Test B")
        self.assertEqual(art["statistics_status"], "completed")
        self.assertTrue(art["statistics_ran"])
        self.assertEqual(art["statistics_input_count"], 1)
        self.assertEqual(art["independent_evidence_count"], 1)
        self.assertEqual(len(art["evidence"]), 1)
        self.assertEqual(art["evidence"][0]["study_id"], "EMP-005")

    def test_artifact_C_mixed_only_emp005_in_artifact(self):
        def mixed_extractor(chunk, hypothesis, config):
            if chunk["chunk_id"] == "EMP-005_CHK_010":
                return _emp005_evidence()
            return _emp001_evidence()

        res = run_analysis(
            question="Does empagliflozin increase genital infection risk?",
            hypothesis_fn=lambda q: _hypothesis(),
            retriever_fn=lambda query, outcome, top_k: [_emp005_chunk(), _emp001_chunk()],
            extractor_fn=mixed_extractor,
            generator_fn=lambda report: _report_for("EMP-005"),
        )
        art = build_artifact(res, evaluation_name="Test C")
        self.assertEqual(art["statistics_input_count"], 1)
        ev_ids = [e["study_id"] for e in art["evidence"]]
        self.assertIn("EMP-005", ev_ids)
        self.assertNotIn("EMP-001", ev_ids)
        self.assertIn("EMP-005", art["retained_publications"])
        self.assertIn("EMP-001", art["excluded_publications"])

    def test_artifact_D_count_monotonicity(self):
        def mixed_extractor(chunk, hypothesis, config):
            if chunk["chunk_id"] == "EMP-005_CHK_010":
                return _emp005_evidence()
            return _emp001_evidence()

        res = run_analysis(
            question="Does empagliflozin increase genital infection risk?",
            hypothesis_fn=lambda q: _hypothesis(),
            retriever_fn=lambda query, outcome, top_k: [_emp005_chunk(), _emp001_chunk()],
            extractor_fn=mixed_extractor,
            generator_fn=lambda report: _report_for("EMP-005"),
        )
        art = build_artifact(res, evaluation_name="Test D")
        rc, ec, vc, sc, ic, sic = (
            art["retrieved_count"], art["extracted_count"], art["validated_count"],
            art["selected_count"], art["independent_evidence_count"],
            art["statistics_input_count"],
        )
        self.assertGreaterEqual(rc, ec)
        self.assertGreaterEqual(ec, vc)
        self.assertGreaterEqual(vc, sc)
        self.assertLessEqual(ic, vc)
        if art["statistics_ran"]:
            self.assertEqual(sic, ic)
        else:
            self.assertEqual(sic, 0)


class TestStaticBoundary(unittest.TestCase):

    def test_boundary_stats_arg_from_independent_evidence(self):
        captured = []
        with patch("pipeline.pipeline.analyze_extracted_evidence", side_effect=captured.append):
            res = run_analysis(
                question="Does empagliflozin increase genital infection risk?",
                hypothesis_fn=lambda q: _hypothesis(),
                retriever_fn=lambda query, outcome, top_k: [_emp005_chunk()],
                extractor_fn=lambda chunk, hypothesis, config: _emp005_evidence(),
                generator_fn=lambda report: _report_for("EMP-005"),
            )
        independent_ids = {ev.study_id for ev in res.independent_evidence}
        for ev in captured:
            self.assertIn(ev.study_id, independent_ids)

    def test_boundary_excluded_evidence_never_reaches_stats_engine(self):
        with patch("pipeline.pipeline.analyze_extracted_evidence") as mock_stats:
            res = run_analysis(
                question="Does empagliflozin increase genital infection risk?",
                hypothesis_fn=lambda q: _hypothesis(),
                retriever_fn=lambda query, outcome, top_k: [_emp001_chunk()],
                extractor_fn=lambda chunk, hypothesis, config: _emp001_evidence(),
                generator_fn=lambda report: _report_for("EMP-001"),
            )
            mock_stats.assert_not_called()
            self.assertEqual(res.stage_status["statistics"], "skipped_no_independent_evidence")

    def test_boundary_no_fallback_pattern_in_source(self):
        import inspect, pipeline.pipeline as pipe_mod
        source = inspect.getsource(pipe_mod)
        self.assertNotIn("independent_evidence else result.validated_evidence", source)
        self.assertNotIn("normalize_study_evidence(result.validated_evidence)", source)
        self.assertNotIn("analyze_extracted_evidence(result.validated_evidence", source)


if __name__ == "__main__":
    unittest.main()
