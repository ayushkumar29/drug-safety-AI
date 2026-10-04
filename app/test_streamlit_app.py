"""
Unit Test Suite for Streamlit MVP UI (app/test_streamlit_app.py)

Tests formatting and helper logic without requiring a browser context:
A. Application module imports successfully.
B. Question formatting handles valid inputs.
C. Evidence formatting normalizes records cleanly.
D. Optional None fields do not crash formatting.
E. Statistics formatting uses pre-computed PipelineResult values (zero UI calculation).
F. HR evidence produces safe non-2x2 messaging flags.
G. No-evidence / statistics-skipped state is handled cleanly.
H. Lineage display data categorizes retained/excluded/context records.
I. Retrieval candidate formatting prepares clean snippets.
J. Forest plot SVG renderer receives pre-calculated statistics rather than calculating them.
"""

import sys
import os
import unittest

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.streamlit_app import (
    format_hypothesis_summary,
    format_evidence_record,
    format_lineage_selection,
    format_statistics_summary,
    format_retrieved_candidate,
    generate_forest_plot_html
)
from pipeline.pipeline import PipelineResult
from llm.hypothesis import StructuredHypothesis
from llm.evidence import ExtractedEvidence
from statistics.risk_ratio import RiskRatioResult
from statistics.odds_ratio import OddsRatioResult
from statistics.hypothesis_test import HypothesisTestResult
from analysis.study_comparison import NormalizedStudyEvidence, StudyComparisonResult


class TestStreamlitAppUI(unittest.TestCase):

    def test_A_module_imports_successfully(self):
        """Test A: Application module imports successfully."""
        import app.streamlit_app
        self.assertTrue(hasattr(app.streamlit_app, "render_app"))
        self.assertTrue(hasattr(app.streamlit_app, "generate_forest_plot_html"))

    def test_B_question_formatting(self):
        """Test B: Structured hypothesis / question formatting."""
        hyp = StructuredHypothesis(
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            population="type 2 diabetes",
            study_type="randomized controlled trial",
            hypothesis_direction="greater_risk",
            null_hypothesis="RR = 1",
            alternative_hypothesis="RR != 1"
        )
        res = format_hypothesis_summary(hyp)
        self.assertEqual(res["Drug"], "empagliflozin")
        self.assertEqual(res["Outcome"], "genital infection")
        self.assertEqual(res["Comparator"], "placebo")
        self.assertEqual(res["Hypothesis Direction"], "greater_risk")

    def test_C_evidence_formatting(self):
        """Test C: Evidence record normalization."""
        ev = ExtractedEvidence(
            study_id="EMP-005",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            study_type="randomized controlled trial",
            treatment_n=4351,
            treatment_events=156,
            control_n=2333,
            control_events=36,
            source_file="EMP-005.pdf",
            page_start=12,
            page_end=12,
            chunk_id="EMP-005_CHK_007",
            evidence_quote="Genital infection reported in 156 of 4351 vs 36 of 2333."
        )
        rec = format_evidence_record(ev)
        self.assertEqual(rec["study_id"], "EMP-005")
        self.assertEqual(rec["treatment_events"], 156)
        self.assertEqual(rec["control_events"], 36)
        self.assertEqual(rec["page_range"], "12")
        self.assertEqual(rec["chunk_id"], "EMP-005_CHK_007")

    def test_D_none_missing_optional_fields(self):
        """Test D: None or missing optional fields do not crash formatting."""
        ev = ExtractedEvidence(
            study_id="EMP-005",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            evidence_quote="genital infection 42 vs 301",
            treatment_n=None,
            treatment_events=None,
            control_n=None,
            control_events=None,
            page_start=None,
            page_end=None
        )
        res = format_evidence_record(ev)
        self.assertEqual(res["study_id"], "EMP-005")
        self.assertEqual(res["page_range"], "N/A")
        self.assertIsNone(res["treatment_events"])

    def test_E_statistics_formatting_uses_precomputed(self):
        """Test E: Statistics summary extracts pre-computed statistics from PipelineResult (zero UI calculation)."""
        from statistics.risk_ratio import calculate_risk_ratio
        from statistics.odds_ratio import calculate_odds_ratio
        from statistics.hypothesis_test import run_hypothesis_test

        rr = calculate_risk_ratio(treatment_events=156, treatment_n=4351, control_events=36, control_n=2333)
        or_res = calculate_odds_ratio(treatment_events=156, treatment_n=4351, control_events=36, control_n=2333)
        ht = run_hypothesis_test(treatment_events=156, treatment_n=4351, control_events=36, control_n=2333)

        rec = NormalizedStudyEvidence(
            study_id="EMP-005",
            comparison_key="empagliflozin__vs__placebo__for__genital infection",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            study_type="randomized controlled trial",
            treatment_events=156,
            treatment_n=4351,
            control_events=36,
            control_n=2333,
            evidence_quote="Genital infection in 156 vs 36",
            statistical_representation="2x2_event_count",
            statistical_eligibility=True,
            risk_ratio_result=rr,
            odds_ratio_result=or_res,
            hypothesis_test_result=ht
        )
        comp = StudyComparisonResult(
            total_input_records=1,
            unique_evidence_records=1,
            study_records=[rec],
            comparison_groups={}
        )

        pipeline_res = PipelineResult(
            question="Does empagliflozin increase genital infection risk?",
            study_comparison=comp,
            stage_status={"statistics": "completed"}
        )

        stats = format_statistics_summary(pipeline_res)
        self.assertEqual(stats["status"], "completed")
        self.assertEqual(len(stats["records"]), 1)
        self.assertAlmostEqual(stats["records"][0]["risk_ratio"], 2.32, places=1)
        self.assertAlmostEqual(stats["records"][0]["odds_ratio"], 2.37, places=1)
        self.assertLess(stats["records"][0]["fisher_p_value"], 0.01)

    def test_F_hr_safe_messaging(self):
        """Test F: Hazard Ratio evidence produces safe non-2x2 messaging flag."""
        hr_ev = ExtractedEvidence(
            study_id="EMP-005",
            drug="empagliflozin",
            comparator="placebo",
            outcome="3-point MACE",
            extracted_metric_type="hazard_ratio",
            evidence_quote="HR 0.86 (95% CI 0.74 to 0.99)"
        )
        res = PipelineResult(
            question="MACE question",
            validated_evidence=[hr_ev],
            stage_status={"statistics": "skipped_no_independent_evidence"}
        )
        stats = format_statistics_summary(res)
        self.assertTrue(stats["has_hr"], "HR evidence must trigger has_hr flag for safe non-2x2 UI warning.")

    def test_G_no_evidence_state(self):
        """Test G: No-evidence state formatting."""
        res = PipelineResult(
            question="Unknown drug question",
            independent_evidence=[],
            stage_status={"statistics": "skipped_no_independent_evidence"}
        )
        stats = format_statistics_summary(res)
        self.assertEqual(stats["status"], "skipped_no_independent_evidence")
        self.assertEqual(stats["records"], [])
        lineage = format_lineage_selection([])
        self.assertEqual(lineage["retained"], [])

    def test_H_lineage_display_data(self):
        """Test H: Lineage display data categorizes records accurately."""
        sel_records = [
            {"selection": {"publication_id": "EMP-005", "selection_status": "retained_for_independent_analysis", "reason": None}},
            {"selection": {"publication_id": "EMP-001", "selection_status": "excluded_due_to_overlap", "reason": "Overlaps with EMPA-REG_OUTCOME"}}
        ]
        lineage = format_lineage_selection(sel_records)
        self.assertEqual(len(lineage["retained"]), 1)
        self.assertEqual(lineage["retained"][0]["publication_id"], "EMP-005")
        self.assertEqual(len(lineage["excluded"]), 1)
        self.assertEqual(lineage["excluded"][0]["publication_id"], "EMP-001")

    def test_I_retrieval_candidate_formatting(self):
        """Test I: Retrieval candidate formatting prepares clean snippets."""
        chk = {
            "chunk_id": "EMP-005_CHK_007",
            "study_id": "EMP-005",
            "source_file": "EMP-005.pdf",
            "page_start": 5,
            "page_end": 5,
            "combined_score": 0.89514,
            "text": "This is a detailed excerpt from the EMP-005 study detailing adverse event frequencies..."
        }
        res = format_retrieved_candidate(chk, rank=1)
        self.assertEqual(res["rank"], 1)
        self.assertEqual(res["chunk_id"], "EMP-005_CHK_007")
        self.assertEqual(res["relevance_score"], 0.8951)
        self.assertTrue("detailed excerpt" in res["snippet"])

    def test_J_forest_plot_uses_existing_statistics(self):
        """Test J: Forest plot SVG renderer receives existing pre-computed statistics rather than calculating them."""
        records = [
            {
                "study_id": "EMP-005",
                "risk_ratio": 2.32,
                "rr_ci_lower": 1.63,
                "rr_ci_upper": 3.32
            }
        ]
        svg_html = generate_forest_plot_html(records)
        self.assertIn("<svg", svg_html)
        self.assertIn("EMP-005", svg_html)
        self.assertIn("RR 2.32 [1.63, 3.32]", svg_html)
        self.assertIn("Line of No Effect", svg_html)


if __name__ == "__main__":
    unittest.main()

