"""
Unit test suite for Grounded LLM Evidence Analysis Report Generator (llm/report.py).
"""

import unittest
from unittest.mock import MagicMock

try:
    from llm.hypothesis import StructuredHypothesis
    from llm.evidence import ExtractedEvidence
    from analysis.study_comparison import normalize_study_evidence
    from analysis.conflict_analysis import analyze_multi_study_conflicts
    from report.report_builder import build_evidence_analysis_report, EvidenceAnalysisReport
    from llm.config import LLMConfig
    from llm.report import (
        generate_evidence_report,
        validate_generated_report,
        GeneratedReportValidationResult
    )
except ImportError:
    from hypothesis import StructuredHypothesis
    from evidence import ExtractedEvidence
    from study_comparison import normalize_study_evidence
    from conflict_analysis import analyze_multi_study_conflicts
    from report_builder import build_evidence_analysis_report, EvidenceAnalysisReport
    from config import LLMConfig
    from report import (
        generate_evidence_report,
        validate_generated_report,
        GeneratedReportValidationResult
    )




class TestGroundedLLMReportGenerator(unittest.TestCase):

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

        self.base_evidence = ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=4221,
            treatment_events=259,
            control_n=4203,
            control_events=67,
            extracted_metric_type="event_count",
            dose="10 mg",
            source_file="EMP-001.pdf",
            chunk_id="EMP-001_CHK_004",
            page_start=7,
            page_end=10,
            evidence_quote="Events consistent with genital infection 259 67"
        )

        comp = normalize_study_evidence([self.base_evidence])
        conf = analyze_multi_study_conflicts(comp)
        self.report_obj = build_evidence_analysis_report(
            "Does empagliflozin increase the risk of genital infection compared with placebo?",
            self.hypothesis,
            comp,
            conf
        )

    def test_A_valid_grounded_report_accepted(self):
        """Test A: Valid grounded report passing all checks is accepted by validator."""
        valid_text = (
            "# Research Question\nDoes empagliflozin increase the risk of genital infection compared with placebo?\n\n"
            "# Structured Hypothesis\n- Drug: empagliflozin\n- Comparator: placebo\n- Outcome: genital infection\n\n"
            "# Evidence Overview\nTotal Studies: 1. Total Evidence Records: 1.\n\n"
            "# Study-Level Evidence\n"
            "Study EMP-001 reported treatment N=4221 with 259 events vs control N=4203 with 67 events. "
            "Risk Ratio was 3.8492 (95% CI 2.9525 to 5.0183, Fisher p-value 9.9954e-29). "
            "[EMP-001, chunk EMP-001_CHK_004, pp. 7-10]\n\n"
            "# Cross-Study Observations\nNo directional conflict observed.\n\n"
            "# Limitations\nCross-study observations are based on the available evidence records.\n\n"
            "# Source Traceability\nPreserved from EMP-001.pdf, chunk EMP-001_CHK_004."
        )

        res = validate_generated_report(valid_text, self.report_obj)
        self.assertTrue(res.valid)
        self.assertEqual(len(res.errors), 0)

    def test_B_unknown_study_id_rejected(self):
        """Test B: Report citing unknown study ID is rejected by validator."""
        invalid_text = "Study EMP-999 reported elevated risk. [EMP-999, chunk EMP-001_CHK_004, pp. 7-10]"
        res = validate_generated_report(invalid_text, self.report_obj)
        self.assertFalse(res.valid)
        self.assertTrue(any("EMP-999" in err for err in res.errors))

    def test_C_unknown_chunk_id_rejected(self):
        """Test C: Report citing unknown chunk ID is rejected by validator."""
        invalid_text = "Study EMP-001 reported data in [EMP-001, chunk EMP-001_CHK_999, pp. 7-10]."
        res = validate_generated_report(invalid_text, self.report_obj)
        self.assertFalse(res.valid)
        self.assertTrue(any("EMP-001_CHK_999" in err for err in res.errors))

    def test_D_unsupported_numeric_value_rejected(self):
        """Test D: Completely unsupported numeric value (e.g. RR = 98.4) is rejected."""
        invalid_text = (
            "Study EMP-001 reported Risk Ratio was 98.45. [EMP-001, chunk EMP-001_CHK_004, pp. 7-10]"
        )
        res = validate_generated_report(invalid_text, self.report_obj)
        self.assertFalse(res.valid)
        self.assertTrue(any("untraced" in err.lower() for err in res.errors))

    def test_E_valid_rounded_numeric_value_accepted(self):
        """Test E: Acceptable rounded numeric value (e.g. 3.85 for 3.8492) is accepted."""
        valid_rounded_text = (
            "Study EMP-001 reported Risk Ratio was approximately 3.85. [EMP-001, chunk EMP-001_CHK_004, pp. 7-10]"
        )
        res = validate_generated_report(valid_rounded_text, self.report_obj)
        self.assertTrue(res.valid)

    def test_F_unsupported_clinical_conclusion_rejected(self):
        """Test F: Unsupported clinical/regulatory claim ('the drug is safe') is rejected."""
        invalid_text = (
            "Study EMP-001 reported data. Therefore, the drug is safe and should be approved. "
            "[EMP-001, chunk EMP-001_CHK_004, pp. 7-10]"
        )
        res = validate_generated_report(invalid_text, self.report_obj)
        self.assertFalse(res.valid)
        self.assertTrue(any("prohibited claim" in err.lower() for err in res.errors))

    def test_G_hr_rr_conversion_claim_rejected(self):
        """Test G: Claiming conversion of HR to RR is rejected."""
        invalid_text = (
            "We converted hazard ratio into risk ratio for EMP-001. [EMP-001, chunk EMP-001_CHK_004, pp. 7-10]"
        )
        res = validate_generated_report(invalid_text, self.report_obj)
        self.assertFalse(res.valid)
        self.assertTrue(any("statistical metric conversion" in err.lower() for err in res.errors))

    def test_H_missing_source_number_rejected(self):
        """Test H: Number not present in source report object is rejected."""
        invalid_text = "Sample size was N=99999 for EMP-001. [EMP-001, chunk EMP-001_CHK_004, pp. 7-10]"
        res = validate_generated_report(invalid_text, self.report_obj)
        self.assertFalse(res.valid)

    def test_I_citation_traceability_accepted(self):
        """Test I: Valid study/chunk/page citation format is accepted."""
        valid_text = (
            "Results were observed in EMP-001 [EMP-001, chunk EMP-001_CHK_004, pp. 7-10]."
        )
        res = validate_generated_report(valid_text, self.report_obj)
        self.assertTrue(res.valid)

    def test_J_llm_provider_failure_raises_exception(self):
        """Test J: LLM provider API failure raises explicit Exception without fallback fabrication."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API connection timeout")

        cfg = LLMConfig(provider="lmstudio", model="gemma", base_url="http://localhost:1234/v1", api_key="lm-studio")

        with self.assertRaises(RuntimeError):
            generate_evidence_report(self.report_obj, client=mock_client, config=cfg)

    def test_K_existing_provider_configuration_used(self):
        """Test K: Uses existing LLMConfig mechanism correctly."""
        cfg = LLMConfig.from_env()
        self.assertIn(cfg.provider, ["lmstudio", "openai", "fallback"])


if __name__ == "__main__":
    unittest.main()
