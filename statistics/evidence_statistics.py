"""
Evidence-to-Statistics Orchestrator Module.

Accepts validated ExtractedEvidence object, verifies statistical eligibility,
and routes valid event-count 2x2 data to Risk Ratio, Odds Ratio, and Hypothesis Testing engines.
Does NOT perform statistical calculations directly or modify statistical engines.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from llm.evidence import ExtractedEvidence
from statistics.risk_ratio import calculate_risk_ratio, RiskRatioResult
from statistics.odds_ratio import calculate_odds_ratio, OddsRatioResult
from statistics.hypothesis_test import run_hypothesis_test, HypothesisTestResult


class EvidenceStatisticalAnalysisResult(BaseModel):
    """
    Structured outcome of evidence-to-statistics orchestration containing
    provenance, eligibility status, and executed statistical engine outputs.
    """
    study_id: str
    source_file: Optional[str] = None
    chunk_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    evidence_quote: str

    statistical_eligibility: bool = Field(description="True if evidence is eligible for quantitative 2x2 meta-analysis")
    ineligibility_reason: Optional[str] = Field(default=None, description="Reason if not analyzable")

    risk_ratio_result: Optional[RiskRatioResult] = None
    odds_ratio_result: Optional[OddsRatioResult] = None
    hypothesis_test_result: Optional[HypothesisTestResult] = None


def analyze_extracted_evidence(evidence: ExtractedEvidence) -> EvidenceStatisticalAnalysisResult:
    """
    Orchestrates statistical analysis on an ExtractedEvidence instance.

    Validates:
    1. Metric type is 'event_count' (rejects incidence_rate, percentage, hazard_ratio, etc.).
    2. Presence of all 4 integer counts (treatment_n, treatment_events, control_n, control_events).
    3. N > 0, events >= 0, events <= N.

    Routes valid counts to Risk Ratio, Odds Ratio, and Hypothesis Testing engines.
    """
    # Extract provenance
    provenance = {
        "study_id": evidence.study_id,
        "source_file": evidence.source_file,
        "chunk_id": evidence.chunk_id,
        "page_start": evidence.page_start,
        "page_end": evidence.page_end,
        "evidence_quote": evidence.evidence_quote
    }

    # 1. Metric Type Compatibility Check
    metric_type = (evidence.extracted_metric_type or "").lower().strip()
    if metric_type != "event_count":
        return EvidenceStatisticalAnalysisResult(
            **provenance,
            statistical_eligibility=False,
            ineligibility_reason=f"Metric type '{evidence.extracted_metric_type}' is not an event-count 2x2 table."
        )

    # 2. Presence Check for 4 numerical counts
    tr_e = evidence.treatment_events
    tr_n = evidence.treatment_n
    ctrl_e = evidence.control_events
    ctrl_n = evidence.control_n

    if tr_e is None or tr_n is None or ctrl_e is None or ctrl_n is None:
        return EvidenceStatisticalAnalysisResult(
            **provenance,
            statistical_eligibility=False,
            ineligibility_reason="Missing one or more required 2x2 numerical count fields (treatment_n, treatment_events, control_n, control_events)."
        )

    # 3. Numeric Bounds Validation
    if tr_n <= 0 or ctrl_n <= 0:
        return EvidenceStatisticalAnalysisResult(
            **provenance,
            statistical_eligibility=False,
            ineligibility_reason=f"Sample sizes must be positive (treatment_n={tr_n}, control_n={ctrl_n})."
        )

    if tr_e < 0 or ctrl_e < 0:
        return EvidenceStatisticalAnalysisResult(
            **provenance,
            statistical_eligibility=False,
            ineligibility_reason=f"Event counts cannot be negative (treatment_events={tr_e}, control_events={ctrl_e})."
        )

    if tr_e > tr_n or ctrl_e > ctrl_n:
        return EvidenceStatisticalAnalysisResult(
            **provenance,
            statistical_eligibility=False,
            ineligibility_reason=f"Event count exceeds sample size (treatment: {tr_e}/{tr_n}, control: {ctrl_e}/{ctrl_n})."
        )

    # 4. Route RAW counts to deterministic engines
    try:
        rr_res = calculate_risk_ratio(
            treatment_events=tr_e,
            treatment_n=tr_n,
            control_events=ctrl_e,
            control_n=ctrl_n
        )
        or_res = calculate_odds_ratio(
            treatment_events=tr_e,
            treatment_n=tr_n,
            control_events=ctrl_e,
            control_n=ctrl_n
        )
        hyp_res = run_hypothesis_test(
            treatment_events=tr_e,
            treatment_n=tr_n,
            control_events=ctrl_e,
            control_n=ctrl_n
        )

        return EvidenceStatisticalAnalysisResult(
            **provenance,
            statistical_eligibility=True,
            ineligibility_reason=None,
            risk_ratio_result=rr_res,
            odds_ratio_result=or_res,
            hypothesis_test_result=hyp_res
        )
    except Exception as e:
        return EvidenceStatisticalAnalysisResult(
            **provenance,
            statistical_eligibility=False,
            ineligibility_reason=f"Engine calculation failure: {str(e)}"
        )
