"""
Deterministic Evidence Analysis Report Builder Module.

Assembles validated StructuredHypothesis, StudyComparisonResult, and ConflictAnalysisResult
objects into a structured, citation-preserving EvidenceAnalysisReport payload.
Does NOT call LLMs, re-run retrieval, perform statistics, or make clinical claims.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from llm.hypothesis import StructuredHypothesis
from analysis.study_comparison import (
    StudyComparisonResult,
    NormalizedStudyEvidence,
    ComparisonGroup
)
from analysis.conflict_analysis import (
    GroupConflictAnalysisResult,
    MultiStudyConflictAnalysisPayload,
    StudyEffectSummary
)


class EvidenceCitation(BaseModel):
    """
    Provenance citation reference preserving source citation metadata.
    """
    study_id: str
    source_file: Optional[str] = None
    chunk_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    evidence_quote: str


class ReportEvidenceSummary(BaseModel):
    """
    Quantitative breakdowns of analyzed study records by statistical representation.
    """
    total_studies: int
    total_evidence_records: int
    statistical_event_count_records: int
    hazard_ratio_records: int
    risk_ratio_records: int
    odds_ratio_records: int
    incidence_rate_records: int
    records_without_usable_statistics: int


class ReportStudyResult(BaseModel):
    """
    Preserved study-level parameters, counts, and statistical estimates.
    """
    study_id: str
    drug: str
    comparator: str
    outcome: str
    study_type: Optional[str] = "unknown"
    dose: Optional[str] = "unknown"
    follow_up: Optional[str] = "unknown"
    publication_year: Optional[int] = None
    statistical_representation: str

    # 2x2 Counts
    treatment_n: Optional[int] = None
    treatment_events: Optional[int] = None
    control_n: Optional[int] = None
    control_events: Optional[int] = None

    # Statistical Estimates
    risk_ratio: Optional[float] = None
    risk_ratio_ci_lower: Optional[float] = None
    risk_ratio_ci_upper: Optional[float] = None
    odds_ratio: Optional[float] = None
    odds_ratio_ci_lower: Optional[float] = None
    odds_ratio_ci_upper: Optional[float] = None
    fisher_p_value: Optional[float] = None
    chi_square_p_value: Optional[float] = None

    # Citation metadata
    source_file: Optional[str] = None
    chunk_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    evidence_quote: str


class CrossStudyObservations(BaseModel):
    """
    Objective observations preserved directly from conflict analysis without clinical conclusions.
    """
    directional_conflict: bool
    confidence_interval_null_relation_differs: bool
    effect_measure_groups_are_separate: bool
    effect_measure_groups: List[str]
    observations: List[str]


class EvidenceAnalysisReport(BaseModel):
    """
    Complete intermediate structured representation payload for downstream consumption.
    """
    research_question: str
    hypothesis: StructuredHypothesis
    evidence_summary: ReportEvidenceSummary
    study_results: List[ReportStudyResult]
    cross_study_observations: Dict[str, CrossStudyObservations]
    limitations: List[str]
    citations: List[EvidenceCitation]


def build_evidence_summary(comparison: StudyComparisonResult) -> ReportEvidenceSummary:
    """
    Calculates deterministic summary counts across all normalized study records.
    """
    records = comparison.study_records
    total_recs = len(records)
    unique_studies = len(set(r.study_id for r in records))

    event_count = 0
    hr_count = 0
    rr_count = 0
    or_count = 0
    inc_count = 0
    no_stat_count = 0

    for r in records:
        rep = r.statistical_representation
        if rep == "2x2_event_count":
            event_count += 1
        elif rep == "hazard_ratio":
            hr_count += 1
        elif rep == "risk_ratio":
            rr_count += 1
        elif rep == "odds_ratio":
            or_count += 1
        elif rep == "incidence_rate":
            inc_count += 1

        if not r.statistical_eligibility:
            no_stat_count += 1

    return ReportEvidenceSummary(
        total_studies=unique_studies,
        total_evidence_records=total_recs,
        statistical_event_count_records=event_count,
        hazard_ratio_records=hr_count,
        risk_ratio_records=rr_count,
        odds_ratio_records=or_count,
        incidence_rate_records=inc_count,
        records_without_usable_statistics=no_stat_count
    )


def extract_report_study_result(record: NormalizedStudyEvidence) -> ReportStudyResult:
    """
    Extracts study parameters and statistical outputs without altering numerical precision.
    """
    rr_val = record.risk_ratio_result.risk_ratio if record.risk_ratio_result else None
    rr_l = record.risk_ratio_result.ci_lower if record.risk_ratio_result else None
    rr_u = record.risk_ratio_result.ci_upper if record.risk_ratio_result else None

    or_val = record.odds_ratio_result.odds_ratio if record.odds_ratio_result else None
    or_l = record.odds_ratio_result.ci_lower if record.odds_ratio_result else None
    or_u = record.odds_ratio_result.ci_upper if record.odds_ratio_result else None

    f_p = record.hypothesis_test_result.fisher_p_value if record.hypothesis_test_result else None
    chi_p = record.hypothesis_test_result.chi_square_p_value if record.hypothesis_test_result else None

    return ReportStudyResult(
        study_id=record.study_id,
        drug=record.drug,
        comparator=record.comparator,
        outcome=record.outcome,
        study_type=record.study_type,
        dose=record.dose,
        follow_up=record.follow_up,
        publication_year=record.publication_year,
        statistical_representation=record.statistical_representation,
        treatment_n=record.treatment_n,
        treatment_events=record.treatment_events,
        control_n=record.control_n,
        control_events=record.control_events,
        risk_ratio=rr_val,
        risk_ratio_ci_lower=rr_l,
        risk_ratio_ci_upper=rr_u,
        odds_ratio=or_val,
        odds_ratio_ci_lower=or_l,
        odds_ratio_ci_upper=or_u,
        fisher_p_value=f_p,
        chi_square_p_value=chi_p,
        source_file=record.source_file,
        chunk_id=record.chunk_id,
        page_start=record.page_start,
        page_end=record.page_end,
        evidence_quote=record.evidence_quote
    )


def derive_methodological_limitations(
    comparison: StudyComparisonResult,
    conflict_payload: MultiStudyConflictAnalysisPayload
) -> List[str]:
    """
    Derives objective methodological limitations supported strictly by dataset data.
    """
    limitations: List[str] = []
    records = comparison.study_records
    summary = build_evidence_summary(comparison)

    # 1. Unusable statistics presence
    if summary.records_without_usable_statistics > 0:
        limitations.append(
            "Some evidence records did not contain sufficient quantitative data for the configured statistical analysis."
        )

    # 2. Multiple statistical representations
    reps_present = set(r.statistical_representation for r in records)
    if len(reps_present) > 1:
        limitations.append(
            "Different effect measures were reported across the evidence and were kept as separate statistical representations."
        )

    # 3. Missing CI information
    missing_ci = any(
        (r.statistical_representation in ("hazard_ratio", "risk_ratio", "odds_ratio") and r.risk_ratio_result is None and r.odds_ratio_result is None)
        for r in records
    )
    if missing_ci:
        limitations.append(
            "Confidence interval information was unavailable for some study-level estimates."
        )

    # 4. Incidence rates preserved separately
    if summary.incidence_rate_records > 0:
        limitations.append(
            "Incidence-rate evidence was preserved separately and was not converted into 2x2 event counts."
        )

    # 5. Small study count limitation
    if summary.total_studies <= 3:
        limitations.append(
            "Cross-study observations are based on the available evidence records in the curated corpus."
        )

    return limitations


def build_evidence_analysis_report(
    research_question: str,
    hypothesis: StructuredHypothesis,
    comparison: StudyComparisonResult,
    conflict_payload: MultiStudyConflictAnalysisPayload
) -> EvidenceAnalysisReport:
    """
    Constructs an EvidenceAnalysisReport intermediate payload from validated input objects.
    """
    summary = build_evidence_summary(comparison)
    study_results = [extract_report_study_result(r) for r in comparison.study_records]

    cross_study_obs: Dict[str, CrossStudyObservations] = {}
    for c_key, group_analysis in conflict_payload.results.items():
        sep_groups = "effect_measure_groups_are_separate" in group_analysis.observations
        cross_study_obs[c_key] = CrossStudyObservations(
            directional_conflict=group_analysis.directional_conflict,
            confidence_interval_null_relation_differs=group_analysis.confidence_interval_null_relation_differs,
            effect_measure_groups_are_separate=sep_groups,
            effect_measure_groups=group_analysis.effect_measure_groups,
            observations=group_analysis.observations
        )

    limitations = derive_methodological_limitations(comparison, conflict_payload)

    citations: List[EvidenceCitation] = [
        EvidenceCitation(
            study_id=r.study_id,
            source_file=r.source_file,
            chunk_id=r.chunk_id,
            page_start=r.page_start,
            page_end=r.page_end,
            evidence_quote=r.evidence_quote
        )
        for r in comparison.study_records
    ]

    return EvidenceAnalysisReport(
        research_question=research_question,
        hypothesis=hypothesis,
        evidence_summary=summary,
        study_results=study_results,
        cross_study_observations=cross_study_obs,
        limitations=limitations,
        citations=citations
    )
