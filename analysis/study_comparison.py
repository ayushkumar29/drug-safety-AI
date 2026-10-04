"""
Multi-Study Evidence Normalizer Module.

Provides study-level normalization, deterministic comparison grouping, and study-level
statistical enrichment without pooling, meta-analysis, or metric conversions.
"""

import re
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from llm.evidence import ExtractedEvidence
from statistics.evidence_statistics import analyze_extracted_evidence, EvidenceStatisticalAnalysisResult
from statistics.risk_ratio import RiskRatioResult
from statistics.odds_ratio import OddsRatioResult
from statistics.hypothesis_test import HypothesisTestResult


def normalize_comparison_key(drug: str, comparator: str, outcome: str) -> str:
    """
    Generates a deterministic comparison key from drug, comparator, and outcome.
    Normalizes case and whitespace without fuzzy matching or stem altering.
    """
    norm_drug = re.sub(r'\s+', ' ', (drug or '').strip().lower())
    norm_comp = re.sub(r'\s+', ' ', (comparator or '').strip().lower())
    norm_out = re.sub(r'\s+', ' ', (outcome or '').strip().lower())
    return f"{norm_drug}__vs__{norm_comp}__for__{norm_out}"


def map_statistical_representation(metric_type: Optional[str]) -> str:
    """
    Maps extracted_metric_type to explicit statistical representation.
    Does NOT infer representations from prose.
    """
    if not metric_type:
        return "unknown"

    m = metric_type.strip().lower()
    mapping = {
        "event_count": "2x2_event_count",
        "incidence_rate": "incidence_rate",
        "hazard_ratio": "hazard_ratio",
        "risk_ratio": "risk_ratio",
        "odds_ratio": "odds_ratio",
        "sample_size": "sample_size",
        "unknown": "unknown"
    }
    return mapping.get(m, "unknown")


class NormalizedStudyEvidence(BaseModel):
    """
    Normalized study-level evidence record preserving all quantitative inputs,
    provenance citation fields, and attached 2x2 study-level statistics if eligible.
    """
    study_id: str
    comparison_key: str
    drug: str
    comparator: str
    outcome: str
    study_type: Optional[str] = "unknown"
    dose: Optional[str] = "unknown"
    follow_up: Optional[str] = "unknown"
    publication_year: Optional[int] = None
    source: Optional[str] = "unknown"

    # Provenance
    source_file: Optional[str] = None
    chunk_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    evidence_quote: str

    # Metric & Representation
    extracted_metric_type: Optional[str] = "unknown"
    statistical_representation: str

    # Raw Quantitative Inputs
    treatment_n: Optional[int] = None
    treatment_events: Optional[int] = None
    control_n: Optional[int] = None
    control_events: Optional[int] = None
    raw_reported_treatment_metric: Optional[str] = None
    raw_reported_control_metric: Optional[str] = None

    # Study-Level Statistics (Only populated if statistical_representation == '2x2_event_count')
    statistical_eligibility: bool = False
    ineligibility_reason: Optional[str] = None
    risk_ratio_result: Optional[RiskRatioResult] = None
    odds_ratio_result: Optional[OddsRatioResult] = None
    hypothesis_test_result: Optional[HypothesisTestResult] = None


class ComparisonGroup(BaseModel):
    """
    Group of normalized study evidence records sharing the exact same comparison_key.
    """
    comparison_key: str
    drug: str
    comparator: str
    outcome: str
    record_count: int
    study_records: List[NormalizedStudyEvidence]


class StudyComparisonResult(BaseModel):
    """
    Container payload holding normalized multi-study evidence comparison results.
    """
    total_input_records: int
    unique_evidence_records: int
    study_records: List[NormalizedStudyEvidence]
    comparison_groups: Dict[str, ComparisonGroup]


def compute_record_dedup_hash(e: ExtractedEvidence) -> tuple:
    """
    Computes a deterministic tuple signature for evidence record deduplication.
    Records with identical study_id, drug, comparator, outcome, dose, follow_up,
    counts, metrics, and chunk/quote provenance are treated as exact duplicates.
    """
    return (
        (e.study_id or '').strip().lower(),
        (e.drug or '').strip().lower(),
        (e.comparator or '').strip().lower(),
        (e.outcome or '').strip().lower(),
        (e.dose or '').strip().lower(),
        (e.follow_up or '').strip().lower(),
        e.treatment_n,
        e.treatment_events,
        e.control_n,
        e.control_events,
        (e.extracted_metric_type or '').strip().lower(),
        (e.source_file or '').strip().lower(),
        (e.chunk_id or '').strip().lower(),
        e.page_start,
        e.page_end,
        (e.evidence_quote or '').strip()
    )


def normalize_study_evidence(records: List[ExtractedEvidence]) -> StudyComparisonResult:
    """
    Accepts a list of validated ExtractedEvidence objects, performs deterministic
    deduplication, maps statistical representations, executes 2x2 statistics via the
    orchestrator for event_count records, and groups by comparison_key.

    Guarantees:
    - No pooling of event counts across studies.
    - No metric conversions (e.g. HR/rates -> event counts).
    - No LLM calls.
    - No merging of different doses, outcomes, or follow-ups.
    """
    total_input = len(records)
    seen_hashes = set()
    deduped_records: List[ExtractedEvidence] = []

    for r in records:
        h = compute_record_dedup_hash(r)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_records.append(r)

    unique_count = len(deduped_records)
    normalized_list: List[NormalizedStudyEvidence] = []
    groups_map: Dict[str, List[NormalizedStudyEvidence]] = {}

    for ev in deduped_records:
        comp_key = normalize_comparison_key(ev.drug, ev.comparator, ev.outcome)
        stat_rep = map_statistical_representation(ev.extracted_metric_type)

        # Run 2x2 statistics via orchestrator if event_count
        stat_eligibility = False
        inelig_reason = None
        rr_res = None
        or_res = None
        hyp_res = None

        if stat_rep == "2x2_event_count":
            stat_analysis: EvidenceStatisticalAnalysisResult = analyze_extracted_evidence(ev)
            stat_eligibility = stat_analysis.statistical_eligibility
            inelig_reason = stat_analysis.ineligibility_reason
            rr_res = stat_analysis.risk_ratio_result
            or_res = stat_analysis.odds_ratio_result
            hyp_res = stat_analysis.hypothesis_test_result
        else:
            inelig_reason = f"Representation '{stat_rep}' is not an event-count 2x2 table."

        norm_rec = NormalizedStudyEvidence(
            study_id=ev.study_id,
            comparison_key=comp_key,
            drug=ev.drug,
            comparator=ev.comparator,
            outcome=ev.outcome,
            study_type=ev.study_type,
            dose=ev.dose,
            follow_up=ev.follow_up,
            publication_year=ev.publication_year,
            source=ev.source,
            source_file=ev.source_file,
            chunk_id=ev.chunk_id,
            page_start=ev.page_start,
            page_end=ev.page_end,
            evidence_quote=ev.evidence_quote,
            extracted_metric_type=ev.extracted_metric_type,
            statistical_representation=stat_rep,
            treatment_n=ev.treatment_n,
            treatment_events=ev.treatment_events,
            control_n=ev.control_n,
            control_events=ev.control_events,
            raw_reported_treatment_metric=ev.raw_reported_treatment_metric,
            raw_reported_control_metric=ev.raw_reported_control_metric,
            statistical_eligibility=stat_eligibility,
            ineligibility_reason=inelig_reason,
            risk_ratio_result=rr_res,
            odds_ratio_result=or_res,
            hypothesis_test_result=hyp_res
        )

        normalized_list.append(norm_rec)
        groups_map.setdefault(comp_key, []).append(norm_rec)

    # Build comparison groups container
    comparison_groups: Dict[str, ComparisonGroup] = {}
    for c_key, items in groups_map.items():
        first = items[0]
        comparison_groups[c_key] = ComparisonGroup(
            comparison_key=c_key,
            drug=first.drug,
            comparator=first.comparator,
            outcome=first.outcome,
            record_count=len(items),
            study_records=items
        )

    return StudyComparisonResult(
        total_input_records=total_input,
        unique_evidence_records=unique_count,
        study_records=normalized_list,
        comparison_groups=comparison_groups
    )
