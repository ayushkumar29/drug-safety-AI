"""
Multi-Study Conflict Analysis Module.

Performs deterministic study-level conflict and heterogeneity inspection on compatible
effect measure groups (RR, OR, HR) without meta-analysis, pooling, or clinical conclusions.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from analysis.study_comparison import (
    StudyComparisonResult,
    NormalizedStudyEvidence,
    ComparisonGroup
)


class StudyEffectSummary(BaseModel):
    """
    Normalized study-level effect summary representing single study point estimate and CI.
    """
    study_id: str
    comparison_key: str
    statistical_representation: str
    effect_measure: str  # "risk_ratio", "odds_ratio", "hazard_ratio", or "none"
    effect_estimate: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    effect_direction: str  # "higher_effect", "lower_effect", "null_effect", "unknown"
    ci_relation: str      # "above_null", "below_null", "includes_null", "not_available"
    publication_year: Optional[int] = None
    dose: Optional[str] = "unknown"
    follow_up: Optional[str] = "unknown"
    source_file: Optional[str] = None
    chunk_id: Optional[str] = None


class EffectMeasureGroupAnalysis(BaseModel):
    """
    Deterministic conflict analysis within a single compatible effect-measure category.
    """
    effect_measure: str
    study_count: int
    directional_conflict: bool = False
    confidence_interval_null_relation_differs: bool = False
    effect_directions_present: List[str] = Field(default_factory=list)
    ci_relations_present: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    study_effects: List[StudyEffectSummary] = Field(default_factory=list)


class GroupConflictAnalysisResult(BaseModel):
    """
    Overall conflict analysis output for a single comparison group (drug vs comparator for outcome).
    """
    comparison_key: str
    total_studies: int
    effect_measure_groups: List[str]
    directional_conflict: bool = False
    confidence_interval_null_relation_differs: bool = False
    observations: List[str] = Field(default_factory=list)
    representation_groups: Dict[str, EffectMeasureGroupAnalysis]
    study_effects: List[StudyEffectSummary]


class MultiStudyConflictAnalysisPayload(BaseModel):
    """
    Container payload holding multi-study conflict analysis results indexed by comparison_key.
    """
    total_comparison_groups: int
    results: Dict[str, GroupConflictAnalysisResult]


def determine_effect_direction(val: Optional[float], tol: float = 1e-12) -> str:
    """
    Determines effect direction relative to null = 1.0.
    Does NOT describe directions as 'safe' or 'unsafe'.
    """
    if val is None:
        return "unknown"

    if abs(val - 1.0) <= tol:
        return "null_effect"
    elif val > 1.0:
        return "higher_effect"
    else:
        return "lower_effect"


def determine_ci_relation(ci_lower: Optional[float], ci_upper: Optional[float], tol: float = 1e-12) -> str:
    """
    Determines confidence interval relationship to null = 1.0.
    """
    if ci_lower is None or ci_upper is None:
        return "not_available"

    # Lower bound > 1.0
    if ci_lower > (1.0 + tol):
        return "above_null"
    # Upper bound < 1.0
    elif ci_upper < (1.0 - tol):
        return "below_null"
    # Null included in interval
    else:
        return "includes_null"


def extract_study_effect_summary(record: NormalizedStudyEvidence) -> StudyEffectSummary:
    """
    Extracts explicit point estimate and CI bounds from a NormalizedStudyEvidence record.
    Uses existing RiskRatioResult for 2x2_event_count. Does NOT extract from prose.
    """
    rep = record.statistical_representation
    effect_measure = "none"
    estimate = None
    ci_l = None
    ci_u = None

    if rep == "2x2_event_count" and record.risk_ratio_result:
        effect_measure = "risk_ratio"
        estimate = record.risk_ratio_result.risk_ratio
        ci_l = record.risk_ratio_result.ci_lower
        ci_u = record.risk_ratio_result.ci_upper
    elif rep == "risk_ratio" and record.risk_ratio_result:
        effect_measure = "risk_ratio"
        estimate = record.risk_ratio_result.risk_ratio
        ci_l = record.risk_ratio_result.ci_lower
        ci_u = record.risk_ratio_result.ci_upper
    elif rep in ("odds_ratio", "hazard_ratio", "risk_ratio"):
        effect_measure = rep
        if record.odds_ratio_result and rep == "odds_ratio":
            estimate = record.odds_ratio_result.odds_ratio
            ci_l = record.odds_ratio_result.ci_lower
            ci_u = record.odds_ratio_result.ci_upper
        elif record.risk_ratio_result and rep == "risk_ratio":
            estimate = record.risk_ratio_result.risk_ratio
            ci_l = record.risk_ratio_result.ci_lower
            ci_u = record.risk_ratio_result.ci_upper

    direction = determine_effect_direction(estimate)
    ci_rel = determine_ci_relation(ci_l, ci_u)

    return StudyEffectSummary(
        study_id=record.study_id,
        comparison_key=record.comparison_key,
        statistical_representation=rep,
        effect_measure=effect_measure,
        effect_estimate=estimate,
        ci_lower=ci_l,
        ci_upper=ci_u,
        effect_direction=direction,
        ci_relation=ci_rel,
        publication_year=record.publication_year,
        dose=record.dose,
        follow_up=record.follow_up,
        source_file=record.source_file,
        chunk_id=record.chunk_id
    )


def analyze_group_conflicts(group: ComparisonGroup) -> GroupConflictAnalysisResult:
    """
    Performs conflict analysis across all study records within a single ComparisonGroup.
    Separates incompatible effect measures (RR vs HR vs OR) into isolated representation groups.
    """
    study_summaries: List[StudyEffectSummary] = [
        extract_study_effect_summary(rec) for rec in group.study_records
    ]

    # Group by compatible effect measure category
    measure_buckets: Dict[str, List[StudyEffectSummary]] = {}
    for s in study_summaries:
        measure_buckets.setdefault(s.effect_measure, []).append(s)

    representation_analyses: Dict[str, EffectMeasureGroupAnalysis] = {}
    effect_groups_present = sorted(list(measure_buckets.keys()))

    group_directional_conflict = False
    group_ci_null_differs = False
    group_observations: List[str] = []

    if len(effect_groups_present) > 1 and "none" not in effect_groups_present:
        group_observations.append("effect_measure_groups_are_separate")

    for measure, summaries in measure_buckets.items():
        if measure == "none":
            representation_analyses[measure] = EffectMeasureGroupAnalysis(
                effect_measure="none",
                study_count=len(summaries),
                directional_conflict=False,
                confidence_interval_null_relation_differs=False,
                effect_directions_present=[],
                ci_relations_present=[],
                observations=["non_numerical_or_unsupported_effect_measure_preserved_without_direct_comparison"],
                study_effects=summaries
            )
            continue

        directions = set(s.effect_direction for s in summaries if s.effect_direction != "unknown")
        ci_rels = set(s.ci_relation for s in summaries if s.ci_relation != "not_available")

        # Directional Conflict: higher_effect AND lower_effect present in same measure
        dir_conflict = ("higher_effect" in directions and "lower_effect" in directions)
        
        # Null-relation difference: e.g. above_null AND includes_null present
        ci_differs = len(ci_rels) > 1

        obs: List[str] = []
        if dir_conflict:
            obs.append("study_effects_have_different_directions")
            group_directional_conflict = True

        if ci_differs:
            obs.append("confidence_interval_null_relation_differs")
            group_ci_null_differs = True

        if any(r == "includes_null" for r in ci_rels):
            obs.append("some_confidence_intervals_include_the_null")

        representation_analyses[measure] = EffectMeasureGroupAnalysis(
            effect_measure=measure,
            study_count=len(summaries),
            directional_conflict=dir_conflict,
            confidence_interval_null_relation_differs=ci_differs,
            effect_directions_present=sorted(list(directions)),
            ci_relations_present=sorted(list(ci_rels)),
            observations=obs,
            study_effects=summaries
        )

    return GroupConflictAnalysisResult(
        comparison_key=group.comparison_key,
        total_studies=len(group.study_records),
        effect_measure_groups=effect_groups_present,
        directional_conflict=group_directional_conflict,
        confidence_interval_null_relation_differs=group_ci_null_differs,
        observations=group_observations,
        representation_groups=representation_analyses,
        study_effects=study_summaries
    )


def analyze_multi_study_conflicts(comparison_result: StudyComparisonResult) -> MultiStudyConflictAnalysisPayload:
    """
    Processes a StudyComparisonResult payload to compute deterministic conflict
    analysis across all comparison groups.
    """
    res_map: Dict[str, GroupConflictAnalysisResult] = {}
    for c_key, group in comparison_result.comparison_groups.items():
        res_map[c_key] = analyze_group_conflicts(group)

    return MultiStudyConflictAnalysisPayload(
        total_comparison_groups=len(res_map),
        results=res_map
    )
