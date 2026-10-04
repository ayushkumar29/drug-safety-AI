"""
Evidence Selection and Overlap-Safe Analysis Module.

Provides deterministic classification of evidence records to prevent overlapping
pooled/meta-analytic publications from being double-counted in independent statistical analysis.
Does NOT alter quantitative values or delete evidence records.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator

try:
    from analysis.study_lineage import StudyLineage, EvidenceLineage
    from analysis.lineage_resolver import get_study_lineage, attach_evidence_lineage
except ImportError:
    from study_lineage import StudyLineage, EvidenceLineage
    from lineage_resolver import get_study_lineage, attach_evidence_lineage


SelectionStatusType = Literal[
    "retained_for_independent_analysis",
    "excluded_due_to_overlap",
    "context_only"
]


class EvidenceSelection(BaseModel):
    """
    Selection decision payload attached to an evidence record.
    """
    publication_id: str = Field(..., description="Publication identifier (e.g. EMP-005)")
    underlying_study_ids: List[str] = Field(default_factory=list, description="Underlying trial dataset IDs")
    selection_status: SelectionStatusType = Field(..., description="Deterministic selection status")
    selection_reason: str = Field(..., description="Rationale for the selection decision")
    primary_source: bool = Field(..., description="True if from a primary RCT publication")
    overlaps_with: List[str] = Field(default_factory=list, description="List of publication IDs this evidence overlaps with")

    @field_validator("publication_id")
    @classmethod
    def validate_publication_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("publication_id cannot be empty or whitespace.")
        return v.strip()


class EvidenceSelectionResult(BaseModel):
    """
    Complete selection result wrapping both annotated evidence records and the subset
    retained for independent statistical analysis.
    """
    all_selected_evidence: List[Dict[str, Any]] = Field(..., description="All evidence records annotated with EvidenceSelection metadata")
    independent_evidence: List[Any] = Field(..., description="Subset of ExtractedEvidence objects retained for independent statistical analysis")
    retained_count: int = Field(0, description="Count of retained independent evidence records")
    excluded_overlap_count: int = Field(0, description="Count of evidence records excluded due to overlap")
    context_only_count: int = Field(0, description="Count of context-only evidence records")


def get_technical_priority(lineage: StudyLineage) -> int:
    """
    Returns technical selection priority rank (lower number = higher priority for dataset representation).
    1: Primary RCT publication
    2: Direct primary-study quantitative evidence
    3: Pooled analysis
    4: Meta-analysis
    5: Narrative review / unknown
    """
    if lineage.primary_study and lineage.independence_status == "independent_primary":
        return 1
    if lineage.independence_status == "independent_primary":
        return 2
    if lineage.independence_status == "pooled_analysis":
        return 3
    if lineage.independence_status == "meta_analysis":
        return 4
    return 5


def select_independent_evidence(evidence_records: List[Any]) -> EvidenceSelectionResult:
    """
    Scans a list of ExtractedEvidence records and deterministically selects records for
    independent statistical analysis based on publication lineage and underlying study datasets.

    Prevents double-counting of overlapping trials while preserving all original evidence objects.
    """
    if not evidence_records:
        return EvidenceSelectionResult(
            all_selected_evidence=[],
            independent_evidence=[],
            retained_count=0,
            excluded_overlap_count=0,
            context_only_count=0
        )

    # Attach lineage and technical priority to each record
    annotated = []
    primary_rct_datasets = set()

    for rec in evidence_records:
        pub_id = getattr(rec, "study_id", None) or getattr(rec, "source_file", "unknown").split(".")[0]
        lineage = get_study_lineage(pub_id)
        priority = get_technical_priority(lineage)

        if lineage.primary_study and lineage.independence_status == "independent_primary":
            for uid in lineage.underlying_study_ids:
                primary_rct_datasets.add(uid)

        annotated.append({
            "record": rec,
            "pub_id": pub_id,
            "lineage": lineage,
            "priority": priority
        })

    all_selected = []
    independent_ev = []
    retained_cnt = 0
    excluded_cnt = 0
    context_cnt = 0

    for item in annotated:
        rec = item["record"]
        pub_id = item["pub_id"]
        lineage = item["lineage"]
        priority = item["priority"]

        # Calculate overlapping publication list
        overlaps = [
            other["pub_id"] for other in annotated
            if other["pub_id"] != pub_id and (
                set(lineage.underlying_study_ids).intersection(set(other["lineage"].underlying_study_ids)) or
                (lineage.overlap_group_id and lineage.overlap_group_id == other["lineage"].overlap_group_id)
            )
        ]

        # Determine selection status
        if lineage.independence_status == "review" or "review" in lineage.publication_type.lower():
            status = "context_only"
            reason = f"Narrative review publication '{pub_id}' retained as qualitative context only."
        elif not lineage.primary_study and any(uid in primary_rct_datasets for uid in lineage.underlying_study_ids):
            status = "excluded_due_to_overlap"
            reason = f"Publication '{pub_id}' ({lineage.publication_type}) excluded from independent statistical analysis to prevent double-counting. Underlying trial dataset(s) {lineage.underlying_study_ids} are represented by a primary RCT publication."
        elif lineage.primary_study and lineage.independence_status == "independent_primary":
            status = "retained_for_independent_analysis"
            reason = f"Publication '{pub_id}' retained as an independent primary RCT clinical trial observation."
        elif not lineage.primary_study and (lineage.independence_status in ["pooled_analysis", "meta_analysis"]):
            # If no primary RCT is present in the dataset, check if this is the highest priority secondary
            status = "excluded_due_to_overlap"
            reason = f"Secondary publication '{pub_id}' ({lineage.publication_type}) excluded due to potential overlap with primary trial datasets."
        else:
            status = "retained_for_independent_analysis"
            reason = f"Publication '{pub_id}' retained for independent analysis."

        sel = EvidenceSelection(
            publication_id=pub_id,
            underlying_study_ids=lineage.underlying_study_ids,
            selection_status=status,
            selection_reason=reason,
            primary_source=lineage.primary_study,
            overlaps_with=overlaps
        )

        rec_dict = rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
        rec_dict["selection"] = sel.model_dump()
        all_selected.append(rec_dict)

        if status == "retained_for_independent_analysis":
            independent_ev.append(rec)
            retained_cnt += 1
        elif status == "excluded_due_to_overlap":
            excluded_cnt += 1
        elif status == "context_only":
            context_cnt += 1

    return EvidenceSelectionResult(
        all_selected_evidence=all_selected,
        independent_evidence=independent_ev,
        retained_count=retained_cnt,
        excluded_overlap_count=excluded_cnt,
        context_only_count=context_cnt
    )
