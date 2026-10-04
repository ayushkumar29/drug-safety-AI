"""
Study Lineage and Provenance Module.

Defines structured Pydantic models for tracking the lineage of clinical publications,
underlying clinical trials/datasets, and extracted evidence records.
Distinguishes publication/document identity from underlying study/dataset identity.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


IndependenceStatusType = Literal[
    "independent_primary",
    "overlapping_primary",
    "pooled_analysis",
    "meta_analysis",
    "review",
    "unknown"
]


class StudyLineage(BaseModel):
    """
    Metadata representation of a publication's provenance and underlying study lineage.
    """
    publication_id: str = Field(..., description="Unique publication/document identifier (e.g. EMP-001)")
    source_file: str = Field(..., description="Source document filename (e.g. EMP-001.pdf)")
    publication_type: str = Field(..., description="Classification of the publication (e.g. Pooled Analysis)")
    underlying_study_ids: List[str] = Field(default_factory=list, description="Identifiers of underlying clinical trials or trial datasets")
    primary_study: bool = Field(..., description="True if this publication represents a single primary clinical trial")
    overlap_group_id: Optional[str] = Field(None, description="Group identifier for related trial/publication families")
    independence_status: IndependenceStatusType = Field("unknown", description="Categorical independence status")
    lineage_notes: str = Field("", description="Detailed provenance notes")

    @field_validator("publication_id")
    @classmethod
    def validate_publication_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("publication_id cannot be empty or whitespace.")
        return v.strip()


class EvidenceLineage(BaseModel):
    """
    Lineage metadata payload attached to an ExtractedEvidence record.
    """
    publication_id: str = Field(..., description="Publication identifier from which evidence was extracted")
    underlying_study_ids: List[str] = Field(default_factory=list, description="Underlying trial datasets included in this evidence")
    overlap_group_id: Optional[str] = Field(None, description="Group identifier for related trial/publication families")
    independence_status: IndependenceStatusType = Field("unknown", description="Categorical independence status")

    @field_validator("publication_id")
    @classmethod
    def validate_publication_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("publication_id cannot be empty or whitespace.")
        return v.strip()
