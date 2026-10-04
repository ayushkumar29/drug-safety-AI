"""
Evidence Extraction Schema Module
Defines strict Pydantic model for raw clinical evidence extraction with traceability metadata and validators.
Does NOT perform statistical calculations or generate qualitative conclusions.
"""

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ExtractedEvidence(BaseModel):
    """
    Structured representation of raw clinical trial evidence extracted from a source chunk.
    Enforces strict numeric bounds, missing value handling, and traceability metadata.
    """

    # Core Identifiers & Study Metadata
    study_id: str = Field(description="Unique study identifier (e.g. EMP-001)")
    drug: str = Field(description="Primary intervention drug under investigation")
    comparator: str = Field(description="Comparator drug or placebo")
    outcome: str = Field(description="Health outcome or adverse event under investigation")
    study_type: Optional[str] = Field(default="unknown", description="Study design/type")

    # Sample Sizes & Raw Event Counts (Optional[int] preserves missing vs zero distinction)
    treatment_n: Optional[int] = Field(default=None, description="Number of patients in treatment group")
    treatment_events: Optional[int] = Field(default=None, description="Confirmed integer count of adverse events in treatment group")
    control_n: Optional[int] = Field(default=None, description="Number of patients in control/comparator group")
    control_events: Optional[int] = Field(default=None, description="Confirmed integer count of adverse events in control/comparator group")

    # Explicit Observation Metric Types (Distinguishes raw counts from rates, %, or ratios)
    extracted_metric_type: Optional[str] = Field(
        default="unknown",
        description="Observed metric type: 'event_count', 'percentage', 'incidence_rate', 'risk_ratio', 'odds_ratio', 'hazard_ratio', 'sample_size', or 'unknown'"
    )
    raw_reported_treatment_metric: Optional[str] = Field(default=None, description="Original raw metric string reported for treatment group (e.g. '1.6 per 100 PY' or '4.2%')")
    raw_reported_control_metric: Optional[str] = Field(default=None, description="Original raw metric string reported for control group (e.g. '0.9 per 100 PY' or '1.5%')")

    # Clinical Context & Source Information
    dose: Optional[str] = Field(default="unknown", description="Drug dosage details if reported")
    follow_up: Optional[str] = Field(default="unknown", description="Follow-up duration")
    source: Optional[str] = Field(default="unknown", description="Publication/journal or PMC citation source")
    publication_year: Optional[int] = Field(default=None, description="Publication year if reported")

    # Traceability & Citation Metadata
    source_file: Optional[str] = Field(default=None, description="Source PDF filename")
    chunk_id: Optional[str] = Field(default=None, description="Originating RAG chunk ID")
    page_start: Optional[int] = Field(default=None, description="Starting page number")
    page_end: Optional[int] = Field(default=None, description="Ending page number")
    evidence_quote: str = Field(description="Exact verbatim supporting quote from the source text chunk")

    @model_validator(mode="after")
    def validate_evidence_integrity(self) -> "ExtractedEvidence":
        """
        Validate non-empty strings, non-negative event/sample counts, and sample size constraints.
        """
        # 1. Non-empty string checks for mandatory fields
        if not self.study_id or not self.study_id.strip():
            raise ValueError("Validation Error: 'study_id' must be a non-empty string.")
        if not self.drug or not self.drug.strip():
            raise ValueError("Validation Error: 'drug' must be a non-empty string.")
        if not self.comparator or not self.comparator.strip():
            raise ValueError("Validation Error: 'comparator' must be a non-empty string.")
        if not self.outcome or not self.outcome.strip():
            raise ValueError("Validation Error: 'outcome' must be a non-empty string.")
        if not self.evidence_quote or not self.evidence_quote.strip():
            raise ValueError("Validation Error: 'evidence_quote' must be a non-empty string.")

        # 2. Non-negative bounds check
        if self.treatment_n is not None and self.treatment_n < 0:
            raise ValueError(f"Validation Error: 'treatment_n' cannot be negative ({self.treatment_n}).")
        if self.treatment_events is not None and self.treatment_events < 0:
            raise ValueError(f"Validation Error: 'treatment_events' cannot be negative ({self.treatment_events}).")
        if self.control_n is not None and self.control_n < 0:
            raise ValueError(f"Validation Error: 'control_n' cannot be negative ({self.control_n}).")
        if self.control_events is not None and self.control_events < 0:
            raise ValueError(f"Validation Error: 'control_events' cannot be negative ({self.control_events}).")

        # 3. Events cannot exceed sample size
        if (
            self.treatment_events is not None
            and self.treatment_n is not None
            and self.treatment_events > self.treatment_n
        ):
            raise ValueError(
                f"Validation Error: treatment_events ({self.treatment_events}) cannot exceed treatment_n ({self.treatment_n})."
            )

        if (
            self.control_events is not None
            and self.control_n is not None
            and self.control_events > self.control_n
        ):
            raise ValueError(
                f"Validation Error: control_events ({self.control_events}) cannot exceed control_n ({self.control_n})."
            )

        return self
