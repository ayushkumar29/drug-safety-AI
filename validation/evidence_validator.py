"""
Evidence Validator Module
Provides deterministic post-extraction validation and sanitization for ExtractedEvidence objects.
Ensures rates, percentages, and ratio statistics are NEVER misclassified as raw integer event counts.
Does NOT use an LLM or modify RAG vector store.
"""

import re
from typing import Dict, Any, Tuple
from llm.evidence import ExtractedEvidence


def is_rate_or_percentage_quote(quote: str) -> Tuple[bool, str]:
    """
    Deterministically analyze an evidence quote to check if numbers represent
    percentages, incidence rates per 100 patient-years, or hazard/risk ratios.

    Returns (is_non_count_metric, detected_metric_type)
    """
    if not quote:
        return False, "unknown"

    q_lower = quote.lower()

    # 1. Check for ratio metrics (HR, RR, OR, CI, p-value)
    if (
        "hazard ratio" in q_lower
        or "risk ratio" in q_lower
        or "odds ratio" in q_lower
        or re.search(r"\b(hr|rr|or|ci)\b", q_lower)
        or "p=" in q_lower
        or "p<" in q_lower
    ):
        return True, "hazard_ratio"

    # 2. Check for incidence rate indicators
    if (
        "per 100" in q_lower
        or "patient-years" in q_lower
        or "patient years" in q_lower
        or "incidence rate" in q_lower
        or "rate per" in q_lower
    ):
        return True, "incidence_rate"

    # 3. Check for percentage signs or words
    if "%" in quote or "percent" in q_lower or "percentage" in q_lower:
        return True, "percentage"

    # 4. Check for decimal numbers in table rows/quotes associated with rates (e.g. "1.6 0.9", "3.5", "0.9")
    if re.search(r"\b\d+\.\d+\b", quote):
        return True, "incidence_rate"

    return False, "event_count"


def validate_and_sanitize_evidence(evidence: ExtractedEvidence) -> ExtractedEvidence:
    """
    Deterministically validate and sanitize an ExtractedEvidence object.

    Rules:
    1. Validates integer types for event counts.
    2. Inspects evidence_quote for decimals, rates, percentages, and ratios.
    3. Handles mixed metric tables (where quote contains integers alongside decimal rates/percentages):
       If treatment_events and control_events are valid integers present in the quote,
       and are supported by surrounding context (e.g. 259 and 67 appear in quote and 259 <= N),
       preserve extracted_metric_type = "event_count".
    4. If event counts were populated from misread decimal rate values (e.g. 6.1 or 1.6),
       safely resets treatment_events and control_events to None while setting extracted_metric_type.
    5. Validates that treatment_events <= treatment_n and control_events <= control_n.
    6. Preserves original evidence_quote and study traceability intact.
    """
    if not evidence:
        raise ValueError("Evidence object cannot be None.")

    quote = evidence.evidence_quote or ""
    is_non_count, detected_type = is_rate_or_percentage_quote(quote)

    treatment_ev = evidence.treatment_events
    control_ev = evidence.control_events

    # Extract raw reported decimal metrics if present
    decimals = re.findall(r"\b\d+\.\d+\b", quote)
    if decimals and len(decimals) >= 2:
        evidence.raw_reported_treatment_metric = decimals[0]
        evidence.raw_reported_control_metric = decimals[1]
    elif decimals and len(decimals) == 1:
        evidence.raw_reported_treatment_metric = decimals[0]

    # Verify if extracted event counts are valid integers present in quote
    has_valid_event_counts = False
    if (
        isinstance(treatment_ev, int)
        and isinstance(control_ev, int)
        and treatment_ev >= 0
        and control_ev >= 0
    ):
        # Verify that the extracted integer counts actually appear in the quote
        str_tr = str(treatment_ev)
        str_ctrl = str(control_ev)
        # Check standalone integer token match in quote (not followed/preceded by decimal point or %)
        tr_in_quote = bool(re.search(rf"(?<![\d\.])\b{str_tr}\b(?![\.\d%])", quote))
        ctrl_in_quote = bool(re.search(rf"(?<![\d\.])\b{str_ctrl}\b(?![\.\d%])", quote))

        if tr_in_quote and ctrl_in_quote:
            has_valid_event_counts = True

    if has_valid_event_counts and evidence.extracted_metric_type == "event_count":
        # Mixed table case: Quote contains decimal rates/percentages alongside valid integer counts.
        # Preserve event counts and metric_type = event_count!
        pass
    elif is_non_count:
        evidence.extracted_metric_type = detected_type
        # Reset misread event counts if they don't represent genuine integer counts from the quote
        if treatment_ev is not None or control_ev is not None:
            print(f"[EVIDENCE VALIDATOR] Resetting misread event counts (tr={treatment_ev}, ctrl={control_ev}) to None for non-count quote metric '{detected_type}': \"{quote[:80]}...\"")
            evidence.treatment_events = None
            evidence.control_events = None
    else:
        if evidence.extracted_metric_type == "unknown":
            evidence.extracted_metric_type = "event_count"

    # Strict bounds checks
    if evidence.treatment_events is not None and evidence.treatment_n is not None:
        if evidence.treatment_events > evidence.treatment_n:
            raise ValueError(f"Validation Error: treatment_events ({evidence.treatment_events}) > treatment_n ({evidence.treatment_n})")

    if evidence.control_events is not None and evidence.control_n is not None:
        if evidence.control_events > evidence.control_n:
            raise ValueError(f"Validation Error: control_events ({evidence.control_events}) > control_n ({evidence.control_n})")

    return evidence
