"""
Grounded LLM Evidence Analysis Report Generator Module.

Assembles deterministic EvidenceAnalysisReport objects into natural language summaries
via LLM, enforced by a strict deterministic grounding validator.
"""

import json
import re
import math
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

try:
    from llm.config import LLMConfig
except ImportError:
    from config import LLMConfig

try:
    from report.report_builder import EvidenceAnalysisReport
except ImportError:
    from report_builder import EvidenceAnalysisReport




class GeneratedReportValidationResult(BaseModel):
    """
    Structured result returned by deterministic report validation.
    """
    valid: bool
    errors: List[str] = Field(default_factory=list)


SYSTEM_GROUNDING_PROMPT = """You are generating a research evidence summary from a structured, precomputed evidence-analysis object.

The supplied JSON is the ONLY source of factual information.

Do not add facts from your prior knowledge.
Do not invent study details, authors, journals, DOIs, URLs, sample sizes, event counts, effect estimates, confidence intervals, p-values, or citations.
Do not perform calculations.
Use numerical values exactly as supplied.
Do not convert one statistical representation into another.
Do not treat HR, RR, OR, and incidence rates as interchangeable.
Do not infer missing values.
Do not create unsupported causal claims.
Do not state that a drug is safe or unsafe.
Do not recommend approval, rejection, treatment, prescribing, or regulatory action.
Describe conflicting evidence neutrally.

Every study-specific factual claim must be traceable to the supplied study result/citation information.

Structure your markdown report using these exact headings:
# Research Question
# Structured Hypothesis
# Evidence Overview
# Study-Level Evidence
# Cross-Study Observations
# Limitations
# Source Traceability

Format study references in text using stable brackets: [study_id, chunk chunk_id, pp. page_range]
Only use study_ids and chunk_ids present in the supplied JSON object.
"""


def extract_numerical_values(text: str) -> List[float]:
    """
    Extracts numerical floats and scientific notation numbers from text.
    Strips out study identifiers, chunk IDs, and page ranges (e.g. pp. 7-10) first.
    """
    clean_text = re.sub(r'\bEMP-\d{3}(?:_CHK_\d{3})?\b', '', text)
    clean_text = re.sub(r'\bpp?\.\s*\d+(?:-\d+)?\b', '', clean_text)
    numbers = []
    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', clean_text)
    for m in matches:
        try:
            val = float(m)
            numbers.append(val)
        except ValueError:
            continue
    return numbers


def is_number_traced(target: float, source_pool: List[float], tol_pct: float = 0.02, abs_tol: float = 1e-3) -> bool:
    """
    Checks if a target numeric value matches any source float within 2% relative or 1e-3 absolute tolerance.
    """
    for s in source_pool:
        diff = abs(target - s)
        # Use relative tolerance if s >= 10.0 to prevent 98.45 matching structural 100.0
        if s >= 10.0:
            max_allowed = s * tol_pct
        else:
            max_allowed = max(abs_tol, s * tol_pct)

        if diff <= max_allowed:
            return True

        if target < 1e-10 and s < 1e-10 and target > 0 and s > 0:
            if abs(math.log10(target) - math.log10(s)) < 0.8:
                return True
    return False


def validate_generated_report(
    generated_text: str,
    source_report: EvidenceAnalysisReport
) -> GeneratedReportValidationResult:
    """
    Deterministically validates that the LLM-generated report text is strictly grounded in source_report.

    Verifies:
    A. Every mentioned study_id exists in source_report.
    B. Every mentioned chunk_id exists in source_report.
    C. No prohibited clinical/regulatory conclusions (safe/unsafe/prescribe/approval).
    D. No unsupported statistical conversion claims (e.g. "converted HR to RR").
    E. Numeric values presented as statistics are traceable to supplied source numbers.
    """
    errors: List[str] = []

    # Collect valid source identifiers
    valid_study_ids = set(r.study_id for r in source_report.study_results)
    valid_chunk_ids = set(r.chunk_id for r in source_report.study_results if r.chunk_id)

    # 1. Study ID Verification
    # Extract bracketed or uppercase study IDs matching pattern EMP-\d{3}
    found_studies = set(re.findall(r'\bEMP-\d{3}\b', generated_text))
    for sid in found_studies:
        if sid not in valid_study_ids:
            errors.append(f"Invalid Study ID cited in report: '{sid}' (not in source evidence).")

    # 2. Chunk ID Verification
    found_chunks = set(re.findall(r'\bEMP-\d{3}_CHK_\d{3}\b', generated_text))
    for cid in found_chunks:
        if cid not in valid_chunk_ids:
            errors.append(f"Invalid Chunk ID cited in report: '{cid}' (not in source evidence).")

    # 3. Prohibited Content & Regulatory Claims
    prohibited_patterns = [
        (r'\bdrug is safe\b', "Contains prohibited claim 'drug is safe'."),
        (r'\bdrug is unsafe\b', "Contains prohibited claim 'drug is unsafe'."),
        (r'\bsafe drug\b', "Contains prohibited claim 'safe drug'."),
        (r'\bunsafe drug\b', "Contains prohibited claim 'unsafe drug'."),
        (r'\bshould be approved\b', "Contains prohibited regulatory recommendation 'should be approved'."),
        (r'\bshould be rejected\b', "Contains prohibited regulatory recommendation 'should be rejected'."),
        (r'\brecommended for treatment\b', "Contains prohibited clinical recommendation."),
        (r'\bshould prescribe\b', "Contains prohibited clinical recommendation."),
        (r'\bshould not prescribe\b', "Contains prohibited clinical recommendation.")
    ]

    # Clean out quotes to avoid false positives on literal quotes from papers
    narrative_text = re.sub(r'".*?"', '', generated_text, flags=re.DOTALL)
    for pattern, err_msg in prohibited_patterns:
        if re.search(pattern, narrative_text, re.IGNORECASE):
            errors.append(err_msg)

    # 4. Prohibited Statistical Conversion Claims
    conversion_patterns = [
        r'converted\s+hazard\s+ratio\s+into\s+risk\s+ratio',
        r'converted\s+(?:hr|hazard ratio|incidence rate)\s+(?:to|into)\s+(?:rr|risk ratio|event count|2x2)',
        r'treated\s+hazard ratio\s+as\s+risk ratio'
    ]
    for pattern in conversion_patterns:
        if re.search(pattern, generated_text, re.IGNORECASE):
            errors.append("Contains prohibited statistical metric conversion claim.")

    # 5. Traceability of Numerical Statistics
    source_floats: List[float] = []
    
    # Collect numbers strictly from quantitative hypothesis, study_results, and summary fields
    for r in source_report.study_results:
        for val_field in [r.treatment_n, r.treatment_events, r.control_n, r.control_events,
                          r.risk_ratio, r.risk_ratio_ci_lower, r.risk_ratio_ci_upper,
                          r.odds_ratio, r.odds_ratio_ci_lower, r.odds_ratio_ci_upper,
                          r.fisher_p_value, r.chi_square_p_value]:
            if val_field is not None:
                source_floats.append(float(val_field))
        if r.page_start is not None:
            source_floats.append(float(r.page_start))
        if r.page_end is not None:
            source_floats.append(float(r.page_end))
        if r.dose:
            for d_num in extract_numerical_values(r.dose):
                source_floats.append(d_num)
        if r.follow_up:
            for f_num in extract_numerical_values(r.follow_up):
                source_floats.append(f_num)
        if r.publication_year is not None:
            source_floats.append(float(r.publication_year))


    summary = source_report.evidence_summary
    for s_field in [summary.total_studies, summary.total_evidence_records,
                    summary.statistical_event_count_records, summary.hazard_ratio_records,
                    summary.risk_ratio_records, summary.odds_ratio_records,
                    summary.incidence_rate_records, summary.records_without_usable_statistics]:
        source_floats.append(float(s_field))

    # Structural/markdown numbers (bullet indices, 95% CIs)
    source_floats.extend([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 95.0, 1.96])

    gen_floats = extract_numerical_values(generated_text)
    untraced_numbers = []
    for g_val in gen_floats:
        # Skip small structural integers
        if g_val in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 95.0]:
            continue
        if not is_number_traced(g_val, source_floats, tol_pct=0.02, abs_tol=1e-3):
            untraced_numbers.append(g_val)

    if untraced_numbers:
        errors.append(f"Report contains untraced or unsupported numerical values: {untraced_numbers[:3]}")

    return GeneratedReportValidationResult(
        valid=(len(errors) == 0),
        errors=errors
    )


def generate_evidence_report(
    report: EvidenceAnalysisReport,
    client: Optional[Any] = None,
    config: Optional[LLMConfig] = None
) -> str:
    """
    Generates grounded natural-language report text from EvidenceAnalysisReport using LLM,
    and enforces validate_generated_report().
    """
    cfg = config or LLMConfig.from_env()

    if cfg.provider == "fallback":
        # Deterministic fallback text generator when no LLM is configured
        fallback_text = (
            f"# Research Question\n{report.research_question}\n\n"
            f"# Structured Hypothesis\n- Drug: {report.hypothesis.drug}\n- Comparator: {report.hypothesis.comparator}\n- Outcome: {report.hypothesis.outcome}\n\n"
            f"# Evidence Overview\nTotal Studies: {report.evidence_summary.total_studies}\n\n"
            f"# Study-Level Evidence\n"
        )
        for r in report.study_results:
            fallback_text += f"## {r.study_id}\n- Dose: {r.dose}\n- Citation: [{r.study_id}, chunk {r.chunk_id or 'N/A'}, pp. {r.page_start or 0}-{r.page_end or 0}]\n"
            if r.risk_ratio is not None:
                fallback_text += f"- Risk Ratio: {r.risk_ratio:.4f}\n"

        fallback_text += "\n# Cross-Study Observations\nSee report payload for details.\n\n# Limitations\n"
        for lim in report.limitations:
            fallback_text += f"- {lim}\n"
        fallback_text += "\n# Source Traceability\nAll evidence preserved from curated corpus.\n"

        val_res = validate_generated_report(fallback_text, report)
        if not val_res.valid:
            raise ValueError(f"Fallback report validation failed: {val_res.errors}")
        return fallback_text

    # Live LLM execution via OpenAI-compatible client
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for LLM report generation.")

    if client is None:
        client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or "lm-studio"
        )

    user_payload_json = report.model_dump_json(indent=2)

    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": SYSTEM_GROUNDING_PROMPT},
            {"role": "user", "content": f"Generate evidence summary report for this payload:\n{user_payload_json}"}
        ],
        temperature=0.0,
        max_tokens=1000
    )

    generated_text = response.choices[0].message.content or ""

    # Validate output
    val_res = validate_generated_report(generated_text, report)
    if not val_res.valid:
        raise ValueError(f"Generated LLM report failed grounding validation: {val_res.errors}")

    return generated_text
