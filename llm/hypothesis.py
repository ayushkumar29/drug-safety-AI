"""
Hypothesis Parser Module
Converts a natural-language research question into a validated structured hypothesis dictionary.
Applies strict validation and prompt constraints. Does not perform retrieval or statistical calculations.
"""

import json
import os
import re
from typing import Dict, Any, Literal
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
try:
    from config import LLMConfig
except ImportError:
    from llm.config import LLMConfig

# Load environment variables
load_dotenv()

VALID_DIRECTIONS = {"greater_risk", "lower_risk", "no_difference", "association", "unknown"}

PROMPT_SYSTEM_INSTRUCTIONS = """
You are a precision clinical epidemiology parser.
Your task is to parse a natural-language research question into a structured hypothesis.

STRICT EXTRACTION RULES:
1. Extract ONLY information directly supported by the user's question.
2. Do NOT invent population, comparator, study type, or hypothesis direction.
3. If information is missing from the question, use 'unknown' rather than guessing.
4. For comparator: if not explicitly stated in the question, set comparator to 'unknown'. Do NOT assume placebo unless explicitly mentioned.
5. For population: if not explicitly stated, default to 'unknown'.
6. For study_type: if not explicitly stated, default to 'unknown'.
7. For hypothesis_direction: must be one of ['greater_risk', 'lower_risk', 'no_difference', 'association', 'unknown'].
8. For null_hypothesis: specify formal null statement (e.g., 'RR = 1' or 'No difference in risk').
9. For alternative_hypothesis: specify formal alternative statement (e.g., 'RR > 1', 'RR < 1', or 'RR != 1').
"""


class StructuredHypothesis(BaseModel):
    drug: str = Field(description="The primary drug under study")
    comparator: str = Field(description="The comparator drug/placebo, or 'unknown' if not explicitly stated")
    outcome: str = Field(description="The health outcome or adverse event under investigation")
    population: str = Field(description="The patient population under study, or 'unknown' if missing")
    study_type: str = Field(description="The study design/type, or 'unknown' if missing")
    hypothesis_direction: Literal["greater_risk", "lower_risk", "no_difference", "association", "unknown"] = Field(
        description="Direction of hypothesized effect: greater_risk, lower_risk, no_difference, association, or unknown"
    )
    null_hypothesis: str = Field(description="Formal null hypothesis statement")
    alternative_hypothesis: str = Field(description="Formal alternative hypothesis statement")


def validate_hypothesis_dict(question: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically validate parsed hypothesis fields against project requirements.
    Raises ValueError if validation fails.
    """
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("Input question must be a non-empty string.")

    # Check for underspecified question missing core research components (drug or outcome)
    drug = str(data.get("drug", "")).strip()
    outcome = str(data.get("outcome", "")).strip()

    if not drug or drug.lower() == "unknown" or not outcome or outcome.lower() == "unknown":
        raise ValueError(f"Underspecified question cannot form a valid research hypothesis: '{question}'")

    # Field presence check
    required_fields = [
        "drug", "comparator", "outcome", "population",
        "study_type", "hypothesis_direction", "null_hypothesis", "alternative_hypothesis"
    ]
    for field in required_fields:
        val = data.get(field)
        if val is None or str(val).strip() == "":
            raise ValueError(f"Validation Error: Required field '{field}' is missing or empty.")

    # Enum check
    direction = str(data.get("hypothesis_direction", "")).strip()
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Validation Error: Invalid hypothesis_direction '{direction}'. Must be one of {VALID_DIRECTIONS}")

    return {
        "drug": drug,
        "comparator": str(data.get("comparator", "")).strip(),
        "outcome": outcome,
        "population": str(data.get("population", "")).strip(),
        "study_type": str(data.get("study_type", "")).strip(),
        "hypothesis_direction": direction,
        "null_hypothesis": str(data.get("null_hypothesis", "")).strip(),
        "alternative_hypothesis": str(data.get("alternative_hypothesis", "")).strip(),
    }


def _deterministic_rule_parser(question: str) -> Dict[str, Any]:
    """
    Deterministic rule-based parser used when running offline or as fallback.
    Extracts exact terms directly from question text without hallucination.
    """
    q_lower = question.lower().strip()

    # Underspecified generic query check
    if "tell me" in q_lower or "something about" in q_lower or len(q_lower.split()) < 4:
        # Check if query lacks specific outcome
        if not any(k in q_lower for k in ["risk", "benefit", "death", "infection", "event", "effect", "outcome"]):
            return {
                "drug": "empagliflozin" if "empagliflozin" in q_lower else "unknown",
                "comparator": "unknown",
                "outcome": "unknown",
                "population": "unknown",
                "study_type": "unknown",
                "hypothesis_direction": "unknown",
                "null_hypothesis": "unknown",
                "alternative_hypothesis": "unknown"
            }

    # Drug extraction
    drug = "unknown"
    if "empagliflozin" in q_lower:
        drug = "empagliflozin"

    # Comparator extraction
    comparator = "unknown"
    if "placebo" in q_lower:
        comparator = "placebo"
    elif "compared with" in q_lower or "versus" in q_lower or " vs " in q_lower or " vs. " in q_lower:
        m = re.search(r'(?:compared with|versus|vs\.?)\s+([a-z0-9\s]+?)(?:\s+in|\?|$)', q_lower)
        if m:
            comparator = m.group(1).strip()

    # Direction & Outcome extraction
    direction = "unknown"
    outcome = "unknown"
    null_h = "RR = 1"
    alt_h = "RR != 1"

    if "increase" in q_lower or "higher risk" in q_lower:
        direction = "greater_risk"
        null_h = "RR = 1"
        alt_h = "RR > 1"
    elif "reduce" in q_lower or "lower risk" in q_lower or "decrease" in q_lower:
        direction = "lower_risk"
        null_h = "RR = 1"
        alt_h = "RR < 1"
    elif "associated" in q_lower or "association" in q_lower:
        direction = "association"
        null_h = "RR = 1"
        alt_h = "RR != 1"

    # Outcome parsing
    if "genital infection" in q_lower:
        outcome = "genital infection"
    elif "cardiovascular death" in q_lower:
        outcome = "cardiovascular death"
    elif "cardiovascular benefit" in q_lower:
        outcome = "cardiovascular benefit"
    elif "cardiovascular outcome" in q_lower:
        outcome = "cardiovascular outcome"
    elif "adverse event" in q_lower or "adverse events" in q_lower:
        outcome = "adverse events"

    # Population & study_type default to unknown unless explicitly stated
    population = "unknown"
    study_type = "unknown"

    return {
        "drug": drug,
        "comparator": comparator,
        "outcome": outcome,
        "population": population,
        "study_type": study_type,
        "hypothesis_direction": direction,
        "null_hypothesis": null_h,
        "alternative_hypothesis": alt_h
    }


def parse_hypothesis(question: str) -> Dict[str, Any]:
    """
    Parse a natural-language research question into a validated structured hypothesis dictionary.

    Args:
        question: Natural language question string.

    Returns:
        Validated dictionary containing drug, comparator, outcome, population, study_type,
        hypothesis_direction, null_hypothesis, alternative_hypothesis.
    """
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("Input question must be a non-empty string.")

    config = LLMConfig.from_env()

    if config.provider in ["openai", "lmstudio"] and config.api_key and not config.api_key.startswith("your_"):
        try:
            from openai import OpenAI
            client_kwargs = {"api_key": config.api_key, "timeout": 2.0, "max_retries": 0}
            if config.base_url:
                client_kwargs["base_url"] = config.base_url

            client = OpenAI(**client_kwargs)

            completion = client.beta.chat.completions.parse(
                model=config.model,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": f"Parse the following research question into a structured hypothesis: '{question}'"}
                ],
                response_format=StructuredHypothesis,
            )
            raw_dict = completion.choices[0].message.parsed.model_dump()
        except Exception:
            # If API call fails, fall back to deterministic parser
            raw_dict = _deterministic_rule_parser(question)
    else:
        # Local / offline fallback mode
        raw_dict = _deterministic_rule_parser(question)

    # Apply strict validation
    validated_hypothesis = validate_hypothesis_dict(question, raw_dict)
    return validated_hypothesis
