"""
Unit Test Suite for Evidence Validator
Tests deterministic validation rules for integer event counts, decimal incidence rates,
percentages, hazard ratios, events > N, and the exact Task 8.3 failure case.
"""

import sys
from pathlib import Path
from pydantic import ValidationError

# Ensure root directory is on sys.path for imports
sys.path.insert(0, str(Path(".").resolve()))

from llm.evidence import ExtractedEvidence
from validation.evidence_validator import validate_and_sanitize_evidence, is_rate_or_percentage_quote


def test_A_valid_integer_event_counts():
    """Test A: Valid integer event counts are preserved."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=8417,
        treatment_events=288,
        control_n=4203,
        control_events=38,
        evidence_quote="A total of 288 patients in empagliflozin group and 38 in placebo group experienced genital infections."
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events == 288
    assert validated.control_events == 38
    assert validated.extracted_metric_type == "event_count"
    print("  [OK] Test A (Valid integer event counts): PASS")


def test_B_decimal_incidence_rates():
    """Test B: Decimal incidence rates in quote cause event counts to be reset to None."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=4221,
        treatment_events=16,  # Incorrectly parsed from 1.6 rate
        control_n=4203,
        control_events=9,   # Incorrectly parsed from 0.9 rate
        evidence_quote="Incidence rates of genital infection were 1.6 vs 0.9 per 100 patient-years."
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None, f"Expected None, got {validated.treatment_events}"
    assert validated.control_events is None, f"Expected None, got {validated.control_events}"
    assert validated.extracted_metric_type == "incidence_rate"
    assert validated.raw_reported_treatment_metric == "1.6"
    assert validated.raw_reported_control_metric == "0.9"
    print("  [OK] Test B (Decimal incidence rates): PASS (Event counts reset to None)")


def test_C_percentages():
    """Test C: Quote with percentage signs resets misread event counts to None."""
    ev = ExtractedEvidence(
        study_id="EMP-004",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=1000,
        treatment_events=42,  # Parsed from 4.2%
        control_n=1000,
        control_events=15,  # Parsed from 1.5%
        evidence_quote="Genital infections occurred in 4.2% vs 1.5% of patients."
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None
    assert validated.control_events is None
    assert validated.extracted_metric_type == "percentage"
    print("  [OK] Test C (Percentages): PASS (Event counts reset to None)")


def test_D_hazard_ratios():
    """Test D: Quote with Hazard Ratio / Risk Ratio metrics resets misread event counts to None."""
    ev = ExtractedEvidence(
        study_id="EMP-004",
        drug="empagliflozin",
        comparator="placebo",
        outcome="3-point MACE",
        treatment_events=86,
        control_events=74,
        evidence_quote="HR 0.86; 95% CI 0.74 to 0.99; p=0.04 for 3-point MACE reduction."
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None
    assert validated.control_events is None
    assert validated.extracted_metric_type == "hazard_ratio"
    print("  [OK] Test D (Hazard Ratios): PASS (Event counts reset to None)")


def test_E_missing_counts():
    """Test E: Missing counts remain None."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=4221,
        control_n=4203,
        evidence_quote="Genital infection incidence was higher with empagliflozin."
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None
    assert validated.control_events is None
    print("  [OK] Test E (Missing counts): PASS")


def test_F_events_greater_than_N():
    """Test F: Events > N raises validation error."""
    try:
        ev = ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=100,
            treatment_events=150,  # 150 > 100
            evidence_quote="Sample text"
        )
        validate_and_sanitize_evidence(ev)
        assert False, "Failed to reject events > N"
    except (ValueError, ValidationError):
        print("  [OK] Test F (Events > N): PASS (Correctly rejected)")


def test_G_decimal_numbers_as_event_counts():
    """Test G: Quote with decimal numbers resets event counts."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_events=3,
        control_events=1,
        evidence_quote="Incidence was 3.5 per 100 PY vs 1.2 per 100 PY"
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None
    assert validated.control_events is None
    print("  [OK] Test G (Decimal numbers as event counts): PASS")


def test_H_exact_task_8_3_failure_case():
    """
    Test H: Exact Task 8.3 failure case.
    Input:
      treatment_n=4221, control_n=4203
      treatment_events=9, control_events=1
      quote="Events consistent with genital infectionb 67 1.6 0.9"
    Expected:
      Validator detects decimals 1.6 and 0.9 in rate quote, safely resets
      treatment_events and control_events to None, sets extracted_metric_type="incidence_rate",
      and preserves quote and sample sizes intact!
    """
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="risk of genital infection",
        treatment_n=4221,
        treatment_events=9,
        control_n=4203,
        control_events=1,
        dose="10 mg",
        evidence_quote="Events consistent with genital infectionb 67 1.6 0.9"
    )

    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None, f"Expected None for treatment_events, got {validated.treatment_events}"
    assert validated.control_events is None, f"Expected None for control_events, got {validated.control_events}"
    assert validated.treatment_n == 4221
    assert validated.control_n == 4203
    assert validated.extracted_metric_type == "incidence_rate"
    assert validated.raw_reported_treatment_metric == "1.6"
    assert validated.raw_reported_control_metric == "0.9"
    assert validated.evidence_quote == "Events consistent with genital infectionb 67 1.6 0.9"

    print("  [OK] Test H (Exact Task 8.3 Failure Case): PASS!")
    print("       -> Misread event counts (9, 1) SAFELY CONVERTED TO None!")
    print("       -> Preserved treatment_n=4221, control_n=4203, rates ('1.6', '0.9') and original quote intact.")


def test_step5_1_mixed_table_valid_counts():
    """Test Step 5.1: Mixed table quote with valid integer event counts preserved."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=4221,
        treatment_events=259,
        control_n=4203,
        control_events=67,
        extracted_metric_type="event_count",
        evidence_quote="Events consistent with genital infection 67 1.6 0.9 259 6.1 3.5 per 100 patient-years"
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events == 259
    assert validated.control_events == 67
    assert validated.extracted_metric_type == "event_count"
    print("  [OK] Test Step 5.1 (Mixed table valid counts): PASS")


def test_step5_2_decimal_incorrectly_extracted_as_events():
    """Test Step 5.2: Decimal numbers (6.1, 1.6) incorrectly extracted as events are sanitized."""
    ev = ExtractedEvidence.model_construct(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=4221,
        treatment_events=6,  # Misread from 6.1 float
        control_n=4203,
        control_events=1,  # Misread from 1.6 float
        extracted_metric_type="event_count",
        evidence_quote="Events consistent with genital infection rate 1.6 per 100 PY vs 6.1 per 100 PY"
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None
    assert validated.control_events is None
    assert validated.extracted_metric_type == "incidence_rate"
    print("  [OK] Test Step 5.2 (Decimal misread as events): PASS")


def test_step5_3_incidence_rate_only_evidence():
    """Test Step 5.3: Incidence rate only evidence (no event counts)."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=4221,
        control_n=4203,
        evidence_quote="Rate/100 patient-years: treatment 3.5, placebo 0.9"
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None
    assert validated.control_events is None
    assert validated.extracted_metric_type == "incidence_rate"
    print("  [OK] Test Step 5.3 (Incidence rate only evidence): PASS")


def test_step5_4_percentage_only_evidence():
    """Test Step 5.4: Percentage only evidence (no event counts)."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=4221,
        control_n=4203,
        evidence_quote="Genital infection occurred in 6.1% vs 1.6%."
    )
    validated = validate_and_sanitize_evidence(ev)

    assert validated.treatment_events is None
    assert validated.control_events is None
    assert validated.extracted_metric_type == "percentage"
    print("  [OK] Test Step 5.4 (Percentage only evidence): PASS")


def test_step5_5_existing_misread_regression():
    """Test Step 5.5: Incorrect 9/1 counts in quote with 67 and 259 are rejected/sanitized."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        treatment_n=4221,
        treatment_events=9,
        control_n=4203,
        control_events=1,
        extracted_metric_type="event_count",
        evidence_quote="Events consistent with genital infectionb 67 1.6 0.9 259 6.1 3.5"
    )
    validated = validate_and_sanitize_evidence(ev)

    # 9 and 1 do not appear in the quote as standalone word boundary integers, so they must be reset!
    assert validated.treatment_events is None
    assert validated.control_events is None
    assert validated.extracted_metric_type == "incidence_rate"
    print("  [OK] Test Step 5.5 (Misread 9/1 count rejection): PASS")


def run_all_validation_tests():
    print("=" * 80)
    print("TASK 8.4: STRENGTHENED EVIDENCE VALIDATOR UNIT TESTS")
    print("=" * 80)
    test_A_valid_integer_event_counts()
    test_B_decimal_incidence_rates()
    test_C_percentages()
    test_D_hazard_ratios()
    test_E_missing_counts()
    test_F_events_greater_than_N()
    test_G_decimal_numbers_as_event_counts()
    test_H_exact_task_8_3_failure_case()
    test_step5_1_mixed_table_valid_counts()
    test_step5_2_decimal_incorrectly_extracted_as_events()
    test_step5_3_incidence_rate_only_evidence()
    test_step5_4_percentage_only_evidence()
    test_step5_5_existing_misread_regression()
    print("=" * 80)
    print("ALL STRENGTHENED EVIDENCE VALIDATOR TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_validation_tests()
