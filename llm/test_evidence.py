"""
Unit Test Suite for ExtractedEvidence Pydantic Schema
Tests schema validation rules, boundary conditions, missing numeric value preservation, and error handling.
"""

from pydantic import ValidationError
try:
    from llm.evidence import ExtractedEvidence
except ImportError:
    from evidence import ExtractedEvidence



def test_valid_evidence():
    """Test creation of valid complete evidence object."""
    ev = ExtractedEvidence(
        study_id="EMP-001",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        study_type="pooled phase 1-3 trials",
        treatment_n=8417,
        treatment_events=288,
        control_n=4203,
        control_events=38,
        dose="10mg and 25mg",
        follow_up="7369-7782 patient-years",
        source="Springer / Current Medical Research and Opinion",
        publication_year=2017,
        source_file="EMP-001.pdf",
        chunk_id="EMP-001_CHK_008",
        page_start=16,
        page_end=17,
        evidence_quote="Events consistent with genital infection occurred more frequently in participants treated with empagliflozin 10 and 25 mg (3.5 and 3.4/100 patient-years) than placebo (0.9/100 patient-years)."
    )

    assert ev.study_id == "EMP-001"
    assert ev.treatment_n == 8417
    assert ev.treatment_events == 288
    assert ev.control_n == 4203
    assert ev.control_events == 38
    assert ev.publication_year == 2017
    assert ev.chunk_id == "EMP-001_CHK_008"
    print("  [OK] Valid complete evidence test passed.")


def test_missing_numeric_values_preserved():
    """Test that missing numeric values remain None and are not converted to 0."""
    ev = ExtractedEvidence(
        study_id="EMP-002",
        drug="empagliflozin",
        comparator="placebo",
        outcome="genital infection",
        evidence_quote="Sample text mentioning genital infections."
    )

    assert ev.treatment_n is None, f"Expected None, got {ev.treatment_n}"
    assert ev.treatment_events is None, f"Expected None, got {ev.treatment_events}"
    assert ev.control_n is None, f"Expected None, got {ev.control_n}"
    assert ev.control_events is None, f"Expected None, got {ev.control_events}"
    assert ev.publication_year is None, f"Expected None, got {ev.publication_year}"
    print("  [OK] Missing numeric values preserved as None (not converted to 0).")


def test_zero_events_valid():
    """Test that explicit zero events (0) are valid and distinct from None."""
    ev = ExtractedEvidence(
        study_id="EMP-003",
        drug="empagliflozin",
        comparator="placebo",
        outcome="ketoacidosis",
        treatment_n=500,
        treatment_events=0,
        control_n=500,
        control_events=0,
        evidence_quote="No cases of ketoacidosis were observed in either group."
    )

    assert ev.treatment_events == 0, f"Expected 0, got {ev.treatment_events}"
    assert ev.control_events == 0, f"Expected 0, got {ev.control_events}"
    print("  [OK] Explicit zero events (0) test passed.")


def test_negative_events_rejected():
    """Test that negative event counts are rejected by validator."""
    try:
        ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=1000,
            treatment_events=-5,
            evidence_quote="Sample text quote"
        )
        assert False, "Failed to reject negative treatment_events"
    except (ValueError, ValidationError) as e:
        print("  [OK] Successfully rejected negative treatment_events.")


def test_negative_sample_size_rejected():
    """Test that negative sample sizes are rejected."""
    try:
        ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=-100,
            treatment_events=10,
            evidence_quote="Sample text quote"
        )
        assert False, "Failed to reject negative treatment_n"
    except (ValueError, ValidationError) as e:
        print("  [OK] Successfully rejected negative treatment_n.")


def test_events_exceeding_sample_size_rejected():
    """Test that event counts exceeding sample size (events > N) are rejected."""
    try:
        ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            treatment_n=100,
            treatment_events=150,  # 150 > 100
            evidence_quote="Sample text quote"
        )
        assert False, "Failed to reject treatment_events > treatment_n"
    except (ValueError, ValidationError) as e:
        print("  [OK] Successfully rejected treatment_events > treatment_n.")


def test_missing_study_id_rejected():
    """Test that empty study_id is rejected."""
    try:
        ExtractedEvidence(
            study_id="",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            evidence_quote="Sample text quote"
        )
        assert False, "Failed to reject empty study_id"
    except (ValueError, ValidationError) as e:
        print("  [OK] Successfully rejected empty study_id.")


def test_missing_evidence_quote_rejected():
    """Test that empty evidence_quote is rejected."""
    try:
        ExtractedEvidence(
            study_id="EMP-001",
            drug="empagliflozin",
            comparator="placebo",
            outcome="genital infection",
            evidence_quote="   "  # Whitespace only
        )
        assert False, "Failed to reject empty evidence_quote"
    except (ValueError, ValidationError) as e:
        print("  [OK] Successfully rejected empty evidence_quote.")


def run_all_schema_tests():
    print("=" * 80)
    print("TASK 8.1: EVIDENCE EXTRACTION SCHEMA UNIT TESTS")
    print("=" * 80)
    test_valid_evidence()
    test_missing_numeric_values_preserved()
    test_zero_events_valid()
    test_negative_events_rejected()
    test_negative_sample_size_rejected()
    test_events_exceeding_sample_size_rejected()
    test_missing_study_id_rejected()
    test_missing_evidence_quote_rejected()
    print("=" * 80)
    print("ALL EVIDENCE SCHEMA TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_schema_tests()
