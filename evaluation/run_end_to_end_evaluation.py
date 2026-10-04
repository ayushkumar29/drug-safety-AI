"""
End-to-End Evaluation Runner (evaluation/run_end_to_end_evaluation.py)

Executes benchmark cases from evaluation/end_to_end_cases.json through the live production pipeline,
evaluates pipeline outputs against expectations, records failure modes, and outputs
structured results to data/processed/evaluation/end_to_end_evaluation.json.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from pipeline.pipeline import run_analysis, PipelineResult
from evaluation.artifact_generator import build_artifact


CASES_FILE = Path("evaluation/end_to_end_cases.json")
OUTPUT_DIR = Path("data/processed/evaluation")
OUTPUT_FILE = OUTPUT_DIR / "end_to_end_evaluation.json"


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    case_id = case["case_id"]
    question = case["question"]
    category = case.get("category", "unknown")
    exp_ev = case.get("evidence_expectation", {})
    exp_stat = case.get("statistical_expectation", {})

    print(f"\n[EVAL] Running {case_id} ({category}): '{question}'...")
    t0 = time.time()

    try:
        result = run_analysis(question, top_k=25)
    except Exception as e:
        return {
            "case_id": case_id,
            "question": question,
            "category": category,
            "expected_behavior": case.get("expected_behavior"),
            "actual_behavior": f"Pipeline crashed with error: {e}",
            "passed": False,
            "retrieved_count": 0,
            "extracted_count": 0,
            "validated_count": 0,
            "selected_count": 0,
            "independent_evidence_count": 0,
            "statistics_status": "failed",
            "statistics_input_count": 0,
            "evidence_provenance": [],
            "statistics": [],
            "failure_component": "infrastructure/runtime_failure",
            "failure_reason": str(e)
        }

    # Extract PipelineResult statistics
    retrieved_count = len(result.retrieved_chunks)
    extracted_count = len(result.extracted_evidence)
    validated_count = len(result.validated_evidence)
    selected_count = len(result.selected_evidence)
    independent_count = len(result.independent_evidence)
    stats_status = result.stage_status.get("statistics", "pending")
    stats_input_count = len(result.independent_evidence) if stats_status == "completed" else 0

    evidence_prov = [
        {
            "study_id": ev.study_id,
            "chunk_id": ev.chunk_id,
            "source_file": ev.source_file,
            "page_start": ev.page_start,
            "page_end": ev.page_end,
            "treatment_events": ev.treatment_events,
            "treatment_n": ev.treatment_n,
            "control_events": ev.control_events,
            "control_n": ev.control_n,
            "evidence_quote": ev.evidence_quote
        }
        for ev in result.validated_evidence
    ]

    stats_records = []
    if result.study_comparison and result.study_comparison.study_records:
        for rec in result.study_comparison.study_records:
            rec_dict = {
                "study_id": rec.study_id,
                "study_type": rec.study_type,
                "dose": rec.dose,
                "treatment_events": rec.treatment_events,
                "treatment_n": rec.treatment_n,
                "control_events": rec.control_events,
                "control_n": rec.control_n,
            }
            if rec.risk_ratio_result:
                rec_dict["risk_ratio"] = rec.risk_ratio_result.risk_ratio
                rec_dict["rr_ci_lower"] = rec.risk_ratio_result.ci_lower
                rec_dict["rr_ci_upper"] = rec.risk_ratio_result.ci_upper
            if rec.odds_ratio_result:
                rec_dict["odds_ratio"] = rec.odds_ratio_result.odds_ratio
                rec_dict["or_ci_lower"] = rec.odds_ratio_result.ci_lower
                rec_dict["or_ci_upper"] = rec.odds_ratio_result.ci_upper
            if rec.hypothesis_test_result:
                rec_dict["fisher_p_value"] = rec.hypothesis_test_result.fisher_p_value
                rec_dict["chi2_p_value"] = rec.hypothesis_test_result.chi_square_p_value
            stats_records.append(rec_dict)

    # Evaluate expectations
    passed = True
    failure_component = "none"
    failure_reason = ""
    actual_behavior = ""

    # Case-by-Case Validation Logic
    if case_id in ["case_A1", "case_H1"] or (category == "A_normal_quantitative" and case_id != "case_A2"):
        if independent_count < exp_ev.get("min_extracted", 1):
            passed = False
            failure_component = "independence_selection_failure"
            failure_reason = f"Expected at least {exp_ev.get('min_extracted', 1)} independent evidence record, got {independent_count}."
        elif not any(ev.study_id == exp_ev.get("target_study_id", "EMP-005") for ev in result.independent_evidence):
            passed = False
            failure_component = "lineage_failure"
            failure_reason = f"Target study {exp_ev.get('target_study_id')} not present in independent evidence."
        elif exp_stat.get("should_run_statistics", True) and stats_status != "completed":
            passed = False
            failure_component = "statistics_failure"
            failure_reason = f"Statistics expected to complete, got status '{stats_status}'."
        elif exp_stat.get("expected_rr"):
            exp_rr = exp_stat["expected_rr"]
            tol = exp_stat.get("tolerance", 0.05)
            actual_rr = stats_records[0].get("risk_ratio") if stats_records else None
            if actual_rr is None or abs(actual_rr - exp_rr) > tol:
                passed = False
                failure_component = "statistics_failure"
                failure_reason = f"Risk Ratio disagreement: expected ~{exp_rr}, got {actual_rr}."

    elif category == "A_hazard_ratio" or case_id == "case_A2":
        fabricated_counts = False
        for ev in result.validated_evidence:
            if ev.extracted_metric_type == "hazard_ratio" and (ev.treatment_events is not None or ev.control_events is not None):
                fabricated_counts = True
                break
        
        if fabricated_counts:
            passed = False
            failure_component = "validation_failure"
            failure_reason = "Hazard ratio metric was converted into integer 2x2 event counts."
        elif stats_status == "completed" and any(r.get("risk_ratio") is not None for r in stats_records):
            passed = False
            failure_component = "statistics_failure"
            failure_reason = "Hazard ratio metric entered 2x2 Risk Ratio statistics calculation."
        else:
            actual_behavior = "expected_non_2x2_skip"

    elif category == "B_missing_unsupported":
        if independent_count > exp_ev.get("expected_independent_count", 0):
            passed = False
            failure_component = "validation_failure"
            failure_reason = f"Expected 0 independent evidence records for missing outcome, got {independent_count}."
        elif stats_status == "completed":
            passed = False
            failure_component = "statistics_failure"
            failure_reason = "Statistics ran on unsupported / missing outcome evidence."

    elif category == "C_multiarm_evidence":
        if validated_count == 0:
            passed = False
            failure_component = "extraction_failure"
            failure_reason = "Failed to extract multi-arm table evidence."
        else:
            for ev in result.validated_evidence:
                if ev.treatment_events and ev.treatment_n and ev.treatment_events > ev.treatment_n:
                    passed = False
                    failure_component = "validation_failure"
                    failure_reason = "Invalid treatment counts: treatment_events exceeds treatment_n."

    elif category == "D_overlapping_studies":
        retained = [sel["selection"]["publication_id"] for sel in result.selected_evidence if sel.get("selection", {}).get("selection_status") == "retained_for_independent_analysis"]
        excluded = [sel["selection"]["publication_id"] for sel in result.selected_evidence if sel.get("selection", {}).get("selection_status") == "excluded_due_to_overlap"]
        if "EMP-005" not in retained:
            passed = False
            failure_component = "independence_selection_failure"
            failure_reason = f"EMP-005 primary study was not retained (retained={retained})."
        elif "EMP-001" in retained or "EMP-003" in retained:
            passed = False
            failure_component = "lineage_failure"
            failure_reason = f"Overlapping pooled study retained in error (retained={retained})."

    elif category == "E_secondary_evidence":
        retained = [ev.study_id for ev in result.independent_evidence]
        if "EMP-002" in retained or "EMP-004" in retained:
            passed = False
            failure_component = "lineage_failure"
            failure_reason = f"Secondary systematic review / meta-analysis wrongly retained as independent primary RCT: {retained}"

    elif category.startswith("F_wrong_metric"):
        for ev in result.validated_evidence:
            if ev.extracted_metric_type in ["hazard_ratio", "incidence_rate"] and (ev.treatment_events is not None or ev.control_events is not None):
                passed = False
                failure_component = "validation_failure"
                failure_reason = f"Non-2x2 metric '{ev.extracted_metric_type}' was misparsed with integer event counts."

    elif category == "G_conflict_multi_study":
        actual_behavior = "no_conflict_case_available_in_current_corpus"

    if passed and not actual_behavior:
        actual_behavior = f"Successfully processed {category} case with {independent_count} independent study records and statistics status '{stats_status}'."

    print(f"  --> [{ 'PASS' if passed else 'FAIL' }] Duration: {time.time()-t0:.2f}s | Stats: {stats_status} | Reason: {failure_reason or 'OK'}")

    return {
        "case_id": case_id,
        "question": question,
        "category": category,
        "expected_behavior": case.get("expected_behavior"),
        "actual_behavior": actual_behavior,
        "passed": passed,
        "retrieved_count": retrieved_count,
        "extracted_count": extracted_count,
        "validated_count": validated_count,
        "selected_count": selected_count,
        "independent_evidence_count": independent_count,
        "statistics_status": stats_status,
        "statistics_input_count": stats_input_count,
        "evidence_provenance": evidence_prov,
        "statistics": stats_records,
        "failure_component": failure_component,
        "failure_reason": failure_reason
    }


def main():
    if not CASES_FILE.exists():
        print(f"Error: Dataset file '{CASES_FILE}' not found.")
        sys.exit(1)

    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} evaluation benchmark cases.")

    results = []
    for case in cases:
        res = evaluate_case(case)
        results.append(res)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation benchmark results to '{OUTPUT_FILE}'.")

    # Print Summary Table
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count
    pass_rate = (passed_count / total * 100.0) if total > 0 else 0.0

    print("\n================ EVALUATION BENCHMARK SUMMARY ================")
    print(f"Total Cases : {total}")
    print(f"Passed      : {passed_count}")
    print(f"Failed      : {failed_count}")
    print(f"Pass Rate   : {pass_rate:.1f}%\n")

    print(f"{'Case ID':<10} | {'Category':<25} | {'Status':<6} | {'Failure Component':<30}")
    print("-" * 77)
    for r in results:
        status_str = "PASS" if r["passed"] else "FAIL"
        print(f"{r['case_id']:<10} | {r['category']:<25} | {status_str:<6} | {r['failure_component']:<30}")


if __name__ == "__main__":
    main()
