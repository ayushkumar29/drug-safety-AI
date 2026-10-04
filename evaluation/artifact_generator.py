import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from pipeline.pipeline import PipelineResult
except ImportError:
    from pipeline import PipelineResult


def _ev_to_dict(ev) -> Dict[str, Any]:
    return {
        "study_id": ev.study_id,
        "drug": ev.drug,
        "comparator": ev.comparator,
        "outcome": ev.outcome,
        "study_type": ev.study_type,
        "dose": ev.dose,
        "follow_up": ev.follow_up,
        "publication_year": ev.publication_year,
        "treatment_n": ev.treatment_n,
        "treatment_events": ev.treatment_events,
        "control_n": ev.control_n,
        "control_events": ev.control_events,
        "extracted_metric_type": ev.extracted_metric_type,
        "source_file": ev.source_file,
        "chunk_id": ev.chunk_id,
        "page_start": ev.page_start,
        "page_end": ev.page_end,
        "evidence_quote": ev.evidence_quote,
    }


def _statistics_records(result) -> List[Dict[str, Any]]:
    if result.stage_status.get("statistics") != "completed":
        return []
    if result.study_comparison is None:
        return []
    out = []
    for sr in result.study_comparison.study_records:
        rr = sr.risk_ratio_result
        orr = sr.odds_ratio_result
        ht = sr.hypothesis_test_result
        out.append({
            "study_id": sr.study_id,
            "comparison_key": sr.comparison_key,
            "statistical_representation": sr.statistical_representation,
            "treatment_n": sr.treatment_n,
            "treatment_events": sr.treatment_events,
            "control_n": sr.control_n,
            "control_events": sr.control_events,
            "source_file": sr.source_file,
            "chunk_id": sr.chunk_id,
            "page_start": sr.page_start,
            "page_end": sr.page_end,
            "risk_ratio": rr.risk_ratio if rr else None,
            "risk_ratio_ci_lower": rr.ci_lower if rr else None,
            "risk_ratio_ci_upper": rr.ci_upper if rr else None,
            "odds_ratio": orr.odds_ratio if orr else None,
            "odds_ratio_ci_lower": orr.ci_lower if orr else None,
            "odds_ratio_ci_upper": orr.ci_upper if orr else None,
            "fisher_p_value": ht.fisher_p_value if ht else None,
            "chi_square_p_value": ht.chi_square_p_value if ht else None,
        })
    return out



def build_artifact(result, evaluation_name: str = "") -> Dict[str, Any]:
    stats_status = result.stage_status.get("statistics", "pending")
    statistics_ran = (stats_status == "completed")
    statistics_input_count = len(result.independent_evidence) if statistics_ran else 0
    study_comparison_count = 0
    if result.study_comparison is not None:
        study_comparison_count = len(result.study_comparison.study_records)
    report_citation_count = 0
    if result.deterministic_report is not None:
        report_citation_count = len(result.deterministic_report.citations)
    retained_ids = [
        sel["selection"]["publication_id"]
        for sel in result.selected_evidence
        if sel.get("selection", {}).get("selection_status") == "retained_for_independent_analysis"
    ]
    excluded_ids = [
        sel["selection"]["publication_id"]
        for sel in result.selected_evidence
        if sel.get("selection", {}).get("selection_status") == "excluded_due_to_overlap"
    ]
    return {
        "evaluation_name": evaluation_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": result.question,
        "stage_status": dict(result.stage_status),
        "retrieved_count": len(result.retrieved_chunks),
        "extracted_count": len(result.extracted_evidence),
        "validated_count": len(result.validated_evidence),
        "rejected_count": len(result.rejected_evidence),
        "lineage_evidence_count": len(result.lineage_evidence),
        "selected_count": len(result.selected_evidence),
        "independent_evidence_count": len(result.independent_evidence),
        "retained_publications": retained_ids,
        "excluded_publications": excluded_ids,
        "statistics_status": stats_status,
        "statistics_ran": statistics_ran,
        "statistics_input_count": statistics_input_count,
        "study_comparison_count": study_comparison_count,
        "report_citation_count": report_citation_count,
        "evidence": [_ev_to_dict(ev) for ev in result.independent_evidence],
        "statistics": _statistics_records(result),
        "errors": result.errors,
    }


def save_artifact(
    result,
    output_path: str,
    evaluation_name: str = "Pipeline Evaluation Artifact",
) -> Dict[str, Any]:
    artifact = build_artifact(result, evaluation_name=evaluation_name)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=str)
    return artifact
