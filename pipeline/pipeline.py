"""
End-to-End Drug Safety Evidence Analysis Pipeline Orchestrator Module.

Coordinates existing deterministic and LLM-grounded analysis components
from hypothesis extraction through retrieval, extraction, validation,
statistics, normalization, conflict analysis, report building, and validated LLM report generation.

Does NOT duplicate business logic, statistical formulas, or retrieval algorithms.
"""

import sys
import os
import time
import traceback
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from llm.hypothesis import parse_hypothesis, StructuredHypothesis
from rag.retriever import retrieve
from llm.evidence_extractor import extract_evidence_from_chunk, select_candidate_chunks
from llm.evidence import ExtractedEvidence
from validation.evidence_validator import validate_and_sanitize_evidence
from statistics.evidence_statistics import analyze_extracted_evidence, EvidenceStatisticalAnalysisResult
from analysis.study_comparison import normalize_study_evidence, StudyComparisonResult
from analysis.conflict_analysis import analyze_multi_study_conflicts, MultiStudyConflictAnalysisPayload
from analysis.lineage_resolver import attach_evidence_lineage
from analysis.evidence_selection import select_independent_evidence
from report.report_builder import build_evidence_analysis_report, EvidenceAnalysisReport
from llm.report import generate_evidence_report, validate_generated_report, GeneratedReportValidationResult
from llm.config import LLMConfig



STAGE_NAMES = [
    "hypothesis",
    "retrieval",
    "evidence_extraction",
    "validation",
    "statistics",
    "study_comparison",
    "conflict_analysis",
    "report_building",
    "llm_report",
    "llm_report_validation"
]


class PipelineResult(BaseModel):
    """
    Complete structured output payload of the End-to-End Drug Safety Analysis Pipeline.
    """
    question: str
    hypothesis: Optional[StructuredHypothesis] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_evidence: List[ExtractedEvidence] = Field(default_factory=list)
    validated_evidence: List[ExtractedEvidence] = Field(default_factory=list)
    rejected_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    lineage_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    selected_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    independent_evidence: List[ExtractedEvidence] = Field(default_factory=list)
    study_comparison: Optional[StudyComparisonResult] = None
    conflict_analysis: Optional[MultiStudyConflictAnalysisPayload] = None
    deterministic_report: Optional[EvidenceAnalysisReport] = None
    generated_report: Optional[str] = None
    generated_report_validation: Optional[GeneratedReportValidationResult] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    stage_status: Dict[str, str] = Field(default_factory=dict)


def run_analysis(
    question: str,
    top_k: int = 10,
    config: Optional[LLMConfig] = None,
    retriever_fn: Optional[Any] = None,
    hypothesis_fn: Optional[Any] = None,
    extractor_fn: Optional[Any] = None,
    generator_fn: Optional[Any] = None
) -> PipelineResult:
    """
    Executes the 10-stage end-to-end drug safety analysis pipeline.

    :param question: The research question string.
    :param top_k: Number of RAG chunks to retrieve (default 10).
    :param config: Optional LLMConfig instance.
    :param retriever_fn: Optional override for retrieval function (useful in unit tests).
    :param hypothesis_fn: Optional override for hypothesis parsing.
    :param extractor_fn: Optional override for evidence extraction.
    :param generator_fn: Optional override for LLM report generation.
    :return: PipelineResult containing all intermediate artifacts, errors, and stage statuses.
    """
    result = PipelineResult(
        question=question,
        stage_status={stage: "pending" for stage in STAGE_NAMES},
        errors=[]
    )
    stage_status = result.stage_status
    errors = result.errors

    def mark_remaining_skipped(start_index: int):
        for idx in range(start_index, len(STAGE_NAMES)):
            st = STAGE_NAMES[idx]
            if stage_status[st] == "pending":
                stage_status[st] = "skipped"


    # --------------------------------------------------
    # Stage 1 — Hypothesis
    # --------------------------------------------------
    stage_status["hypothesis"] = "running"
    print("[PIPELINE] hypothesis START")
    t0 = time.time()
    try:
        parse_func = hypothesis_fn or parse_hypothesis
        hypothesis_obj = parse_func(question)
        if not hypothesis_obj:
            raise ValueError("Hypothesis parsing returned None or empty hypothesis.")
        if isinstance(hypothesis_obj, dict):
            hypothesis_obj = StructuredHypothesis(**hypothesis_obj)
        result.hypothesis = hypothesis_obj
        stage_status["hypothesis"] = "completed"
        print(f"[PIPELINE] hypothesis DONE {time.time()-t0:.2f}s")
    except Exception as e:
        stage_status["hypothesis"] = "failed"
        errors.append({
            "stage": "hypothesis",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        mark_remaining_skipped(1)
        return result

    # --------------------------------------------------
    # Stage 2 — Retrieval
    # --------------------------------------------------
    stage_status["retrieval"] = "running"
    print("[PIPELINE] retrieval START")
    t0 = time.time()
    try:
        retrieve_func = retriever_fn or retrieve
        outcome_val = getattr(result.hypothesis, "outcome", None)
        if outcome_val is None and isinstance(result.hypothesis, dict):
            outcome_val = result.hypothesis.get("outcome")
        chunks = retrieve_func(query=question, outcome=outcome_val, top_k=top_k)
        result.retrieved_chunks = chunks or []
        stage_status["retrieval"] = "completed"
        print(f"[PIPELINE] retrieval DONE {time.time()-t0:.2f}s, chunks={len(result.retrieved_chunks)}")
    except Exception as e:
        stage_status["retrieval"] = "failed"
        errors.append({
            "stage": "retrieval",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        mark_remaining_skipped(2)
        return result

    # --------------------------------------------------
    # Stage 3 — Evidence Extraction
    # --------------------------------------------------
    stage_status["evidence_extraction"] = "running"
    print("[PIPELINE] evidence_extraction START")
    t0 = time.time()
    extracted_records: List[ExtractedEvidence] = []
    try:
        hyp_dict = result.hypothesis.model_dump() if hasattr(result.hypothesis, "model_dump") else result.hypothesis.dict()

        if extractor_fn:
            target_chunks = result.retrieved_chunks
            extract_func = extractor_fn
        else:
            target_chunks = select_candidate_chunks(result.retrieved_chunks, hyp_dict, max_candidates=15)
            extract_func = extract_evidence_from_chunk

        print(f"[PIPELINE] candidate_chunks selected={len(target_chunks)}")
        for idx, chunk in enumerate(target_chunks, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            print(f"[EXTRACTION] chunk {idx}/{len(target_chunks)} ({chunk_id}) START")
            tc0 = time.time()
            ev = extract_func(chunk=chunk, hypothesis=hyp_dict, config=config)
            print(f"[EXTRACTION] chunk {idx}/{len(target_chunks)} ({chunk_id}) DONE {time.time()-tc0:.2f}s (extracted={'YES' if ev else 'NO'})")
            if ev is not None:
                extracted_records.append(ev)

        result.extracted_evidence = extracted_records
        stage_status["evidence_extraction"] = "completed"
        print(f"[PIPELINE] evidence_extraction DONE {time.time()-t0:.2f}s, extracted={len(extracted_records)}")
    except Exception as e:
        stage_status["evidence_extraction"] = "failed"
        errors.append({
            "stage": "evidence_extraction",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        mark_remaining_skipped(3)
        return result

    # --------------------------------------------------
    # Stage 4 — Evidence Validation & Lineage / Selection
    # --------------------------------------------------
    stage_status["validation"] = "running"
    print("[PIPELINE] validation START")
    t0 = time.time()
    validated_records: List[ExtractedEvidence] = []
    rejected_records: List[Dict[str, Any]] = []

    try:
        for ev in result.extracted_evidence:
            try:
                sanitized_ev = validate_and_sanitize_evidence(ev)
                validated_records.append(sanitized_ev)
            except Exception as val_err:
                rejected_records.append({
                    "reason": str(val_err),
                    "provenance": {
                        "study_id": ev.study_id,
                        "chunk_id": ev.chunk_id,
                        "source_file": ev.source_file,
                        "page_start": ev.page_start,
                        "page_end": ev.page_end,
                        "evidence_quote": ev.evidence_quote
                    },
                    "original_record": ev.model_dump() if hasattr(ev, "model_dump") else ev.dict()
                })

        result.validated_evidence = validated_records
        result.rejected_evidence = rejected_records

        # Attach Lineage and Run Evidence Selection
        lineage_records = []
        for ev in validated_records:
            ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
            ev_dict["lineage"] = attach_evidence_lineage(ev).model_dump()
            lineage_records.append(ev_dict)

        result.lineage_evidence = lineage_records

        selection_res = select_independent_evidence(validated_records)
        result.selected_evidence = selection_res.all_selected_evidence
        result.independent_evidence = selection_res.independent_evidence

        stage_status["validation"] = "completed"
        print(f"[PIPELINE] validation DONE {time.time()-t0:.2f}s, validated={len(validated_records)}, independent={len(result.independent_evidence)}")
    except Exception as e:
        stage_status["validation"] = "failed"
        errors.append({
            "stage": "validation",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        mark_remaining_skipped(4)
        return result

    # --------------------------------------------------
    # Stage 5 — Study-Level Statistics
    # --------------------------------------------------
    stage_status["statistics"] = "running"
    print("[PIPELINE] statistics START")
    t0 = time.time()
    try:
        if not result.independent_evidence:
            # Strict boundary: no fallback to overlapping validated evidence.
            # When no independently-analysable evidence exists, statistics is skipped.
            stage_status["statistics"] = "skipped_no_independent_evidence"
            print("[PIPELINE] statistics SKIPPED — no independent evidence available (overlap boundary enforced)")
        else:
            for ev in result.independent_evidence:
                if ev.extracted_metric_type == "event_count":
                    analyze_extracted_evidence(ev)
            stage_status["statistics"] = "completed"
            print(f"[PIPELINE] statistics DONE {time.time()-t0:.2f}s")
    except Exception as e:
        stage_status["statistics"] = "failed"
        errors.append({
            "stage": "statistics",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        mark_remaining_skipped(5)
        return result

    # --------------------------------------------------
    # Stage 6 — Study Comparison
    # --------------------------------------------------
    stage_status["study_comparison"] = "running"
    print("[PIPELINE] study_comparison START")
    t0 = time.time()
    try:
        # Strict boundary: compare only independent evidence — never overlapping validated evidence.
        comparison_res = normalize_study_evidence(result.independent_evidence)
        result.study_comparison = comparison_res
        stage_status["study_comparison"] = "completed"
        print(f"[PIPELINE] study_comparison DONE {time.time()-t0:.2f}s")
    except Exception as e:
        stage_status["study_comparison"] = "failed"
        errors.append({
            "stage": "study_comparison",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        mark_remaining_skipped(6)
        return result

    # --------------------------------------------------
    # Stage 7 — Conflict Analysis
    # --------------------------------------------------
    stage_status["conflict_analysis"] = "running"
    print("[PIPELINE] conflict_analysis START")
    t0 = time.time()
    try:
        conflict_res = analyze_multi_study_conflicts(result.study_comparison)
        result.conflict_analysis = conflict_res
        stage_status["conflict_analysis"] = "completed"
        print(f"[PIPELINE] conflict_analysis DONE {time.time()-t0:.2f}s")
    except Exception as e:
        stage_status["conflict_analysis"] = "failed"
        errors.append({
            "stage": "conflict_analysis",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        mark_remaining_skipped(7)
        return result

    # --------------------------------------------------
    # Stage 8 — Deterministic Report
    # --------------------------------------------------
    stage_status["report_building"] = "running"
    print("[PIPELINE] report_building START")
    t0 = time.time()
    try:
        det_report = build_evidence_analysis_report(
            research_question=question,
            hypothesis=result.hypothesis,
            comparison=result.study_comparison,
            conflict_payload=result.conflict_analysis
        )
        result.deterministic_report = det_report
        stage_status["report_building"] = "completed"
        print(f"[PIPELINE] report_building DONE {time.time()-t0:.2f}s")
    except Exception as e:
        stage_status["report_building"] = "failed"
        errors.append({
            "stage": "report_building",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        mark_remaining_skipped(8)
        return result

    # --------------------------------------------------
    # Stage 9 — Grounded LLM Report
    # --------------------------------------------------
    stage_status["llm_report"] = "running"
    print("[PIPELINE] llm_report START")
    t0 = time.time()
    try:
        gen_func = generator_fn or generate_evidence_report
        llm_text = gen_func(result.deterministic_report)
        if not llm_text:
            raise ValueError("LLM report generator returned empty text.")
        result.generated_report = llm_text
        stage_status["llm_report"] = "completed"
        print(f"[PIPELINE] llm_report DONE {time.time()-t0:.2f}s")
    except Exception as e:
        stage_status["llm_report"] = "failed"
        errors.append({
            "stage": "llm_report",
            "error_type": type(e).__name__,
            "message": str(e)
        })
        stage_status["llm_report_validation"] = "skipped"
        return result

    # --------------------------------------------------
    # Stage 10 — LLM Report Validation
    # --------------------------------------------------
    stage_status["llm_report_validation"] = "running"
    print("[PIPELINE] llm_report_validation START")
    t0 = time.time()
    try:
        val_res = validate_generated_report(result.generated_report, result.deterministic_report)
        result.generated_report_validation = val_res

        if not val_res.valid:
            stage_status["llm_report_validation"] = "failed"
            errors.append({
                "stage": "llm_report_validation",
                "error_type": "ValidationError",
                "message": f"Generated LLM report failed grounding validation: {val_res.errors}"
            })
        else:
            stage_status["llm_report_validation"] = "completed"
        print(f"[PIPELINE] llm_report_validation DONE {time.time()-t0:.2f}s (valid={val_res.valid})")
    except Exception as e:
        stage_status["llm_report_validation"] = "failed"
        errors.append({
            "stage": "llm_report_validation",
            "error_type": type(e).__name__,
            "message": str(e)
        })

    return result
