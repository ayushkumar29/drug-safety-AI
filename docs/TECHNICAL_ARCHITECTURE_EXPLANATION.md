# 🏗️ Technical Architecture Explanation & Data Flow Guide

This document explains the end-to-end technical architecture, data flow, validation checkpoints, and failure-handling strategies in interview-oriented language.

---

## 1. End-to-End System Architecture

```text
  [ User Question Input ]
             │
             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 1: Hypothesis Generation (llm/hypothesis.py)      │
  │ Parses Question -> Pydantic StructuredHypothesis Schema│
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 2: RAG Retrieval (rag/retriever.py)             │
  │ Encodes query (all-MiniLM-L6-v2) -> Searches FAISS     │
  │ Reranks candidate chunks based on outcome relevance    │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 3: Evidence Extraction (llm/extraction.py)       │
  │ Extracts Structured JSON -> ExtractedEvidence Schema   │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 4: Evidence Validation (validation/)             │
  │ Checks N > 0, events <= N, metric_type == 'event_count'│
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 5: Study Lineage Resolution (analysis/)          │
  │ Maps pooled studies to primary RCTs -> Excludes overlap│
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 6: Deterministic Statistics (statistics/)        │
  │ Calculates Risk Ratio, Odds Ratio, Fisher, Chi-Square  │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Stage 7: Grounded Report Building (report/ & llm/)     │
  │ Synthesizes cited narrative report grounded in ev quote │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  [ Streamlit UI Output (app/streamlit_app.py) ]
```

---

## 2. Component Responsibilities & Data Flow

### A. Hypothesis Generation (`llm/hypothesis.py`)
- **Input**: Raw text question string (e.g. `"Does empagliflozin increase the risk of genital infection compared with placebo?"`).
- **Processing**: Prompts LLM to extract structured fields (`drug`, `comparator`, `outcome`, `population`, `study_type`, `hypothesis_direction`).
- **Output**: `StructuredHypothesis` Pydantic object.

### B. Candidate Retrieval & Re-ranking (`rag/retriever.py`)
- **Input**: User question and target outcome string from hypothesis.
- **Processing**: Encodes query into 384-dimensional vector using `all-MiniLM-L6-v2`, queries FAISS vector index for top 25 chunks, and applies outcome-aware keyword score boosting.
- **Output**: Ranked list of 25 candidate PDF text chunks (`List[Dict[str, Any]]`).

### C. Evidence Extraction (`llm/extraction.py`)
- **Input**: Top candidate chunks + `StructuredHypothesis`.
- **Processing**: Prompts LLM in structured JSON mode to extract 2×2 numerical cell counts ($a, b, c, d$), study ID, dose, follow-up, metric type, and verbatim `evidence_quote`. Deterministic regex table fallback executes if LLM times out.
- **Output**: `List[ExtractedEvidence]`.

### D. Evidence Validation (`validation/evidence_validator.py`)
- **Input**: `ExtractedEvidence` objects.
- **Processing**: Enforces mathematical bounds ($N > 0$, $events \ge 0$, $events \le N$). Blocks non-2×2 metric types (`hazard_ratio`, `incidence_rate`, `percentage`).
- **Output**: Validated evidence list (`validated_evidence`) and excluded invalid records with explicit failure reasons.

### E. Study Lineage & Overlap Protection (`analysis/study_lineage.py`)
- **Input**: Validated evidence list.
- **Processing**: Builds study provenance graph using `data/metadata/study_lineage.json`. Identifies primary RCTs (e.g. `EMP-005`), marks overlapping pooled publications (e.g. `EMP-001`, `EMP-003`) as `excluded_due_to_overlap`.
- **Output**: `selected_evidence` container holding `independent_evidence` records.

### F. Deterministic Statistics Engine (`statistics/`)
- **Input**: `independent_evidence` event counts ($a, b, c, d$).
- **Processing**: Computes Risk Ratio ($RR$), Odds Ratio ($OR$), 95% CIs via Katz/Woolf log-normal approximations with Haldane-Anscombe continuity correction for zero cells, Fisher exact $p$-value, and Pearson $\chi^2$ $p$-value.
- **Output**: `StudyComparisonResult` holding `NormalizedStudyEvidence` records with attached statistical objects.

### G. Presentation Layer (`app/streamlit_app.py`)
- **Input**: `PipelineResult` payload.
- **Processing**: Formats 8 clean UI sections, metric cards, expandable evidence cards, lineage status badges, and zero-dependency inline SVG Forest Plot.
- **Output**: Interactive web interface. **Zero statistical math execution.**

---

## 3. Provenance Preservation & Failure Handling

1. **Source Provenance**: Every extracted record retains `source_file`, `chunk_id`, `page_start`, `page_end`, and verbatim `evidence_quote`.
2. **Hazard Ratio Boundary**: `extracted_metric_type == 'hazard_ratio'` skips 2×2 statistics safely without crashing pipeline; UI displays survival analysis warning banner.
3. **No-Evidence Handling**: Unstudied outcomes return 0 quantitative chunks; pipeline skips statistics engine and UI displays clean absence-of-evidence notification.
