# 🛡️ Drug Safety AI — Technical Defense & Architecture Guide

This document provides concise, technically rigorous answers to critical architectural, statistical, and engineering questions for technical lead and clinical AI review.

---

### 1. Why use an LLM here?
> We use an LLM for **natural language hypothesis understanding**, **semi-structured document entity extraction**, and **grounded narrative report synthesis**. Natural language clinical trial text and PDF tables contain heterogeneous formatting (e.g. "301 (6.4%) vs 42 (1.8%)") that standard regex cannot parse across diverse publications.

---

### 2. Why not let the LLM calculate statistics?
> LLMs are non-deterministic autoregressive token predictors. They frequently hallucinate arithmetic calculations, miscalculate confidence intervals, or round numbers inconsistently. In pharmacovigilance, statistical errors can lead to false safety conclusions. Therefore, we restrict the LLM to candidate chunk retrieval and structured JSON data extraction, while all 2×2 statistical tests ($RR$, $OR$, Fisher exact, $\chi^2$) are computed deterministically in standard Python code.

---

### 3. Why RAG (Retrieval-Augmented Generation)?
> Full clinical trial documents contain thousands of tokens of protocol narrative, inclusion criteria, and references that exceed context limits and introduce noise. RAG allows us to segment publications into semantic chunks, generate vector embeddings, and retrieve only the top candidate quantitative tables and outcome sections relevant to the specific user question.

---

### 4. How do you prevent hallucinations?
> We enforce grounding through three mechanisms:
> 1. **Strict JSON Schema Extraction**: The LLM must output evidence in a Pydantic schema with mandatory verbatim `evidence_quote` strings.
> 2. **Deterministic Code Validation**: Numerical event counts ($a, b, c, d$) are checked against strict mathematical bounds ($events \le N, N > 0$).
> 3. **Verbatim Quote Citation**: The UI and final report display the exact source chunk quote alongside all statistical outputs.

---

### 5. How do you validate extracted evidence?
> Extracted records pass through `validation/evidence_validator.py`, which verifies:
> - Metric type is explicitly `event_count` (blocking `hazard_ratio`, `incidence_rate`, or `percentage`).
> - Presence of all 4 cell values ($a, b, c, d$).
> - Integer types and positive sample sizes ($N > 0$).
> - Event counts do not exceed group denominators ($a \le N_{treatment}$, $c \le N_{control}$).

---

### 6. How do you prevent double counting in meta-analyses?
> Post-hoc pooled safety publications (e.g. `EMP-001` or `EMP-003`) analyze patient cohorts drawn directly from primary randomized controlled trials (e.g. `EMP-005` / EMPA-REG OUTCOME). Including both in a meta-analysis double-counts event frequencies. Our `analysis/study_lineage.py` module constructs a study provenance graph that identifies primary RCTs, marks overlapping pooled analyses as `excluded_due_to_overlap`, and passes only independent primary evidence to the statistical engine.

---

### 7. Why Risk Ratio (RR) and Odds Ratio (OR)?
> - **Risk Ratio ($RR = p_1 / p_2$)** measures the relative probability of an adverse event in the treatment group versus control, providing an intuitive clinical metric for prospective trial cohorts.
> - **Odds Ratio ($OR = \frac{a/b}{c/d}$)** provides an symmetric measure of association that is mathematically stable across retrospective and prospective study designs. Standard log-transformed normal approximation (Katz et al.) with Haldane-Anscombe continuity correction (+0.5 for zero-event cells) is used for 95% Confidence Intervals.

---

### 8. Why Fisher's Exact Test?
> Chi-square tests rely on large-sample asymptotic normal approximations that become invalid when expected cell counts fall below 5 (common in rare drug safety adverse events). Fisher's exact test computes exact hyper-geometric probabilities without asymptotic assumptions, guaranteeing valid $p$-values across rare adverse event counts.

---

### 9. What happens with Hazard Ratios (HR)?
> Hazard Ratios represent time-to-event survival analysis metrics derived from Cox proportional hazards models. They cannot be converted into 2×2 event count tables without raw patient-level follow-up times. Our validator detects `extracted_metric_type == 'hazard_ratio'`, preserves the evidence quote for display, and safely skips 2×2 statistical engine calculations while displaying a clear non-2×2 safety notice.

---

### 10. How did you evaluate retrieval?
> We constructed a benchmark set of 11 end-to-end evaluation questions (`evaluation/end_to_end_cases.json`). We measured **Candidate Reachability** (verifying that the target quantitative chunk reaches the top-25 candidate pool, achieved at 100%) and **Average Precision@5** (0.80 across initial query audits).

---

### 11. How did you evaluate extraction?
> We conducted a 55-chunk quantitative evidence audit (`data/processed/evaluation/quantitative_chunk_audit.json`) categorizing 54 quantitative chunks into 27 raw event-count tables and 27 derived metric chunks. Extraction accuracy was evaluated against deterministic ground truth extractions across benchmark cases.

---

### 12. What are the biggest system limitations?
> 1. **Survival Analysis**: Survival curve parsing and Cox regression are not implemented; HRs are displayed as reported metrics only.
> 2. **Multi-Row Tabular Extraction**: Complex multi-arm tables require LLM JSON output mode when live; deterministic fallback parses single-row 2×2 comparisons per chunk.
> 3. **Corpus Scope**: Curated to 5 empagliflozin publications (`EMP-001` through `EMP-005`).

---

### 13. What would you improve in a production system?
> 1. Integrate an OCR/PDF vision parser (e.g. LayoutLM / Unstructured) for multi-column tabular parsing.
> 2. Implement Cox proportional hazards survival model parsing for Hazard Ratio evidence.
> 3. Expand the vector index to a distributed hybrid BM25 + dense vector store (e.g. Qdrant / Weaviate).

---

### 14. Why is this NOT a clinical decision system?
> This system is an **evidence-analysis decision-support prototype**. It aggregates, filters, and computes statistics on published literature. It does not replace medical judgment, clinical diagnosis, patient context, or regulatory pharmacovigilance review.

---

### 15. What happens if there is no evidence?
> If a query asks about an unstudied adverse outcome (e.g. "Does empagliflozin cause pulmonary fibrosis?"), retrieval returns 0 matching quantitative event chunks. The pipeline skips statistical calculations, produces 0 validated evidence records, and displays an intentional no-evidence message explaining that no evidence was found in the curated corpus.
