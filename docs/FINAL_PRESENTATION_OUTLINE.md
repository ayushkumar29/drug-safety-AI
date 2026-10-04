# 📊 AI-Assisted Drug Safety Evidence Analysis — Final Presentation Deck

**Project:** AI-Assisted Drug Safety Evidence Analysis using LLM, RAG and Statistical Hypothesis Testing  
**Audience:** Technical Interviewers / Senior AI/ML Engineering Leads  
**Duration:** 10–15 Minutes  

---

## Slide 1 — Project Title
### **AI-Assisted Drug Safety Evidence Analysis using LLM, RAG and Statistical Hypothesis Testing**
- **Presenter:** Ayush Kumar
- **Domain:** Pharmacovigilance AI & Hybrid Deterministic Medical Evidence Engineering
- **Repository:** `ayushkumar29/drug-safety-AI`

---

## Slide 2 — Problem Statement
### **Challenges in Automated Pharmacovigilance**
1. **Unstructured Data Dispersion**: Critical drug safety evidence is buried across hundreds of multi-page PDF clinical trial reports, journal publications, and post-hoc safety analyses.
2. **LLM Limitations in Quantitative Reasoning**: Standard LLMs hallucinate numbers, misparse tabular event counts, round confidence intervals inconsistently, and cannot be trusted for critical medical statistics.
3. **Double Counting in Pooled Analyses**: Meta-analyses and pooled safety reports analyze patient cohorts drawn directly from primary RCTs. Treating pooled studies as independent leads to duplicated event counts and false statistical significance.

---

## Slide 3 — Proposed Solution
### **Hybrid AI / Deterministic Evidence Pipeline**
```text
Natural-language Question
        ↓
Structured Hypothesis Generation (LLM)
        ↓
RAG Retrieval & Candidate Selection (FAISS Vector Store)
        ↓
Structured Evidence Extraction (LLM JSON Schema)
        ↓
Validation & Bounds Verification (Python Rules)
        ↓
Study Lineage & Overlap Protection (Python Resolution Engine)
        ↓
Independent Evidence Selection
        ↓
Deterministic 2×2 Statistics Engine (Python math)
        ↓
Grounded Report & Citation Synthesis (LLM Narrative)
```

---

## Slide 4 — System Architecture & Component Boundaries
### **Strict Separation of Responsibilities**
- **Streamlit UI (`app/streamlit_app.py`)**: Presentation layer only; zero statistical formulas or math execution.
- **RAG & Vector Index (`rag/`)**: Candidate chunk retrieval & outcome-aware re-ranking only.
- **LLM Engine (`llm/`)**: Natural language hypothesis parsing, structured JSON extraction, and narrative reporting.
- **Validation Engine (`validation/`)**: Deterministic bounds checking ($events \le N$, $N > 0$, non-negative counts).
- **Lineage Engine (`analysis/`)**: Study provenance resolution, overlap protection, and independent study selection.
- **Statistics Engine (`statistics/`)**: Deterministic 2×2 Risk Ratio ($RR$), Odds Ratio ($OR$), Fisher Exact, and $\chi^2$ calculations.

> **Key Design Decision**: The LLM is used strictly for natural language candidate retrieval and structured extraction. All mathematical calculations and statistical hypothesis tests are computed deterministically in standard Python code.

---

## Slide 5 — RAG Pipeline & Retrieval Evaluation
- **Ingestion**: PyMuPDF page parsing, text cleaning, and 500-token semantic chunking with page-range tracking.
- **Vector Index**: 384-dimensional dense embeddings using SentenceTransformers `all-MiniLM-L6-v2` with FAISS index.
- **Retrieval Metrics (Curated 5-Doc Corpus, 55 Chunks)**:
  - Candidate Pool ($top\_k$): `25` candidate chunks.
  - Candidate Pool Reachability: **100.0%** across benchmark evaluation cases.
  - Current Evaluation Average Precision@5 ($AP@5$): **0.80** across initial query retrieval audits.
- *Note on Retrieval Evaluation*: Historical retrieval evaluation artifacts measured different benchmark definitions (e.g. initial dense vector retrieval vs. outcome-aware re-ranked candidate pools) and are not presented as a single combined metric.

---

## Slide 6 — Structured Evidence Extraction
- **Pydantic Evidence Schema**:
  - `study_id`, `drug`, `comparator`, `outcome`, `study_type`
  - `treatment_n`, `treatment_events`, `control_n`, `control_events`
  - `dose`, `follow_up`, `source_file`, `chunk_id`, `page_range`, `evidence_quote`
  - `extracted_metric_type` (`event_count`, `hazard_ratio`, `incidence_rate`, `percentage`)
- **Extraction Audit**: Audited across 55 corpus chunks (27 raw event-count chunks, 27 derived-metric chunks).
- *Extraction Metric Note*: Extraction performance was evaluated against deterministic audit extractions; global percentage accuracy is not claimed due to the absence of a large gold-standard human-annotated dataset.

---

## Slide 7 — Study Lineage & Overlap Protection
- **Why Lineage Matters**: Post-hoc pooled analyses (e.g. `EMP-001`, `EMP-003`) analyze patient cohorts drawn directly from primary RCTs (e.g. `EMP-005` / EMPA-REG OUTCOME).
- **Lineage Resolution**:
  - Primary RCT (`EMP-005`): **RETAINED** for independent statistical analysis.
  - Pooled Analyses (`EMP-001`, `EMP-003`): **EXCLUDED** from 2×2 statistical engine with reason `"Overlaps with EMPA-REG_OUTCOME"`.
- **Hierarchy for Selection**: Primary RCT > Direct Quantitative Evidence > Pooled Analysis > Systematic Review.
- *Clarification*: This hierarchy is used strictly for overlap protection and selection filtering, not as a universal scientific ranking of overall literature quality.

---

## Slide 8 — Deterministic Statistical Engine
- **Risk Ratio ($RR$)**: $RR = p_1 / p_2$ with log-transformed normal approximation (Katz et al.) for 95% CI.
- **Odds Ratio ($OR$)**: $OR = \frac{a/b}{c/d}$ with Woolf log-odds normal approximation for 95% CI.
- **Zero-Cell Handling**: Haldane-Anscombe continuity correction (+0.5 added to all 2×2 cells) applied automatically when $a=0$ or $c=0$.
- **Hypothesis Testing**: Exact hyper-geometric probabilities for Fisher Exact Test; Pearson $\chi^2$ test with 1 degree of freedom.
- *Audit Guarantee*: **100% Deterministic Python math** using `math` module algorithms. No LLM arithmetic.

---

## Slide 9 — Golden Path Demo Case (EMPA-REG OUTCOME)
- **Question**: *"Does empagliflozin increase the risk of genital infection compared with placebo?"*
- **Retained Study**: `EMP-005` (Table 2)
- **Extracted Counts**:
  - Empagliflozin Pooled 10mg + 25mg: **301** events / **4687** N
  - Placebo: **42** events / **2333** N
- **Deterministic Statistics Output**:
  - **Risk Ratio ($RR$)**: `3.5673` (95% CI: `2.5930 – 4.9077`)
  - **Odds Ratio ($OR$)**: `3.7435` (95% CI: `2.7000 – 5.1902`)
  - **Fisher Exact Test**: $p = 7.7241 \times 10^{-20}$ ($p < 0.0001$)
  - **Chi-Square Test**: $p = 2.6415 \times 10^{-17}$ ($p < 0.0001$)
- *Presentation Note*: The system presents these values as deterministic statistical evidence, not as an autonomous clinical safety declaration.

---

## Slide 10 — Safety Boundaries & Failure Modes
- **Hazard Ratio (HR) Evidence**: Preserved with verbatim quote; validator flags `extracted_metric_type == 'hazard_ratio'` and blocks 2×2 engine. UI displays explicit non-2×2 survival analysis limitation notice.
- **No-Evidence Queries** (e.g. *"Does empagliflozin cause pulmonary fibrosis?"*): Produces 0 validated evidence records, skips 2×2 statistics safely, and displays clear absence-of-evidence notification without hallucinating numbers.
- **Multi-Arm Safety**: Uses verified reported pooled treatment arm ($301 / 4687$) without fabricating combined denominators.

---

## Slide 11 — Systematic Evaluation Results
- **End-to-End Evaluation Benchmark**: **11 / 11 benchmark cases passing (100% pass rate)** in `evaluation/end_to_end_cases.json`.
- **Full Project Regression Suite**: **167 / 167 unit and integration tests passing** (`python -m unittest discover -s . -p "test_*.py"`).
- **Safety Boundary Coverage**: Verified 100% protection against overlap leakage, HR misparsing, and missing evidence hallucination.

---

## Slide 12 — System Limitations
1. **Corpus Scale**: Curated to 5 empagliflozin publications (`EMP-001` through `EMP-005`).
2. **Survival Analysis**: Cox proportional hazards models and time-to-event survival curves are not implemented.
3. **Tabular Parsing**: Complex multi-row tables rely on LLM structured JSON output mode when live; deterministic fallback extracts single 2×2 comparison rows per chunk.
4. **Prototype Scope**: Research/demo decision-support tool; **not a production clinical decision system**.

---

## Slide 13 — Future Engineering Work
- Expand vector index to full open-access PubMed/PMC biomedical literature using hybrid BM25 + dense retrieval.
- Implement Cox proportional hazards model parsing for Hazard Ratio evidence.
- Build a gold-standard human-annotated extraction benchmark dataset for global precision/recall metrics.
- Support prospective observational registry data and network meta-analyses.

---

## Slide 14 — Key Engineering Lessons
1. **Separation of Concerns**: LLMs should never be trusted with critical numerical calculation; hybrid AI + deterministic architectures are essential for medical AI.
2. **Provenance is Mandatory**: Every extracted metric must remain strictly linked to its original file, chunk ID, page number, and verbatim source quote.
3. **Lineage Prevents Double Counting**: Overlap protection based on study lineage graphs is critical when aggregating clinical literature.
4. **Test Negative Cases**: Robust evaluation benchmarks must explicitly test unstudied outcomes and unsupported metric types.

---

## Slide 15 — Summary & Conclusion
1. **Rigorous Hybrid Architecture**: Combines LLM language flexibility with Python deterministic statistical precision.
2. **Strict Grounding & Traceability**: Verbatim quotes, chunk provenance, and study lineage protect against hallucinations and double counting.
3. **Validated Performance**: 100% pass rate across 11 end-to-end benchmark cases and 167 unit/integration regression tests.
