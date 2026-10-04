# ⏱️ 60-Second Interview Elevator Pitch

> *"I built a **Hybrid AI and Deterministic Pharmacovigilance Pipeline** for drug safety evidence analysis using RAG, LLM extraction, study lineage protection, and deterministic 2×2 statistical testing.*
>
> *The problem with standard LLMs in medical AI is that they hallucinate numbers and cannot be trusted for statistical calculations. Furthermore, naive RAG systems double-count patients when post-hoc pooled safety analyses overlap with primary clinical trials.*
>
> *Our system solves this with a strict architectural boundary: we use the LLM solely for natural language hypothesis parsing, candidate chunk retrieval, and structured JSON table extraction. Once extracted, a Python validation engine verifies numerical bounds, a study-lineage graph excludes overlapping pooled publications to protect against double counting, and a deterministic Python engine computes the Risk Ratio, Odds Ratio, Fisher exact test, and Chi-square statistics.*
>
> *Across an 11-case end-to-end benchmark and a 167-test regression suite, the system achieved a 100% pass rate while safely blocking non-2×2 Hazard Ratio metrics and unstudied outcomes without hallucination."*
