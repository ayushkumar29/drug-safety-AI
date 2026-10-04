# 🎬 5-Minute Technical Live Demo Sequence & Checklist

---

## 📋 Pre-Demo Environment Checklist
- [ ] Terminal 1: Virtual environment activated (`venv\Scripts\activate`).
- [ ] Verify Streamlit app launches: `streamlit run app/streamlit_app.py`.
- [ ] Browser window open at `http://localhost:8501`.

---

## ⏱️ Step-by-Step Live Demo Sequence

### Step 1: Launch Application (30s)
- **Action**: Open `http://localhost:8501`.
- **Narrative**: *"Welcome to the AI-Assisted Drug Safety Evidence Analyzer. This application combines LLM-assisted retrieval and extraction with a deterministic Python statistical engine and study-lineage protection."*

### Step 2: Enter Golden Path Question (30s)
- **Action**: In Section 1, enter or verify:  
  `"Does empagliflozin increase the risk of genital infection compared with placebo?"`
- **Action**: Click **Analyze Evidence**.
- **Narrative**: *"We submit our research question. The pipeline executes natural-language hypothesis parsing, RAG retrieval across our 55-chunk corpus, structured table extraction, validation, lineage protection, and deterministic 2×2 statistical testing."*

### Step 3: Show Executive Summary & Structured Hypothesis (30s)
- **Action**: Scroll to Section 2.
- **Point out**: `Drug: empagliflozin`, `Comparator: placebo`, `Outcome: genital infection`, `Population: type 2 diabetes`, `Study Type: randomized controlled trial`.
- **Narrative**: *"The LLM structures the raw question into formal Pydantic parameters and generates testable null and alternative hypotheses."*

### Step 4: Show Evidence Cards & Verbatim Quotes (40s)
- **Action**: Scroll to Section 3 and expand Evidence Card #1 (`EMP-005`).
- **Point out**: Treatment events `301 / 4687`, Control events `42 / 2333`, Pages `8-10`, Chunk `EMP-005_CHK_006`.
- **Narrative**: *"Here is our primary evidence record extracted from EMP-005. Notice the verbatim quote display — we retain exact text quotes so clinical reviewers can verify numbers against source PDFs."*

### Step 5: Show Lineage & Overlap Protection (40s)
- **Action**: Scroll to Section 4 (Study Lineage).
- **Point out**:
  - ✅ **RETAINED**: `EMP-005` (Primary RCT)
  - 🛑 **EXCLUDED**: `EMP-001` and `EMP-003` (Pooled safety analyses; Reason: `Overlaps with EMPA-REG_OUTCOME`)
- **Narrative**: *"This is our overlap protection engine. EMP-001 and EMP-003 analyze the exact same EMPA-REG OUTCOME patient cohort. Our lineage resolver excludes them from 2×2 statistical calculations to prevent double counting."*

### Step 6: Show Deterministic Statistics & SVG Forest Plot (50s)
- **Action**: Scroll to Section 5 (Statistical Analysis).
- **Point out**:
  - **Risk Ratio (RR)**: `3.57` (95% CI `2.59 – 4.91`)
  - **Odds Ratio (OR)**: `3.74` (95% CI `2.70 – 5.19`)
  - **Fisher Exact Test**: $p < 0.0001$ (`7.72e-20`)
- **Point out Forest Plot**: Point to the SVG Forest Plot displaying $RR = 3.57$ relative to the dashed line ($RR = 1.0$).
- **Narrative**: *"All four statistical metrics were calculated deterministically in standard Python code outside Streamlit. The UI contains zero math formulas."*

### Step 7: Demonstrate Hazard Ratio Safety Boundary (40s)
- **Action**: Enter question:  
  `"Does empagliflozin reduce 3-point MACE cardiovascular death compared with placebo in EMPA-REG OUTCOME?"`
- **Action**: Click **Analyze Evidence**.
- **Point out**: Warning banner: *"Hazard Ratio evidence identified. Survival-analysis statistics are not currently implemented in this MVP. The evidence is preserved but is not converted into 2×2 event counts."*
- **Narrative**: *"Notice how the system handles Hazard Ratios. It extracts the HR evidence but blocks the 2×2 event-count engine, preventing invalid metric conversions."*

### Step 8: Demonstrate No-Evidence Handling (30s)
- **Action**: Enter question:  
  `"Does empagliflozin cause pulmonary fibrosis?"`
- **Action**: Click **Analyze Evidence**.
- **Point out**: Info banner: *"No independent evidence supporting this question was found in the current curated corpus."*
- **Narrative**: *"For unstudied outcomes, the system extracts zero evidence, skips statistics safely, and refrains from fabricating numbers."*

### Step 9: Close with System Limitations (30s)
- **Narrative**: *"In conclusion, this research prototype demonstrates end-to-end grounded evidence extraction and statistical protection. It is designed as a clinical decision-support tool and does not replace regulatory judgment."*
