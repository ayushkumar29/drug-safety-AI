"""
Streamlit MVP Prototype UI for Drug Safety AI (app/streamlit_app.py)

Evidence-grounded analysis of drug safety questions using LLM-assisted retrieval
and extraction with deterministic 2x2 statistics and study-lineage protection.

This application is a presentation layer. It consumes pre-computed PipelineResult objects
and performs ZERO statistical calculations or formula evaluations inside the UI.
"""

import sys
import os
import html
from typing import Dict, Any, List, Optional

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from pipeline.pipeline import run_analysis, PipelineResult


# ==========================================
# Pure Formatting & Helper Functions for Testability
# ==========================================

def format_hypothesis_summary(hypothesis: Any) -> Dict[str, str]:
    """Extracts clean display strings from StructuredHypothesis object or dict."""
    if not hypothesis:
        return {}

    if hasattr(hypothesis, "model_dump"):
        data = hypothesis.model_dump()
    elif isinstance(hypothesis, dict):
        data = hypothesis
    else:
        data = getattr(hypothesis, "__dict__", {})

    return {
        "Drug": str(data.get("drug") or "N/A"),
        "Comparator": str(data.get("comparator") or "N/A"),
        "Outcome": str(data.get("outcome") or "N/A"),
        "Population": str(data.get("population") or "N/A"),
        "Study Type": str(data.get("study_type") or "N/A"),
        "Hypothesis Direction": str(data.get("hypothesis_direction") or "N/A"),
        "Null Hypothesis": str(data.get("null_hypothesis") or "N/A"),
        "Alternative Hypothesis": str(data.get("alternative_hypothesis") or "N/A"),
    }


def format_evidence_record(ev: Any) -> Dict[str, Any]:
    """Normalizes an ExtractedEvidence object or dict into a display-safe record."""
    if not ev:
        return {}

    if hasattr(ev, "model_dump"):
        data = ev.model_dump()
    elif isinstance(ev, dict):
        data = ev
    else:
        data = getattr(ev, "__dict__", {})

    p_start = data.get("page_start")
    p_end = data.get("page_end")
    if p_start is not None and p_end is not None:
        page_str = f"{p_start} - {p_end}" if p_start != p_end else str(p_start)
    elif p_start is not None:
        page_str = str(p_start)
    else:
        page_str = "N/A"

    return {
        "study_id": str(data.get("study_id") or "N/A"),
        "drug": str(data.get("drug") or "N/A"),
        "comparator": str(data.get("comparator") or "N/A"),
        "outcome": str(data.get("outcome") or "N/A"),
        "study_type": str(data.get("study_type") or "N/A"),
        "treatment_n": data.get("treatment_n"),
        "treatment_events": data.get("treatment_events"),
        "control_n": data.get("control_n"),
        "control_events": data.get("control_events"),
        "dose": str(data.get("dose") or "N/A"),
        "follow_up": str(data.get("follow_up") or "N/A"),
        "source": str(data.get("source") or "N/A"),
        "publication_year": data.get("publication_year"),
        "source_file": str(data.get("source_file") or "N/A"),
        "chunk_id": str(data.get("chunk_id") or "N/A"),
        "page_range": page_str,
        "evidence_quote": str(data.get("evidence_quote") or ""),
        "extracted_metric_type": str(data.get("extracted_metric_type") or "unknown"),
    }


def format_lineage_selection(selected_evidence: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorizes lineage selection records into retained, excluded, and context-only groups."""
    retained = []
    excluded = []
    context = []

    for sel in selected_evidence or []:
        sec = sel.get("selection", {}) if isinstance(sel, dict) else getattr(sel, "selection", {})
        pub_id = sec.get("publication_id", "Unknown") if isinstance(sec, dict) else getattr(sec, "publication_id", "Unknown")
        status = sec.get("selection_status", "unknown") if isinstance(sec, dict) else getattr(sec, "selection_status", "unknown")
        reason = sec.get("reason", "N/A") if isinstance(sec, dict) else getattr(sec, "reason", "N/A")

        rec = {"publication_id": pub_id, "status": status, "reason": reason}

        if status == "retained_for_independent_analysis":
            retained.append(rec)
        elif status == "excluded_due_to_overlap":
            excluded.append(rec)
        else:
            context.append(rec)

    return {"retained": retained, "excluded": excluded, "context": context}


def format_statistics_summary(result: Any) -> Dict[str, Any]:
    """Consumes PipelineResult and extracts pre-calculated statistical values safely."""
    if not result:
        return {"status": "none", "records": [], "has_hr": False, "has_rate": False}

    stats_status = getattr(result, "stage_status", {}).get("statistics", "pending")
    records = []
    has_hr = False
    has_rate = False

    validated = getattr(result, "validated_evidence", []) or []
    for ev in validated:
        m_type = getattr(ev, "extracted_metric_type", "") if hasattr(ev, "extracted_metric_type") else (ev.get("extracted_metric_type") if isinstance(ev, dict) else "")
        if m_type == "hazard_ratio":
            has_hr = True
        elif m_type in ["incidence_rate", "percentage"]:
            has_rate = True

    comp = getattr(result, "study_comparison", None)
    if comp and getattr(comp, "study_records", None):
        for rec in comp.study_records:
            item = {
                "study_id": getattr(rec, "study_id", "N/A"),
                "study_type": getattr(rec, "study_type", "N/A"),
                "dose": getattr(rec, "dose", "N/A"),
                "treatment_events": getattr(rec, "treatment_events", None),
                "treatment_n": getattr(rec, "treatment_n", None),
                "control_events": getattr(rec, "control_events", None),
                "control_n": getattr(rec, "control_n", None),
            }
            rr = getattr(rec, "risk_ratio_result", None)
            if rr:
                item["risk_ratio"] = getattr(rr, "risk_ratio", None)
                item["rr_ci_lower"] = getattr(rr, "ci_lower", None)
                item["rr_ci_upper"] = getattr(rr, "ci_upper", None)

            or_res = getattr(rec, "odds_ratio_result", None)
            if or_res:
                item["odds_ratio"] = getattr(or_res, "odds_ratio", None)
                item["or_ci_lower"] = getattr(or_res, "ci_lower", None)
                item["or_ci_upper"] = getattr(or_res, "ci_upper", None)

            ht = getattr(rec, "hypothesis_test_result", None)
            if ht:
                item["fisher_p_value"] = getattr(ht, "fisher_p_value", None)
                item["chi2_p_value"] = getattr(ht, "chi_square_p_value", None)

            records.append(item)

    return {
        "status": stats_status,
        "records": records,
        "has_hr": has_hr,
        "has_rate": has_rate
    }


def format_retrieved_candidate(chunk: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Formats a single retrieved RAG chunk dictionary for presentation."""
    p_start = chunk.get("page_start")
    p_end = chunk.get("page_end")
    if p_start is not None and p_end is not None:
        page_str = f"{p_start} - {p_end}" if p_start != p_end else str(p_start)
    elif p_start is not None:
        page_str = str(p_start)
    else:
        page_str = "N/A"

    text = chunk.get("text", "")
    snippet = text[:200] + ("..." if len(text) > 200 else "")

    return {
        "rank": rank,
        "chunk_id": str(chunk.get("chunk_id") or "N/A"),
        "study_id": str(chunk.get("study_id") or "N/A"),
        "source_file": str(chunk.get("source_file") or "N/A"),
        "page_range": page_str,
        "relevance_score": round(float(chunk.get("combined_score", 0.0)), 4),
        "snippet": snippet
    }


def generate_forest_plot_html(records: List[Dict[str, Any]]) -> str:
    """
    Renders a zero-dependency SVG Forest Plot for existing pre-computed 2x2 statistics.
    Executes ZERO statistical calculations in UI code; only reads pre-calculated Risk Ratios and 95% CIs.
    """
    valid_recs = [r for r in records if r.get("risk_ratio") is not None and r.get("rr_ci_lower") is not None and r.get("rr_ci_upper") is not None]
    if not valid_recs:
        return ""

    width = 650
    row_height = 50
    height = 60 + len(valid_recs) * row_height
    x_min_val, x_max_val = 0.5, 10.0
    margin_left, margin_right = 160, 200
    plot_width = width - margin_left - margin_right

    def val_to_x(v):
        v_clamped = max(x_min_val, min(x_max_val, v))
        return margin_left + ((v_clamped - x_min_val) / (x_max_val - x_min_val)) * plot_width

    x_null = val_to_x(1.0)

    svg_lines = [
        f'<svg width="{width}" height="{height}" style="background-color: #1e1e1e; font-family: sans-serif; border-radius: 8px;">',
        f'<text x="{width/2}" y="22" fill="#ffffff" font-size="14" font-weight="bold" text-anchor="middle">Forest Plot — Risk Ratio (95% CI)</text>',
        f'<line x1="{x_null}" y1="35" x2="{x_null}" y2="{height-20}" stroke="#ff6b6b" stroke-dasharray="4,4" stroke-width="2"/>',
        f'<text x="{x_null}" y="{height-5}" fill="#ff6b6b" font-size="11" text-anchor="middle">Line of No Effect (RR=1.0)</text>'
    ]

    y = 55
    for r in valid_recs:
        s_id = html.escape(r.get("study_id", "Study"))
        rr = r.get("risk_ratio", 1.0)
        ci_l = r.get("rr_ci_lower", 1.0)
        ci_u = r.get("rr_ci_upper", 1.0)

        cx = val_to_x(rr)
        lx = val_to_x(ci_l)
        rx = val_to_x(ci_u)

        svg_lines.append(f'<text x="15" y="{y+4}" fill="#ffffff" font-size="12" font-weight="bold">{s_id}</text>')
        svg_lines.append(f'<line x1="{lx}" y1="{y}" x2="{rx}" y2="{y}" stroke="#4dabf7" stroke-width="3"/>')
        svg_lines.append(f'<line x1="{lx}" y1="{y-6}" x2="{lx}" y2="{y+6}" stroke="#4dabf7" stroke-width="2"/>')
        svg_lines.append(f'<line x1="{rx}" y1="{y-6}" x2="{rx}" y2="{y+6}" stroke="#4dabf7" stroke-width="2"/>')
        svg_lines.append(f'<rect x="{cx-5}" y="{y-5}" width="10" height="10" fill="#38d9a9" />')
        svg_lines.append(f'<text x="{width-180}" y="{y+4}" fill="#ffffff" font-size="12">RR {rr:.2f} [{ci_l:.2f}, {ci_u:.2f}]</text>')
        y += row_height

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


# ==========================================
# Streamlit UI Rendering Functions
# ==========================================

def render_app():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit library not installed. Please install streamlit to launch UI.")
        return

    st.set_page_config(
        page_title="AI-Assisted Drug Safety Evidence Analyzer",
        page_icon="💊",
        layout="wide"
    )

    # Title & Subtitle
    st.title("💊 AI-Assisted Drug Safety Evidence Analyzer")
    st.markdown(
        "**Evidence-grounded analysis of drug safety questions using "
        "LLM-assisted retrieval and extraction with deterministic statistics.**"
    )
    st.divider()

    # 1. Research Question Input Area
    st.subheader("❓ 1. Research Question")
    question_input = st.text_area(
        "Enter your drug safety question:",
        value="Does empagliflozin increase the risk of genital infection compared with placebo?",
        height=90,
        placeholder="e.g. Does empagliflozin increase the risk of genital infection compared with placebo?"
    )

    analyze_btn = st.button("🔍 Analyze Evidence", type="primary")

    if not analyze_btn and "pipeline_result" not in st.session_state:
        st.info("💡 **Getting Started**: Enter a drug safety research question above and click **Analyze Evidence** to execute the 10-stage analysis pipeline.")
        st.divider()

    if analyze_btn:
        if not question_input or not question_input.strip():
            st.warning("Please enter a valid drug safety question before submitting.")
            return

        with st.spinner("Analyzing evidence across corpus (retrieval, extraction, validation, lineage, statistics)..."):
            try:
                result = run_analysis(question_input.strip(), top_k=25)
                st.session_state["pipeline_result"] = result
            except Exception as e:
                st.error(f"Pipeline Execution Failed: {str(e)}")
                return

    # Render Results if available in Session State
    if "pipeline_result" in st.session_state:
        result: PipelineResult = st.session_state["pipeline_result"]

        # --------------------------------------------------
        # 2. Executive Summary & Structured Hypothesis
        # --------------------------------------------------
        st.subheader("📋 2. Executive Summary & Structured Hypothesis")
        if result.hypothesis:
            hyp = format_hypothesis_summary(result.hypothesis)
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Drug", hyp.get("Drug", "N/A"))
            with c2:
                st.metric("Comparator", hyp.get("Comparator", "N/A"))
            with c3:
                st.metric("Outcome", hyp.get("Outcome", "N/A"))
            with c4:
                st.metric("Population", hyp.get("Population", "N/A"))
            with c5:
                st.metric("Study Type", hyp.get("Study Type", "N/A"))
            with c6:
                st.metric("Direction", hyp.get("Hypothesis Direction", "N/A"))

            with st.expander("Formal Hypothesis Statements", expanded=False):
                st.markdown(f"**Null Hypothesis ($H_0$):** {hyp.get('Null Hypothesis')}")
                st.markdown(f"**Alternative Hypothesis ($H_1$):** {hyp.get('Alternative Hypothesis')}")
        else:
            st.warning("No structured hypothesis parsed.")

        st.divider()

        # --------------------------------------------------
        # 3. Independent Primary Evidence & Evidence Cards
        # --------------------------------------------------
        st.subheader("📚 3. Evidence Summary & Independent Evidence")
        ind_evidence = result.independent_evidence or []

        if not ind_evidence:
            st.warning("ℹ️ **No Independent Evidence Found**: No independent primary evidence supporting this question was found in the current curated corpus. The system did not fabricate evidence or perform unsupported statistical calculations.")
        else:
            # Table View
            if PANDAS_AVAILABLE:
                table_rows = []
                for ev_obj in ind_evidence:
                    rec = format_evidence_record(ev_obj)
                    table_rows.append({
                        "Study ID": rec["study_id"],
                        "Drug": rec["drug"],
                        "Comparator": rec["comparator"],
                        "Trt Events": rec["treatment_events"],
                        "Trt N": rec["treatment_n"],
                        "Ctrl Events": rec["control_events"],
                        "Ctrl N": rec["control_n"],
                        "Metric": rec["extracted_metric_type"],
                        "Source File": rec["source_file"],
                        "Pages": rec["page_range"]
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # Expandable Evidence Cards
            for idx, raw_ev in enumerate(ind_evidence, 1):
                ev = format_evidence_record(raw_ev)
                with st.expander(f"📄 Evidence Card #{idx}: {ev['study_id']} ({ev['study_type']})", expanded=True):
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.write(f"**Treatment ({ev['drug']}):** {ev['treatment_events']} / {ev['treatment_n']}")
                        st.write(f"**Control ({ev['comparator']}):** {ev['control_events']} / {ev['control_n']}")
                    with m2:
                        st.write(f"**Dose:** {ev['dose']}")
                        st.write(f"**Follow-up:** {ev['follow_up']}")
                    with m3:
                        st.write(f"**Publication Year:** {ev['publication_year']}")
                        st.write(f"**Metric Type:** `{ev['extracted_metric_type']}`")
                    with m4:
                        st.write(f"**Source File:** `{ev['source_file']}`")
                        st.write(f"**Chunk ID:** `{ev['chunk_id']}` (p. {ev['page_range']})")

                    st.markdown("**Source Evidence:**")
                    st.code(ev["evidence_quote"], language="text")

        st.divider()

        # --------------------------------------------------
        # 4. Study Lineage & Overlap Protection
        # --------------------------------------------------
        st.subheader("🧬 4. Study Lineage & Evidence Selection")
        lineage = format_lineage_selection(result.selected_evidence)

        l1, l2, l3 = st.columns(3)
        with l1:
            st.markdown("#### ✅ RETAINED (Independent)")
            if lineage["retained"]:
                for r in lineage["retained"]:
                    st.success(f"**{r['publication_id']}**\nPrimary Trial\n✓ Independent")
            else:
                st.write("None")

        with l2:
            st.markdown("#### 🛑 EXCLUDED (Overlap)")
            if lineage["excluded"]:
                for r in lineage["excluded"]:
                    st.error(f"**{r['publication_id']}**\nPooled analysis\n→ {r['reason']}")
            else:
                st.write("None")

        with l3:
            st.markdown("#### 📖 CONTEXT (Secondary)")
            if lineage["context"]:
                for r in lineage["context"]:
                    st.info(f"**{r['publication_id']}**\nSystematic Review / Context")
            else:
                st.write("None")

        st.caption("🛡️ **Overlap Protection**: Overlapping publications (such as post-hoc pooled safety analyses) are identified via study lineage and excluded from independent statistical calculations to prevent double counting.")

        st.divider()

        # --------------------------------------------------
        # 5. Statistical Analysis & Forest-Style Visualization
        # --------------------------------------------------
        st.subheader("📊 5. Statistical Analysis")
        stats = format_statistics_summary(result)

        if stats["has_hr"]:
            st.warning("⚠️ **Hazard Ratio Evidence Identified**: Survival-analysis statistics are not currently implemented in this MVP. The evidence is preserved but is not converted into 2×2 event counts.")

        if stats["has_rate"]:
            st.info("ℹ️ **Incidence Rates / Percentages Identified**: Rates and percentages are preserved as reported metrics and are not treated as raw 2×2 event counts.")

        if stats["status"] == "completed" and stats["records"]:
            for rec in stats["records"]:
                st.markdown(f"#### Study: {rec['study_id']} (Dose: {rec['dose']})")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    rr = rec.get("risk_ratio")
                    l_rr = rec.get("rr_ci_lower")
                    u_rr = rec.get("rr_ci_upper")
                    if rr is not None:
                        st.metric("Risk Ratio (RR)", f"{rr:.2f}", f"95% CI: {l_rr:.2f} – {u_rr:.2f}")
                    else:
                        st.metric("Risk Ratio", "N/A")

                with m2:
                    odds = rec.get("odds_ratio")
                    l_or = rec.get("or_ci_lower")
                    u_or = rec.get("or_ci_upper")
                    if odds is not None:
                        st.metric("Odds Ratio (OR)", f"{odds:.2f}", f"95% CI: {l_or:.2f} – {u_or:.2f}")
                    else:
                        st.metric("Odds Ratio", "N/A")

                with m3:
                    fp = rec.get("fisher_p_value")
                    st.metric("Fisher Exact Test", f"p = {fp:.6f}" if fp is not None else "N/A")

                with m4:
                    cp = rec.get("chi2_p_value")
                    st.metric("Chi-Square Test", f"p = {cp:.6f}" if cp is not None else "N/A")

            # Simple Forest-Style SVG Plot (Consuming pre-calculated statistics only)
            forest_html = generate_forest_plot_html(stats["records"])
            if forest_html:
                st.markdown("#### Forest Plot Visualization")
                st.markdown(forest_html, unsafe_allow_html=True)

        else:
            st.info("ℹ️ **Statistics Skipped**: 2×2 statistical analysis was not performed because no eligible independent event-count evidence was available.")

        st.divider()

        # --------------------------------------------------
        # 6. Evidence-Based Analysis (Grounded Report)
        # --------------------------------------------------
        st.subheader("📄 6. Evidence-Based Analysis")
        if result.generated_report:
            st.markdown(result.generated_report)
        elif result.deterministic_report:
            st.markdown(f"**Research Question:** {result.deterministic_report.research_question}")
            st.markdown(f"**Citations:** {len(result.deterministic_report.citations)} source citations attached.")
        else:
            st.write("No report generated.")

        st.divider()

        # --------------------------------------------------
        # 7. Retrieved Evidence Sources (Expander)
        # --------------------------------------------------
        with st.expander("🔍 7. Retrieved Evidence Sources", expanded=False):
            retrieved_rows = []
            for rank, chk in enumerate(result.retrieved_chunks, 1):
                retrieved_rows.append(format_retrieved_candidate(chk, rank))

            if PANDAS_AVAILABLE and retrieved_rows:
                st.dataframe(pd.DataFrame(retrieved_rows), use_container_width=True)
            else:
                for r in retrieved_rows:
                    st.markdown(f"**#{r['rank']} {r['chunk_id']}** (Study: `{r['study_id']}`, Score: `{r['relevance_score']}`)\n*{r['snippet']}*")

        # --------------------------------------------------
        # 8. Pipeline Execution Details (Expander)
        # --------------------------------------------------
        with st.expander("🛠️ 8. Pipeline Execution Details", expanded=False):
            st.json(result.stage_status)
            if result.errors:
                st.error("Pipeline Errors:")
                st.json(result.errors)

    # --------------------------------------------------
    # Persistent Disclaimer
    # --------------------------------------------------
    st.divider()
    st.caption(
        "⚠️ **Disclaimer**: This prototype is a research-support and evidence-analysis tool. "
        "It does not determine whether a drug is clinically safe or unsafe and is not a substitute for medical, "
        "clinical, or regulatory judgment."
    )


if __name__ == "__main__":
    render_app()
