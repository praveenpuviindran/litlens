"""LitLens Streamlit frontend entry point."""

import time
from typing import Any, Optional

import streamlit as st

st.set_page_config(
    page_title="LitLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #1A3A5C;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #117A65;
        border-color: #117A65;
        color: white;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0E6655;
        border-color: #0E6655;
    }
    .mesh-tag {
        display: inline-block;
        background-color: #D5F5E3;
        color: #1D6A48;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-strong   { background: #1D8348; color: white; border-radius: 4px; padding: 2px 10px; font-weight: 600; }
    .badge-moderate { background: #D4AC0D; color: white; border-radius: 4px; padding: 2px 10px; font-weight: 600; }
    .badge-mixed    { background: #E67E22; color: white; border-radius: 4px; padding: 2px 10px; font-weight: 600; }
    .badge-weak     { background: #C0392B; color: white; border-radius: 4px; padding: 2px 10px; font-weight: 600; }
    .synthesis-box {
        border: 1px solid #117A65;
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #F0FBF8;
        margin-bottom: 1rem;
    }
    .intent-badge-definitional    { background: #2980B9; color: white; border-radius: 4px; padding: 2px 10px; font-size: 0.82rem; font-weight: 600; }
    .intent-badge-comparative     { background: #16A085; color: white; border-radius: 4px; padding: 2px 10px; font-size: 0.82rem; font-weight: 600; }
    .intent-badge-search          { background: #8E44AD; color: white; border-radius: 4px; padding: 2px 10px; font-size: 0.82rem; font-weight: 600; }
    .intent-badge-mechanistic     { background: #D35400; color: white; border-radius: 4px; padding: 2px 10px; font-size: 0.82rem; font-weight: 600; }
    .intent-badge-epidemiological { background: #C0392B; color: white; border-radius: 4px; padding: 2px 10px; font-size: 0.82rem; font-weight: 600; }
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from frontend.components.analytics_dashboard import render_analytics_dashboard
from frontend.components.contradiction_panel import render_contradiction_panel
from frontend.components.eval_dashboard import render_eval_dashboard
from frontend.components.result_card import render_paper_card
from frontend.components.search_bar import render_search_bar
from frontend.utils import api_client

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 LitLens")
    st.markdown("*Biomedical Literature Intelligence*")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Search", "Eval Dashboard", "Analytics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Powered by PubMed · Semantic Scholar · GPT-4o-mini")


# ── Helper: intent badge ───────────────────────────────────────────────────────
def _intent_badge_html(intent: str | None) -> str:
    if not intent:
        return ""
    css_class = f"intent-badge-{intent.lower()}"
    return f'<span class="{css_class}">{intent.capitalize()}</span>'


def _quality_badge(quality: str) -> str:
    label = quality.capitalize()
    css_class = f"badge-{quality.lower()}"
    return f'<span class="{css_class}">{label} Evidence</span>'


def render_synthesis_card(synthesis: dict[str, Any], query_id: str | None = None) -> None:
    """Render the evidence synthesis result card with feedback buttons."""
    quality = synthesis.get("evidence_quality", "mixed").lower()
    badge_html = _quality_badge(quality)

    st.markdown('<div class="synthesis-box">', unsafe_allow_html=True)
    st.markdown(f"### 📋 Evidence Synthesis &nbsp; {badge_html}", unsafe_allow_html=True)
    st.markdown(f"> {synthesis.get('consensus_statement', '')}")

    findings = synthesis.get("key_findings", [])
    if findings:
        st.markdown("**Key Findings**")
        for finding in findings:
            citations = finding.get("citations", [])
            cite_str = " ".join(f"[{c}]" for c in citations) if citations else ""
            conf = finding.get("confidence")
            conf_str = f" *(confidence: {conf})*" if conf else ""
            st.markdown(f"- {finding.get('finding', '')} {cite_str}{conf_str}")

    gaps = synthesis.get("gaps", [])
    if gaps:
        st.markdown("**Research Gaps**")
        for gap in gaps:
            st.markdown(f"- {gap}")

    limitations = synthesis.get("limitations", "")
    if limitations:
        st.caption(f"*Limitations: {limitations}*")

    # ── Recommended follow-up searches ────────────────────────────────────────
    next_searches = synthesis.get("recommended_next_searches", [])
    if next_searches:
        st.markdown("**Recommended follow-up searches:**")
        cols = st.columns(len(next_searches))
        for i, ns in enumerate(next_searches):
            if cols[i].button(f"🔍 {ns}", key=f"followup_{i}_{ns[:20]}"):
                st.session_state["prefill_query"] = ns
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Feedback widget ────────────────────────────────────────────────────────
    if query_id:
        feedback_key = f"feedback_sent_{query_id}"
        if st.session_state.get(feedback_key):
            st.success("Thanks for the feedback!")
        else:
            st.markdown("**Was this synthesis helpful?**")
            fcol1, fcol2, _ = st.columns([1, 1, 6])
            if fcol1.button("👍 Helpful", key=f"helpful_{query_id}"):
                api_client.submit_feedback(query_id, rating=5)
                st.session_state[feedback_key] = True
                st.rerun()
            if fcol2.button("👎 Not helpful", key=f"not_helpful_{query_id}"):
                api_client.submit_feedback(query_id, rating=1)
                st.session_state[feedback_key] = True
                st.rerun()


# ── Pages ──────────────────────────────────────────────────────────────────────
if page == "Search":
    st.markdown("# 🔬 LitLens  -  Biomedical Literature Search")
    st.markdown(
        "Enter a clinical or research question. LitLens searches PubMed and Semantic Scholar, "
        "deduplicates results, reranks with a cross-encoder, and synthesises the evidence."
    )
    st.divider()

    # Pre-fill query if a follow-up button was clicked
    prefill = st.session_state.pop("prefill_query", "")
    query, submitted = render_search_bar(default_value=prefill)

    if submitted and query:
        progress_bar = st.progress(0, text="Classifying query intent…")
        spinner_placeholder = st.empty()

        with spinner_placeholder.container():
            with st.spinner("Searching PubMed and Semantic Scholar…"):
                for pct in range(0, 80, 5):
                    progress_bar.progress(pct, text="Fetching and processing papers…")
                    time.sleep(0.05)

                try:
                    result = api_client.search(query)
                    progress_bar.progress(100, text="Done!")
                    time.sleep(0.3)
                    progress_bar.empty()
                except RuntimeError as exc:
                    progress_bar.empty()
                    st.error(f"Search failed: {exc}")
                    st.stop()

        spinner_placeholder.empty()

        if result.get("cached"):
            st.info("⚡ Result from cache  -  no API calls made.")

        # ── Intent badge + routing info ────────────────────────────────────────
        intent = result.get("intent")
        if intent:
            from backend.services.intent_classifier import INTENT_PIPELINE_CONFIG, QueryIntent
            intent_badge = _intent_badge_html(intent)
            try:
                config = INTENT_PIPELINE_CONFIG.get(QueryIntent(intent), {})
                max_p = config.get("max_papers", 10)
                style = config.get("synthesis_style", "standard")
                routing_note = f"Searched for up to {max_p} papers · {style} synthesis"
            except (ValueError, KeyError):
                routing_note = ""
            st.markdown(
                f"**Query intent:** {intent_badge} &nbsp; <span style='color:#555;font-size:0.9rem'>{routing_note}</span>",
                unsafe_allow_html=True,
            )

        # ── Synthesis ─────────────────────────────────────────────────────────
        synthesis = result.get("synthesis")
        query_id = str(result.get("query_id", ""))
        if synthesis:
            render_synthesis_card(synthesis, query_id=query_id or None)
        else:
            st.warning("No synthesis was generated. Review the papers below directly.")

        # ── Contradictions ────────────────────────────────────────────────────
        contradictions = result.get("contradictions", [])
        render_contradiction_panel(contradictions)

        # ── Papers grid ───────────────────────────────────────────────────────
        papers = result.get("papers", [])
        if papers:
            st.markdown("---")
            st.markdown(f"### 📄 Top {len(papers)} Papers")
            for i, paper in enumerate(papers, start=1):
                render_paper_card(paper, i)
        else:
            st.info("No papers were returned. Try broadening your query.")

        # ── Query info footer ─────────────────────────────────────────────────
        st.markdown("---")
        expanded_query = result.get("expanded_pubmed_query", "")
        latency_ms = result.get("latency_ms", 0)

        if expanded_query:
            with st.expander("Expanded PubMed Query", expanded=False):
                st.code(expanded_query, language="text")

        st.caption(
            f"Total retrieved: {result.get('total_retrieved', 0)} papers · "
            f"Latency: {latency_ms:,} ms"
        )

elif page == "Eval Dashboard":
    render_eval_dashboard()

elif page == "Analytics":
    render_analytics_dashboard()
