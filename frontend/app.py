"""LitLens Streamlit frontend entry point."""

import time
from typing import Any, Optional

import streamlit as st

# ── Page config  -  must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="LitLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A3A5C;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
    }

    /* Primary buttons */
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

    /* MeSH tags */
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

    /* Evidence quality badges */
    .badge-strong   { background: #1D8348; color: white; border-radius: 4px; padding: 2px 10px; font-weight: 600; }
    .badge-moderate { background: #D4AC0D; color: white; border-radius: 4px; padding: 2px 10px; font-weight: 600; }
    .badge-mixed    { background: #E67E22; color: white; border-radius: 4px; padding: 2px 10px; font-weight: 600; }
    .badge-weak     { background: #C0392B; color: white; border-radius: 4px; padding: 2px 10px; font-weight: 600; }

    /* Synthesis card */
    .synthesis-box {
        border: 1px solid #117A65;
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #F0FBF8;
        margin-bottom: 1rem;
    }

    /* General font */
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from frontend.components.contradiction_panel import render_contradiction_panel
from frontend.components.eval_dashboard import render_eval_dashboard
from frontend.components.result_card import render_paper_card
from frontend.components.search_bar import render_search_bar
from frontend.utils import api_client


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 LitLens")
    st.markdown("*Biomedical Literature Intelligence*")
    st.divider()
    page = st.radio("Navigation", ["Search", "Eval Dashboard"], label_visibility="collapsed")
    st.divider()
    st.caption("Powered by PubMed · Semantic Scholar · GPT-4o-mini")


def _quality_badge(quality: str) -> str:
    """Return an HTML badge for the given evidence quality level.

    Args:
        quality: One of 'strong', 'moderate', 'mixed', 'weak'.

    Returns:
        HTML span with appropriate colour class.
    """
    label = quality.capitalize()
    css_class = f"badge-{quality.lower()}"
    return f'<span class="{css_class}">{label} Evidence</span>'


def render_synthesis_card(synthesis: dict[str, Any]) -> None:
    """Render the evidence synthesis result card.

    Args:
        synthesis: Synthesis dict from the API response.
    """
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
            st.markdown(f"- {finding.get('finding', '')} {cite_str}")

    gaps = synthesis.get("gaps", [])
    if gaps:
        st.markdown("**Research Gaps**")
        for gap in gaps:
            st.markdown(f"- {gap}")

    limitations = synthesis.get("limitations", "")
    if limitations:
        st.caption(f"*Limitations: {limitations}*")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────────────────────
if page == "Search":
    st.markdown("# 🔬 LitLens  -  Biomedical Literature Search")
    st.markdown(
        "Enter a clinical or research question. LitLens searches PubMed and Semantic Scholar, "
        "deduplicates results, reranks with a cross-encoder, and synthesises the evidence."
    )
    st.divider()

    query, submitted = render_search_bar()

    if submitted and query:
        progress_bar = st.progress(0, text="Expanding query with MeSH terms…")
        spinner_placeholder = st.empty()

        with spinner_placeholder.container():
            with st.spinner("Searching PubMed and Semantic Scholar…"):
                # Animate progress bar while waiting.
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

        # ── Synthesis ─────────────────────────────────────────────────────────
        synthesis = result.get("synthesis")
        if synthesis:
            render_synthesis_card(synthesis)
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

