"""Contradiction panel component  -  displays detected contradictions between papers."""

from typing import Any

import streamlit as st


def render_contradiction_panel(contradictions: list[dict[str, Any]]) -> None:
    """Render the contradiction detection panel.

    Only displayed when contradictions were found. Shows a warning banner and
    a two-column layout for each contradiction.

    Args:
        contradictions: List of contradiction dicts from the API response.
    """
    if not contradictions:
        return

    st.markdown("---")
    st.warning(f"⚠️ {len(contradictions)} contradiction(s) detected in retrieved literature")

    for i, c in enumerate(contradictions, start=1):
        st.markdown(f"**Contradiction {i}**")

        col_a, col_vs, col_b = st.columns([5, 1, 5])

        with col_a:
            title_a = c.get("paper_a_title", "Paper A")
            claim_a = c.get("claim_a", "No claim extracted.")
            st.markdown(f"**{title_a}**")
            st.info(claim_a)

        with col_vs:
            st.markdown("<div style='text-align:center;font-weight:bold;padding-top:2rem'>vs.</div>", unsafe_allow_html=True)

        with col_b:
            title_b = c.get("paper_b_title", "Paper B")
            claim_b = c.get("claim_b", "No claim extracted.")
            st.markdown(f"**{title_b}**")
            st.info(claim_b)

        note = c.get("methodological_note")
        if note:
            st.caption(f"💡 {note}")

        conf = c.get("confidence")
        if conf is not None:
            st.caption(f"Confidence: {conf:.0%}")

        if i < len(contradictions):
            st.divider()
