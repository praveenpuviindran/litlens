"""Search bar component for the LitLens Streamlit app."""

from typing import Optional

import streamlit as st


def render_search_bar() -> tuple[Optional[str], bool]:
    """Render the query input area and search button.

    Returns:
        Tuple of (query string or None, submitted bool).
        submitted is True only on the frame where the button was clicked.
    """
    st.markdown("### Ask a Research Question")
    query = st.text_area(
        label="Research question",
        placeholder=(
            "What is the effect of metformin on cardiovascular outcomes in Type 2 diabetes?"
        ),
        height=100,
        label_visibility="collapsed",
        key="search_query",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submitted = st.button("🔬 Search Literature", type="primary", use_container_width=True)
    with col2:
        st.caption("Searches PubMed and Semantic Scholar · Powered by GPT-4o-mini")

    if submitted and not query.strip():
        st.warning("Please enter a research question before searching.")
        return None, False

    return query.strip() if query.strip() else None, submitted
