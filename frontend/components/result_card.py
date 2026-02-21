"""Result card component — renders a single paper in a collapsible expander."""

from typing import Any

import streamlit as st


def render_paper_card(paper: dict[str, Any], index: int) -> None:
    """Render a single paper as a styled expander card.

    Args:
        paper: Paper dict from the API response.
        index: 1-based position in the results list (displayed as a badge).
    """
    title = paper.get("title", "Untitled")
    abstract = paper.get("abstract", "No abstract available.")
    authors: list[str] = paper.get("authors") or []
    journal = paper.get("journal", "")
    year = paper.get("publication_year", "")
    citations = paper.get("citation_count", 0)
    mesh_terms: list[str] = paper.get("mesh_terms") or []
    pubmed_id = paper.get("pubmed_id")
    oa_url = paper.get("open_access_url")
    source = paper.get("source", "")

    # Truncate author list to first 3 + "et al."
    if len(authors) > 3:
        author_display = ", ".join(authors[:3]) + " et al."
    else:
        author_display = ", ".join(authors) if authors else "Unknown Authors"

    label = f"[{index}] {title}"

    with st.expander(label, expanded=False):
        # Author / journal / year line
        meta_parts = [author_display]
        if journal:
            meta_parts.append(f"*{journal}*")
        if year:
            meta_parts.append(str(year))
        st.markdown(" · ".join(meta_parts))

        # Citation count and source badge
        badge_col, link_col = st.columns([3, 1])
        with badge_col:
            if citations:
                st.caption(f"📊 {citations:,} citations")
            if source:
                source_label = {"pubmed": "PubMed", "semantic_scholar": "Semantic Scholar", "both": "PubMed + S2"}.get(source, source)
                st.caption(f"Source: {source_label}")

        with link_col:
            if oa_url:
                st.link_button("📄 Full Text", oa_url, use_container_width=True)
            elif pubmed_id:
                st.link_button(
                    "🔗 PubMed",
                    f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                    use_container_width=True,
                )

        # MeSH tags
        if mesh_terms:
            tags_html = " ".join(
                f'<span class="mesh-tag">{term}</span>' for term in mesh_terms[:8]
            )
            st.markdown(tags_html, unsafe_allow_html=True)

        # Abstract
        st.markdown("**Abstract**")
        st.markdown(abstract)
