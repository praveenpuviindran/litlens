"""LitLens - Biomedical Literature Intelligence Engine.

Built by Praveen Puviindran.
Deployed on Streamlit Community Cloud. No API keys required.
"""

import time

import streamlit as st


@st.cache_resource(show_spinner="Loading semantic model...")
def _load_encoder():
    """Load sentence-transformers encoder once and cache across sessions."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

st.set_page_config(
    page_title="LitLens | Biomedical Literature Search",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2545 0%, #1A3A5C 100%);
}
section[data-testid="stSidebar"] * { color: #E8F4FD !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] .stRadio label { color: #E8F4FD !important; }
section[data-testid="stSidebar"] hr { border-color: #2D5F8A; }

.stButton > button[kind="primary"] {
    background: #0E7C6B; border: none; color: white;
    font-weight: 700; font-size: 1rem; padding: 0.6rem 1.4rem;
    border-radius: 6px; transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover { background: #0B6559; }

.synthesis-card {
    border: 1.5px solid #0E7C6B; border-radius: 10px;
    padding: 1.6rem 1.8rem; background: #F0FAF8;
    margin-bottom: 1.2rem;
}
.direct-answer {
    font-size: 1.08rem; font-weight: 600; color: #0B2545;
    border-left: 4px solid #0E7C6B; padding-left: 1rem;
    margin-bottom: 1rem; line-height: 1.6;
}
.about-card {
    border: 1px solid #CBD5E0; border-radius: 10px;
    padding: 1.6rem 1.8rem; background: #FAFBFC;
    margin-bottom: 1rem;
}
.badge-strong   { background:#1D8348; color:white; border-radius:5px; padding:3px 12px; font-weight:700; font-size:.85rem; }
.badge-moderate { background:#D4AC0D; color:white; border-radius:5px; padding:3px 12px; font-weight:700; font-size:.85rem; }
.badge-mixed    { background:#E67E22; color:white; border-radius:5px; padding:3px 12px; font-weight:700; font-size:.85rem; }
.badge-weak     { background:#C0392B; color:white; border-radius:5px; padding:3px 12px; font-weight:700; font-size:.85rem; }
.mesh-tag {
    display:inline-block; background:#D5F5E3; color:#1D6A48;
    border-radius:4px; padding:2px 8px; margin:2px;
    font-size:.74rem; font-weight:600;
}
.step-badge {
    display:inline-block; background:#0B2545; color:#E8F4FD;
    border-radius:50%; width:28px; height:28px; line-height:28px;
    text-align:center; font-weight:700; margin-right:8px; font-size:.85rem;
}
html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## LitLens")
    st.markdown("*Biomedical Literature Intelligence*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Search", "About This Project", "How It Works"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
**Data sources**
- PubMed (NCBI E-utilities)
- Semantic Scholar

No API keys required.
All processing uses open-source libraries.
    """)
    st.markdown("---")
    st.markdown("""
Built by **Praveen Puviindran**

[GitHub](https://github.com/praveenpuviindran) · [LinkedIn](https://linkedin.com/in/praveenpuviindran)
    """)


# ── SEARCH PAGE ────────────────────────────────────────────────────────────────
if page == "Search":
    st.markdown("# LitLens")
    st.markdown(
        "Type a clinical or research question. LitLens searches PubMed and Semantic Scholar "
        "simultaneously, deduplicates the results, reranks them by relevance, and synthesises "
        "the key evidence from the top papers."
    )
    st.caption(
        'Example: "Do SGLT2 inhibitors reduce heart failure hospitalisation in type 2 diabetes?"'
    )

    with st.form("search_form"):
        query = st.text_area(
            "Research question",
            placeholder="What is the effect of metformin on cardiovascular outcomes in Type 2 diabetes?",
            height=90,
            label_visibility="collapsed",
        )
        col_btn, col_note = st.columns([1, 4])
        with col_btn:
            submitted = st.form_submit_button("Search Literature", type="primary", use_container_width=True)
        with col_note:
            st.caption("Searches PubMed + Semantic Scholar · Deduplicates · Reranks · Synthesises")

    if submitted:
        if not query or len(query.strip()) < 5:
            st.warning("Please enter a research question.")
            st.stop()

        query = query.strip()

        from src.synthesizer import _detect_intent
        intent = _detect_intent(query)
        if intent == "term":
            st.info(
                "Tip: For a more targeted synthesis, phrase your input as a clinical question. "
                'Example: "What are the main causes of heavy menstrual bleeding?" or '
                '"Do tranexamic acid reduce heavy menstrual bleeding?" '
                "Proceeding with your search term below."
            )

        encoder = _load_encoder()

        progress = st.progress(0, text="Searching PubMed and Semantic Scholar...")
        t_start = time.time()

        from src.fetcher import fetch_all
        from src.deduplicator import deduplicate
        from src.reranker import rerank
        from src.synthesizer import synthesise, detect_contradictions

        raw_papers = fetch_all(query, max_results=25)
        progress.progress(35, text=f"Retrieved {len(raw_papers)} papers. Deduplicating...")

        papers = deduplicate(raw_papers)
        progress.progress(55, text=f"{len(papers)} unique papers. Ranking by relevance...")

        top_papers = rerank(query, papers, top_k=10, encoder=encoder)
        progress.progress(75, text="Synthesising evidence...")

        synthesis = synthesise(query, top_papers, encoder=encoder)
        contradictions = detect_contradictions(top_papers)
        progress.progress(100, text="Complete.")
        time.sleep(0.3)
        progress.empty()

        elapsed = int((time.time() - t_start) * 1000)

        if not top_papers:
            st.error(
                "No papers returned for this query. "
                "Try rephrasing using specific medical terminology."
            )
            st.stop()

        # Synthesis card
        quality = synthesis.evidence_quality
        badge_class = f"badge-{quality}"
        badge_label = quality.capitalize() + " Evidence"

        st.markdown('<div class="synthesis-card">', unsafe_allow_html=True)
        st.markdown(
            f"### Evidence Synthesis &nbsp;&nbsp;"
            f'<span class="{badge_class}">{badge_label}</span>',
            unsafe_allow_html=True,
        )

        # Direct answer at the top
        st.markdown(
            f'<div class="direct-answer">{synthesis.direct_answer}</div>',
            unsafe_allow_html=True,
        )

        # Broader consensus context
        if synthesis.consensus_statement != synthesis.direct_answer:
            st.markdown(synthesis.consensus_statement)

        if synthesis.key_findings:
            st.markdown("**Supporting Evidence**")
            for f in synthesis.key_findings:
                st.markdown(f"- {f.finding} **[{f.citation}]**")

        if synthesis.gaps:
            st.markdown("**Research Gaps**")
            for g in synthesis.gaps:
                st.markdown(f"- {g}")

        if synthesis.limitations:
            st.caption(synthesis.limitations)

        st.markdown("</div>", unsafe_allow_html=True)

        # Contradictions panel
        if contradictions:
            st.warning(
                f"{len(contradictions)} potential contradiction(s) detected. "
                "The papers below appear to reach opposing conclusions on the same topic."
            )
            for i, c in enumerate(contradictions, 1):
                st.markdown(f"**Contradiction {i} - Topic: {c['shared_topic']}**")
                col_a, col_mid, col_b = st.columns([5, 1, 5])
                with col_a:
                    st.markdown(f"**{c['paper_a_title'][:80]}**")
                    st.success(f"Direction: {c['direction_a'].capitalize()}")
                with col_mid:
                    st.markdown("<div style='text-align:center;padding-top:1.5rem;font-weight:700'>vs.</div>", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"**{c['paper_b_title'][:80]}**")
                    st.error(f"Direction: {c['direction_b'].capitalize()}")
                st.caption(c['note'])
                if i < len(contradictions):
                    st.divider()

        # Papers grid
        st.markdown(f"---\n### Top {len(top_papers)} Papers")
        for i, paper in enumerate(top_papers, start=1):
            authors = paper.authors or []
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            journal_year = " · ".join(filter(None, [paper.journal, str(paper.publication_year or "")]))

            with st.expander(f"[{i}] {paper.title}", expanded=False):
                st.markdown(f"*{author_str}*" + (f" · {journal_year}" if journal_year else ""))

                col_meta, col_link = st.columns([3, 1])
                with col_meta:
                    if paper.citation_count:
                        st.caption(f"{paper.citation_count:,} citations")
                    source_map = {
                        "pubmed": "PubMed",
                        "semantic_scholar": "Semantic Scholar",
                        "both": "PubMed + Semantic Scholar",
                    }
                    st.caption(f"Source: {source_map.get(paper.source, paper.source)}")
                with col_link:
                    if paper.open_access_url:
                        st.link_button("Full Text", paper.open_access_url, use_container_width=True)
                    elif paper.pubmed_id:
                        st.link_button("PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{paper.pubmed_id}/", use_container_width=True)

                if paper.mesh_terms:
                    tags_html = " ".join(f'<span class="mesh-tag">{t}</span>' for t in paper.mesh_terms[:8])
                    st.markdown(tags_html, unsafe_allow_html=True)

                st.markdown("**Abstract**")
                st.markdown(paper.abstract)

        st.markdown("---")
        st.caption(
            f"Retrieved {len(raw_papers)} papers, deduplicated to {len(papers)}, "
            f"showing top {len(top_papers)} · {elapsed:,} ms · "
            "Data from PubMed (NCBI) and Semantic Scholar"
        )


# ── ABOUT PAGE ─────────────────────────────────────────────────────────────────
elif page == "About This Project":
    st.markdown("# About LitLens")
    st.markdown("---")

    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("## Why I Built This")
    st.markdown("""
During my research, I spent a significant amount of time manually searching PubMed each week,
copying abstracts into documents, and trying to piece together what the evidence actually said.
The process was slow, error-prone, and did not scale. Two papers on the same topic could
contradict each other, and I would not catch it until I had read both in full.

I built LitLens to automate this process. The goal was not to replace the researcher, but to
handle the retrieval and organisation work so the researcher can focus on interpretation and
judgment. A well-designed retrieval pipeline should surface the most relevant evidence, flag
conflicts, and present findings in a structured format without requiring any manual curation.

LitLens does this end-to-end: a single question returns a ranked, deduplicated, synthesised
evidence summary in under 10 seconds, at no cost and with no account required.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("## Pipeline")
    st.markdown("""
When you submit a question, LitLens runs five sequential stages:

<span class="step-badge">1</span>**Parallel retrieval** - queries PubMed (NCBI E-utilities) and
Semantic Scholar simultaneously, fetching up to 50 candidate papers.

<span class="step-badge">2</span>**Deduplication** - merges records from both sources using exact
DOI matching, then fuzzy title similarity via rapidfuzz token sort ratio (threshold: 85).
The richer abstract is retained; sources are merged into a single record.

<span class="step-badge">3</span>**Relevance reranking** - encodes each abstract and the query using a
sentence-transformers model (all-MiniLM-L6-v2), then combines semantic cosine similarity
(65%) with Okapi BM25 lexical scoring (35%) to return the top 10 papers.

<span class="step-badge">4</span>**Evidence synthesis** - detects the query intent (intervention question,
descriptive question, or search term), scores each abstract sentence semantically against
the query, deduplicates near-identical sentences, and selects the top results as cited key
findings. Framing adapts to the query type: intervention questions receive directional evidence
framing; descriptive queries and search terms receive an informative literature summary.

<span class="step-badge">5</span>**Contradiction detection** - for each pair of papers sharing
a MeSH term, compares directional signals in the abstracts. Pairs where one paper reports
a positive effect and the other a negative effect on the same topic are flagged for review.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("## Tools and Technologies")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Language**
- Python 3.11

**Data Retrieval**
- PubMed via NCBI E-utilities REST API
- Semantic Scholar Graph API
- `requests`, `lxml` for HTTP and XML parsing

**Text Processing and NLP**
- `sentence-transformers` - semantic sentence embeddings (all-MiniLM-L6-v2)
- `scikit-learn` - TF-IDF vectorisation, cosine similarity, fallback scoring
- `rank-bm25` - Okapi BM25 lexical relevance ranking
- `rapidfuzz` - fuzzy string matching for deduplication
- `numpy` - numerical operations
        """)
    with col2:
        st.markdown("""
**Frontend and Deployment**
- `streamlit` - interactive web application
- Streamlit Community Cloud - free hosting, no infrastructure required

**Data Sources**
- PubMed (NCBI E-utilities) - free, no API key required
- Semantic Scholar - free public API, no key required

**Version Control**
- Git, GitHub

**Design Decisions**
- No database - all retrieval is live and stateless
- No paid APIs - all NLP runs on open-source libraries
- Extractive synthesis - every output sentence traces to a source paper
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("## About Me")
    st.markdown("""
I am **Praveen Puviindran**, a researcher with interests in computational biology,
biomedical informatics, and data-driven approaches to scientific discovery. LitLens
demonstrates my ability to design and ship a full-stack data pipeline from retrieval
through synthesis, deployed as a production web application.

- GitHub: [github.com/praveenpuviindran](https://github.com/praveenpuviindran)
- LinkedIn: [linkedin.com/in/praveenpuviindran](https://linkedin.com/in/praveenpuviindran)
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ── HOW IT WORKS PAGE ──────────────────────────────────────────────────────────
elif page == "How It Works":
    st.markdown("# How LitLens Works")
    st.markdown("---")

    st.markdown("## System Architecture")
    st.code("""
Your Question
      |
      v
Query Intent Detection
  - Intervention: "Does X reduce Y?"
  - Descriptive:  "What causes Y?"
  - Term:         "heavy menstrual bleeding"
      |
      v
PubMed (NCBI E-utilities)  +  Semantic Scholar  (parallel, free)
      |
      v
Deduplication
  Pass 1: Exact DOI match
  Pass 2: Fuzzy title match (rapidfuzz, threshold 85)
      |
      v
Semantic Reranking  ->  Top 10 papers
  65% sentence-transformers cosine similarity
  35% Okapi BM25 lexical score
      |
      v
Semantic Evidence Synthesis
  - Sentence scoring via sentence-transformers embeddings
  - Near-duplicate sentence removal
  - Intent-aware framing (intervention vs. descriptive vs. term)
  - Highest-scoring sentence as direct answer
  - Remaining top sentences as cited key findings
      |
      v
Contradiction Detection
  - MeSH term overlap check (shared topic)
  - Directional signal comparison (positive vs. negative)
      |
      v
Streamlit UI
    """, language="text")

    st.markdown("---")
    st.markdown("## Frequently Asked Questions")

    with st.expander("Why are some queries slow?"):
        st.markdown("""
PubMed's free tier permits 3 requests per second. LitLens makes two sequential
requests to PubMed per search (ESearch for IDs, EFetch for full records), with a
mandatory delay between them. Semantic Scholar is queried in parallel. Total
retrieval time is typically 3 to 8 seconds depending on query complexity and
network conditions.
        """)

    with st.expander("Why is a paper I know about not in the results?"):
        st.markdown("""
LitLens retrieves up to 25 papers per source before deduplication, then returns
the top 10 by relevance. A paper may be absent if:

- It has no abstract in PubMed or Semantic Scholar (abstracts are required for synthesis)
- It ranked below the top 10 by BM25 score for this query
- It was merged with a higher-scoring duplicate from the other source

Rephrasing with more specific terminology, including author names or journal names,
can surface specific papers.
        """)

    with st.expander("How is the synthesis generated?"):
        st.markdown("""
The synthesis is extractive. Every sentence in the output is taken directly from one
of the retrieved abstracts.

Sentences are scored using semantic similarity: the query and all abstract sentences
are encoded as dense vectors using a sentence-transformers model (all-MiniLM-L6-v2),
and sentences are ranked by cosine similarity to the query vector.

The query intent is detected before synthesis. If the query is an intervention question
("Does X reduce Y?"), the synthesis applies directional framing based on the overall
signal across the abstracts. If the query is descriptive or a search term, the synthesis
presents the most informative sentences as a literature overview without applying
intervention-specific framing.

No new text is generated. Every output sentence is traceable to a specific source paper.
        """)

    with st.expander("What does the evidence quality label mean?"):
        st.markdown("""
LitLens estimates evidence quality from study design keywords detected in the abstracts:

| Label | Basis |
|-------|-------|
| Strong | Meta-analysis or systematic review detected |
| Moderate | Randomised controlled trial detected |
| Weak | Primarily observational studies (cohort, case-control) |
| Mixed | No clear study design signal detected |

This is a keyword heuristic, not a formal GRADE or AMSTAR assessment.
        """)

    with st.expander("How does contradiction detection work?"):
        st.markdown("""
For each pair of papers that share at least one MeSH term, LitLens compares
directional signals in the abstract text.

Positive signals include words like: reduced, improved, effective, significant, beneficial.
Negative signals include: increased risk, no benefit, no significant difference, ineffective.

If one paper shows predominantly positive signals and the other predominantly negative
signals on the same MeSH topic, the pair is flagged. This is a conservative heuristic -
it will miss nuanced contradictions and may occasionally flag papers discussing different
aspects of the same topic. Treat flagged pairs as a prompt for further manual review, not
as a definitive finding.
        """)

    with st.expander("Is this suitable for clinical decision making?"):
        st.markdown("""
No. LitLens is a research assistance tool. The synthesis is automated and based on
abstract text only. Always consult full-text papers, systematic reviews, and current
clinical guidelines before making patient care decisions.
        """)

    st.markdown("---")
    st.markdown("## Source Code")
    st.markdown("""
The full source code is available on GitHub:

[github.com/praveenpuviindran/litlens](https://github.com/praveenpuviindran/litlens)
    """)
