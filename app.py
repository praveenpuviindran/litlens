"""LitLens — Biomedical Literature Intelligence Engine.

Built by Praveen Puviindran.
Deployed on Streamlit Community Cloud. No API keys required.
"""

import time

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LitLens | Biomedical Literature Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2545 0%, #1A3A5C 100%);
}
section[data-testid="stSidebar"] * { color: #E8F4FD !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] .stRadio label { color: #E8F4FD !important; }
section[data-testid="stSidebar"] hr { border-color: #2D5F8A; }

/* ── Primary button ──────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #0E7C6B; border: none; color: white;
    font-weight: 700; font-size: 1rem; padding: 0.6rem 1.4rem;
    border-radius: 6px; transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover { background: #0B6559; }

/* ── Cards ───────────────────────────────────────────────────────── */
.synthesis-card {
    border: 1.5px solid #0E7C6B; border-radius: 10px;
    padding: 1.6rem 1.8rem; background: #F0FAF8;
    margin-bottom: 1.2rem;
}
.about-card {
    border: 1px solid #CBD5E0; border-radius: 10px;
    padding: 1.6rem 1.8rem; background: #FAFBFC;
    margin-bottom: 1rem;
}

/* ── Badges ──────────────────────────────────────────────────────── */
.badge-strong   { background:#1D8348; color:white; border-radius:5px; padding:3px 12px; font-weight:700; font-size:.85rem; }
.badge-moderate { background:#D4AC0D; color:white; border-radius:5px; padding:3px 12px; font-weight:700; font-size:.85rem; }
.badge-mixed    { background:#E67E22; color:white; border-radius:5px; padding:3px 12px; font-weight:700; font-size:.85rem; }
.badge-weak     { background:#C0392B; color:white; border-radius:5px; padding:3px 12px; font-weight:700; font-size:.85rem; }

/* ── MeSH tags ───────────────────────────────────────────────────── */
.mesh-tag {
    display:inline-block; background:#D5F5E3; color:#1D6A48;
    border-radius:4px; padding:2px 8px; margin:2px;
    font-size:.74rem; font-weight:600;
}

/* ── Step badges (pipeline diagram) ─────────────────────────────── */
.step-badge {
    display:inline-block; background:#0B2545; color:#E8F4FD;
    border-radius:50%; width:28px; height:28px; line-height:28px;
    text-align:center; font-weight:700; margin-right:8px; font-size:.85rem;
}

/* ── General ─────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 LitLens")
    st.markdown("*Biomedical Literature Intelligence*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Search", "📖 About This Project", "⚙️ How It Works"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
**Data sources**
- PubMed (NCBI E-utilities)
- Semantic Scholar

**No API keys required.**
All processing runs free and open-source.
    """)
    st.markdown("---")
    st.markdown("""
Built by **Praveen Puviindran**
[GitHub](https://github.com/praveenpuviindran) · [LinkedIn](https://linkedin.com/in/praveenpuviindran)
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEARCH
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Search":
    st.markdown("# 🔬 LitLens")
    st.markdown(
        "**Biomedical Literature Intelligence** — type a clinical or research question, "
        "and LitLens searches PubMed and Semantic Scholar simultaneously, deduplicates "
        "the results, reranks them by relevance, and synthesises the key evidence."
    )
    st.info(
        "💡 **Tip:** The more specific your question, the better the results. "
        'Try: *"Do SGLT2 inhibitors reduce heart failure hospitalisation in type 2 diabetes?"*'
    )

    # ── Query input ────────────────────────────────────────────────────────────
    with st.form("search_form"):
        query = st.text_area(
            "Your research question",
            placeholder="What is the effect of metformin on cardiovascular outcomes in Type 2 diabetes?",
            height=90,
            label_visibility="collapsed",
        )
        col_btn, col_note = st.columns([1, 4])
        with col_btn:
            submitted = st.form_submit_button("🔬 Search Literature", type="primary", use_container_width=True)
        with col_note:
            st.caption("Searches PubMed + Semantic Scholar · Deduplicates · Reranks · Synthesises")

    if submitted:
        if not query or len(query.strip()) < 5:
            st.warning("Please enter a research question (at least 5 characters).")
            st.stop()

        query = query.strip()

        # ── Pipeline execution ─────────────────────────────────────────────────
        progress = st.progress(0, text="🔎 Searching PubMed and Semantic Scholar…")
        t_start = time.time()

        from src.fetcher import fetch_all
        from src.deduplicator import deduplicate
        from src.reranker import rerank
        from src.synthesizer import synthesise, detect_contradictions

        raw_papers = fetch_all(query, max_results=25)
        progress.progress(35, text=f"📄 Retrieved {len(raw_papers)} raw papers. Deduplicating…")

        papers = deduplicate(raw_papers)
        progress.progress(55, text=f"✅ {len(papers)} unique papers. Ranking by relevance…")

        top_papers = rerank(query, papers, top_k=10)
        progress.progress(75, text="🧠 Synthesising evidence…")

        synthesis = synthesise(query, top_papers)
        contradictions = detect_contradictions(top_papers)
        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()

        elapsed = int((time.time() - t_start) * 1000)

        if not top_papers:
            st.error(
                "No papers were returned for this query. "
                "Try rephrasing — use specific medical terms rather than abbreviations."
            )
            st.stop()

        # ── Synthesis card ─────────────────────────────────────────────────────
        quality = synthesis.evidence_quality
        badge_class = f"badge-{quality}"
        badge_label = quality.capitalize() + " Evidence"

        st.markdown('<div class="synthesis-card">', unsafe_allow_html=True)
        st.markdown(
            f"### 📋 Evidence Synthesis &nbsp;&nbsp;"
            f'<span class="{badge_class}">{badge_label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"> {synthesis.consensus_statement}")

        if synthesis.key_findings:
            st.markdown("**Key Findings**")
            for f in synthesis.key_findings:
                st.markdown(f"- {f.finding} **[{f.citation}]**")

        if synthesis.gaps:
            st.markdown("**Research Gaps**")
            for g in synthesis.gaps:
                st.markdown(f"- {g}")

        if synthesis.limitations:
            st.caption(f"*{synthesis.limitations}*")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Contradictions panel ───────────────────────────────────────────────
        if contradictions:
            st.warning(
                f"⚠️ {len(contradictions)} potential contradiction(s) detected in retrieved literature. "
                "Papers below appear to reach opposing conclusions on the same topic."
            )
            for i, c in enumerate(contradictions, 1):
                st.markdown(f"**Contradiction {i} — Topic: {c['shared_topic']}**")
                col_a, col_mid, col_b = st.columns([5, 1, 5])
                with col_a:
                    st.markdown(f"**{c['paper_a_title'][:80]}…**")
                    st.success(f"Direction: {c['direction_a'].capitalize()}")
                with col_mid:
                    st.markdown("<div style='text-align:center;padding-top:1.5rem;font-weight:700'>vs.</div>", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"**{c['paper_b_title'][:80]}…**")
                    st.error(f"Direction: {c['direction_b'].capitalize()}")
                st.caption(f"💡 {c['note']}")
                if i < len(contradictions):
                    st.divider()

        # ── Papers grid ────────────────────────────────────────────────────────
        st.markdown(f"---\n### 📄 Top {len(top_papers)} Papers")
        for i, paper in enumerate(top_papers, start=1):
            authors = paper.authors or []
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            journal_year = " · ".join(filter(None, [paper.journal, str(paper.publication_year or "")]))
            label = f"[{i}] {paper.title}"

            with st.expander(label, expanded=False):
                st.markdown(f"*{author_str}*" + (f" · {journal_year}" if journal_year else ""))

                col_meta, col_link = st.columns([3, 1])
                with col_meta:
                    if paper.citation_count:
                        st.caption(f"📊 {paper.citation_count:,} citations")
                    source_map = {"pubmed": "PubMed", "semantic_scholar": "Semantic Scholar", "both": "PubMed + S2"}
                    st.caption(f"Source: {source_map.get(paper.source, paper.source)}")
                with col_link:
                    if paper.open_access_url:
                        st.link_button("📄 Full Text", paper.open_access_url, use_container_width=True)
                    elif paper.pubmed_id:
                        st.link_button("🔗 PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{paper.pubmed_id}/", use_container_width=True)

                if paper.mesh_terms:
                    tags_html = " ".join(f'<span class="mesh-tag">{t}</span>' for t in paper.mesh_terms[:8])
                    st.markdown(tags_html, unsafe_allow_html=True)

                st.markdown("**Abstract**")
                st.markdown(paper.abstract)

        # ── Footer ─────────────────────────────────────────────────────────────
        st.markdown("---")
        st.caption(
            f"Retrieved {len(raw_papers)} papers → deduplicated to {len(papers)} → "
            f"showing top {len(top_papers)} · Total time: {elapsed:,} ms · "
            "Data from PubMed (NCBI) and Semantic Scholar"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 About This Project":
    st.markdown("# 📖 About LitLens")
    st.markdown("---")

    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("## Why I Built This")
    st.markdown("""
I built LitLens because I kept running into the same problem during my research: searching PubMed
returns hundreds of abstracts, and synthesising them into something actionable takes hours of
manual work — skimming titles, copy-pasting abstracts, trying to spot what the evidence actually
says, and then noticing halfway through that two papers seem to contradict each other.

I wanted a tool that could do the tedious part automatically. Not just a search engine — something
that would actually read the papers and tell me: *here's what the evidence shows, here's where the
gaps are, and here are two papers that seem to disagree with each other.*

LitLens is that tool. I designed it so that a clinician, researcher, or student can ask a
natural-language question and get a structured evidence summary in under 10 seconds, for free,
with no account or API key required.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("## What It Does")
    st.markdown("""
When you submit a question, LitLens runs a five-stage pipeline:

<span class="step-badge">1</span>**Parallel retrieval** — queries PubMed (NCBI E-utilities) and
Semantic Scholar simultaneously, fetching up to 50 candidate papers.

<span class="step-badge">2</span>**Deduplication** — merges records from both sources using exact
DOI matching and fuzzy title similarity (rapidfuzz), keeping the richest version of each paper.

<span class="step-badge">3</span>**Relevance reranking** — scores each paper's abstract against
your query using BM25, a probabilistic retrieval function used in production search systems like
Elasticsearch, and returns the top 10.

<span class="step-badge">4</span>**Evidence synthesis** — extracts the most query-relevant,
non-redundant sentences from the top abstracts using TF-IDF cosine similarity, then structures
them as key findings with citation markers.

<span class="step-badge">5</span>**Contradiction detection** — identifies pairs of papers that
share a topic (via MeSH terms) but show opposing directional signals, flagging potential conflicts
in the evidence.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("## Technical Details")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Data Sources**
- PubMed via NCBI E-utilities (free tier)
- Semantic Scholar public API

**Text Processing**
- Deduplication: rapidfuzz token sort ratio
- Reranking: BM25 (rank-bm25)
- Synthesis: TF-IDF cosine similarity (scikit-learn)

**Deployment**
- Streamlit Community Cloud
- Zero infrastructure cost
- No API keys required
        """)
    with col2:
        st.markdown("""
**Why no LLM?**

I deliberately built LitLens without relying on any paid LLM API. This means:

- It's completely free to run and deploy
- It works offline / without rate limits
- It's reproducible — the same query always returns the same synthesis
- It demonstrates that you don't need GPT-4 to build useful NLP tools

The synthesis uses *extractive* NLP (selecting the most relevant real sentences
from the papers) rather than *abstractive* generation (having an LLM write new text).
This also means every sentence in the output is traceable to a specific source paper.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="about-card">', unsafe_allow_html=True)
    st.markdown("## About Me")
    st.markdown("""
I'm **Praveen Puviindran**, a researcher with a background in data science and computational biology.
I built LitLens as a portfolio project to demonstrate end-to-end software engineering:
designing a data pipeline, integrating multiple public APIs, applying NLP techniques,
and deploying a production-ready web application.

If you have feedback or want to connect:

- 🐙 **GitHub:** [github.com/praveenpuviindran](https://github.com/praveenpuviindran)
- 💼 **LinkedIn:** [linkedin.com/in/praveenpuviindran](https://linkedin.com/in/praveenpuviindran)
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ How It Works":
    st.markdown("# ⚙️ How LitLens Works")
    st.markdown("---")

    st.markdown("## Pipeline Architecture")
    st.markdown("""
```
Your Question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              Parallel Data Retrieval                │
│  PubMed (NCBI E-utilities)  │  Semantic Scholar     │
│  • ESearch → PubMed IDs     │  • Graph API search   │
│  • EFetch → XML → Papers    │  • JSON → Papers      │
└──────────────────┬──────────────────────────────────┘
                   │ Up to 50 raw papers
                   ▼
┌─────────────────────────────────────────────────────┐
│                  Deduplication                      │
│  Pass 1: Exact DOI match                            │
│  Pass 2: Fuzzy title match (rapidfuzz, ≥ 85%)       │
│  → Merge records, keep richer abstract              │
│  → Drop papers with no abstract                     │
└──────────────────┬──────────────────────────────────┘
                   │ 10–40 unique papers
                   ▼
┌─────────────────────────────────────────────────────┐
│              BM25 Relevance Reranking               │
│  Scores each abstract against your query            │
│  using Okapi BM25 probabilistic ranking             │
│  → Returns top 10 most relevant papers              │
└──────────────────┬──────────────────────────────────┘
                   │ Top 10 papers
                   ▼
┌─────────────────────────────────────────────────────┐
│           TF-IDF Evidence Synthesis                 │
│  1. Split all abstracts into sentences              │
│  2. Score sentences vs. query with TF-IDF cosine   │
│  3. Deduplicate near-identical sentences            │
│  4. Select top N as key findings (with citations)   │
│  5. Detect evidence direction (positive/negative)   │
│  6. Estimate quality (RCT/meta-analysis heuristic)  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│           Contradiction Detection                   │
│  For each pair sharing a MeSH term:                 │
│  Compare directional signals in abstracts           │
│  Flag opposing pairs for the user                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
            Results displayed
```
""")

    st.markdown("---")
    st.markdown("## Frequently Asked Questions")

    with st.expander("Why are some queries slow?"):
        st.markdown("""
PubMed's free tier limits requests to 3 per second. LitLens makes two requests
to PubMed (ESearch to get IDs, EFetch to get abstracts), so there is a mandatory
~0.35 second delay between them. Semantic Scholar is queried in parallel.

Total retrieval time is typically 3–8 seconds depending on query complexity and
network conditions.
        """)

    with st.expander("Why doesn't my paper appear in the results?"):
        st.markdown("""
LitLens retrieves up to 25 papers per source (50 total before deduplication),
then returns the top 10 by relevance. A paper may not appear if:

- It's not in PubMed or Semantic Scholar
- Its abstract is missing (LitLens requires abstracts to synthesise)
- It ranked below the top 10 for your query
- It was merged with a duplicate from the other source

Try rephrasing your query with more specific terminology to surface niche papers.
        """)

    with st.expander("How is the synthesis generated?"):
        st.markdown("""
The synthesis is **extractive** — every sentence in the Key Findings section is
an actual sentence from one of the retrieved abstracts, selected because it was
most similar to your query by TF-IDF cosine similarity.

No language model generates new text. This means:
- Every finding is directly traceable to a specific source paper (the [N] citation)
- The output is deterministic — the same query always returns the same synthesis
- There is no risk of hallucinated or fabricated findings
        """)

    with st.expander("What does the evidence quality badge mean?"):
        st.markdown("""
LitLens assigns a quality badge based on study design signals in the abstracts:

| Badge | Meaning |
|-------|---------|
| **Strong** | At least one meta-analysis or systematic review detected |
| **Moderate** | At least one RCT or randomised controlled trial detected |
| **Weak** | Primarily observational studies (cohort, case-control) |
| **Mixed** | No clear study design signal detected |

This is a heuristic based on keyword matching, not a formal GRADE assessment.
        """)

    with st.expander("How does contradiction detection work?"):
        st.markdown("""
For each pair of papers that share at least one MeSH term (indicating they discuss
the same biomedical concept), LitLens analyses the direction of evidence using
keyword signals:

- **Positive signals:** reduced, lower, improve, effective, significant, beneficial…
- **Negative signals:** increased, higher, no benefit, no significant, ineffective…

If one paper shows positive signals and the other shows negative signals on the
same topic, LitLens flags them as a potential contradiction.

This is a conservative heuristic — it will miss nuanced contradictions and may
occasionally flag papers that discuss different aspects of a topic. Treat flagged
contradictions as a starting point for further manual review.
        """)

    with st.expander("Can I use LitLens for clinical decision making?"):
        st.markdown("""
**No.** LitLens is a research assistance tool, not a clinical decision support system.

The synthesis is automated and based on abstract text only — it has not been validated
for clinical accuracy. Always consult full-text papers, systematic reviews, and clinical
guidelines before making any patient care decisions.
        """)

    st.markdown("---")
    st.markdown("## Open Source")
    st.markdown("""
LitLens is open source. The full source code is available on GitHub:

🐙 [github.com/praveenpuviindran/litlens](https://github.com/praveenpuviindran/litlens)

Feel free to fork, contribute, or use it as the foundation for your own biomedical
NLP project.
    """)
