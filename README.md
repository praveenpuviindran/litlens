# LitLens - Biomedical Literature Intelligence Engine

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Community_Cloud-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![No API Key](https://img.shields.io/badge/API_Keys-None_Required-brightgreen)

**[Live Demo](https://litlens.streamlit.app)** - Built by [Praveen Puviindran](https://github.com/praveenpuviindran)

---

## Overview

LitLens is a biomedical literature intelligence engine that automates the retrieval,
deduplication, ranking, and synthesis of research evidence from PubMed and Semantic Scholar.

A researcher submits a natural-language clinical question. The system returns a structured
evidence summary with a direct answer, cited key findings, detected contradictions between
papers, and identified research gaps - in under 10 seconds, at no cost.

---

## What It Does

1. Queries **PubMed** (NCBI E-utilities) and **Semantic Scholar** in parallel
2. Deduplicates results across both sources using DOI matching and fuzzy title similarity
3. Reranks papers by relevance to the query using Okapi BM25
4. Synthesises the top 10 papers using TF-IDF extractive summarisation
5. Detects contradictions between papers that reach opposing conclusions on the same topic

---

## Deploy Your Own Copy

No API keys. No environment variables. No configuration.

1. Fork this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select this repo, set main file to `app.py`, click **Deploy**

Live in approximately 60 seconds.

---

## Run Locally

```bash
git clone https://github.com/praveenpuviindran/litlens.git
cd litlens
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Tools and Technologies

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |
| Data retrieval | PubMed NCBI E-utilities, Semantic Scholar Graph API |
| HTTP and parsing | `requests`, `lxml` |
| Relevance ranking | `rank-bm25` (Okapi BM25) |
| Evidence synthesis | `scikit-learn` (TF-IDF, cosine similarity) |
| Deduplication | `rapidfuzz` (token sort ratio, threshold 85) |
| Numerical operations | `numpy` |
| Version control | Git, GitHub |

---

## Architecture

```
User query
    |
    v
PubMed (NCBI E-utilities)  +  Semantic Scholar   <- parallel, free, no key
    |
    v
Deduplication
  Pass 1: Exact DOI match
  Pass 2: Fuzzy title match (rapidfuzz)
    |
    v
BM25 Relevance Reranking  ->  Top 10 papers
    |
    v
TF-IDF Extractive Synthesis
  - Sentence scoring by cosine similarity to query
  - Near-duplicate sentence removal
  - Direct answer: highest-scoring sentence
  - Supporting findings: next N sentences with citations
    |
    v
Contradiction Detection
  - MeSH term overlap check
  - Directional signal comparison
    |
    v
Streamlit UI
```

---

## Design Decisions

**No paid APIs.** All NLP runs on open-source libraries. The synthesis is extractive -
every output sentence is taken verbatim from a source abstract and cited by paper number.
No text is generated or invented, ensuring every output sentence traces directly to a source paper.

**No database.** All retrieval is live and stateless. Results reflect the current state
of PubMed and Semantic Scholar at the time of the query.

**No infrastructure cost.** Streamlit Community Cloud hosts the app for free. There are
no servers, databases, or API subscriptions to maintain.

---

## Project Structure

```
litlens/
├── app.py              # Streamlit entry point
├── src/
│   ├── fetcher.py      # PubMed + Semantic Scholar retrieval
│   ├── deduplicator.py # DOI + fuzzy title deduplication
│   ├── reranker.py     # BM25 relevance reranking
│   └── synthesizer.py  # TF-IDF synthesis and contradiction detection
├── .streamlit/
│   └── config.toml     # Theme configuration
├── requirements.txt
└── README.md
```

---

## License

MIT

---

*Praveen Puviindran - [GitHub](https://github.com/praveenpuviindran) - [LinkedIn](https://www.linkedin.com/in/praveen-puviindran-218998220) - [Website](https://praveenpuviindran.github.io/)*
