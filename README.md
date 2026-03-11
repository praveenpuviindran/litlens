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

---

## Model Monitoring & Evaluation

Merges to `main` are gated by a faithfulness threshold of 0.75. The CI pipeline runs the
full Ragas evaluation suite on every push to main using `eval/run_ragas_eval.py`.

### Ragas Eval Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Faithfulness** | Are all synthesis claims supported by retrieved paper abstracts? | ≥ 0.75 |
| **Answer Relevancy** | Does the consensus statement address the research question? | ≥ 0.70 |
| **Context Precision** | Are the retrieved papers relevant to the query? | ≥ 0.65 |
| **Retrieval Precision** | Do retrieved papers contain expected keywords? | ≥ 0.60 |

Run the evaluation locally:
```bash
python eval/run_ragas_eval.py --sample-size 20
```

Results are appended to `eval/eval_history.json` and visualised in the **Eval Dashboard** tab of the Streamlit app.

---

## Query Intent Classification

Every search query is classified into one of five intent types before the pipeline runs.
Intent routing selects the most appropriate fetching strategy and synthesis style.

| Intent | Example Query | Routing |
|--------|--------------|---------|
| **Definitional** | "What is metformin?" | Review articles, up to 5 papers |
| **Comparative** | "Compare statin vs PCSK9i for LDL" | RCTs preferred, up to 10 papers |
| **Search** | "Recent trials on GLP-1 agonists" | Sorted by recency, up to 15 papers |
| **Mechanistic** | "Why does amyloid cause neurodegeneration?" | Basic science, up to 8 papers |
| **Epidemiological** | "Prevalence of T2D by region" | Cohort studies, up to 10 papers |

The classified intent and routing decision are shown as a badge on every search result.
Synthesis responses include `recommended_next_searches` displayed as clickable buttons.

---

## Analytics

LitLens tracks query patterns and synthesis quality to support ongoing improvement.

**Analytics API:** `GET /analytics/summary`  
**Analytics dashboard:** available in the Streamlit app under the **Analytics** page

### Metrics tracked

- Query volume and latency over time (daily, by intent)
- Topic frequency across all queries (word frequency from raw queries)
- Contradiction detection rate per week
- Faithfulness scores by query intent
- User feedback ratings (Helpful / Not helpful)

### Deployment

## GitHub Secrets Required for CI Eval Gate

In your GitHub repository settings → **Secrets** → **Actions**, add:

| Secret | Description |
|--------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM calls and Ragas evaluation |
| `NCBI_API_KEY` | NCBI E-utilities key (optional but increases rate limits) |
| `NCBI_EMAIL` | Email address for NCBI API usage |
| `DATABASE_URL` | Render PostgreSQL connection string (`postgresql+asyncpg://...`) |
| `BACKEND_URL` | Deployed backend URL for the eval gate to call `/search` |
