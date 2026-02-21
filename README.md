# 🔬 LitLens — Biomedical Literature Intelligence Engine

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Community_Cloud-FF4B4B?logo=streamlit)
![PubMed](https://img.shields.io/badge/Data-PubMed%20%2B%20Semantic%20Scholar-2E86AB)
![License](https://img.shields.io/badge/License-MIT-green)
![No API Key](https://img.shields.io/badge/API%20Keys-None%20Required-brightgreen)

**[▶ Live Demo](https://litlens.streamlit.app)** · Built by [Praveen Puviindran](https://github.com/praveenpuviindran)

---

## What It Does

LitLens is a biomedical literature intelligence engine. You type a clinical research question; it:

1. Searches **PubMed** and **Semantic Scholar** simultaneously (up to 50 papers)
2. **Deduplicates** results across both sources using DOI + fuzzy title matching
3. **Reranks** papers by relevance to your query using BM25
4. **Synthesises** the top 10 papers into structured key findings with citations
5. **Flags contradictions** between papers that appear to reach opposing conclusions

Everything runs free. No API key. No account. No database.

---

## Deploy Your Own Copy in 3 Steps

> **No API keys needed. No environment variables to set.**

### Step 1 — Fork the repository

Click **Fork** on this GitHub page.

### Step 2 — Go to Streamlit Community Cloud

Visit [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account (free).

### Step 3 — Click "New app" and deploy

- **Repository:** `your-username/litlens`
- **Branch:** `main`
- **Main file path:** `app.py`

Click **Deploy**. That's it. Your app will be live in ~60 seconds.

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

## Architecture

```
Your Question
      │
      ▼
PubMed (NCBI)  +  Semantic Scholar     ← free public APIs, no key
      │
      ▼
Deduplication (DOI exact + fuzzy title)  ← rapidfuzz
      │
      ▼
BM25 Relevance Reranking               ← rank-bm25, top 10
      │
      ▼
TF-IDF Evidence Synthesis              ← scikit-learn
      │
      ▼
Contradiction Detection                ← directional signal heuristic
      │
      ▼
Streamlit UI
```

---

## Why No LLM?

I deliberately built this without any paid LLM API. The synthesis is **extractive** — every sentence
in the output is a real sentence from a real abstract, selected because it was most similar to your
query by TF-IDF cosine similarity. No text is invented or hallucinated.

This makes LitLens:
- **Free** — no API costs, ever
- **Reproducible** — the same query always returns the same synthesis
- **Traceable** — every finding has a [citation] pointing to its source paper

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Data retrieval | PubMed E-utilities + Semantic Scholar API |
| Deduplication | rapidfuzz |
| Reranking | rank-bm25 (Okapi BM25) |
| Synthesis | scikit-learn TF-IDF |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
litlens/
├── app.py              # Main Streamlit app (entry point)
├── src/
│   ├── fetcher.py      # PubMed + Semantic Scholar retrieval
│   ├── deduplicator.py # DOI + fuzzy title deduplication
│   ├── reranker.py     # BM25 relevance reranking
│   └── synthesizer.py  # TF-IDF synthesis + contradiction detection
├── .streamlit/
│   └── config.toml     # Streamlit theme
├── requirements.txt    # Dependencies
└── README.md
```

---

## License

MIT — use it, fork it, build on it.

---

*Built by Praveen Puviindran · [GitHub](https://github.com/praveenpuviindran) · [LinkedIn](https://linkedin.com/in/praveenpuviindran)*
