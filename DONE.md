# LitLens — Build Completion Checklist

## Public Demo URL (placeholder — update after deploying to Render)
https://litlens-frontend.onrender.com

## GitHub Repository URL (placeholder — update after pushing)
https://github.com/praveenpuviindran/litlens

---

## Deliverables Checklist

### Code
- [x] `pyproject.toml` with all dependencies and optional groups (`frontend`, `dev`)
- [x] `backend/config.py` — Pydantic Settings with full validation
- [x] `backend/database.py` — SQLAlchemy 2.x async engine, pgvector init, FTS trigger
- [x] `backend/models.py` — `papers`, `queries`, `contradictions` ORM models with indexes
- [x] `backend/schemas.py` — All Pydantic request/response schemas
- [x] `backend/services/query_expansion.py` — gpt-4o-mini MeSH query generation
- [x] `backend/services/fetcher.py` — Async PubMed + Semantic Scholar with rate limiting and retry
- [x] `backend/services/deduplicator.py` — DOI exact + rapidfuzz fuzzy title deduplication
- [x] `backend/services/embedder.py` — OpenAI embeddings, pgvector + FAISS hybrid retrieval, RRF
- [x] `backend/services/reranker.py` — cross-encoder/ms-marco-MiniLM-L-6-v2 lazy singleton
- [x] `backend/services/generator.py` — Synthesis + pairwise contradiction detection
- [x] `backend/services/evaluator.py` — LLM-based faithfulness scoring
- [x] `backend/routers/health.py` — GET /health
- [x] `backend/routers/search.py` — POST /search with cache, full pipeline, error handling
- [x] `backend/routers/papers.py` — GET /papers with pagination and filters
- [x] `backend/main.py` — FastAPI app with CORS, lifespan, global exception handler
- [x] `frontend/app.py` — Streamlit search + eval dashboard, custom CSS
- [x] `frontend/components/` — search_bar, result_card, contradiction_panel, eval_dashboard
- [x] `frontend/utils/api_client.py` — httpx client with retry
- [x] `eval/test_set.json` — 50 synthetic BioASQ questions across 5 categories
- [x] `eval/contradiction_labels.json` — 40 labeled paper pairs (20 true, 20 false)
- [x] `eval/run_eval.py` — Full eval pipeline with threshold gating
- [x] `eval/bioasq_loader.py` — Test set loader utility

### Infrastructure
- [x] `Dockerfile.backend` with health check
- [x] `Dockerfile.frontend` with health check
- [x] `docker-compose.yml` — postgres + backend + frontend with health dependency
- [x] `.github/workflows/ci.yml` — lint, type-check, test, Docker build
- [x] `render.yaml` — backend service, frontend service, PostgreSQL add-on

### Tests
- [x] `tests/conftest.py` — shared fixtures
- [x] `tests/test_fetcher.py`
- [x] `tests/test_deduplicator.py`
- [x] `tests/test_reranker.py`
- [x] `tests/test_generator.py`
- [x] `tests/test_api.py`
- [x] `tests/test_eval_gate.py`

### Documentation
- [x] `README.md` with Mermaid architecture diagram and all 13 sections
- [x] `.env.example` with comments on every variable
- [x] `.gitignore`

---

## Next Steps After First Deployment

1. Push to GitHub: `git push -u origin main`
2. Connect repo to Render via Blueprint
3. Set `OPENAI_API_KEY`, `NCBI_EMAIL`, `NCBI_API_KEY` in Render dashboard
4. Once live: `python eval/run_eval.py` against the production URL
5. Update `README.md` Evaluation Results table with real numbers
6. Update demo URL in README and this file
