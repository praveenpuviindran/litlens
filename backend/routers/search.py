"""Search router — POST /search."""

import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend import models
from backend.database import get_db
from backend.schemas import (
    ContradictionResponse,
    PaperResponse,
    SearchRequest,
    SearchResponse,
    Synthesis,
)
from backend.services.deduplicator import deduplicate
from backend.services.embedder import embed_papers, embed_query, retrieve_papers, store_in_faiss
from backend.services.fetcher import fetch_all
from backend.services.generator import detect_contradictions, synthesise
from backend.services.query_expansion import expand_query
from backend.services.reranker import rerank
from backend.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


async def _store_papers(papers, session: AsyncSession) -> list[models.Paper]:
    """Upsert paper records and their embeddings into PostgreSQL.

    Returns the list of ORM Paper models with their database IDs.

    Args:
        papers: List of (Paper schema, embedding vector) tuples.
        session: Active async database session.

    Returns:
        List of persisted ORM Paper objects.
    """
    orm_papers: list[models.Paper] = []
    for paper, vector in papers:
        # Try to find existing record by pubmed_id or doi to avoid duplicates.
        existing = None
        if paper.pubmed_id:
            result = await session.execute(
                select(models.Paper).where(models.Paper.pubmed_id == paper.pubmed_id)
            )
            existing = result.scalar_one_or_none()
        if existing is None and paper.doi:
            result = await session.execute(
                select(models.Paper).where(models.Paper.doi == paper.doi)
            )
            existing = result.scalar_one_or_none()

        if existing:
            # Update embedding if not yet set.
            if existing.embedding is None:
                existing.embedding = vector
            orm_papers.append(existing)
        else:
            orm_paper = models.Paper(
                pubmed_id=paper.pubmed_id,
                s2_id=paper.s2_id,
                doi=paper.doi,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                journal=paper.journal,
                publication_year=paper.publication_year,
                mesh_terms=paper.mesh_terms,
                keywords=paper.keywords,
                citation_count=paper.citation_count,
                open_access_url=paper.open_access_url,
                source=paper.source,
                embedding=vector,
            )
            session.add(orm_paper)
            orm_papers.append(orm_paper)

    await session.flush()
    return orm_papers


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search biomedical literature",
    description=(
        "Expands the natural language query into PubMed MeSH syntax, fetches papers from "
        "PubMed and Semantic Scholar concurrently, deduplicates and embeds results, reranks "
        "the top candidates with a cross-encoder, and generates an evidence synthesis. "
        "Results are cached by query string; repeat queries return instantly."
    ),
)
async def search(
    body: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Execute the full RAG pipeline for a biomedical literature query."""
    t_start = time.monotonic()

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_result = await db.execute(
        select(models.Query).where(
            models.Query.raw_query.ilike(body.query)
        )
    )
    cached_query = cache_result.scalar_one_or_none()

    if cached_query and cached_query.synthesis:
        import json as _json
        try:
            synth_data = _json.loads(cached_query.synthesis)
            synthesis = Synthesis(**synth_data)
        except Exception:
            synthesis = None

        contradictions = []
        if cached_query.contradictions:
            contradictions = [
                ContradictionResponse(**c) for c in cached_query.contradictions
            ]

        latency_ms = int((time.monotonic() - t_start) * 1000)
        logger.info("returning cached query result", query=body.query[:60])
        return SearchResponse(
            query_id=cached_query.id,
            raw_query=cached_query.raw_query,
            expanded_pubmed_query=cached_query.expanded_query,
            papers=[],
            synthesis=synthesis,
            contradictions=contradictions,
            faithfulness_score=cached_query.faithfulness,
            total_retrieved=cached_query.papers_retrieved or 0,
            latency_ms=latency_ms,
            cached=True,
        )

    # ── Query expansion ───────────────────────────────────────────────────────
    try:
        expanded = await expand_query(body.query)
    except Exception as exc:
        logger.error("query expansion failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Query expansion service unavailable.")

    # ── Fetch ─────────────────────────────────────────────────────────────────
    try:
        raw_papers = await fetch_all(expanded.pubmed_query, expanded.s2_query)
    except Exception as exc:
        logger.error("fetch failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Literature fetch failed.")

    # ── Deduplication ─────────────────────────────────────────────────────────
    papers = deduplicate(raw_papers)
    logger.info("deduplicated papers", count=len(papers))

    if not papers:
        query_id = uuid.uuid4()
        latency_ms = int((time.monotonic() - t_start) * 1000)
        return SearchResponse(
            query_id=query_id,
            raw_query=body.query,
            expanded_pubmed_query=expanded.pubmed_query,
            papers=[],
            synthesis=None,
            contradictions=[],
            total_retrieved=0,
            latency_ms=latency_ms,
        )

    # ── Embedding ─────────────────────────────────────────────────────────────
    try:
        vectors = await embed_papers(papers)
    except Exception as exc:
        logger.error("embedding failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Embedding service unavailable.")

    # ── Store ─────────────────────────────────────────────────────────────────
    try:
        if settings.use_faiss_fallback:
            await store_in_faiss(papers, vectors)
            # For FAISS path, rerank directly from fetched papers.
            top_k_papers = papers
        else:
            await _store_papers(list(zip(papers, vectors)), db)
    except Exception as exc:
        logger.warning("storage step failed — continuing without persistence", error=str(exc))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    try:
        query_vector = await embed_query(body.query)
        if settings.use_faiss_fallback:
            candidate_papers = await retrieve_papers(
                body.query, query_vector, top_k=20,
                year_from=body.year_from, year_to=body.year_to
            )
        else:
            candidate_papers = await retrieve_papers(
                body.query, query_vector, session=db, top_k=20,
                year_from=body.year_from, year_to=body.year_to
            )
    except Exception as exc:
        logger.warning("retrieval failed — using fetched papers directly", error=str(exc))
        candidate_papers = papers[:20]

    # ── Reranking ─────────────────────────────────────────────────────────────
    top_papers = await rerank(body.query, candidate_papers)

    # ── Synthesis ─────────────────────────────────────────────────────────────
    try:
        synthesis = await synthesise(body.query, top_papers)
    except Exception as exc:
        logger.error("synthesis failed", error=str(exc))
        synthesis = None

    # ── Contradiction detection ───────────────────────────────────────────────
    try:
        contradictions = await detect_contradictions(top_papers)
    except Exception as exc:
        logger.warning("contradiction detection failed", error=str(exc))
        contradictions = []

    # ── Persist query record ──────────────────────────────────────────────────
    import json as _json

    query_record = models.Query(
        raw_query=body.query,
        expanded_query=expanded.pubmed_query,
        papers_retrieved=len(papers),
        synthesis=synthesis.model_dump_json() if synthesis else None,
        contradictions=[c.model_dump() for c in contradictions] if contradictions else None,
    )
    db.add(query_record)
    await db.flush()

    latency_ms = int((time.monotonic() - t_start) * 1000)

    # ── Build response ────────────────────────────────────────────────────────
    paper_responses: list[PaperResponse] = []
    for p in top_papers:
        paper_responses.append(
            PaperResponse(
                id=uuid.uuid4(),  # placeholder if not stored in DB
                pubmed_id=p.pubmed_id,
                s2_id=p.s2_id,
                doi=p.doi,
                title=p.title,
                abstract=p.abstract,
                authors=p.authors,
                journal=p.journal,
                publication_year=p.publication_year,
                mesh_terms=p.mesh_terms,
                keywords=p.keywords,
                citation_count=p.citation_count,
                open_access_url=p.open_access_url,
                source=p.source,
            )
        )

    return SearchResponse(
        query_id=query_record.id,
        raw_query=body.query,
        expanded_pubmed_query=expanded.pubmed_query,
        papers=paper_responses,
        synthesis=synthesis,
        contradictions=contradictions,
        faithfulness_score=None,
        total_retrieved=len(papers),
        latency_ms=latency_ms,
        cached=False,
    )
