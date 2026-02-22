"""Embedding generation, storage, and retrieval service.

Supports two storage backends:
- PostgreSQL + pgvector (production): hybrid retrieval via vector similarity + FTS + RRF.
- FAISS in-memory index (development fallback): pure vector similarity.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from openai import AsyncOpenAI

from backend.config import settings
from backend.schemas import Paper

logger = structlog.get_logger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100  # OpenAI recommends batches of ≤ 100 texts
MAX_CHARS = 8000  # Truncate before embedding to stay within token limits

FAISS_INDEX_PATH = Path("./data/faiss.index")
FAISS_META_PATH = Path("./data/faiss_metadata.json")

# Lazy-loaded module-level FAISS state (only used when USE_FAISS_FALLBACK=true)
_faiss_index = None
_faiss_metadata: list[dict] = []


def _get_openai_client() -> AsyncOpenAI:
    """Return a new AsyncOpenAI client configured with the application API key."""
    return AsyncOpenAI(api_key=settings.openai_api_key)


def _text_for_embedding(paper: Paper) -> str:
    """Build the embedding input text for a paper.

    Concatenates title and abstract, truncated to MAX_CHARS.

    Args:
        paper: The paper to embed.

    Returns:
        A single string ready for the embedding API.
    """
    text = f"{paper.title}. {paper.abstract or ''}".strip()
    return text[:MAX_CHARS]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI text-embedding-3-small.

    Processes texts in batches of BATCH_SIZE concurrently.

    Args:
        texts: List of strings to embed.

    Returns:
        List of 1536-dimensional float vectors, in the same order as input.
    """
    client = _get_openai_client()
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    async def embed_batch(batch: list[str]) -> list[list[float]]:
        """Embed a single batch and return its vectors."""
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        return [item.embedding for item in response.data]

    results = await asyncio.gather(*[embed_batch(b) for b in batches])
    # Flatten list of lists
    return [vec for batch_vecs in results for vec in batch_vecs]


async def embed_papers(papers: list[Paper]) -> list[list[float]]:
    """Generate embeddings for a list of papers.

    Args:
        papers: Papers to embed.

    Returns:
        List of embedding vectors in the same order as input papers.
    """
    texts = [_text_for_embedding(p) for p in papers]
    logger.info("embedding papers", count=len(papers))
    vectors = await embed_texts(texts)
    logger.info("embedding complete", count=len(vectors))
    return vectors


async def embed_query(query: str) -> list[float]:
    """Embed a single query string.

    Args:
        query: Raw natural language query.

    Returns:
        1536-dimensional embedding vector.
    """
    vectors = await embed_texts([query[:MAX_CHARS]])
    return vectors[0]


# ── FAISS fallback ────────────────────────────────────────────────────────────


def _load_faiss_index() -> None:
    """Load a persisted FAISS index from disk, or create a new one.

    Called lazily on first use. Sets module globals ``_faiss_index`` and
    ``_faiss_metadata``.
    """
    global _faiss_index, _faiss_metadata
    try:
        import faiss  # type: ignore[import]
    except ImportError:
        logger.error("faiss-cpu is not installed  -  cannot use FAISS fallback")
        raise

    if FAISS_INDEX_PATH.exists():
        _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        _faiss_metadata = json.loads(FAISS_META_PATH.read_text()) if FAISS_META_PATH.exists() else []
        logger.info("FAISS index loaded from disk", vectors=_faiss_index.ntotal)
    else:
        # Inner product on L2-normalised vectors == cosine similarity.
        _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        _faiss_metadata = []
        logger.info("new FAISS index created in memory")


def _save_faiss_index() -> None:
    """Persist the FAISS index and metadata to disk."""
    import faiss  # type: ignore[import]

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_faiss_index, str(FAISS_INDEX_PATH))
    FAISS_META_PATH.write_text(json.dumps(_faiss_metadata))
    logger.debug("FAISS index saved", vectors=_faiss_index.ntotal)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows of a 2D float32 array for cosine similarity via inner product.

    Args:
        vectors: Shape (n, dim) float32 array.

    Returns:
        Normalised array of same shape.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


async def store_in_faiss(papers: list[Paper], vectors: list[list[float]]) -> None:
    """Add paper embeddings to the FAISS index and persist to disk.

    Args:
        papers: Paper objects (must be in the same order as *vectors*).
        vectors: Embedding vectors for each paper.
    """
    global _faiss_index, _faiss_metadata

    if _faiss_index is None:
        await asyncio.to_thread(_load_faiss_index)

    arr = np.array(vectors, dtype=np.float32)
    arr = _normalise(arr)

    start_idx = _faiss_index.ntotal
    _faiss_index.add(arr)  # type: ignore[union-attr]

    for i, paper in enumerate(papers):
        _faiss_metadata.append(
            {
                "faiss_idx": start_idx + i,
                "title": paper.title,
                "abstract": paper.abstract,
                "pubmed_id": paper.pubmed_id,
                "s2_id": paper.s2_id,
                "doi": paper.doi,
                "authors": paper.authors,
                "journal": paper.journal,
                "publication_year": paper.publication_year,
                "mesh_terms": paper.mesh_terms,
                "citation_count": paper.citation_count,
                "open_access_url": paper.open_access_url,
                "source": paper.source,
            }
        )

    await asyncio.to_thread(_save_faiss_index)
    logger.info("stored in FAISS", new_vectors=len(papers), total=_faiss_index.ntotal)


async def retrieve_from_faiss(query_vector: list[float], top_k: int = 20) -> list[Paper]:
    """Retrieve the top-k most similar papers from the FAISS index.

    Args:
        query_vector: Embedding of the search query.
        top_k: Number of results to return.

    Returns:
        List of Paper objects sorted by descending cosine similarity.
    """
    global _faiss_index, _faiss_metadata

    if _faiss_index is None:
        await asyncio.to_thread(_load_faiss_index)

    if _faiss_index.ntotal == 0:  # type: ignore[union-attr]
        logger.warning("FAISS index is empty  -  no results to return")
        return []

    q = np.array([query_vector], dtype=np.float32)
    q = _normalise(q)
    k = min(top_k, _faiss_index.ntotal)  # type: ignore[union-attr]

    # Blocking FAISS search offloaded to a thread pool.
    _distances, indices = await asyncio.to_thread(_faiss_index.search, q, k)  # type: ignore[union-attr]
    indices = indices[0]

    papers: list[Paper] = []
    for idx in indices:
        if idx < 0 or idx >= len(_faiss_metadata):
            continue
        meta = _faiss_metadata[idx]
        papers.append(
            Paper(
                pubmed_id=meta.get("pubmed_id"),
                s2_id=meta.get("s2_id"),
                doi=meta.get("doi"),
                title=meta["title"],
                abstract=meta.get("abstract"),
                authors=meta.get("authors") or [],
                journal=meta.get("journal"),
                publication_year=meta.get("publication_year"),
                mesh_terms=meta.get("mesh_terms") or [],
                keywords=[],
                citation_count=meta.get("citation_count") or 0,
                open_access_url=meta.get("open_access_url"),
                source=meta.get("source", "unknown"),
            )
        )
    return papers


# ── PostgreSQL + pgvector retrieval ───────────────────────────────────────────


async def retrieve_hybrid_pgvector(
    query: str,
    query_vector: list[float],
    session,
    top_k: int = 20,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> list[Paper]:
    """Hybrid retrieval: vector similarity + PostgreSQL full-text search fused with RRF.

    Reciprocal Rank Fusion (RRF) with k=60 combines the two ranked lists.
    Returns up to top_k Paper objects.

    Args:
        query: Raw query string (used for FTS).
        query_vector: Embedded query vector.
        session: SQLAlchemy async session.
        top_k: Number of results to return.
        year_from: Optional lower bound on publication_year.
        year_to: Optional upper bound on publication_year.

    Returns:
        Reranked list of Paper objects.
    """
    from sqlalchemy import text as sa_text

    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
    year_filter = ""
    if year_from:
        year_filter += f" AND publication_year >= {year_from}"
    if year_to:
        year_filter += f" AND publication_year <= {year_to}"

    # Vector similarity search
    vector_sql = sa_text(f"""
        SELECT id, title, abstract, pubmed_id, s2_id, doi, authors,
               journal, publication_year, mesh_terms, keywords,
               citation_count, open_access_url, source,
               ROW_NUMBER() OVER (ORDER BY embedding <=> '{vector_str}') AS rank
        FROM papers
        WHERE embedding IS NOT NULL {year_filter}
        ORDER BY embedding <=> '{vector_str}'
        LIMIT :limit
    """)

    # Full-text search
    fts_sql = sa_text(f"""
        SELECT id, title, abstract, pubmed_id, s2_id, doi, authors,
               journal, publication_year, mesh_terms, keywords,
               citation_count, open_access_url, source,
               ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', :q)) DESC) AS rank
        FROM papers
        WHERE fts_vector @@ plainto_tsquery('english', :q) {year_filter}
        ORDER BY ts_rank(fts_vector, plainto_tsquery('english', :q)) DESC
        LIMIT :limit
    """)

    RRF_K = 60
    vector_results = await session.execute(vector_sql, {"limit": top_k * 2})
    fts_results = await session.execute(fts_sql, {"q": query, "limit": top_k * 2})

    # Build RRF score map: paper_id -> score
    rrf_scores: dict[str, float] = {}
    row_map: dict[str, dict] = {}

    for row in vector_results.mappings():
        pid = str(row["id"])
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + row["rank"])
        row_map[pid] = dict(row)

    for row in fts_results.mappings():
        pid = str(row["id"])
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + row["rank"])
        if pid not in row_map:
            row_map[pid] = dict(row)

    sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)
    papers: list[Paper] = []
    for pid in sorted_ids[:top_k]:
        row = row_map[pid]
        papers.append(
            Paper(
                pubmed_id=row.get("pubmed_id"),
                s2_id=row.get("s2_id"),
                doi=row.get("doi"),
                title=row["title"],
                abstract=row.get("abstract"),
                authors=row.get("authors") or [],
                journal=row.get("journal"),
                publication_year=row.get("publication_year"),
                mesh_terms=row.get("mesh_terms") or [],
                keywords=row.get("keywords") or [],
                citation_count=row.get("citation_count") or 0,
                open_access_url=row.get("open_access_url"),
                source=row.get("source", "unknown"),
            )
        )

    logger.info("hybrid retrieval complete", rrf_results=len(papers))
    return papers


async def retrieve_papers(
    query: str,
    query_vector: list[float],
    session=None,
    top_k: int = 20,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> list[Paper]:
    """Retrieve papers using the configured backend (FAISS or PostgreSQL).

    Args:
        query: Raw query string.
        query_vector: Embedded query vector.
        session: SQLAlchemy async session (required for PostgreSQL path).
        top_k: Number of results to return.
        year_from: Optional publication year lower bound.
        year_to: Optional publication year upper bound.

    Returns:
        Top-k relevant Paper objects.
    """
    if settings.use_faiss_fallback:
        return await retrieve_from_faiss(query_vector, top_k)
    else:
        if session is None:
            raise ValueError("A database session is required for PostgreSQL retrieval.")
        return await retrieve_hybrid_pgvector(query, query_vector, session, top_k, year_from, year_to)
