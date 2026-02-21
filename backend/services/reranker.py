"""Cross-encoder reranking service.

Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` from HuggingFace to score
query-paper relevance pairs and return the top-10 most relevant papers.

The model is loaded once at startup via a lazy singleton to avoid paying the
load cost on every request.
"""

import asyncio
import time
from typing import Optional

import structlog

from backend.schemas import Paper

logger = structlog.get_logger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 10
# Log a warning if reranking exceeds this duration.
PERF_WARNING_SECONDS = 5.0

# ── Lazy singleton ────────────────────────────────────────────────────────────
_cross_encoder = None
_load_attempted: bool = False


def _load_model() -> None:
    """Load the cross-encoder model into the module-level singleton.

    This is a blocking operation and must be called via ``asyncio.to_thread``
    from async contexts.
    """
    global _cross_encoder, _load_attempted
    _load_attempted = True
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]

        _cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("cross-encoder model loaded", model=RERANKER_MODEL)
    except Exception as exc:
        logger.warning(
            "cross-encoder failed to load — will fall back to similarity ranking",
            model=RERANKER_MODEL,
            error=str(exc),
        )
        _cross_encoder = None


async def warm_reranker() -> None:
    """Pre-load the cross-encoder at application startup.

    Call this once from the FastAPI lifespan/startup event to ensure the first
    user request does not pay the model-load latency.
    """
    global _load_attempted
    if not _load_attempted:
        await asyncio.to_thread(_load_model)


def _score_pairs(query: str, abstracts: list[str]) -> list[float]:
    """Run synchronous cross-encoder inference.

    Args:
        query: The original user query.
        abstracts: List of paper abstract strings to score.

    Returns:
        List of relevance scores, one per abstract.
    """
    pairs = [(query, abstract or "") for abstract in abstracts]
    scores: list[float] = _cross_encoder.predict(pairs).tolist()  # type: ignore[union-attr]
    return scores


async def rerank(query: str, papers: list[Paper]) -> list[Paper]:
    """Rerank papers by relevance to *query* using the cross-encoder.

    Falls back gracefully to the original ordering if the model is unavailable.
    Returns at most TOP_K papers.

    Args:
        query: The original user query string.
        papers: Up to 20 candidate papers from embedding retrieval.

    Returns:
        Top-10 papers sorted by descending cross-encoder score.
    """
    global _cross_encoder, _load_attempted

    # Load model on first call if warm_reranker was not called at startup.
    if not _load_attempted:
        await asyncio.to_thread(_load_model)

    if _cross_encoder is None:
        logger.warning("reranker unavailable — returning top-k by original order")
        return papers[:TOP_K]

    abstracts = [p.abstract or "" for p in papers]

    t_start = time.monotonic()
    try:
        scores = await asyncio.to_thread(_score_pairs, query, abstracts)
    except Exception as exc:
        logger.error("cross-encoder inference failed", error=str(exc))
        return papers[:TOP_K]
    elapsed = time.monotonic() - t_start

    if elapsed > PERF_WARNING_SECONDS:
        logger.warning(
            "reranker exceeded performance threshold",
            elapsed_seconds=round(elapsed, 2),
            threshold=PERF_WARNING_SECONDS,
        )

    scored = sorted(zip(scores, papers), key=lambda x: x[0], reverse=True)
    top_papers = [p for _, p in scored[:TOP_K]]
    logger.info(
        "reranking complete",
        input_count=len(papers),
        output_count=len(top_papers),
        elapsed_seconds=round(elapsed, 2),
    )
    return top_papers
