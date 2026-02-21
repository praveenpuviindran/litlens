"""BM25-based reranker. No models, no API calls, no GPU.

BM25 is a probabilistic retrieval function widely used in production search
systems (Elasticsearch default). It scores each paper's abstract against the
user's query using term frequency and inverse document frequency.
"""

from rank_bm25 import BM25Okapi

from src.fetcher import Paper


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + lowercase tokeniser."""
    return text.lower().split()


def rerank(query: str, papers: list[Paper], top_k: int = 10) -> list[Paper]:
    """Rerank papers by BM25 relevance to query.

    Args:
        query: The user's original research question.
        papers: Candidate papers retrieved from the fetch layer.
        top_k: Number of top papers to return.

    Returns:
        Top-k papers sorted by descending BM25 score.
    """
    if not papers:
        return []

    corpus = [
        _tokenise(f"{p.title} {p.abstract or ''}")
        for p in papers
    ]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenise(query)
    scores = bm25.get_scores(query_tokens)

    scored = sorted(zip(scores, papers), key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]
