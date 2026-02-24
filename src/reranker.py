"""BM25 + semantic reranker for biomedical paper retrieval.

When a sentence-transformers encoder is provided, combines BM25 (lexical)
with semantic cosine similarity for better relevance ranking. Falls back
to pure BM25 if no encoder is available.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from src.fetcher import Paper


def _tokenise(text: str) -> list[str]:
    return text.lower().split()


def rerank(query: str, papers: list[Paper], top_k: int = 10, encoder=None) -> list[Paper]:
    """Rerank papers by relevance to query.

    Combines BM25 (term frequency) with semantic similarity when an encoder
    is available. Without an encoder, uses pure BM25.

    Args:
        query: The user's research question.
        papers: Candidate papers to rerank.
        top_k: Number of top papers to return.
        encoder: Optional sentence-transformers encoder.

    Returns:
        Top-k papers sorted by descending combined score.
    """
    if not papers:
        return []

    corpus = [f"{p.title} {p.abstract or ''}" for p in papers]

    # BM25 scoring
    bm25_scores = np.zeros(len(papers))
    try:
        bm25 = BM25Okapi([_tokenise(doc) for doc in corpus])
        bm25_scores = np.array(bm25.get_scores(_tokenise(query)))
    except Exception:
        pass

    # Semantic scoring
    semantic_scores = np.zeros(len(papers))
    if encoder is not None:
        try:
            paper_embs = encoder.encode(corpus, show_progress_bar=False, batch_size=32)
            query_emb = encoder.encode([query], show_progress_bar=False)
            semantic_scores = cosine_similarity(query_emb, paper_embs)[0]
        except Exception:
            pass

    # Combine: 35% BM25, 65% semantic when encoder is available
    if encoder is not None and semantic_scores.max() > 0:
        bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores
        sem_norm = (semantic_scores + 1) / 2  # shift cosine from [-1,1] to [0,1]
        combined = 0.35 * bm25_norm + 0.65 * sem_norm
    else:
        combined = bm25_scores

    ranked_idx = sorted(range(len(papers)), key=lambda i: combined[i], reverse=True)
    return [papers[i] for i in ranked_idx[:top_k]]
