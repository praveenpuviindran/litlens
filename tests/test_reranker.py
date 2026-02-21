"""Tests for backend/services/reranker.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.schemas import Paper
from backend.services.reranker import TOP_K, rerank


def make_papers(n: int) -> list[Paper]:
    """Create n Paper objects with sequential titles for testing."""
    return [
        Paper(
            title=f"Paper {i}",
            abstract=f"This is the abstract for paper number {i}.",
            source="pubmed",
        )
        for i in range(n)
    ]


class TestRerankerOutput:
    """Tests for the rerank function's output."""

    @pytest.mark.asyncio
    async def test_rerank_returns_exactly_top_k_papers(self) -> None:
        """rerank returns exactly TOP_K papers when given more than TOP_K inputs."""
        papers = make_papers(20)
        # Assign descending scores so paper 0 scores highest.
        scores = list(range(20, 0, -1))

        with patch("backend.services.reranker._load_attempted", True):
            with patch("backend.services.reranker._cross_encoder") as mock_encoder:
                import numpy as np
                mock_encoder.predict.return_value = np.array(scores)
                result = await rerank("test query", papers)

        assert len(result) == TOP_K

    @pytest.mark.asyncio
    async def test_rerank_output_sorted_descending_by_score(self) -> None:
        """rerank returns papers sorted by score from highest to lowest."""
        papers = make_papers(5)
        # Score paper 4 highest, paper 0 lowest.
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]

        with patch("backend.services.reranker._load_attempted", True):
            with patch("backend.services.reranker._cross_encoder") as mock_encoder:
                import numpy as np
                mock_encoder.predict.return_value = np.array(scores)
                result = await rerank("test query", papers)

        # Paper index 4 scored 5.0 (highest), should be first.
        assert result[0].title == "Paper 4"

    @pytest.mark.asyncio
    async def test_rerank_falls_back_when_model_unavailable(self) -> None:
        """rerank returns top-k papers by original order when cross-encoder is None."""
        papers = make_papers(20)

        with patch("backend.services.reranker._load_attempted", True):
            with patch("backend.services.reranker._cross_encoder", None):
                result = await rerank("test query", papers)

        assert len(result) == TOP_K
        # Should return the first TOP_K papers in original order.
        assert result[0].title == "Paper 0"

    @pytest.mark.asyncio
    async def test_rerank_handles_fewer_than_top_k_papers(self) -> None:
        """rerank returns all papers when fewer than TOP_K are provided."""
        papers = make_papers(3)
        scores = [0.9, 0.5, 0.7]

        with patch("backend.services.reranker._load_attempted", True):
            with patch("backend.services.reranker._cross_encoder") as mock_encoder:
                import numpy as np
                mock_encoder.predict.return_value = np.array(scores)
                result = await rerank("test query", papers)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_rerank_falls_back_on_inference_exception(self) -> None:
        """rerank returns top-k originals when cross-encoder.predict raises."""
        papers = make_papers(15)

        with patch("backend.services.reranker._load_attempted", True):
            with patch("backend.services.reranker._cross_encoder") as mock_encoder:
                mock_encoder.predict.side_effect = RuntimeError("CUDA OOM")
                result = await rerank("test query", papers)

        assert len(result) == TOP_K
