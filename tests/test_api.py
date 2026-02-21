"""FastAPI integration tests using TestClient."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.schemas import Paper, Synthesis, KeyFinding


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient with DB init and reranker warm-up mocked out."""
    with (
        patch("backend.database.init_db", new=AsyncMock()),
        patch("backend.services.reranker.warm_reranker", new=AsyncMock()),
        patch("backend.database.AsyncSessionLocal"),
    ):
        from backend.main import app
        return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health always returns HTTP 200."""
        with patch("backend.routers.health.AsyncSessionLocal") as mock_session:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.execute = AsyncMock()
            mock_session.return_value = mock_ctx

            response = client.get("/health")

        assert response.status_code == 200

    def test_health_response_has_required_fields(self, client: TestClient) -> None:
        """GET /health response contains status, database, openai, and version fields."""
        with patch("backend.routers.health.AsyncSessionLocal") as mock_session:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.execute = AsyncMock()
            mock_session.return_value = mock_ctx

            response = client.get("/health")

        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "openai" in data
        assert "version" in data

    def test_health_returns_ok_status(self, client: TestClient) -> None:
        """GET /health returns status == 'ok' regardless of sub-component health."""
        with patch("backend.routers.health.AsyncSessionLocal") as mock_session:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.execute = AsyncMock(side_effect=Exception("DB down"))
            mock_session.return_value = mock_ctx

            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestSearchEndpoint:
    """Tests for POST /search."""

    def _mock_search_pipeline(self, mocker) -> None:
        """Patch all external services for the search pipeline."""
        from backend.schemas import ContradictionResponse

        fake_paper = Paper(
            title="Metformin Cardiovascular Outcomes",
            abstract="RCT showing 15% reduction in MACE.",
            source="pubmed",
            pubmed_id="12345",
        )
        fake_synthesis = Synthesis(
            consensus_statement="Evidence supports metformin.",
            key_findings=[KeyFinding(finding="Reduces risk.", citations=[1])],
            evidence_quality="moderate",
            gaps=["Long-term data needed."],
            limitations="Observational studies only.",
        )

        mocker.patch(
            "backend.routers.search.expand_query",
            new=AsyncMock(return_value=MagicMock(
                pubmed_query="metformin[MeSH]", s2_query="metformin cardiovascular"
            )),
        )
        mocker.patch(
            "backend.routers.search.fetch_all",
            new=AsyncMock(return_value=[fake_paper]),
        )
        mocker.patch(
            "backend.routers.search.deduplicate",
            return_value=[fake_paper],
        )
        mocker.patch(
            "backend.routers.search.embed_papers",
            new=AsyncMock(return_value=[[0.0] * 1536]),
        )
        mocker.patch(
            "backend.routers.search.embed_query",
            new=AsyncMock(return_value=[0.0] * 1536),
        )
        mocker.patch(
            "backend.routers.search.store_in_faiss",
            new=AsyncMock(),
        )
        mocker.patch(
            "backend.routers.search.retrieve_papers",
            new=AsyncMock(return_value=[fake_paper]),
        )
        mocker.patch(
            "backend.routers.search.rerank",
            new=AsyncMock(return_value=[fake_paper]),
        )
        mocker.patch(
            "backend.routers.search.synthesise",
            new=AsyncMock(return_value=fake_synthesis),
        )
        mocker.patch(
            "backend.routers.search.detect_contradictions",
            new=AsyncMock(return_value=[]),
        )

    def test_search_returns_200_with_valid_body(self, client: TestClient, mocker) -> None:
        """POST /search with a valid body returns HTTP 200."""
        self._mock_search_pipeline(mocker)

        # Also mock the DB session used in the router
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        with patch("backend.routers.search.get_db", return_value=mock_session):
            response = client.post("/search", json={"query": "metformin cardiovascular diabetes"})

        assert response.status_code == 200

    def test_search_returns_422_when_query_missing(self, client: TestClient) -> None:
        """POST /search without a query field returns HTTP 422."""
        response = client.post("/search", json={"max_results": 10})
        assert response.status_code == 422

    def test_search_returns_422_when_query_too_short(self, client: TestClient) -> None:
        """POST /search with a query shorter than 3 chars returns HTTP 422."""
        response = client.post("/search", json={"query": "ab"})
        assert response.status_code == 422

    def test_search_response_contains_required_fields(self, client: TestClient, mocker) -> None:
        """POST /search response contains query_id, raw_query, papers, and synthesis fields."""
        self._mock_search_pipeline(mocker)

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        with patch("backend.routers.search.get_db", return_value=mock_session):
            response = client.post("/search", json={"query": "statin therapy LDL reduction"})

        data = response.json()
        assert "query_id" in data
        assert "raw_query" in data
        assert "papers" in data
        assert "synthesis" in data


class TestPapersEndpoint:
    """Tests for GET /papers."""

    def test_papers_returns_200(self, client: TestClient) -> None:
        """GET /papers returns HTTP 200."""
        mock_session = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        papers_result = MagicMock()
        papers_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(side_effect=[count_result, papers_result])

        with patch("backend.routers.papers.get_db", return_value=mock_session):
            response = client.get("/papers")

        assert response.status_code == 200

    def test_papers_response_has_pagination_fields(self, client: TestClient) -> None:
        """GET /papers response contains total, page, page_size, and pages fields."""
        mock_session = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        papers_result = MagicMock()
        papers_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(side_effect=[count_result, papers_result])

        with patch("backend.routers.papers.get_db", return_value=mock_session):
            response = client.get("/papers")

        data = response.json()
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "pages" in data
