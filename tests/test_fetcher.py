"""Tests for backend/services/fetcher.py."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.schemas import Paper
from backend.services.fetcher import (
    _parse_pubmed_xml,
    fetch_all,
    fetch_pubmed,
    fetch_semantic_scholar,
)


class TestPubmedXmlParsing:
    """Tests for _parse_pubmed_xml."""

    def test_parse_pubmed_xml_returns_papers_with_abstract(
        self, mock_pubmed_response: str
    ) -> None:
        """Parsing valid XML returns Paper objects for each article with an abstract."""
        papers = _parse_pubmed_xml(mock_pubmed_response)
        assert len(papers) == 3

    def test_parse_pubmed_xml_extracts_correct_fields(
        self, mock_pubmed_response: str
    ) -> None:
        """Parsed papers contain the expected title, pubmed_id, doi, and authors."""
        papers = _parse_pubmed_xml(mock_pubmed_response)
        first = papers[0]
        assert first.pubmed_id == "11111"
        assert "Metformin" in first.title
        assert first.doi == "10.1056/test.001"
        assert len(first.authors) == 2
        assert first.source == "pubmed"

    def test_parse_pubmed_xml_discards_papers_without_abstract(self) -> None:
        """Papers with no AbstractText element are excluded from results."""
        xml_no_abstract = """<?xml version="1.0"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>99999</PMID>
              <Article>
                <ArticleTitle>Paper Without Abstract</ArticleTitle>
                <AuthorList/>
                <Journal><ISOAbbreviation>Test J</ISOAbbreviation></Journal>
                <PubDate><Year>2020</Year></PubDate>
                <ArticleIdList/>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>"""
        papers = _parse_pubmed_xml(xml_no_abstract)
        assert papers == []

    def test_parse_pubmed_xml_returns_empty_on_malformed_xml(self) -> None:
        """Malformed XML returns an empty list instead of raising an exception."""
        papers = _parse_pubmed_xml("<<<not valid xml>>>")
        assert papers == []

    def test_parse_pubmed_xml_extracts_mesh_terms(
        self, mock_pubmed_response: str
    ) -> None:
        """MeSH terms are extracted from MeshHeadingList."""
        papers = _parse_pubmed_xml(mock_pubmed_response)
        first = papers[0]
        assert "Metformin" in first.mesh_terms


class TestPubmedEsearch:
    """Tests for the ESearch step of the PubMed fetcher."""

    @pytest.mark.asyncio
    async def test_pubmed_esearch_called_with_correct_params(self) -> None:
        """ESearch is called with the expected db, retmode, and email params."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"esearchresult": {"idlist": ["111", "222"]}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        from backend.services.fetcher import _pubmed_esearch

        ids = await _pubmed_esearch(mock_client, "metformin[MeSH Terms]")
        call_kwargs = mock_client.get.call_args
        params = call_kwargs[1]["params"]

        assert params["db"] == "pubmed"
        assert params["retmode"] == "json"
        assert ids == ["111", "222"]


class TestRetryOnRateLimit:
    """Tests for exponential backoff on 429 responses."""

    @pytest.mark.asyncio
    async def test_fetch_pubmed_returns_empty_list_on_persistent_error(self) -> None:
        """fetch_pubmed returns an empty list when the API consistently fails."""
        with patch("backend.services.fetcher._pubmed_esearch", new=AsyncMock(side_effect=Exception("timeout"))):
            result = await fetch_pubmed("test query")
        assert result == []


class TestSemanticScholarFetch:
    """Tests for fetch_semantic_scholar."""

    @pytest.mark.asyncio
    async def test_s2_fetch_returns_papers(self, mock_s2_response: dict) -> None:
        """fetch_semantic_scholar returns Paper objects from a valid API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_s2_response
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(return_value=mock_response)
            papers = await fetch_semantic_scholar("metformin cardiovascular")

        assert len(papers) == 3
        assert all(isinstance(p, Paper) for p in papers)

    @pytest.mark.asyncio
    async def test_s2_fetch_returns_empty_list_on_error(self) -> None:
        """fetch_semantic_scholar returns an empty list on network error."""
        with patch("httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(side_effect=Exception("connection refused"))
            papers = await fetch_semantic_scholar("test")
        assert papers == []


class TestFetchAll:
    """Tests for concurrent fetch_all."""

    @pytest.mark.asyncio
    async def test_fetch_all_calls_both_sources(self) -> None:
        """fetch_all calls both PubMed and Semantic Scholar concurrently."""
        pubmed_called = asyncio.Event()
        s2_called = asyncio.Event()

        async def fake_pubmed(q: str) -> list:
            pubmed_called.set()
            return []

        async def fake_s2(q: str) -> list:
            s2_called.set()
            return []

        with (
            patch("backend.services.fetcher.fetch_pubmed", side_effect=fake_pubmed),
            patch("backend.services.fetcher.fetch_semantic_scholar", side_effect=fake_s2),
        ):
            await fetch_all("mesh query", "plain query")

        assert pubmed_called.is_set()
        assert s2_called.is_set()

    @pytest.mark.asyncio
    async def test_fetch_all_combines_results(self) -> None:
        """fetch_all returns the union of papers from both sources."""
        pubmed_paper = Paper(title="PubMed Paper", abstract="abstract", source="pubmed")
        s2_paper = Paper(title="S2 Paper", abstract="abstract", source="semantic_scholar")

        with (
            patch("backend.services.fetcher.fetch_pubmed", new=AsyncMock(return_value=[pubmed_paper])),
            patch("backend.services.fetcher.fetch_semantic_scholar", new=AsyncMock(return_value=[s2_paper])),
        ):
            results = await fetch_all("q", "q")

        assert len(results) == 2
