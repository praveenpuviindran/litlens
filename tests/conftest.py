"""Shared pytest fixtures for the LitLens test suite."""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from backend.schemas import Paper


# ── Paper fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def test_papers() -> list[Paper]:
    """Return 5 fully-populated Paper objects for use across tests."""
    return [
        Paper(
            pubmed_id=f"1234{i}",
            s2_id=f"s2_{i}",
            doi=f"10.1000/test.{i}",
            title=f"Effect of Drug {i} on Cardiovascular Outcomes in Type 2 Diabetes",
            abstract=(
                f"Background: This study examines Drug {i} in {100 * i} patients. "
                f"Methods: Randomised controlled trial. "
                f"Results: Drug {i} reduced LDL by {10 + i}%. "
                f"Conclusion: Drug {i} is effective for cardiovascular risk reduction."
            ),
            authors=[f"Smith, John", f"Doe, Jane"],
            journal="J Med Research",
            publication_year=2020 + i,
            mesh_terms=["Diabetes Mellitus, Type 2", "Cardiovascular Diseases"],
            keywords=["diabetes", "cardiovascular"],
            citation_count=100 * i,
            open_access_url=f"https://example.com/paper{i}",
            source="pubmed",
        )
        for i in range(1, 6)
    ]


# ── OpenAI mock ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Return a mock OpenAI client that produces deterministic outputs.

    Embeddings: 1536-dimensional zero vector with a 1.0 at index 0.
    Chat completions: deterministic JSON string.
    """
    client = MagicMock()

    # Embedding response
    embedding_data = MagicMock()
    vector = [0.0] * 1536
    vector[0] = 1.0
    embedding_data.embedding = vector
    embedding_response = MagicMock()
    embedding_response.data = [embedding_data]

    embed_mock = AsyncMock(return_value=embedding_response)
    client.embeddings = MagicMock()
    client.embeddings.create = embed_mock

    # Chat completion response
    choice = MagicMock()
    choice.message.content = (
        '{"consensus_statement": "Test consensus.", '
        '"key_findings": [{"finding": "Drug reduces LDL.", "citations": [1]}], '
        '"evidence_quality": "moderate", '
        '"gaps": ["Long-term data needed."], '
        '"limitations": "All studies observational."}'
    )
    completion = MagicMock()
    completion.choices = [choice]
    chat_mock = AsyncMock(return_value=completion)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = chat_mock

    return client


# ── API response mocks ────────────────────────────────────────────────────────


@pytest.fixture
def mock_pubmed_response() -> str:
    """Return a realistic PubMed EFetch XML string for 3 papers."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111</PMID>
      <Article>
        <ArticleTitle>Metformin and Cardiovascular Risk in Type 2 Diabetes</ArticleTitle>
        <Abstract>
          <AbstractText>Background: Metformin is widely used. Methods: RCT of 500 patients.
          Results: Metformin reduced MACE by 15%. Conclusion: Beneficial effect confirmed.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Johnson</LastName><ForeName>Alice</ForeName></Author>
          <Author><LastName>Chen</LastName><ForeName>Bob</ForeName></Author>
        </AuthorList>
        <Journal><ISOAbbreviation>N Engl J Med</ISOAbbreviation></Journal>
        <PubDate><Year>2021</Year></PubDate>
        <ArticleIdList>
          <ArticleId IdType="doi">10.1056/test.001</ArticleId>
        </ArticleIdList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Metformin</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName>Diabetes Mellitus, Type 2</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222</PMID>
      <Article>
        <ArticleTitle>Statin Therapy in Primary Prevention of Cardiovascular Disease</ArticleTitle>
        <Abstract>
          <AbstractText>Statins reduce LDL cholesterol by 40% in primary prevention settings.
          A meta-analysis of 20 RCTs confirms significant risk reduction.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Williams</LastName><ForeName>Carol</ForeName></Author>
        </AuthorList>
        <Journal><ISOAbbreviation>Lancet</ISOAbbreviation></Journal>
        <PubDate><Year>2022</Year></PubDate>
        <ArticleIdList>
          <ArticleId IdType="doi">10.1016/test.002</ArticleId>
        </ArticleIdList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Hydroxymethylglutaryl-CoA Reductase Inhibitors</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>33333</PMID>
      <Article>
        <ArticleTitle>SGLT2 Inhibitors and Heart Failure Hospitalisation</ArticleTitle>
        <Abstract>
          <AbstractText>SGLT2 inhibitors reduce heart failure hospitalisations by 30% in
          patients with type 2 diabetes and established cardiovascular disease.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Patel</LastName><ForeName>Raj</ForeName></Author>
        </AuthorList>
        <Journal><ISOAbbreviation>JAMA</ISOAbbreviation></Journal>
        <PubDate><Year>2023</Year></PubDate>
        <ArticleIdList>
          <ArticleId IdType="doi">10.1001/test.003</ArticleId>
        </ArticleIdList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


@pytest.fixture
def mock_s2_response() -> dict[str, Any]:
    """Return a realistic Semantic Scholar API JSON response for 3 papers."""
    return {
        "data": [
            {
                "paperId": "s2abc001",
                "title": "Metformin and Cardiovascular Risk in Type 2 Diabetes",
                "abstract": "Metformin reduces MACE by 15% in RCT of 500 patients.",
                "authors": [{"name": "Johnson, Alice"}, {"name": "Chen, Bob"}],
                "year": 2021,
                "citationCount": 350,
                "externalIds": {"DOI": "10.1056/test.001", "PubMed": "11111"},
                "publicationDate": "2021-03-15",
                "fieldsOfStudy": ["Medicine"],
                "openAccessPdf": {"url": "https://example.com/paper1.pdf"},
            },
            {
                "paperId": "s2abc002",
                "title": "GLP-1 Receptor Agonists for Weight Loss in Obesity",
                "abstract": "GLP-1 agonists produce 10-15% body weight reduction in obese adults.",
                "authors": [{"name": "Garcia, Maria"}],
                "year": 2023,
                "citationCount": 120,
                "externalIds": {"DOI": "10.1000/glp1.001"},
                "publicationDate": "2023-06-01",
                "fieldsOfStudy": ["Medicine"],
                "openAccessPdf": None,
            },
            {
                "paperId": "s2abc003",
                "title": "Aspirin in Primary Prevention: A Systematic Review",
                "abstract": "Low-dose aspirin does not reduce all-cause mortality in primary prevention.",
                "authors": [{"name": "Thompson, David"}, {"name": "Lee, Sarah"}],
                "year": 2022,
                "citationCount": 280,
                "externalIds": {"DOI": "10.1002/aspirin.001"},
                "publicationDate": "2022-11-20",
                "fieldsOfStudy": ["Medicine"],
                "openAccessPdf": {"url": "https://example.com/paper3.pdf"},
            },
        ]
    }


# ── FastAPI test client ───────────────────────────────────────────────────────


@pytest.fixture
def test_client(mock_openai_client: MagicMock, mocker: Any) -> Any:
    """Return a FastAPI TestClient with all DB and external calls mocked."""
    # Mock DB session dependency
    mocker.patch("backend.database.init_db", new=AsyncMock())
    mocker.patch("backend.database.AsyncSessionLocal", new=MagicMock())

    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)
