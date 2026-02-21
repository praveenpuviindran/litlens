"""Tests for backend/services/deduplicator.py."""

import pytest

from backend.schemas import Paper
from backend.services.deduplicator import FUZZY_TITLE_THRESHOLD, deduplicate, _merge


def make_paper(
    title: str = "Default Title",
    abstract: str = "Default abstract text.",
    doi: str | None = None,
    source: str = "pubmed",
    pubmed_id: str | None = None,
    s2_id: str | None = None,
    mesh_terms: list[str] | None = None,
    authors: list[str] | None = None,
) -> Paper:
    """Helper to create a Paper with sensible defaults for testing."""
    return Paper(
        pubmed_id=pubmed_id,
        s2_id=s2_id,
        doi=doi,
        title=title,
        abstract=abstract,
        authors=authors or ["Smith, John"],
        journal="Test Journal",
        publication_year=2022,
        mesh_terms=mesh_terms or [],
        keywords=[],
        citation_count=10,
        open_access_url=None,
        source=source,
    )


class TestDoiDeduplication:
    """Tests for Pass 1: exact DOI matching."""

    def test_doi_exact_match_merges_duplicates(self) -> None:
        """Two papers sharing a DOI are merged into a single record."""
        a = make_paper(title="Paper A", doi="10.1000/test", source="pubmed")
        b = make_paper(title="Paper B", doi="10.1000/test", source="semantic_scholar")
        result = deduplicate([a, b])
        assert len(result) == 1

    def test_doi_match_sets_source_to_both(self) -> None:
        """Merged DOI duplicate has source == 'both'."""
        a = make_paper(doi="10.1000/test", source="pubmed")
        b = make_paper(doi="10.1000/test", source="semantic_scholar")
        result = deduplicate([a, b])
        assert result[0].source == "both"

    def test_doi_match_prefers_longer_abstract(self) -> None:
        """Merged record keeps the abstract from the paper with the longer abstract."""
        a = make_paper(doi="10.1000/test", abstract="Short.", source="pubmed")
        b = make_paper(doi="10.1000/test", abstract="A much longer and more detailed abstract that has more information.", source="semantic_scholar")
        result = deduplicate([a, b])
        assert result[0].abstract == b.abstract

    def test_different_dois_not_merged(self) -> None:
        """Papers with distinct DOIs are not merged."""
        a = make_paper(doi="10.1000/aaa", source="pubmed")
        b = make_paper(doi="10.1000/bbb", source="pubmed")
        result = deduplicate([a, b])
        assert len(result) == 2


class TestFuzzyTitleDeduplication:
    """Tests for Pass 2: fuzzy title matching."""

    def test_high_similarity_title_merges_as_duplicate(self) -> None:
        """Titles with token_sort_ratio >= FUZZY_TITLE_THRESHOLD are merged."""
        # Identical titles should always score 100.
        a = make_paper(title="Effect of Metformin on Cardiovascular Outcomes in Type 2 Diabetes")
        b = make_paper(title="Effect of Metformin on Cardiovascular Outcomes in Type 2 Diabetes")
        result = deduplicate([a, b])
        assert len(result) == 1

    def test_fuzzy_title_merge_sets_source_to_both(self) -> None:
        """Fuzzy-title-merged records from different sources get source == 'both'."""
        a = make_paper(
            title="Metformin cardiovascular outcomes diabetes type 2",
            source="pubmed",
        )
        b = make_paper(
            title="Metformin cardiovascular outcomes diabetes type 2",
            source="semantic_scholar",
        )
        result = deduplicate([a, b])
        assert result[0].source == "both"

    def test_clearly_different_titles_not_merged(self) -> None:
        """Titles with very low similarity are kept as separate records."""
        a = make_paper(title="Aspirin in Primary Prevention of Cardiovascular Disease")
        b = make_paper(title="CRISPR Gene Editing for Sickle Cell Disease Treatment")
        result = deduplicate([a, b])
        assert len(result) == 2


class TestAbstractFilter:
    """Tests for the empty-abstract filter applied after deduplication."""

    def test_paper_with_no_abstract_is_removed(self) -> None:
        """Papers with None or empty abstract are dropped from output."""
        a = make_paper(abstract="Has abstract.")
        b = make_paper(doi="10.1000/no-abs", abstract=None)
        b_copy = b.model_copy(update={"abstract": None})
        result = deduplicate([a, b_copy])
        assert len(result) == 1
        assert result[0].abstract == "Has abstract."

    def test_paper_with_empty_string_abstract_is_removed(self) -> None:
        """Papers with an empty string abstract are also dropped."""
        a = make_paper(abstract="Real content here.")
        b = make_paper(abstract="   ")
        result = deduplicate([a, b])
        assert len(result) == 1


class TestMerge:
    """Unit tests for the _merge helper."""

    def test_merge_combines_mesh_terms_from_primary(self) -> None:
        """Merged record inherits MeSH terms from the record with the longer abstract."""
        a = make_paper(
            abstract="Short.",
            mesh_terms=["Diabetes"],
            source="pubmed",
        )
        b = make_paper(
            abstract="Much longer abstract with more detail provided here.",
            mesh_terms=["Metformin"],
            source="semantic_scholar",
        )
        merged = _merge(a, b)
        # b has the longer abstract, so its mesh_terms should be used
        assert merged.mesh_terms == ["Metformin"]

    def test_merge_takes_max_citation_count(self) -> None:
        """Merged record takes the higher citation count of the two."""
        a = make_paper(source="pubmed")
        a_high = a.model_copy(update={"citation_count": 500})
        b = make_paper(source="semantic_scholar")
        b_low = b.model_copy(update={"citation_count": 50})
        merged = _merge(a_high, b_low)
        assert merged.citation_count == 500
