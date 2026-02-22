"""Two-pass deduplication for merged PubMed + Semantic Scholar results.

Pass 1: Exact DOI match.
Pass 2: Fuzzy title match (rapidfuzz token_sort_ratio ≥ 85).
"""

import re
import string
from typing import Optional

from rapidfuzz import fuzz

from src.fetcher import Paper

FUZZY_THRESHOLD = 85


def _normalise(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    title = title.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", title).strip()


def _merge(a: Paper, b: Paper) -> Paper:
    """Merge two Papers, keeping the richer abstract and combining metadata."""
    primary, secondary = (b, a) if len(b.abstract or "") > len(a.abstract or "") else (a, b)

    source = "both" if a.source != b.source or "both" in (a.source, b.source) else primary.source
    merged = Paper(
        title=primary.title,
        source=source,
        pubmed_id=primary.pubmed_id or secondary.pubmed_id,
        s2_id=primary.s2_id or secondary.s2_id,
        doi=primary.doi or secondary.doi,
        abstract=primary.abstract,
        authors=primary.authors if primary.authors else secondary.authors,
        journal=primary.journal or secondary.journal,
        publication_year=primary.publication_year or secondary.publication_year,
        mesh_terms=primary.mesh_terms if primary.mesh_terms else secondary.mesh_terms,
        keywords=list(set((primary.keywords or []) + (secondary.keywords or []))),
        citation_count=max(primary.citation_count, secondary.citation_count),
        open_access_url=primary.open_access_url or secondary.open_access_url,
    )
    return merged


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers in two passes, then drop those with no abstract.

    Args:
        papers: Combined list from all fetch sources.

    Returns:
        Deduplicated list of Papers, all with non-empty abstracts.
    """
    # Pass 1  -  DOI exact match
    doi_map: dict[str, Paper] = {}
    no_doi: list[Paper] = []
    for p in papers:
        if p.doi:
            if p.doi in doi_map:
                doi_map[p.doi] = _merge(doi_map[p.doi], p)
            else:
                doi_map[p.doi] = p
        else:
            no_doi.append(p)

    after_pass1 = list(doi_map.values()) + no_doi

    # Pass 2  -  fuzzy title match
    unique: list[Paper] = []
    for candidate in after_pass1:
        norm_c = _normalise(candidate.title)
        merged_idx: Optional[int] = None
        for idx, existing in enumerate(unique):
            if fuzz.token_sort_ratio(norm_c, _normalise(existing.title)) >= FUZZY_THRESHOLD:
                unique[idx] = _merge(existing, candidate)
                merged_idx = idx
                break
        if merged_idx is None:
            unique.append(candidate)

    # Drop papers with no usable abstract
    return [p for p in unique if p.abstract and p.abstract.strip()]
