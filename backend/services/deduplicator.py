"""Two-pass deduplication for merged PubMed + Semantic Scholar results.

Pass 1: Exact DOI match.
Pass 2: Fuzzy title match using rapidfuzz token_sort_ratio (threshold ≥ 85).

Merged records inherit the richer of the two versions, and ``source`` is set
to ``'both'`` when papers from different sources are merged.
"""

import re
import string
from typing import Optional

import structlog
from rapidfuzz import fuzz

from backend.schemas import Paper

logger = structlog.get_logger(__name__)

# Threshold for fuzzy title matching (0–100 scale).
FUZZY_TITLE_THRESHOLD = 85


def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for title comparison.

    Args:
        title: Raw title string.

    Returns:
        Normalised title suitable for fuzzy comparison.
    """
    title = title.lower()
    title = title.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", title).strip()


def _merge(a: Paper, b: Paper) -> Paper:
    """Merge two Paper records, preferring the richer of the two.

    Richness is determined by abstract length, then author count,
    then presence of MeSH terms.

    Args:
        a: First Paper record.
        b: Second Paper record.

    Returns:
        Single merged Paper with ``source`` set to ``'both'`` if the sources differ.
    """
    # Choose the base record with the longer abstract.
    abstract_a = a.abstract or ""
    abstract_b = b.abstract or ""
    if len(abstract_b) > len(abstract_a):
        primary, secondary = b, a
    else:
        primary, secondary = a, b

    # Inherit fields from secondary when primary is missing them.
    merged_doi = primary.doi or secondary.doi
    merged_pubmed_id = primary.pubmed_id or secondary.pubmed_id
    merged_s2_id = primary.s2_id or secondary.s2_id
    merged_authors = primary.authors if primary.authors else secondary.authors
    merged_journal = primary.journal or secondary.journal
    merged_year = primary.publication_year or secondary.publication_year
    merged_mesh = primary.mesh_terms if primary.mesh_terms else secondary.mesh_terms
    merged_keywords = list(set(primary.keywords + secondary.keywords))
    merged_citations = max(primary.citation_count, secondary.citation_count)
    merged_oa = primary.open_access_url or secondary.open_access_url

    # Mark source as 'both' if the two records came from different sources.
    if a.source != b.source or "both" in (a.source, b.source):
        source = "both"
    else:
        source = primary.source

    return Paper(
        pubmed_id=merged_pubmed_id,
        s2_id=merged_s2_id,
        doi=merged_doi,
        title=primary.title,
        abstract=primary.abstract,
        authors=merged_authors,
        journal=merged_journal,
        publication_year=merged_year,
        mesh_terms=merged_mesh,
        keywords=merged_keywords,
        citation_count=merged_citations,
        open_access_url=merged_oa,
        source=source,
    )


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Deduplicate a list of Papers in two passes.

    Pass 1 uses exact DOI match; Pass 2 uses rapidfuzz fuzzy title similarity.
    Papers with an empty or absent abstract are removed from the final output.

    Args:
        papers: Combined raw list from all fetch sources.

    Returns:
        Deduplicated list of Papers with non-empty abstracts.
    """
    # ── Pass 1: DOI exact match ───────────────────────────────────────────────
    doi_map: dict[str, Paper] = {}  # doi -> canonical paper
    no_doi: list[Paper] = []

    for paper in papers:
        if paper.doi:
            if paper.doi in doi_map:
                doi_map[paper.doi] = _merge(doi_map[paper.doi], paper)
                logger.debug("doi duplicate merged", doi=paper.doi)
            else:
                doi_map[paper.doi] = paper
        else:
            no_doi.append(paper)

    after_pass1 = list(doi_map.values()) + no_doi
    logger.info(
        "deduplication pass 1 complete",
        before=len(papers),
        after=len(after_pass1),
    )

    # ── Pass 2: Fuzzy title match ─────────────────────────────────────────────
    unique: list[Paper] = []
    for candidate in after_pass1:
        norm_candidate = _normalise_title(candidate.title)
        merged_into: Optional[int] = None

        for idx, existing in enumerate(unique):
            score = fuzz.token_sort_ratio(norm_candidate, _normalise_title(existing.title))
            if score >= FUZZY_TITLE_THRESHOLD:
                unique[idx] = _merge(existing, candidate)
                merged_into = idx
                logger.debug(
                    "fuzzy title duplicate merged",
                    title_a=existing.title[:50],
                    title_b=candidate.title[:50],
                    score=score,
                )
                break

        if merged_into is None:
            unique.append(candidate)

    logger.info(
        "deduplication pass 2 complete",
        before=len(after_pass1),
        after=len(unique),
    )

    # ── Filter: remove papers with no abstract ────────────────────────────────
    result = [p for p in unique if p.abstract and p.abstract.strip()]
    dropped = len(unique) - len(result)
    if dropped:
        logger.info("dropped papers with no abstract", count=dropped)

    return result
