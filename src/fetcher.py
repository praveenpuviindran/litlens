"""Synchronous biomedical paper fetcher for PubMed and Semantic Scholar.

No API keys required. PubMed is queried via NCBI E-utilities (free tier,
3 req/sec). Semantic Scholar is queried via their public search API (free,
no authentication needed for basic access).
"""

import time
from typing import Optional
from xml.etree import ElementTree as ET

import requests

# Generic contact email satisfies NCBI's "strongly recommended" policy
# without requiring the user to provide one.
NCBI_EMAIL = "research@litlens.app"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "paperId,title,abstract,authors,year,citationCount,"
    "externalIds,openAccessPdf,fieldsOfStudy"
)

TIMEOUT = 20  # seconds per request


class Paper:
    """Normalised representation of a single biomedical paper."""

    __slots__ = (
        "pubmed_id", "s2_id", "doi", "title", "abstract",
        "authors", "journal", "publication_year", "mesh_terms",
        "keywords", "citation_count", "open_access_url", "source",
    )

    def __init__(
        self,
        title: str,
        source: str,
        pubmed_id: Optional[str] = None,
        s2_id: Optional[str] = None,
        doi: Optional[str] = None,
        abstract: Optional[str] = None,
        authors: Optional[list[str]] = None,
        journal: Optional[str] = None,
        publication_year: Optional[int] = None,
        mesh_terms: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
        citation_count: int = 0,
        open_access_url: Optional[str] = None,
    ) -> None:
        self.title = title
        self.source = source
        self.pubmed_id = pubmed_id
        self.s2_id = s2_id
        self.doi = doi
        self.abstract = abstract
        self.authors = authors or []
        self.journal = journal
        self.publication_year = publication_year
        self.mesh_terms = mesh_terms or []
        self.keywords = keywords or []
        self.citation_count = citation_count
        self.open_access_url = open_access_url

    def to_dict(self) -> dict:
        """Return a plain dict representation."""
        return {k: getattr(self, k) for k in self.__slots__}


# ── PubMed ────────────────────────────────────────────────────────────────────

def _pubmed_esearch(query: str, max_results: int = 25) -> list[str]:
    """Return PubMed IDs matching query via ESearch."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": NCBI_EMAIL,
    }
    try:
        resp = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def _parse_pubmed_xml(xml_text: str) -> list[Paper]:
    """Parse EFetch XML and return Paper objects."""
    papers: list[Paper] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return papers

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        art = medline.find("Article")
        if art is None:
            continue

        pmid_el = medline.find("PMID")
        pubmed_id = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None

        title_el = art.find("ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue

        abstract_parts = [(el.text or "") for el in art.findall(".//AbstractText")]
        abstract = " ".join(p.strip() for p in abstract_parts if p.strip()) or None
        if not abstract:
            continue

        authors = []
        for author in art.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            name = f"{last}, {fore}".strip(", ")
            if name:
                authors.append(name)

        journal_el = art.find(".//Journal/ISOAbbreviation")
        journal = journal_el.text.strip() if journal_el is not None and journal_el.text else None

        year_el = art.find(".//PubDate/Year")
        pub_year = None
        if year_el is not None and year_el.text:
            try:
                pub_year = int(year_el.text.strip())
            except ValueError:
                pass

        mesh_terms = [
            el.text.strip()
            for el in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if el.text
        ]

        doi = None
        for id_el in art.findall(".//ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "doi" and id_el.text:
                doi = id_el.text.strip()
                break

        papers.append(Paper(
            title=title,
            source="pubmed",
            pubmed_id=pubmed_id,
            doi=doi,
            abstract=abstract,
            authors=authors,
            journal=journal,
            publication_year=pub_year,
            mesh_terms=mesh_terms,
        ))

    return papers


def fetch_pubmed(query: str, max_results: int = 25) -> list[Paper]:
    """Fetch papers from PubMed matching query.

    Args:
        query: Search string (plain text or MeSH syntax).
        max_results: Maximum number of papers to fetch.

    Returns:
        List of Paper objects with abstracts.
    """
    ids = _pubmed_esearch(query, max_results)
    if not ids:
        return []
    time.sleep(0.35)  # Respect 3 req/sec free-tier limit
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "rettype": "abstract",
        "email": NCBI_EMAIL,
    }
    try:
        resp = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return _parse_pubmed_xml(resp.text)
    except Exception:
        return []


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def fetch_semantic_scholar(query: str, max_results: int = 25) -> list[Paper]:
    """Fetch papers from Semantic Scholar matching query.

    No API key required  -  uses the free public search endpoint.

    Args:
        query: Plain-text search query.
        max_results: Maximum number of papers to fetch.

    Returns:
        List of Paper objects with abstracts.
    """
    try:
        resp = requests.get(
            S2_SEARCH_URL,
            params={"query": query, "limit": max_results, "fields": S2_FIELDS},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        papers = []
        for item in resp.json().get("data", []):
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip() or None
            if not title or not abstract:
                continue

            authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
            external = item.get("externalIds") or {}
            doi = external.get("DOI")
            pubmed_id = str(external.get("PubMed")) if external.get("PubMed") else None
            oa = item.get("openAccessPdf") or {}

            papers.append(Paper(
                title=title,
                source="semantic_scholar",
                s2_id=item.get("paperId"),
                doi=doi,
                pubmed_id=pubmed_id,
                abstract=abstract,
                authors=authors,
                publication_year=item.get("year"),
                keywords=item.get("fieldsOfStudy") or [],
                citation_count=item.get("citationCount") or 0,
                open_access_url=oa.get("url"),
            ))
        return papers
    except Exception:
        return []


def fetch_all(query: str, max_results: int = 25) -> list[Paper]:
    """Fetch from PubMed and Semantic Scholar, returning combined results.

    Args:
        query: User's natural language research question.
        max_results: Per-source maximum.

    Returns:
        Combined list from both sources.
    """
    pubmed = fetch_pubmed(query, max_results)
    s2 = fetch_semantic_scholar(query, max_results)
    return pubmed + s2
