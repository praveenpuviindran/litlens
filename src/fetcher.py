"""Synchronous biomedical paper fetcher for PubMed and Semantic Scholar.

No API keys required. PubMed is queried via NCBI E-utilities (free tier,
3 req/sec). Semantic Scholar is queried via their public search API (free,
no authentication needed for basic access).

Query strategy:
  PubMed   - receives a stripped, term-based query with common abbreviations
             expanded (KO -> knockout, KD -> knockdown, etc.) so MeSH terms
             are matched correctly. Falls back to a shorter 2-term query if
             the full query yields nothing.
  S2       - receives the original natural language query (Semantic Scholar
             handles NL well). Falls back to the stripped query if NL returns
             nothing.
"""

import time
from typing import Optional
from xml.etree import ElementTree as ET

import requests

# Words that add no retrieval value when sent to PubMed
_SEARCH_STOPWORDS = {
    "what", "how", "why", "does", "do", "is", "are", "can", "should",
    "will", "would", "which", "when", "where", "who", "the", "of",
    "on", "in", "with", "for", "a", "an", "to", "and", "or", "not",
    "effect", "effects", "impact", "role", "association", "relationship",
    "between", "among", "there", "any", "some", "their",
}

# Common biomedical abbreviations that PubMed may not index well as-is.
# These are expanded before building the PubMed search term.
_ABBREV_EXPANSIONS: dict[str, str] = {
    "KO": "knockout",
    "KD": "knockdown",
    "OE": "overexpression",
    "WT": "wild-type",
    "OA": "osteoarthritis",
    "MI": "myocardial infarction",
    "CVD": "cardiovascular disease",
    "HF": "heart failure",
    "DM": "diabetes mellitus",
    "HTN": "hypertension",
    "CKD": "chronic kidney disease",
    "IBD": "inflammatory bowel disease",
    "MS": "multiple sclerosis",
    "RA": "rheumatoid arthritis",
    "SLE": "systemic lupus erythematosus",
    "NASH": "nonalcoholic steatohepatitis",
    "NAFLD": "nonalcoholic fatty liver disease",
    "CAD": "coronary artery disease",
    "AF": "atrial fibrillation",
    "PE": "pulmonary embolism",
    "DVT": "deep vein thrombosis",
    "AKI": "acute kidney injury",
    "ARDS": "acute respiratory distress syndrome",
    "ICU": "intensive care unit",
    "SNP": "single nucleotide polymorphism",
    "GWAS": "genome-wide association study",
    "ChIP": "chromatin immunoprecipitation",
    "PCR": "polymerase chain reaction",
    "qPCR": "quantitative PCR",
    "CRISPR": "CRISPR Cas9",
    "siRNA": "small interfering RNA",
    "shRNA": "short hairpin RNA",
    "mRNA": "messenger RNA",
    "lncRNA": "long noncoding RNA",
}


def _expand_abbreviations(query: str) -> str:
    """Expand common biomedical abbreviations for better PubMed MeSH matching.

    Only expands tokens that exactly match the abbreviation (case-sensitive)
    so that gene/protein names that happen to share abbreviations are not
    wrongly expanded.
    """
    tokens = query.split()
    expanded = []
    for tok in tokens:
        # Strip trailing punctuation for lookup, then re-attach
        clean = tok.rstrip(".,?!")
        suffix = tok[len(clean):]
        if clean in _ABBREV_EXPANSIONS:
            expanded.append(_ABBREV_EXPANSIONS[clean] + suffix)
        else:
            expanded.append(tok)
    return " ".join(expanded)


def _build_pubmed_query(query: str) -> str:
    """Build an optimized term-based query for PubMed ESearch.

    Expands abbreviations, then strips question/stop words while preserving
    uppercase abbreviations (T, B, GADS, TNF, IL-6, etc.).
    """
    expanded = _expand_abbreviations(query)
    original_words = expanded.rstrip("?").split()
    key_terms = []
    for orig in original_words:
        lower = orig.lower()
        if lower in _SEARCH_STOPWORDS:
            continue
        # Preserve uppercase abbreviations (single-char like T, B and multi-char like GADS)
        if orig.isupper() or (len(orig) >= 2 and orig[0].isupper() and any(c.isdigit() for c in orig)):
            key_terms.append(orig)
        elif len(lower) > 2:
            key_terms.append(lower)
    return " ".join(key_terms) if key_terms else expanded.rstrip("?").strip()


def _build_s2_query(query: str) -> str:
    """Build query for Semantic Scholar.

    S2 handles natural language queries well, so pass the original query
    (with abbreviations expanded for better recall) rather than stripping it.
    """
    return _expand_abbreviations(query).rstrip("?").strip()


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

    Uses abbreviation-expanded, stopword-stripped terms for ESearch.
    If the full term set returns nothing, retries with a shorter 2-term
    fallback to maximise recall on niche queries.

    Args:
        query: Search string (plain text or natural language).
        max_results: Maximum number of papers to fetch.

    Returns:
        List of Paper objects with abstracts.
    """
    search_q = _build_pubmed_query(query)
    ids = _pubmed_esearch(search_q, max_results)

    # Fallback 1: drop to first 3 terms if the full query found nothing
    if not ids:
        terms = search_q.split()
        if len(terms) > 3:
            ids = _pubmed_esearch(" ".join(terms[:3]), max_results)

    # Fallback 2: drop to first 2 terms
    if not ids:
        terms = search_q.split()
        if len(terms) > 2:
            ids = _pubmed_esearch(" ".join(terms[:2]), max_results)

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

def _fetch_s2_with_query(search_q: str, max_results: int) -> list[Paper]:
    """Internal helper: run one S2 query and return Paper objects."""
    try:
        resp = requests.get(
            S2_SEARCH_URL,
            params={"query": search_q, "limit": max_results, "fields": S2_FIELDS},
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


def fetch_semantic_scholar(query: str, max_results: int = 25) -> list[Paper]:
    """Fetch papers from Semantic Scholar matching query.

    Tries the natural language query first (S2 handles it well), then
    falls back to a stripped term-based query if NL returns nothing.

    Args:
        query: Plain-text or natural language search query.
        max_results: Maximum number of papers to fetch.

    Returns:
        List of Paper objects with abstracts.
    """
    # Primary: natural language with abbreviations expanded
    papers = _fetch_s2_with_query(_build_s2_query(query), max_results)
    if papers:
        return papers

    # Fallback: stripped term query
    return _fetch_s2_with_query(_build_pubmed_query(query), max_results)


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
