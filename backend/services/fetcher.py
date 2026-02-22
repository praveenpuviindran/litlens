"""Async multi-source biomedical paper fetcher.

Fetches papers concurrently from PubMed (NCBI E-utilities) and Semantic Scholar,
normalises them into a shared ``Paper`` schema, and returns the combined list.
"""

import asyncio
from typing import Optional
from xml.etree import ElementTree as ET

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from backend.config import settings
from backend.schemas import Paper
from backend.utils.rate_limiter import AsyncRateLimiter

logger = structlog.get_logger(__name__)

# ── Rate limiters (module-level singletons) ───────────────────────────────────
# PubMed: 3 req/s without key, 10 req/s with key.
_pubmed_rate = AsyncRateLimiter(
    requests_per_second=10.0 if settings.ncbi_api_key else 3.0,
    source="pubmed",
)

# Semantic Scholar: 1 req/s without key, 10 req/s with key.
_s2_rate = AsyncRateLimiter(
    requests_per_second=10.0 if settings.s2_api_key else 1.0,
    source="semantic_scholar",
)

# ── PubMed ────────────────────────────────────────────────────────────────────
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def _pubmed_esearch(client: httpx.AsyncClient, query: str) -> list[str]:
    """Run ESearch and return a list of PubMed IDs matching *query*.

    Args:
        client: Shared HTTPX async client.
        query: PubMed MeSH search string.

    Returns:
        List of PubMed ID strings (may be empty).
    """
    params: dict = {
        "db": "pubmed",
        "term": query,
        "retmax": 25,
        "retmode": "json",
        "email": settings.ncbi_email,
    }
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key

    await _pubmed_rate.acquire()
    response = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
    response.raise_for_status()
    data = response.json()
    ids: list[str] = data.get("esearchresult", {}).get("idlist", [])
    logger.debug("pubmed esearch", query=query[:60], result_count=len(ids))
    return ids


def _parse_pubmed_xml(xml_text: str) -> list[Paper]:
    """Parse the EFetch XML response and return a list of normalised Papers.

    Args:
        xml_text: Raw XML string returned by NCBI EFetch.

    Returns:
        List of Paper objects.  Papers with no abstract are excluded.
    """
    papers: list[Paper] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("failed to parse PubMed XML", error=str(exc))
        return papers

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue

        art = medline.find("Article")
        if art is None:
            continue

        # PMID
        pmid_el = medline.find("PMID")
        pubmed_id = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None

        # Title
        title_el = art.find("ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue

        # Abstract
        abstract_parts = [
            (el.text or "") for el in art.findall(".//AbstractText")
        ]
        abstract = " ".join(p.strip() for p in abstract_parts if p.strip()) or None

        # Authors
        authors: list[str] = []
        for author in art.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            name = f"{last}, {fore}".strip(", ")
            if name:
                authors.append(name)

        # Journal
        journal_el = art.find(".//Journal/ISOAbbreviation")
        journal = journal_el.text.strip() if journal_el is not None and journal_el.text else None

        # Publication year
        year_el = art.find(".//PubDate/Year")
        pub_year: Optional[int] = None
        if year_el is not None and year_el.text:
            try:
                pub_year = int(year_el.text.strip())
            except ValueError:
                pass

        # MeSH terms
        mesh_terms: list[str] = [
            el.text.strip()
            for el in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if el.text
        ]

        # DOI
        doi: Optional[str] = None
        for id_el in art.findall(".//ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "doi" and id_el.text:
                doi = id_el.text.strip()
                break

        if abstract:
            papers.append(
                Paper(
                    pubmed_id=pubmed_id,
                    s2_id=None,
                    doi=doi,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal=journal,
                    publication_year=pub_year,
                    mesh_terms=mesh_terms,
                    keywords=[],
                    citation_count=0,
                    open_access_url=None,
                    source="pubmed",
                )
            )

    return papers


@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _pubmed_efetch(client: httpx.AsyncClient, ids: list[str]) -> list[Paper]:
    """Run EFetch for *ids* and return parsed Papers.

    Retries with exponential backoff + jitter on 429 responses.

    Args:
        client: Shared HTTPX async client.
        ids: List of PubMed IDs to fetch.

    Returns:
        List of Paper objects.
    """
    params: dict = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "rettype": "abstract",
        "email": settings.ncbi_email,
    }
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key

    await _pubmed_rate.acquire()
    response = await client.get(f"{PUBMED_BASE}/efetch.fcgi", params=params)

    if response.status_code == 429:
        logger.warning("pubmed rate limited  -  retrying with backoff")
        response.raise_for_status()  # triggers tenacity retry

    response.raise_for_status()
    return _parse_pubmed_xml(response.text)


async def fetch_pubmed(query: str) -> list[Paper]:
    """Fetch up to 25 papers from PubMed matching *query*.

    Args:
        query: PubMed MeSH search string produced by query expansion.

    Returns:
        List of normalised Paper objects.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            ids = await _pubmed_esearch(client, query)
            if not ids:
                logger.info("pubmed returned no IDs", query=query[:60])
                return []
            papers = await _pubmed_efetch(client, ids)
            logger.info("pubmed fetch complete", count=len(papers))
            return papers
        except Exception as exc:
            logger.error("pubmed fetch failed", error=str(exc))
            return []


# ── Semantic Scholar ──────────────────────────────────────────────────────────
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "paperId,title,abstract,authors,year,citationCount,"
    "externalIds,publicationDate,fieldsOfStudy,openAccessPdf"
)


@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _s2_search(client: httpx.AsyncClient, query: str) -> list[Paper]:
    """Query the Semantic Scholar paper search API.

    Args:
        client: Shared HTTPX async client.
        query: Plain-text search query (no MeSH syntax).

    Returns:
        List of normalised Paper objects.
    """
    headers: dict = {}
    if settings.s2_api_key:
        headers["x-api-key"] = settings.s2_api_key

    await _s2_rate.acquire()
    response = await client.get(
        S2_SEARCH_URL,
        params={"query": query, "limit": 25, "fields": S2_FIELDS},
        headers=headers,
    )

    if response.status_code == 429:
        logger.warning("semantic scholar rate limited  -  retrying with backoff")
        response.raise_for_status()

    response.raise_for_status()
    data = response.json()
    papers: list[Paper] = []

    for item in data.get("data", []):
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip() or None
        if not title or not abstract:
            continue

        authors = [
            a.get("name", "") for a in item.get("authors", []) if a.get("name")
        ]
        year_raw = item.get("year")
        pub_year: Optional[int] = int(year_raw) if year_raw else None
        external = item.get("externalIds") or {}
        doi = external.get("DOI")
        pubmed_id = str(external.get("PubMed")) if external.get("PubMed") else None
        oa = item.get("openAccessPdf") or {}
        oa_url = oa.get("url")

        papers.append(
            Paper(
                pubmed_id=pubmed_id,
                s2_id=item.get("paperId"),
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=None,  # S2 basic search does not return journal
                publication_year=pub_year,
                mesh_terms=[],
                keywords=item.get("fieldsOfStudy") or [],
                citation_count=item.get("citationCount") or 0,
                open_access_url=oa_url,
                source="semantic_scholar",
            )
        )

    logger.info("semantic scholar fetch complete", count=len(papers))
    return papers


async def fetch_semantic_scholar(query: str) -> list[Paper]:
    """Fetch up to 25 papers from Semantic Scholar matching *query*.

    Args:
        query: Plain-text search query for Semantic Scholar.

    Returns:
        List of normalised Paper objects.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            return await _s2_search(client, query)
        except Exception as exc:
            logger.error("semantic scholar fetch failed", error=str(exc))
            return []


# ── Combined fetch ────────────────────────────────────────────────────────────
async def fetch_all(pubmed_query: str, s2_query: str) -> list[Paper]:
    """Concurrently fetch papers from both PubMed and Semantic Scholar.

    Args:
        pubmed_query: MeSH-syntax query for PubMed.
        s2_query: Plain-text query for Semantic Scholar.

    Returns:
        Combined list of Paper objects from both sources.
    """
    pubmed_task = asyncio.create_task(fetch_pubmed(pubmed_query))
    s2_task = asyncio.create_task(fetch_semantic_scholar(s2_query))
    pubmed_papers, s2_papers = await asyncio.gather(pubmed_task, s2_task)
    combined = pubmed_papers + s2_papers
    logger.info(
        "fetch_all complete",
        pubmed=len(pubmed_papers),
        s2=len(s2_papers),
        total=len(combined),
    )
    return combined
