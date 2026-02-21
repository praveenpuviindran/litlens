"""HTTP client wrapping FastAPI calls from the Streamlit frontend."""

import time
from typing import Any, Optional

import httpx
import structlog

from backend.config import settings

logger = structlog.get_logger(__name__)

# 60-second timeout — synthesis can take up to 10 seconds; leave headroom.
_TIMEOUT = httpx.Timeout(60.0)
_RETRY_DELAY = 2.0


def _base_url() -> str:
    """Return the backend base URL from environment configuration."""
    return settings.backend_url.rstrip("/")


def search(
    query: str,
    max_results: int = 10,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    mesh_filter: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Call POST /search and return the parsed JSON response.

    Retries once on connection error or 503 after a 2-second delay.

    Args:
        query: Natural language research question.
        max_results: Maximum number of papers to return.
        year_from: Optional publication year lower bound.
        year_to: Optional publication year upper bound.
        mesh_filter: Optional list of MeSH terms to filter by.

    Returns:
        Parsed JSON response from the backend.

    Raises:
        RuntimeError: If the backend returns an error or is unreachable.
    """
    payload: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "mesh_filter": mesh_filter or [],
    }
    if year_from is not None:
        payload["year_from"] = year_from
    if year_to is not None:
        payload["year_to"] = year_to

    url = f"{_base_url()}/search"
    last_error: Optional[Exception] = None

    for attempt in range(2):
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                response = client.post(url, json=payload)
                if response.status_code == 503 and attempt == 0:
                    logger.warning("backend returned 503 — retrying after delay")
                    time.sleep(_RETRY_DELAY)
                    continue
                response.raise_for_status()
                return response.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt == 0:
                logger.warning("connection error — retrying", error=str(exc))
                time.sleep(_RETRY_DELAY)
            else:
                break
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Backend error: {exc.response.status_code} — {exc.response.text}")

    raise RuntimeError(f"Backend unreachable: {last_error}")


def get_papers(
    page: int = 1,
    page_size: int = 20,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    mesh_term: Optional[str] = None,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """Call GET /papers and return the parsed JSON response.

    Args:
        page: Page number (1-indexed).
        page_size: Number of results per page.
        year_from: Optional publication year lower bound.
        year_to: Optional publication year upper bound.
        mesh_term: Optional MeSH term filter.
        source: Optional source filter (pubmed, semantic_scholar, both).

    Returns:
        Parsed JSON response from the backend.
    """
    params: dict[str, Any] = {"page": page, "page_size": page_size}
    if year_from:
        params["year_from"] = year_from
    if year_to:
        params["year_to"] = year_to
    if mesh_term:
        params["mesh_term"] = mesh_term
    if source:
        params["source"] = source

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            response = client.get(f"{_base_url()}/papers", params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Backend error: {exc.response.status_code}")


def check_health() -> dict[str, Any]:
    """Call GET /health and return the parsed response.

    Returns:
        Health status dict, or an error dict on failure.
    """
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            response = client.get(f"{_base_url()}/health")
            return response.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}
