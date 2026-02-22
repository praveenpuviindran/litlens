"""Papers router  -  GET /papers."""

import math

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend import models
from backend.database import get_db
from backend.schemas import PaperResponse, PapersListResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get(
    "",
    response_model=PapersListResponse,
    summary="Browse stored papers",
    description="Returns a paginated list of papers stored in the database, with optional filters.",
)
async def list_papers(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of results per page"),
    year_from: int | None = Query(None, ge=1900, le=2100, description="Earliest publication year"),
    year_to: int | None = Query(None, ge=1900, le=2100, description="Latest publication year"),
    mesh_term: str | None = Query(None, description="Filter by MeSH term (case-insensitive)"),
    source: str | None = Query(None, description="Filter by source: pubmed, semantic_scholar, both"),
    db: AsyncSession = Depends(get_db),
) -> PapersListResponse:
    """Return paginated papers with optional year, MeSH, and source filters."""
    stmt = select(models.Paper)

    if year_from is not None:
        stmt = stmt.where(models.Paper.publication_year >= year_from)
    if year_to is not None:
        stmt = stmt.where(models.Paper.publication_year <= year_to)
    if mesh_term is not None:
        # PostgreSQL array containment: check if mesh_terms contains the term
        stmt = stmt.where(models.Paper.mesh_terms.any(mesh_term))  # type: ignore[union-attr]
    if source is not None:
        stmt = stmt.where(models.Paper.source == source)

    # Count total matching rows
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Apply pagination
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size).order_by(models.Paper.created_at.desc())
    result = await db.execute(stmt)
    papers = result.scalars().all()

    pages = math.ceil(total / page_size) if total > 0 else 1

    paper_responses = [
        PaperResponse(
            id=p.id,
            pubmed_id=p.pubmed_id,
            s2_id=p.s2_id,
            doi=p.doi,
            title=p.title,
            abstract=p.abstract,
            authors=p.authors or [],
            journal=p.journal,
            publication_year=p.publication_year,
            mesh_terms=p.mesh_terms or [],
            keywords=p.keywords or [],
            citation_count=p.citation_count or 0,
            open_access_url=p.open_access_url,
            source=p.source or "unknown",
        )
        for p in papers
    ]

    return PapersListResponse(
        papers=paper_responses,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )
