"""Analytics router — GET /analytics/summary, GET /analytics/queries, POST /feedback."""

import math
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend import models
from backend.database import get_db
from backend.schemas import (
    AnalyticsSummaryResponse,
    FeedbackRequest,
    QueryHistoryItem,
    QueryHistoryResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
async def analytics_summary(
    db: AsyncSession = Depends(get_db),
) -> AnalyticsSummaryResponse:
    """Return aggregated analytics metrics for the admin dashboard."""
    now = datetime.now(timezone.utc)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    # ── Total queries ─────────────────────────────────────────────────────────
    total_result = await db.execute(select(func.count()).select_from(models.Query))
    total_queries: int = total_result.scalar_one() or 0

    # ── Queries last 7 days ───────────────────────────────────────────────────
    recent_result = await db.execute(
        select(func.count()).select_from(models.Query).where(
            models.Query.created_at >= cutoff_7d
        )
    )
    queries_last_7_days: int = recent_result.scalar_one() or 0

    # ── Average latency ───────────────────────────────────────────────────────
    lat_result = await db.execute(
        select(func.avg(models.Query.latency_ms)).where(
            models.Query.latency_ms.isnot(None)
        )
    )
    avg_latency_ms: float = float(lat_result.scalar_one() or 0.0)

    # ── Average papers per query ──────────────────────────────────────────────
    papers_result = await db.execute(
        select(func.avg(models.Query.papers_retrieved)).where(
            models.Query.papers_retrieved.isnot(None)
        )
    )
    avg_papers_per_query: float = float(papers_result.scalar_one() or 0.0)

    # ── Contradiction rate ────────────────────────────────────────────────────
    contrad_result = await db.execute(
        select(
            func.count().filter(models.Query.contradictions_found > 0),
            func.count(),
        ).select_from(models.Query).where(
            models.Query.contradictions_found.isnot(None)
        )
    )
    row = contrad_result.one()
    contradiction_rate: float = float(row[0]) / float(row[1]) if row[1] else 0.0

    # ── Top topics (word frequency from raw_query) ────────────────────────────
    queries_result = await db.execute(
        select(models.Query.raw_query).order_by(models.Query.created_at.desc()).limit(500)
    )
    all_queries = [r[0] for r in queries_result.fetchall()]
    word_counter: Counter = Counter()
    stop = {"the", "and", "for", "with", "that", "this", "from", "are", "what", "does",
            "how", "which", "when", "who", "why", "its", "than", "have", "been", "not"}
    for q in all_queries:
        for word in q.lower().split():
            clean = word.strip("?.,:;()[]\"'")
            if len(clean) > 4 and clean not in stop:
                word_counter[clean] += 1
    top_topics = [{"word": w, "count": c} for w, c in word_counter.most_common(20)]

    # ── Queries by day (last 30 days) ─────────────────────────────────────────
    day_result = await db.execute(
        text("""
        SELECT date_trunc('day', created_at)::date AS day, COUNT(*) AS cnt
        FROM queries
        WHERE created_at >= :cutoff
        GROUP BY day
        ORDER BY day
        """),
        {"cutoff": cutoff_30d},
    )
    queries_by_day = [{"day": str(row[0]), "count": row[1]} for row in day_result.fetchall()]

    # ── Latency by intent ─────────────────────────────────────────────────────
    intent_lat_result = await db.execute(
        text("""
        SELECT intent,
               COUNT(*) AS n,
               AVG(latency_ms) AS avg_ms,
               PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
        FROM queries
        WHERE intent IS NOT NULL AND latency_ms IS NOT NULL
        GROUP BY intent
        """)
    )
    latency_by_intent: dict[str, Any] = {}
    for row in intent_lat_result.fetchall():
        latency_by_intent[row[0]] = {
            "n": row[1],
            "avg_ms": round(float(row[2] or 0), 1),
            "p50_ms": round(float(row[3] or 0), 1),
            "p95_ms": round(float(row[4] or 0), 1),
        }

    # ── Faithfulness by intent ────────────────────────────────────────────────
    faith_result = await db.execute(
        text("""
        SELECT intent,
               AVG(faithfulness) AS avg_faith,
               COUNT(*) FILTER (WHERE faithfulness >= 0.75) AS above_threshold,
               COUNT(*) AS total
        FROM queries
        WHERE intent IS NOT NULL AND faithfulness IS NOT NULL
        GROUP BY intent
        """)
    )
    faithfulness_by_intent: dict[str, Any] = {}
    for row in faith_result.fetchall():
        faithfulness_by_intent[row[0]] = {
            "avg_faithfulness": round(float(row[1] or 0), 3),
            "above_threshold": row[2],
            "total": row[3],
        }

    return AnalyticsSummaryResponse(
        total_queries=total_queries,
        queries_last_7_days=queries_last_7_days,
        avg_latency_ms=round(avg_latency_ms, 1),
        avg_papers_per_query=round(avg_papers_per_query, 1),
        contradiction_rate=round(contradiction_rate, 4),
        top_topics=top_topics,
        queries_by_day=queries_by_day,
        latency_by_intent=latency_by_intent,
        faithfulness_by_intent=faithfulness_by_intent,
    )


@router.get("/queries", response_model=QueryHistoryResponse)
async def query_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    intent_filter: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> QueryHistoryResponse:
    """Return paginated query history with optional filters."""
    stmt = select(models.Query)

    if intent_filter:
        stmt = stmt.where(models.Query.intent == intent_filter)
    if date_from:
        try:
            stmt = stmt.where(models.Query.created_at >= datetime.fromisoformat(date_from))
        except ValueError:
            pass
    if date_to:
        try:
            stmt = stmt.where(models.Query.created_at <= datetime.fromisoformat(date_to))
        except ValueError:
            pass

    # Count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one() or 0

    # Paginate
    offset = (page - 1) * page_size
    stmt = stmt.order_by(models.Query.created_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    items = [
        QueryHistoryItem(
            id=row.id,
            raw_query=row.raw_query,
            intent=row.intent,
            papers_retrieved=row.papers_retrieved,
            synthesis_generated=row.synthesis_generated,
            contradictions_found=row.contradictions_found,
            latency_ms=row.latency_ms,
            faithfulness=row.faithfulness,
            created_at=row.created_at.isoformat() if row.created_at else None,
        )
        for row in rows
    ]

    pages = math.ceil(total / page_size) if page_size else 1
    return QueryHistoryResponse(
        queries=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.post("/feedback", status_code=201)
async def submit_feedback(
    body: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Record user feedback (helpful / not helpful) for a search result."""
    # Verify query exists
    query_result = await db.execute(
        select(models.Query).where(models.Query.id == body.query_id)
    )
    query_row = query_result.scalar_one_or_none()
    if query_row is None:
        raise HTTPException(status_code=404, detail="Query not found.")

    feedback = models.QueryFeedback(
        query_id=body.query_id,
        rating=body.rating,
        feedback_text=body.feedback_text,
    )
    db.add(feedback)
    await db.flush()

    logger.info("feedback recorded", query_id=str(body.query_id), rating=body.rating)
    return {"status": "recorded"}
