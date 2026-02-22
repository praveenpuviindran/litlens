"""Health check router  -  GET /health."""

import structlog
from fastapi import APIRouter
from sqlalchemy import text

from backend.config import settings
from backend.database import AsyncSessionLocal
from backend.schemas import HealthResponse

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns the operational status of the API, database, and OpenAI configuration. "
        "Always returns HTTP 200  -  status fields carry the health signal."
    ),
)
async def health_check() -> HealthResponse:
    """Check database connectivity and OpenAI key presence."""
    # Database check
    db_status = "error"
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        logger.warning("health check: database unreachable", error=str(exc))

    # OpenAI key check
    openai_status = "configured" if settings.openai_api_key else "missing"

    return HealthResponse(
        status="ok",
        database=db_status,
        openai=openai_status,
        version="1.0.0",
    )
