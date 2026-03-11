"""FastAPI application entry point for LitLens."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.database import init_db
from backend.routers import health, papers, search
from backend.routers.analytics import router as analytics_router
from backend.services.reranker import warm_reranker
from backend.utils.logging import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan  -  runs startup and shutdown logic."""
    logger.info(
        "LitLens starting up",
        environment=settings.environment,
        faiss_fallback=settings.use_faiss_fallback,
    )

    try:
        await init_db()
    except Exception as exc:
        logger.error("database init failed  -  continuing without DB", error=str(exc))

    try:
        await warm_reranker()
    except Exception as exc:
        logger.warning("reranker warm-up failed  -  will retry on first request", error=str(exc))

    logger.info("LitLens startup complete")
    yield
    logger.info("LitLens shutting down")


app = FastAPI(
    title="LitLens API",
    description="Biomedical Literature Intelligence Engine  -  RAG-powered search and synthesis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

if settings.is_production:
    allowed_origins = [
        "https://litlens-frontend.onrender.com",
        settings.backend_url,
    ]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=exc,
    )
    return JSONResponse(
        status_code=503,
        content={"detail": "An unexpected error occurred. Please try again.", "error_type": type(exc).__name__},
    )


app.include_router(health.router)
app.include_router(search.router)
app.include_router(papers.router)
app.include_router(analytics_router)
