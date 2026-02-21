"""Structured logging configuration using structlog."""

import logging
import structlog

from backend.config import settings


def configure_logging() -> None:
    """Configure structlog with JSON output in production, pretty-print in development."""
    log_level = getattr(logging, settings.log_level, logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        # JSON output for log aggregation services (Render, Datadog, etc.)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable coloured output for local development
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libraries are captured.
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )
