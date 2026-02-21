"""Standalone script to initialise the database.

Run this against the Docker Compose postgres service before starting the backend:

    python scripts/init_db.py

It is safe to run multiple times — all operations are idempotent.
"""

import asyncio
import sys

import structlog

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, ".")

from backend.database import init_db  # noqa: E402

logger = structlog.get_logger(__name__)


async def main() -> None:
    """Run database initialisation."""
    logger.info("starting database initialisation")
    try:
        await init_db()
        logger.info("database initialisation succeeded")
    except Exception as exc:
        logger.error("database initialisation failed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
