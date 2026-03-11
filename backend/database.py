"""SQLAlchemy 2.x async engine, session factory, and database initialisation helpers."""

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings

logger = structlog.get_logger(__name__)

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.environment == "development",
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create tables, enable pgvector extension, set up FTS trigger, and run migrations.

    Called once at application startup. Safe to call multiple times — all
    statements use IF NOT EXISTS guards.
    """
    from backend import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension enabled")

        await conn.run_sync(Base.metadata.create_all)
        logger.info("database tables created or verified")

        # ── Forward-compatible column migrations ──────────────────────────────
        # ALTER TABLE IF NOT EXISTS guards for columns added after initial deploy.
        migration_statements = [
            "ALTER TABLE queries ADD COLUMN IF NOT EXISTS intent TEXT",
            "ALTER TABLE queries ADD COLUMN IF NOT EXISTS synthesis_generated BOOLEAN",
            "ALTER TABLE queries ADD COLUMN IF NOT EXISTS contradictions_found INTEGER",
            "ALTER TABLE queries ADD COLUMN IF NOT EXISTS latency_ms INTEGER",
        ]
        for stmt in migration_statements:
            try:
                await conn.execute(text(stmt))
            except Exception as exc:
                logger.warning("migration statement skipped", stmt=stmt[:60], error=str(exc))
        logger.info("schema migrations applied")

        # ── Full-text search trigger ──────────────────────────────────────────
        await conn.execute(
            text("""
            CREATE OR REPLACE FUNCTION papers_fts_update() RETURNS trigger AS $$
            BEGIN
                NEW.fts_vector := to_tsvector(
                    'english',
                    coalesce(NEW.title, '') || ' ' || coalesce(NEW.abstract, '')
                );
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """)
        )
        await conn.execute(
            text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'papers_fts_trigger'
                ) THEN
                    CREATE TRIGGER papers_fts_trigger
                    BEFORE INSERT OR UPDATE ON papers
                    FOR EACH ROW EXECUTE FUNCTION papers_fts_update();
                END IF;
            END
            $$;
            """)
        )
        logger.info("FTS trigger created or verified")

    logger.info("database initialisation complete")
