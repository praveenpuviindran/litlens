"""SQLAlchemy 2.x async engine, session factory, and database initialisation helpers."""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from backend.config import settings

logger = structlog.get_logger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping ensures stale connections are detected and recycled.
engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.environment == "development",
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ── Dependency ────────────────────────────────────────────────────────────────
async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Initialisation ────────────────────────────────────────────────────────────
async def init_db() -> None:
    """Create tables, enable pgvector extension, and set up FTS trigger.

    Called once at application startup. Safe to call multiple times  -  all
    statements use IF NOT EXISTS guards.
    """
    # Import models so SQLAlchemy registers them before create_all.
    from backend import models  # noqa: F401

    async with engine.begin() as conn:
        # Enable pgvector  -  required before any VECTOR column can be created.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension enabled")

        # Create all tables defined via the ORM.
        await conn.run_sync(Base.metadata.create_all)
        logger.info("database tables created or verified")

        # Full-text search trigger on papers(title, abstract).
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
