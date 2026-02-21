"""SQLAlchemy ORM models for LitLens."""

import uuid
from datetime import date, datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    DATE,
    FLOAT,
    INTEGER,
    TEXT,
    TIMESTAMPTZ,
    Boolean,
    ForeignKey,
    Index,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class Paper(Base):
    """Represents a single biomedical paper retrieved from PubMed or Semantic Scholar."""

    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pubmed_id: Mapped[Optional[str]] = mapped_column(TEXT, unique=True, nullable=True)
    s2_id: Mapped[Optional[str]] = mapped_column(TEXT, unique=True, nullable=True)
    doi: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    title: Mapped[str] = mapped_column(TEXT, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    authors: Mapped[Optional[list[str]]] = mapped_column(ARRAY(TEXT), nullable=True)
    journal: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    publication_year: Mapped[Optional[int]] = mapped_column(INTEGER, nullable=True)
    publication_date: Mapped[Optional[date]] = mapped_column(DATE, nullable=True)
    mesh_terms: Mapped[Optional[list[str]]] = mapped_column(ARRAY(TEXT), nullable=True)
    keywords: Mapped[Optional[list[str]]] = mapped_column(ARRAY(TEXT), nullable=True)
    citation_count: Mapped[int] = mapped_column(INTEGER, default=0, nullable=False)
    open_access_url: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    # 'pubmed', 'semantic_scholar', or 'both'
    source: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    # text-embedding-3-small produces 1536-dimensional vectors
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(1536), nullable=True)
    fts_vector: Mapped[Optional[str]] = mapped_column(TSVECTOR, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, server_default=text("NOW()"), nullable=False
    )

    # ── Indexes ───────────────────────────────────────────────────────────────
    __table_args__ = (
        # GIN index for fast full-text search
        Index("ix_papers_fts_vector", "fts_vector", postgresql_using="gin"),
        # GIN index for MeSH term array containment queries
        Index("ix_papers_mesh_terms", "mesh_terms", postgresql_using="gin"),
        # B-tree index for year range filters
        Index("ix_papers_publication_year", "publication_year"),
        # IVFFlat ANN index for vector similarity search (lists=100 is appropriate for
        # a corpus of ~10k–1M papers; adjust lists= as the table grows)
        Index(
            "ix_papers_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Query(Base):
    """Represents a user search query and its cached results."""

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    raw_query: Mapped[str] = mapped_column(TEXT, nullable=False)
    expanded_query: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    papers_retrieved: Mapped[Optional[int]] = mapped_column(INTEGER, nullable=True)
    synthesis: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    contradictions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    faithfulness: Mapped[Optional[float]] = mapped_column(FLOAT, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, server_default=text("NOW()"), nullable=False
    )

    contradiction_records: Mapped[list["Contradiction"]] = relationship(
        "Contradiction", back_populates="query", cascade="all, delete-orphan"
    )


class Contradiction(Base):
    """Stores a detected contradiction between two papers for a given query."""

    __tablename__ = "contradictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("queries.id"), nullable=False
    )
    paper_a_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False
    )
    paper_b_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False
    )
    claim_a: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    claim_b: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    intervention: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    methodological_note: Mapped[Optional[str]] = mapped_column(TEXT, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(FLOAT, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMPTZ, server_default=text("NOW()"), nullable=False
    )

    query: Mapped["Query"] = relationship("Query", back_populates="contradiction_records")

    __table_args__ = (
        Index("ix_contradictions_query_id", "query_id"),
        Index("ix_contradictions_paper_a_id", "paper_a_id"),
        Index("ix_contradictions_paper_b_id", "paper_b_id"),
    )
