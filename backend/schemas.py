"""Pydantic request/response schemas for the LitLens API."""

import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Intent enum (shared between classifier and schemas) ───────────────────────

class QueryIntent(str, Enum):
    DEFINITIONAL    = "definitional"
    COMPARATIVE     = "comparative"
    SEARCH          = "search"
    MECHANISTIC     = "mechanistic"
    EPIDEMIOLOGICAL = "epidemiological"


# ── Core domain models ────────────────────────────────────────────────────────

class Paper(BaseModel):
    """Normalised representation of a biomedical paper from any source."""

    model_config = ConfigDict(str_strip_whitespace=True)

    pubmed_id: Optional[str] = None
    s2_id: Optional[str] = None
    doi: Optional[str] = None
    title: str
    abstract: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    mesh_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    citation_count: int = 0
    open_access_url: Optional[str] = None
    source: str  # 'pubmed', 'semantic_scholar', or 'both'


class PaperResponse(Paper):
    """Paper with database-assigned ID, returned from /papers endpoints."""

    id: uuid.UUID


class KeyFinding(BaseModel):
    """A single key finding from the evidence synthesis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    finding: str
    citations: list[int] = Field(default_factory=list)
    confidence: Optional[Literal["high", "medium", "low"]] = None


class Synthesis(BaseModel):
    """Structured evidence synthesis produced by the LLM."""

    model_config = ConfigDict(str_strip_whitespace=True)

    intent: Optional[str] = None
    consensus_statement: str
    key_findings: list[KeyFinding] = Field(default_factory=list)
    evidence_quality: str  # 'strong' | 'moderate' | 'weak' | 'mixed'
    gaps: list[str] = Field(default_factory=list)
    limitations: str
    recommended_next_searches: list[str] = Field(default_factory=list)


class ContradictionResponse(BaseModel):
    """A detected contradiction between two papers."""

    model_config = ConfigDict(str_strip_whitespace=True)

    paper_a_title: Optional[str] = None
    paper_b_title: Optional[str] = None
    claim_a: Optional[str] = None
    claim_b: Optional[str] = None
    intervention: Optional[str] = None
    outcome: Optional[str] = None
    methodological_note: Optional[str] = None
    confidence: float


# ── Request schemas ───────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=3, description="Natural language research question")
    max_results: int = Field(10, ge=1, le=25, description="Maximum papers to return")
    year_from: Optional[int] = Field(None, ge=1900, le=2100)
    year_to: Optional[int] = Field(None, ge=1900, le=2100)
    mesh_filter: list[str] = Field(default_factory=list)


class PapersQueryParams(BaseModel):
    """Query parameters for GET /papers."""

    model_config = ConfigDict(str_strip_whitespace=True)

    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    mesh_term: Optional[str] = None
    source: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query_id: uuid.UUID
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None


# ── Response schemas ──────────────────────────────────────────────────────────


class SearchResponse(BaseModel):
    """Full response from POST /search."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query_id: uuid.UUID
    raw_query: str
    expanded_pubmed_query: Optional[str] = None
    intent: Optional[str] = None
    papers: list[PaperResponse] = Field(default_factory=list)
    synthesis: Optional[Synthesis] = None
    contradictions: list[ContradictionResponse] = Field(default_factory=list)
    faithfulness_score: Optional[float] = None
    total_retrieved: int = 0
    latency_ms: int = 0
    cached: bool = False


class PapersListResponse(BaseModel):
    """Paginated list of papers from GET /papers."""

    model_config = ConfigDict(str_strip_whitespace=True)

    papers: list[PaperResponse]
    total: int
    page: int
    page_size: int
    pages: int


class HealthResponse(BaseModel):
    """Response from GET /health."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str
    database: str  # 'connected' | 'error'
    openai: str    # 'configured' | 'missing'
    version: str


class ErrorResponse(BaseModel):
    """Structured error response returned instead of 500s."""

    model_config = ConfigDict(str_strip_whitespace=True)

    detail: str
    error_type: Optional[str] = None


class AnalyticsSummaryResponse(BaseModel):
    """Response from GET /analytics/summary."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total_queries: int
    queries_last_7_days: int
    avg_latency_ms: float
    avg_papers_per_query: float
    contradiction_rate: float
    top_topics: list[dict[str, Any]] = Field(default_factory=list)
    queries_by_day: list[dict[str, Any]] = Field(default_factory=list)
    latency_by_intent: dict[str, Any] = Field(default_factory=dict)
    faithfulness_by_intent: dict[str, Any] = Field(default_factory=dict)


class QueryHistoryItem(BaseModel):
    """A single query record for the analytics history list."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: uuid.UUID
    raw_query: str
    intent: Optional[str] = None
    papers_retrieved: Optional[int] = None
    synthesis_generated: Optional[bool] = None
    contradictions_found: Optional[int] = None
    latency_ms: Optional[int] = None
    faithfulness: Optional[float] = None
    created_at: Optional[str] = None


class QueryHistoryResponse(BaseModel):
    """Paginated query history from GET /analytics/queries."""

    model_config = ConfigDict(str_strip_whitespace=True)

    queries: list[QueryHistoryItem]
    total: int
    page: int
    page_size: int
    pages: int
