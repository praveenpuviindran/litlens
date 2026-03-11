"""LLM-powered evidence synthesis and contradiction detection.

Synthesis: summarises top-10 reranked paper abstracts into a structured
evidence report using gpt-4o-mini with OpenAI Structured Outputs.

Contradiction detection: classifies each pair of papers that share at least
one MeSH term for opposing empirical claims.
"""

import asyncio
import json
from typing import Literal, Optional

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.config import settings
from backend.schemas import (
    ContradictionResponse,
    KeyFinding,
    Paper,
    QueryIntent,
    Synthesis,
)

logger = structlog.get_logger(__name__)

_CONTRADICTION_SEMAPHORE = asyncio.Semaphore(5)
CONTRADICTION_CONFIDENCE_THRESHOLD = 0.70

# ── Structured output schema for synthesis ────────────────────────────────────


class _KeyFindingSchema(BaseModel):
    finding: str
    citations: list[int]
    confidence: Literal["high", "medium", "low"]


class _SynthesisOutputSchema(BaseModel):
    intent: str
    consensus_statement: str
    key_findings: list[_KeyFindingSchema]
    evidence_quality: Literal["strong", "moderate", "weak", "mixed"]
    gaps: list[str]
    limitations: str
    recommended_next_searches: list[str]


# ── Intent-based pipeline configuration ──────────────────────────────────────

INTENT_PIPELINE_CONFIG: dict[str, dict] = {
    QueryIntent.DEFINITIONAL: {
        "max_papers": 5,
        "synthesis_style": "explanatory",
        "require_review_articles": True,
        "date_range_years": None,
    },
    QueryIntent.COMPARATIVE: {
        "max_papers": 10,
        "synthesis_style": "comparative",
        "require_rct": True,
        "date_range_years": 10,
    },
    QueryIntent.SEARCH: {
        "max_papers": 15,
        "synthesis_style": "listing",
        "sort_by": "recency",
        "date_range_years": 3,
    },
    QueryIntent.MECHANISTIC: {
        "max_papers": 8,
        "synthesis_style": "mechanistic",
        "require_basic_science": True,
        "date_range_years": None,
    },
    QueryIntent.EPIDEMIOLOGICAL: {
        "max_papers": 10,
        "synthesis_style": "epidemiological",
        "prefer_cohort_studies": True,
        "date_range_years": 7,
    },
}

SYNTHESIS_SYSTEM_PROMPT = """\
You are an expert biomedical research analyst. You will be given a research question and a set \
of relevant paper abstracts with citation numbers. Your task is to produce a structured evidence \
synthesis.

Your response MUST be a JSON object with these exact fields:
- "intent": one of definitional, comparative, search, mechanistic, epidemiological — classify the query intent.
- "consensus_statement": A 2-4 sentence summary of what the evidence broadly shows. If evidence \
is mixed, say so explicitly. Do not overstate certainty.
- "key_findings": An array of objects, each with:
    "finding" (string), "citations" (array of integers), "confidence" ("high", "medium", or "low").
  Maximum 6 findings.
- "evidence_quality": One of "strong", "moderate", "weak", or "mixed".
- "gaps": An array of 2-4 strings describing what the retrieved literature does NOT answer.
- "limitations": A string noting important caveats.
- "recommended_next_searches": An array of 2-3 follow-up query strings the researcher should try.

Rules:
- Every factual claim in key_findings must have at least one citation.
- If the abstracts do not contain enough information, say so in consensus_statement.
- Do not invent findings not supported by the provided abstracts.
- Return only the JSON object. No preamble, no markdown.\
"""

CONTRADICTION_SYSTEM_PROMPT = """\
You are a biomedical research methodologist. Given two paper abstracts about a similar topic, \
determine if they reach conflicting conclusions.

Return a JSON object with:
- "contradicts": boolean
- "claim_a": string or null
- "claim_b": string or null
- "intervention": string or null
- "outcome": string or null
- "methodological_note": string (only if contradicts is true)
- "confidence": float between 0 and 1

Return only the JSON object.\
"""

_FALLBACK_SYNTHESIS = Synthesis(
    consensus_statement=(
        "Synthesis could not be generated. Please review the source papers directly."
    ),
    key_findings=[],
    evidence_quality="weak",
    gaps=[],
    limitations="Automated synthesis failed.",
    recommended_next_searches=[],
)


def _build_abstract_context(papers: list[Paper]) -> str:
    parts: list[str] = []
    for i, paper in enumerate(papers, start=1):
        parts.append(f"[{i}] {paper.title}\n{paper.abstract or 'No abstract available.'}")
    return "\n\n".join(parts)


async def synthesise(query: str, papers: list[Paper], intent: Optional[str] = None) -> Synthesis:
    """Generate a structured evidence synthesis using OpenAI Structured Outputs.

    Args:
        query: The original user research question.
        papers: Top-10 reranked papers to synthesise.
        intent: Optional pre-classified query intent string.

    Returns:
        A Synthesis object. Returns a fallback on failure.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    context = _build_abstract_context(papers)
    intent_hint = f"\nQuery intent (use this as the 'intent' field): {intent}" if intent else ""
    user_message = f"Research question: {query}{intent_hint}\n\nAbstracts:\n{context}"

    # ── Try OpenAI Structured Outputs (beta.chat.completions.parse) ───────────
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=_SynthesisOutputSchema,
            temperature=0.1,
            max_tokens=1800,
        )
        parsed: _SynthesisOutputSchema = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Structured output returned None")

        findings = [
            KeyFinding(
                finding=f.finding,
                citations=f.citations,
                confidence=f.confidence,
            )
            for f in (parsed.key_findings or [])
        ]

        return Synthesis(
            intent=parsed.intent or intent,
            consensus_statement=parsed.consensus_statement,
            key_findings=findings,
            evidence_quality=parsed.evidence_quality,
            gaps=parsed.gaps or [],
            limitations=parsed.limitations,
            recommended_next_searches=parsed.recommended_next_searches or [],
        )

    except Exception as exc:
        logger.warning("structured output synthesis failed — falling back to JSON mode", error=str(exc))

    # ── Fallback: standard JSON mode ──────────────────────────────────────────
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1800,
        )
        raw = response.choices[0].message.content or ""
        parsed_dict = json.loads(raw)

        findings = [
            KeyFinding(
                finding=f.get("finding", ""),
                citations=f.get("citations", []),
                confidence=f.get("confidence"),
            )
            for f in parsed_dict.get("key_findings", [])
        ]

        return Synthesis(
            intent=parsed_dict.get("intent") or intent,
            consensus_statement=parsed_dict.get("consensus_statement", ""),
            key_findings=findings,
            evidence_quality=parsed_dict.get("evidence_quality", "mixed"),
            gaps=parsed_dict.get("gaps", []),
            limitations=parsed_dict.get("limitations", ""),
            recommended_next_searches=parsed_dict.get("recommended_next_searches", []),
        )

    except json.JSONDecodeError as exc:
        logger.warning("synthesis JSON parse failed", error=str(exc))
        return _FALLBACK_SYNTHESIS

    except Exception as exc:
        logger.error("synthesis LLM call failed", error=str(exc))
        return _FALLBACK_SYNTHESIS


def _papers_share_mesh(a: Paper, b: Paper) -> bool:
    set_a = set(t.lower() for t in (a.mesh_terms or []))
    set_b = set(t.lower() for t in (b.mesh_terms or []))
    return bool(set_a & set_b)


async def _classify_pair(
    client: AsyncOpenAI,
    paper_a: Paper,
    paper_b: Paper,
) -> Optional[ContradictionResponse]:
    user_msg = (
        f"Paper A: {paper_a.title}\n{paper_a.abstract or ''}\n\n"
        f"Paper B: {paper_b.title}\n{paper_b.abstract or ''}"
    )

    async with _CONTRADICTION_SEMAPHORE:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            raw = response.choices[0].message.content or ""
            result = json.loads(raw)
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("contradiction pair classification failed", error=str(exc))
            return None

    if not result.get("contradicts"):
        return None

    confidence = float(result.get("confidence", 0.0))
    if confidence < CONTRADICTION_CONFIDENCE_THRESHOLD:
        return None

    return ContradictionResponse(
        paper_a_title=paper_a.title,
        paper_b_title=paper_b.title,
        claim_a=result.get("claim_a"),
        claim_b=result.get("claim_b"),
        intervention=result.get("intervention"),
        outcome=result.get("outcome"),
        methodological_note=result.get("methodological_note"),
        confidence=confidence,
    )


async def detect_contradictions(papers: list[Paper]) -> list[ContradictionResponse]:
    """Detect contradictions across all pairs in the top-10 papers."""
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    tasks = []

    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            a, b = papers[i], papers[j]
            if not _papers_share_mesh(a, b):
                continue
            tasks.append(_classify_pair(client, a, b))

    logger.info("running contradiction detection", eligible_pairs=len(tasks))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    contradictions: list[ContradictionResponse] = [
        r for r in results if isinstance(r, ContradictionResponse)
    ]
    logger.info("contradiction detection complete", found=len(contradictions))
    return contradictions
