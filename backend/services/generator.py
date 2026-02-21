"""LLM-powered evidence synthesis and contradiction detection.

Synthesis: summarises top-10 reranked paper abstracts into a structured
evidence report using gpt-4o-mini.

Contradiction detection: classifies each pair of papers that share at least
one MeSH term for opposing empirical claims.
"""

import asyncio
import json
from typing import Optional

import structlog
from openai import AsyncOpenAI

from backend.config import settings
from backend.schemas import (
    ContradictionResponse,
    KeyFinding,
    Paper,
    Synthesis,
)

logger = structlog.get_logger(__name__)

# Maximum concurrent pairwise contradiction calls to avoid OpenAI rate limits.
_CONTRADICTION_SEMAPHORE = asyncio.Semaphore(5)
# Only surface contradictions above this confidence threshold.
CONTRADICTION_CONFIDENCE_THRESHOLD = 0.70

SYNTHESIS_SYSTEM_PROMPT = """\
You are an expert biomedical research analyst. You will be given a research question and a set \
of relevant paper abstracts with citation numbers. Your task is to produce a structured evidence \
synthesis.

Your response MUST be a JSON object with these exact fields:
- "consensus_statement": A 2-4 sentence summary of what the evidence broadly shows. If evidence \
is mixed, say so explicitly. Do not overstate certainty.
- "key_findings": An array of objects, each with "finding" (string) and "citations" (array of \
integers referencing the paper numbers provided). Maximum 6 findings.
- "evidence_quality": One of "strong", "moderate", "weak", or "mixed". Base this on study design, \
sample size, and consistency across papers.
- "gaps": An array of 2-4 strings describing what the retrieved literature does NOT answer about \
the research question.
- "limitations": A string noting important caveats about this synthesis.

Rules:
- Every factual claim in key_findings must have at least one citation.
- If the abstracts do not contain enough information to answer the question, say so in \
consensus_statement and return empty key_findings.
- Do not invent findings not supported by the provided abstracts.
- Return only the JSON object. No preamble, no markdown formatting.\
"""

CONTRADICTION_SYSTEM_PROMPT = """\
You are a biomedical research methodologist. Given two paper abstracts about a similar topic, \
determine if they reach conflicting conclusions.

Return a JSON object with:
- "contradicts": boolean — true only if the two papers make directly opposing empirical claims \
about the same intervention and outcome in comparable populations.
- "claim_a": string — the relevant claim from Paper A (or null if no contradiction).
- "claim_b": string — the relevant claim from Paper B (or null if no contradiction).
- "intervention": string — the intervention or exposure being compared (or null).
- "outcome": string — the outcome measure where they disagree (or null).
- "methodological_note": string — one sentence on why results might differ (study design, \
population, dosage, follow-up period, etc.) — only if contradicts is true.
- "confidence": float between 0 and 1 — your confidence that this is a genuine contradiction.

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
)


def _build_abstract_context(papers: list[Paper]) -> str:
    """Format numbered abstract context for injection into the synthesis prompt.

    Args:
        papers: The top-10 reranked papers.

    Returns:
        Numbered string block of title + abstract pairs.
    """
    parts: list[str] = []
    for i, paper in enumerate(papers, start=1):
        parts.append(f"[{i}] {paper.title}\n{paper.abstract or 'No abstract available.'}")
    return "\n\n".join(parts)


async def synthesise(query: str, papers: list[Paper]) -> Synthesis:
    """Generate a structured evidence synthesis for the given query and papers.

    Args:
        query: The original user research question.
        papers: Top-10 reranked papers to synthesise.

    Returns:
        A Synthesis object. Returns a fallback on LLM or parse failure.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    context = _build_abstract_context(papers)
    user_message = f"Research question: {query}\n\nAbstracts:\n{context}"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)

        findings = [
            KeyFinding(
                finding=f.get("finding", ""),
                citations=f.get("citations", []),
            )
            for f in parsed.get("key_findings", [])
        ]

        return Synthesis(
            consensus_statement=parsed.get("consensus_statement", ""),
            key_findings=findings,
            evidence_quality=parsed.get("evidence_quality", "mixed"),
            gaps=parsed.get("gaps", []),
            limitations=parsed.get("limitations", ""),
        )

    except json.JSONDecodeError as exc:
        logger.warning("synthesis JSON parse failed", error=str(exc))
        return _FALLBACK_SYNTHESIS

    except Exception as exc:
        logger.error("synthesis LLM call failed", error=str(exc))
        return _FALLBACK_SYNTHESIS


def _papers_share_mesh(a: Paper, b: Paper) -> bool:
    """Return True if papers a and b share at least one MeSH term.

    Args:
        a: First paper.
        b: Second paper.

    Returns:
        True if any MeSH term appears in both papers.
    """
    set_a = set(t.lower() for t in (a.mesh_terms or []))
    set_b = set(t.lower() for t in (b.mesh_terms or []))
    return bool(set_a & set_b)


async def _classify_pair(
    client: AsyncOpenAI,
    paper_a: Paper,
    paper_b: Paper,
) -> Optional[ContradictionResponse]:
    """Classify a single paper pair for contradiction.

    Uses a semaphore to limit concurrent calls to 5.

    Args:
        client: Shared AsyncOpenAI client.
        paper_a: First paper.
        paper_b: Second paper.

    Returns:
        A ContradictionResponse if confidence >= threshold, else None.
    """
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
    """Detect contradictions across all pairs in the top-10 papers.

    Optimisation: only classifies pairs that share at least one MeSH term,
    then runs all eligible pairs concurrently (semaphore limits to 5 at once).

    Args:
        papers: Top-10 reranked papers.

    Returns:
        List of ContradictionResponse objects where confidence >= threshold.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    tasks = []

    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            a, b = papers[i], papers[j]
            # Skip pairs that share no MeSH terms — unlikely to have comparable claims.
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
