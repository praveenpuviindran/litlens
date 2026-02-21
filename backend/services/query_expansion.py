"""LLM-powered biomedical query expansion.

Converts a natural language research question into:
1. A structured PubMed MeSH search string.
2. A plain-text reformulation for Semantic Scholar.
"""

import json
from typing import Optional

import structlog
from openai import AsyncOpenAI

from backend.config import settings

logger = structlog.get_logger(__name__)

EXPANSION_SYSTEM_PROMPT = """\
You are a biomedical research librarian expert in PubMed search strategy.
Given a natural language research question, you will return a JSON object with two fields:
- "pubmed_query": A valid PubMed search string using MeSH terms with [MeSH Terms] field tags \
and Boolean operators (AND, OR, NOT). Keep it specific enough to return < 200 results but broad \
enough to return > 5. Include publication type filters where appropriate (e.g., \
("Clinical Trial"[Publication Type] OR "Randomized Controlled Trial"[Publication Type])). \
Do not use wildcards.
- "s2_query": A plain English reformulation of the same question, 10-20 words, suitable for \
keyword search. No MeSH syntax.
Return only the JSON object. No preamble, no explanation.\
"""


class ExpandedQuery:
    """Result of LLM-based query expansion."""

    def __init__(self, pubmed_query: str, s2_query: str) -> None:
        """Initialise with both query forms.

        Args:
            pubmed_query: MeSH-syntax PubMed query string.
            s2_query: Plain-English Semantic Scholar query.
        """
        self.pubmed_query = pubmed_query
        self.s2_query = s2_query


async def expand_query(raw_query: str) -> ExpandedQuery:
    """Expand a natural language query into PubMed MeSH and S2 forms.

    If the LLM response cannot be parsed as JSON, falls back to using the
    raw query string for both sources.

    Args:
        raw_query: Natural language research question from the user.

    Returns:
        An ExpandedQuery with pubmed_query and s2_query populated.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                {"role": "user", "content": raw_query},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        pubmed_query = parsed.get("pubmed_query", raw_query)
        s2_query = parsed.get("s2_query", raw_query)
        logger.info(
            "query expansion succeeded",
            raw=raw_query[:60],
            pubmed=pubmed_query[:80],
            s2=s2_query[:60],
        )
        return ExpandedQuery(pubmed_query=pubmed_query, s2_query=s2_query)

    except json.JSONDecodeError as exc:
        logger.warning(
            "query expansion JSON parse failed — using raw query as fallback",
            error=str(exc),
            raw_query=raw_query[:60],
        )
        return ExpandedQuery(pubmed_query=raw_query, s2_query=raw_query)

    except Exception as exc:
        logger.error(
            "query expansion LLM call failed — using raw query as fallback",
            error=str(exc),
            raw_query=raw_query[:60],
        )
        return ExpandedQuery(pubmed_query=raw_query, s2_query=raw_query)
