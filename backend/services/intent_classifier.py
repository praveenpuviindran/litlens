"""Query intent classification using gpt-4o-mini with JSON structured output.

Classifies a natural language biomedical query into one of five intent types
so the pipeline can route it to the most appropriate fetching and synthesis strategy.
"""

import json
import logging
from enum import Enum
from typing import Literal, Optional

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.config import settings

logger = structlog.get_logger(__name__)


class QueryIntent(str, Enum):
    DEFINITIONAL    = "definitional"      # "What is X?" "How does X work?"
    COMPARATIVE     = "comparative"       # "Compare X vs Y" "Which is better: X or Y?"
    SEARCH          = "search"            # "Find papers about X" "Recent RCTs on X"
    MECHANISTIC     = "mechanistic"       # "Why does X cause Y?" "What is the mechanism?"
    EPIDEMIOLOGICAL = "epidemiological"   # "How prevalent is X?" "Risk factors for X?"


class IntentClassification(BaseModel):
    """Result of intent classification for a user query."""

    intent: QueryIntent
    confidence: float                       # 0–1
    reasoning: str                          # one sentence
    suggested_mesh_focus: list[str]         # 2–4 MeSH term suggestions
    recommended_date_filter: Optional[str] = None  # e.g., "last 5 years" for SEARCH


INTENT_PIPELINE_CONFIG: dict[QueryIntent, dict] = {
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

_CLASSIFICATION_SYSTEM_PROMPT = """\
You are a biomedical query classifier. Given a natural language research question,
classify its intent into exactly one of these categories:

- definitional:    "What is X?" or "How does X work?" — asks for an explanation or definition.
- comparative:     "Compare X vs Y" or "Which is better: X or Y?" — asks for a head-to-head comparison.
- search:          "Find papers about X" or "Recent RCTs on X" — asks for a literature list.
- mechanistic:     "Why does X cause Y?" or "What is the mechanism of X?" — asks about biological mechanism.
- epidemiological: "How prevalent is X?" or "Risk factors for X?" — asks about population-level patterns.

Return a JSON object with exactly these fields:
- "intent": one of definitional, comparative, search, mechanistic, epidemiological
- "confidence": float 0.0–1.0 indicating classification confidence
- "reasoning": one sentence explaining why you chose this intent
- "suggested_mesh_focus": array of 2–4 MeSH term strings most relevant to this query
- "recommended_date_filter": string like "last 5 years" or null if no date restriction applies

Return only the JSON object. No preamble.\
"""


async def classify_intent(query: str) -> IntentClassification:
    """Classify the intent of a biomedical research query.

    Calls gpt-4o-mini with JSON mode to classify the query. Falls back to
    SEARCH intent if the LLM call or JSON parse fails.

    Args:
        query: Natural language research question from the user.

    Returns:
        An IntentClassification with intent, confidence, and routing hints.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        intent_str = data.get("intent", "search").lower()
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.SEARCH

        return IntentClassification(
            intent=intent,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            suggested_mesh_focus=data.get("suggested_mesh_focus", []),
            recommended_date_filter=data.get("recommended_date_filter"),
        )

    except Exception as exc:
        logger.warning("intent classification failed — defaulting to SEARCH", error=str(exc))
        return IntentClassification(
            intent=QueryIntent.SEARCH,
            confidence=0.5,
            reasoning="Classification failed; defaulting to search intent.",
            suggested_mesh_focus=[],
            recommended_date_filter=None,
        )
