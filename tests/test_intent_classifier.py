"""Tests for the query intent classifier."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.intent_classifier import (
    IntentClassification,
    QueryIntent,
    classify_intent,
)
from backend.schemas import Synthesis, KeyFinding


def _mock_openai_response(content: str) -> MagicMock:
    """Build a minimal mock OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Intent classification tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_definitional_query_classified_correctly():
    """'What is metformin?' should be classified as DEFINITIONAL."""
    mock_response = _mock_openai_response(json.dumps({
        "intent": "definitional",
        "confidence": 0.95,
        "reasoning": "The query asks what metformin is, indicating a definitional intent.",
        "suggested_mesh_focus": ["Metformin", "Hypoglycemic Agents"],
        "recommended_date_filter": None,
    }))

    with patch("backend.services.intent_classifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classify_intent("What is metformin?")

    assert result.intent == QueryIntent.DEFINITIONAL
    assert result.confidence >= 0.9


@pytest.mark.asyncio
async def test_comparative_query_classified_correctly():
    """'Compare metformin vs SGLT2i for T2D' should be classified as COMPARATIVE."""
    mock_response = _mock_openai_response(json.dumps({
        "intent": "comparative",
        "confidence": 0.92,
        "reasoning": "The query directly compares two treatment options.",
        "suggested_mesh_focus": ["Metformin", "Sodium-Glucose Transporter 2 Inhibitors", "Diabetes Mellitus, Type 2"],
        "recommended_date_filter": "last 10 years",
    }))

    with patch("backend.services.intent_classifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classify_intent("Compare metformin vs SGLT2i for type 2 diabetes")

    assert result.intent == QueryIntent.COMPARATIVE
    assert result.recommended_date_filter is not None


@pytest.mark.asyncio
async def test_search_query_classified_correctly():
    """'Recent RCTs on GLP-1 agonists' should be classified as SEARCH."""
    mock_response = _mock_openai_response(json.dumps({
        "intent": "search",
        "confidence": 0.88,
        "reasoning": "The query asks for recent trials, indicating a literature search intent.",
        "suggested_mesh_focus": ["Glucagon-Like Peptide-1 Receptor Agonists", "Randomized Controlled Trial"],
        "recommended_date_filter": "last 3 years",
    }))

    with patch("backend.services.intent_classifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classify_intent("Recent RCTs on GLP-1 agonists for weight loss")

    assert result.intent == QueryIntent.SEARCH
    assert result.recommended_date_filter == "last 3 years"


@pytest.mark.asyncio
async def test_fallback_on_classification_failure():
    """When OpenAI returns invalid JSON, intent should default to SEARCH."""
    with patch("backend.services.intent_classifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API timeout")
        )

        result = await classify_intent("What is the mechanism of amyloid plaque formation?")

    assert result.intent == QueryIntent.SEARCH
    assert result.confidence == 0.5


@pytest.mark.asyncio
async def test_fallback_on_invalid_json():
    """When OpenAI returns non-JSON, intent should default to SEARCH."""
    mock_response = _mock_openai_response("This is not valid JSON at all")

    with patch("backend.services.intent_classifier.AsyncOpenAI") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classify_intent("What is the mechanism of statins?")

    assert result.intent == QueryIntent.SEARCH


# ── Structured output schema tests ───────────────────────────────────────────

def test_structured_output_validates_correctly():
    """SynthesisOutput (Synthesis schema) validates a valid dict without error."""
    valid_data = {
        "intent": "comparative",
        "consensus_statement": "Statins reduce LDL in primary prevention by 25-35%.",
        "key_findings": [
            {"finding": "Rosuvastatin reduced events by 44% in JUPITER.", "citations": [1], "confidence": "high"},
        ],
        "evidence_quality": "strong",
        "gaps": ["Long-term effects beyond 10 years are unclear."],
        "limitations": "Most trials exclude patients over 75.",
        "recommended_next_searches": ["statin safety elderly", "statin discontinuation outcomes"],
    }
    synthesis = Synthesis.model_validate(valid_data)
    assert synthesis.consensus_statement.startswith("Statins")
    assert synthesis.evidence_quality == "strong"
    assert len(synthesis.key_findings) == 1
    assert synthesis.key_findings[0].confidence == "high"
    assert len(synthesis.recommended_next_searches) == 2


def test_intent_classification_model_validates():
    """IntentClassification validates correctly from a dict."""
    data = IntentClassification(
        intent=QueryIntent.MECHANISTIC,
        confidence=0.87,
        reasoning="The query asks about a biological mechanism.",
        suggested_mesh_focus=["Amyloid beta-Peptides", "Alzheimer Disease"],
        recommended_date_filter=None,
    )
    assert data.intent == QueryIntent.MECHANISTIC
    assert data.confidence == 0.87
    assert "Alzheimer" in data.suggested_mesh_focus[1]
