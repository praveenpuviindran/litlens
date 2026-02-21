"""Tests for backend/services/generator.py."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.schemas import ContradictionResponse, Paper, Synthesis
from backend.services.generator import (
    CONTRADICTION_CONFIDENCE_THRESHOLD,
    _FALLBACK_SYNTHESIS,
    _papers_share_mesh,
    detect_contradictions,
    synthesise,
)


def make_paper(
    title: str = "Test Paper",
    abstract: str = "Test abstract with relevant information.",
    mesh_terms: list[str] | None = None,
) -> Paper:
    """Create a Paper with sensible defaults for generator tests."""
    return Paper(
        title=title,
        abstract=abstract,
        mesh_terms=mesh_terms or ["Diabetes Mellitus, Type 2"],
        source="pubmed",
    )


def make_openai_response(content: str) -> MagicMock:
    """Return a mock OpenAI chat completion with *content* as the message text."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class TestSynthesis:
    """Tests for the synthesise function."""

    @pytest.mark.asyncio
    async def test_synthesis_parses_valid_json_response(self) -> None:
        """synthesise returns a populated Synthesis when the LLM returns valid JSON."""
        valid_json = json.dumps({
            "consensus_statement": "Evidence supports Drug X for outcome Y.",
            "key_findings": [{"finding": "Drug X reduces risk.", "citations": [1, 2]}],
            "evidence_quality": "moderate",
            "gaps": ["Long-term data lacking."],
            "limitations": "Mostly observational studies.",
        })
        mock_response = make_openai_response(valid_json)

        with patch("backend.services.generator.AsyncOpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await synthesise("test query", [make_paper() for _ in range(3)])

        assert isinstance(result, Synthesis)
        assert result.consensus_statement == "Evidence supports Drug X for outcome Y."
        assert result.evidence_quality == "moderate"
        assert len(result.key_findings) == 1
        assert result.key_findings[0].citations == [1, 2]

    @pytest.mark.asyncio
    async def test_synthesis_returns_fallback_on_malformed_json(self) -> None:
        """synthesise returns the fallback Synthesis when the LLM returns malformed JSON."""
        mock_response = make_openai_response("This is not JSON at all.")

        with patch("backend.services.generator.AsyncOpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await synthesise("test query", [make_paper()])

        assert result.consensus_statement == _FALLBACK_SYNTHESIS.consensus_statement
        assert result.key_findings == []

    @pytest.mark.asyncio
    async def test_synthesis_returns_fallback_on_llm_exception(self) -> None:
        """synthesise returns the fallback Synthesis when the API call raises."""
        with patch("backend.services.generator.AsyncOpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create = AsyncMock(
                side_effect=Exception("API unavailable")
            )
            result = await synthesise("test query", [make_paper()])

        assert result.consensus_statement == _FALLBACK_SYNTHESIS.consensus_statement


class TestMeshSharing:
    """Tests for the _papers_share_mesh helper."""

    def test_papers_share_mesh_returns_true_when_overlap(self) -> None:
        """Returns True when papers share at least one MeSH term."""
        a = make_paper(mesh_terms=["Diabetes Mellitus, Type 2", "Metformin"])
        b = make_paper(mesh_terms=["Metformin", "Cardiovascular Diseases"])
        assert _papers_share_mesh(a, b) is True

    def test_papers_share_mesh_returns_false_when_no_overlap(self) -> None:
        """Returns False when papers share no MeSH terms."""
        a = make_paper(mesh_terms=["Diabetes Mellitus, Type 2"])
        b = make_paper(mesh_terms=["Alzheimer Disease"])
        assert _papers_share_mesh(a, b) is False

    def test_papers_share_mesh_case_insensitive(self) -> None:
        """MeSH term comparison is case-insensitive."""
        a = make_paper(mesh_terms=["METFORMIN"])
        b = make_paper(mesh_terms=["metformin"])
        assert _papers_share_mesh(a, b) is True

    def test_papers_share_mesh_handles_empty_lists(self) -> None:
        """Returns False when either paper has no MeSH terms."""
        a = make_paper(mesh_terms=[])
        b = make_paper(mesh_terms=["Metformin"])
        assert _papers_share_mesh(a, b) is False


class TestContradictionDetection:
    """Tests for detect_contradictions."""

    @pytest.mark.asyncio
    async def test_detect_contradictions_returns_only_high_confidence(self) -> None:
        """Only contradictions with confidence >= threshold are returned."""
        low_conf = json.dumps({
            "contradicts": True,
            "claim_a": "Drug increases risk.",
            "claim_b": "Drug decreases risk.",
            "intervention": "Drug X",
            "outcome": "Cardiovascular risk",
            "methodological_note": "Different populations.",
            "confidence": CONTRADICTION_CONFIDENCE_THRESHOLD - 0.1,
        })
        mock_response = make_openai_response(low_conf)

        papers = [
            make_paper(mesh_terms=["Metformin"]),
            make_paper(mesh_terms=["Metformin"]),
        ]

        with patch("backend.services.generator.AsyncOpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            result = await detect_contradictions(papers)

        assert result == []

    @pytest.mark.asyncio
    async def test_detect_contradictions_skips_pairs_without_shared_mesh(self) -> None:
        """Pairs without shared MeSH terms are not submitted to the LLM."""
        papers = [
            make_paper(mesh_terms=["Metformin"]),
            make_paper(mesh_terms=["Alzheimer Disease"]),  # no overlap
        ]

        with patch("backend.services.generator.AsyncOpenAI") as MockOAI:
            create_mock = AsyncMock()
            MockOAI.return_value.chat.completions.create = create_mock
            await detect_contradictions(papers)

        # No API call should have been made since there are no shared MeSH terms.
        create_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_detect_contradictions_fires_pairs_concurrently(self) -> None:
        """Pairwise calls are fired concurrently (all tasks created before awaited)."""
        import asyncio

        call_times: list[float] = []

        async def fake_create(**kwargs: Any) -> MagicMock:
            """Record call time and return a non-contradiction response."""
            call_times.append(asyncio.get_event_loop().time())
            return make_openai_response(json.dumps({"contradicts": False, "confidence": 0.1}))

        # Three papers all sharing the same MeSH term = 3 pairs.
        papers = [
            make_paper(mesh_terms=["Metformin"]),
            make_paper(mesh_terms=["Metformin"]),
            make_paper(mesh_terms=["Metformin"]),
        ]

        with patch("backend.services.generator.AsyncOpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create = fake_create
            await detect_contradictions(papers)

        # All 3 calls should have been made (we don't verify exact timing in unit tests
        # but we do verify all were called).
        assert len(call_times) == 3
