"""CI eval gate  -  fails if the most recent eval run missed any threshold."""

import json
from pathlib import Path

import pytest

EVAL_HISTORY_PATH = Path(__file__).parent.parent / "eval" / "eval_history.json"

THRESHOLDS = {
    "faithfulness": 0.75,
    "contradiction_precision": 0.65,
    "retrieval_precision": 0.55,
}


class TestEvalGate:
    """Gate: the most recent eval entry must meet all metric thresholds."""

    def test_eval_history_exists_and_is_not_empty(self) -> None:
        """eval_history.json must exist and contain at least one entry."""
        if not EVAL_HISTORY_PATH.exists():
            pytest.skip("eval_history.json not found  -  skipping gate on first run")
        history = json.loads(EVAL_HISTORY_PATH.read_text())
        assert len(history) > 0, "eval_history.json is empty  -  run eval/run_eval.py first"

    def test_faithfulness_meets_threshold(self) -> None:
        """Most recent faithfulness score must be >= 0.75."""
        if not EVAL_HISTORY_PATH.exists():
            pytest.skip("eval_history.json not found  -  skipping gate on first run")
        history = json.loads(EVAL_HISTORY_PATH.read_text())
        latest = history[-1]
        score = latest.get("faithfulness", 0.0)
        assert score >= THRESHOLDS["faithfulness"], (
            f"Faithfulness {score:.3f} is below threshold {THRESHOLDS['faithfulness']}"
        )

    def test_contradiction_precision_meets_threshold(self) -> None:
        """Most recent contradiction precision must be >= 0.65."""
        if not EVAL_HISTORY_PATH.exists():
            pytest.skip("eval_history.json not found  -  skipping gate on first run")
        history = json.loads(EVAL_HISTORY_PATH.read_text())
        latest = history[-1]
        score = latest.get("contradiction_precision", 0.0)
        assert score >= THRESHOLDS["contradiction_precision"], (
            f"Contradiction precision {score:.3f} is below threshold {THRESHOLDS['contradiction_precision']}"
        )

    def test_retrieval_precision_meets_threshold(self) -> None:
        """Most recent retrieval precision must be >= 0.55."""
        if not EVAL_HISTORY_PATH.exists():
            pytest.skip("eval_history.json not found  -  skipping gate on first run")
        history = json.loads(EVAL_HISTORY_PATH.read_text())
        latest = history[-1]
        score = latest.get("retrieval_precision", 0.0)
        assert score >= THRESHOLDS["retrieval_precision"], (
            f"Retrieval precision {score:.3f} is below threshold {THRESHOLDS['retrieval_precision']}"
        )
