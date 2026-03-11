"""Ragas-based LLM evaluation harness for LitLens.

Evaluates the full RAG pipeline against a structured test set using Ragas metrics:
- Faithfulness (threshold 0.75): are synthesis claims supported by retrieved contexts?
- Answer Relevancy (threshold 0.70): does the answer address the question?
- Context Precision (threshold 0.65): are retrieved contexts relevant to the question?
- Retrieval Precision (threshold 0.60): keyword overlap proxy for retrieval quality.

Writes timestamped results to eval/eval_history.json and exits with code 1
if any threshold is missed (for CI/CD eval gate).

Usage:
    python eval/run_ragas_eval.py [--sample-size N]
"""

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = structlog.get_logger(__name__)

EVAL_DIR = Path(__file__).parent
TEST_SET_PATH = EVAL_DIR / "test_set.json"
HISTORY_PATH = EVAL_DIR / "eval_history.json"

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_precision": 0.65,
    "retrieval_precision": 0.60,
}

_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
_TIMEOUT = httpx.Timeout(90.0)


# ── Pipeline call ─────────────────────────────────────────────────────────────

def _call_search(question: str) -> dict[str, Any] | None:
    """Call POST /search and return the JSON result, or None on failure."""
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{_BACKEND_URL.rstrip('/')}/search",
                json={"query": question, "max_results": 10},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("search call failed", question=question[:50], error=str(exc))
        return None


# ── Ragas evaluation ──────────────────────────────────────────────────────────

async def run_ragas_eval(sample: list[dict[str, Any]]) -> dict[str, float]:
    """Run Ragas faithfulness, answer_relevancy, and context_precision evaluation.

    Args:
        sample: Subset of test_set items to evaluate.

    Returns:
        Dict with ragas metric names → float scores.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    rows: list[dict[str, Any]] = []
    for item in sample:
        result = _call_search(item["question"])
        if not result or not result.get("synthesis") or not result.get("papers"):
            logger.warning("skipping item — no result", id=item.get("id"))
            continue

        synthesis = result["synthesis"]
        papers = result["papers"]

        answer = synthesis.get("consensus_statement", "")
        contexts = [p.get("abstract", "") for p in papers if p.get("abstract")][:5]

        if not contexts or not answer:
            continue

        rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("ground_truth", ""),
        })

    if not rows:
        logger.warning("no valid rows for Ragas evaluation")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}

    dataset = Dataset.from_list(rows)

    try:
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        return {
            "faithfulness": float(scores["faithfulness"]),
            "answer_relevancy": float(scores["answer_relevancy"]),
            "context_precision": float(scores["context_precision"]),
        }
    except Exception as exc:
        logger.error("Ragas evaluate() failed", error=str(exc))
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}


# ── Retrieval precision ───────────────────────────────────────────────────────

def run_retrieval_precision(sample: list[dict[str, Any]]) -> float:
    """Keyword overlap proxy for retrieval quality.

    Args:
        sample: Subset of test_set items to evaluate.

    Returns:
        Mean fraction of expected_keywords appearing in retrieved papers.
    """
    scores: list[float] = []
    for item in sample:
        result = _call_search(item["question"])
        keywords = [kw.lower() for kw in item.get("expected_keywords", [])]
        if not keywords:
            continue

        papers = result.get("papers", []) if result else []
        all_context = " ".join(
            f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers[:5]
        ).lower()

        hits = sum(1 for kw in keywords if kw.lower() in all_context)
        scores.append(hits / len(keywords))

    return sum(scores) / len(scores) if scores else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_full_eval(sample_size: int = 10) -> int:
    """Run the full Ragas evaluation suite.

    Args:
        sample_size: Number of questions to sample from the test set.

    Returns:
        0 if all thresholds pass, 1 otherwise.
    """
    test_set: list[dict[str, Any]] = json.loads(TEST_SET_PATH.read_text())
    sample = random.sample(test_set, min(sample_size, len(test_set)))

    print("\n" + "=" * 62)
    print("  LitLens Ragas Evaluation Harness")
    print("=" * 62)
    print(f"  Questions sampled: {len(sample)} / {len(test_set)}")

    # ── Ragas metrics ─────────────────────────────────────────────────────────
    print("\n  Running Ragas evaluation (faithfulness, relevancy, precision)…")
    ragas_scores = await run_ragas_eval(sample)

    # ── Retrieval precision ───────────────────────────────────────────────────
    print("  Running retrieval keyword precision…")
    retrieval_precision = run_retrieval_precision(sample)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(sample),
        "faithfulness": ragas_scores["faithfulness"],
        "answer_relevancy": ragas_scores["answer_relevancy"],
        "context_precision": ragas_scores["context_precision"],
        "retrieval_precision": retrieval_precision,
        "thresholds": THRESHOLDS,
        "passed": True,
    }

    # ── Threshold checks ──────────────────────────────────────────────────────
    failures: list[str] = []
    for metric, threshold in THRESHOLDS.items():
        score = results[metric]
        if score < threshold:
            results["passed"] = False
            failures.append(f"{metric}: {score:.3f} < {threshold}")

    if failures:
        results["failures"] = failures

    # ── Persist history ───────────────────────────────────────────────────────
    history: list[dict] = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text())
        except Exception:
            history = []
    history.append(results)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "-" * 62)
    print(f"  Faithfulness:        {results['faithfulness']:.3f}  (threshold: {THRESHOLDS['faithfulness']})")
    print(f"  Answer Relevancy:    {results['answer_relevancy']:.3f}  (threshold: {THRESHOLDS['answer_relevancy']})")
    print(f"  Context Precision:   {results['context_precision']:.3f}  (threshold: {THRESHOLDS['context_precision']})")
    print(f"  Retrieval Precision: {results['retrieval_precision']:.3f}  (threshold: {THRESHOLDS['retrieval_precision']})")
    print("-" * 62)

    if results["passed"]:
        print("  RESULT: ALL THRESHOLDS PASSED ✓")
    else:
        print("  RESULT: EVAL GATE FAILED ✗")
        for f in failures:
            print(f"    — {f}")

    print("=" * 62 + "\n")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LitLens Ragas evaluation harness")
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Number of test questions to evaluate (default: 10)"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run_full_eval(sample_size=args.sample_size)))
