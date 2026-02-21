"""Full evaluation pipeline for LitLens.

Computes:
- Faithfulness score across 50 test questions (subset sampled for efficiency)
- Contradiction detection precision, recall, F1 on 40 labeled pairs
- Retrieval precision (keyword overlap proxy)

Writes results to eval/eval_history.json and exits with code 1 if any
threshold is missed.

Usage:
    python eval/run_eval.py [--questions N] [--sample-size N]
"""

import asyncio
import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog

# Ensure project root is on path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings
from backend.schemas import Paper, Synthesis
from backend.services.generator import detect_contradictions
from backend.services.evaluator import evaluate_faithfulness

logger = structlog.get_logger(__name__)

EVAL_DIR = Path(__file__).parent
TEST_SET_PATH = EVAL_DIR / "test_set.json"
CONTRADICTION_PATH = EVAL_DIR / "contradiction_labels.json"
HISTORY_PATH = EVAL_DIR / "eval_history.json"

FAITHFULNESS_THRESHOLD = 0.75
CONTRADICTION_PRECISION_THRESHOLD = 0.65
RETRIEVAL_PRECISION_THRESHOLD = 0.55

BACKEND_URL = settings.backend_url.rstrip("/")
TIMEOUT = httpx.Timeout(90.0)


# ── Faithfulness evaluation ────────────────────────────────────────────────────

def _call_search(query: str) -> dict[str, Any] | None:
    """Call the /search endpoint synchronously and return the result."""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{BACKEND_URL}/search", json={"query": query, "max_results": 10})
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("search call failed", query=query[:50], error=str(exc))
        return None


async def _evaluate_single_faithfulness(item: dict[str, Any]) -> float:
    """Evaluate faithfulness for a single test question."""
    result = _call_search(item["question"])
    if not result or not result.get("synthesis") or not result.get("papers"):
        return 0.0

    synth_data = result["synthesis"]
    papers_data = result["papers"]

    from backend.schemas import KeyFinding
    synthesis = Synthesis(
        consensus_statement=synth_data.get("consensus_statement", ""),
        key_findings=[
            KeyFinding(finding=f.get("finding", ""), citations=f.get("citations", []))
            for f in synth_data.get("key_findings", [])
        ],
        evidence_quality=synth_data.get("evidence_quality", "mixed"),
        gaps=synth_data.get("gaps", []),
        limitations=synth_data.get("limitations", ""),
    )
    papers = [
        Paper(
            title=p.get("title", ""),
            abstract=p.get("abstract", ""),
            source=p.get("source", "pubmed"),
        )
        for p in papers_data
    ]

    return await evaluate_faithfulness(item["question"], papers, synthesis)


async def run_faithfulness_eval(test_set: list[dict], sample_size: int) -> float:
    """Run faithfulness evaluation on a sample of test questions.

    Args:
        test_set: Full list of test questions.
        sample_size: Number of questions to evaluate (random sample).

    Returns:
        Mean faithfulness score in [0.0, 1.0].
    """
    import random
    sample = random.sample(test_set, min(sample_size, len(test_set)))
    logger.info("running faithfulness eval", n=len(sample))

    scores = []
    for item in sample:
        score = await _evaluate_single_faithfulness(item)
        scores.append(score)
        logger.info("faithfulness", question=item["id"], score=round(score, 3))

    mean_score = sum(scores) / len(scores) if scores else 0.0
    logger.info("faithfulness eval complete", mean=round(mean_score, 3))
    return mean_score


# ── Retrieval precision evaluation ────────────────────────────────────────────

def run_retrieval_precision_eval(test_set: list[dict], sample_size: int) -> float:
    """Evaluate retrieval precision using keyword overlap as a proxy.

    For each query, checks what fraction of top-10 papers contain at least
    one expected keyword in the title or abstract.

    Args:
        test_set: Full test set list.
        sample_size: Number of questions to evaluate.

    Returns:
        Mean retrieval precision score.
    """
    import random
    sample = random.sample(test_set, min(sample_size, len(test_set)))
    logger.info("running retrieval precision eval", n=len(sample))

    precisions = []
    for item in sample:
        result = _call_search(item["question"])
        if not result or not result.get("papers"):
            precisions.append(0.0)
            continue

        keywords = [kw.lower() for kw in item.get("expected_answer_keywords", [])]
        papers = result["papers"]
        relevant = 0
        for p in papers[:10]:
            haystack = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            if any(kw in haystack for kw in keywords):
                relevant += 1

        precision = relevant / min(len(papers), 10) if papers else 0.0
        precisions.append(precision)
        logger.debug("retrieval precision", question=item["id"], precision=round(precision, 3))

    mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
    logger.info("retrieval precision eval complete", mean=round(mean_precision, 3))
    return mean_precision


# ── Contradiction detection evaluation ────────────────────────────────────────

async def run_contradiction_eval(labels: list[dict]) -> dict[str, float]:
    """Evaluate contradiction detection precision and recall.

    Runs pairwise classification directly (bypasses retrieval) on all 40
    labeled pairs and compares against ground truth.

    Args:
        labels: List of labeled paper pairs from contradiction_labels.json.

    Returns:
        Dict with precision, recall, and f1 keys.
    """
    logger.info("running contradiction detection eval", pairs=len(labels))

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for item in labels:
        paper_a = Paper(
            title=item["title_a"],
            abstract=item["abstract_a"],
            mesh_terms=[item.get("intervention", "")],
            source="pubmed",
        )
        paper_b = Paper(
            title=item["title_b"],
            abstract=item["abstract_b"],
            mesh_terms=[item.get("intervention", "")],
            source="semantic_scholar",
        )

        contradictions = await detect_contradictions([paper_a, paper_b])
        predicted = len(contradictions) > 0
        ground_truth = item["label"]

        if predicted and ground_truth:
            true_positives += 1
        elif predicted and not ground_truth:
            false_positives += 1
        elif not predicted and ground_truth:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info(
        "contradiction eval complete",
        tp=true_positives, fp=false_positives, fn=false_negatives,
        precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3),
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(question_sample: int = 10) -> None:
    """Run the full evaluation pipeline."""
    test_set = json.loads(TEST_SET_PATH.read_text())
    contradiction_labels = json.loads(CONTRADICTION_PATH.read_text())

    print("\n" + "="*60)
    print("  LitLens Evaluation Pipeline")
    print("="*60)

    # Faithfulness
    faithfulness = await run_faithfulness_eval(test_set, sample_size=question_sample)

    # Retrieval precision
    retrieval_precision = run_retrieval_precision_eval(test_set, sample_size=question_sample)

    # Contradiction detection
    contradiction_metrics = await run_contradiction_eval(contradiction_labels)

    print("\n" + "-"*60)
    print(f"  Faithfulness Score:          {faithfulness:.3f}  (threshold: {FAITHFULNESS_THRESHOLD})")
    print(f"  Contradiction Precision:     {contradiction_metrics['precision']:.3f}  (threshold: {CONTRADICTION_PRECISION_THRESHOLD})")
    print(f"  Contradiction Recall:        {contradiction_metrics['recall']:.3f}")
    print(f"  Contradiction F1:            {contradiction_metrics['f1']:.3f}")
    print(f"  Retrieval Precision:         {retrieval_precision:.3f}  (threshold: {RETRIEVAL_PRECISION_THRESHOLD})")
    print("-"*60)

    # Write history
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "faithfulness": faithfulness,
        "contradiction_precision": contradiction_metrics["precision"],
        "contradiction_recall": contradiction_metrics["recall"],
        "contradiction_f1": contradiction_metrics["f1"],
        "retrieval_precision": retrieval_precision,
        "queries_evaluated": question_sample,
    }

    history: list[dict] = []
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text())
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    logger.info("eval history written", path=str(HISTORY_PATH))

    # Check thresholds
    passed = True
    if faithfulness < FAITHFULNESS_THRESHOLD:
        print(f"  ❌ FAIL: Faithfulness {faithfulness:.3f} < {FAITHFULNESS_THRESHOLD}")
        passed = False
    else:
        print(f"  ✅ PASS: Faithfulness {faithfulness:.3f}")

    if contradiction_metrics["precision"] < CONTRADICTION_PRECISION_THRESHOLD:
        print(f"  ❌ FAIL: Contradiction Precision {contradiction_metrics['precision']:.3f} < {CONTRADICTION_PRECISION_THRESHOLD}")
        passed = False
    else:
        print(f"  ✅ PASS: Contradiction Precision {contradiction_metrics['precision']:.3f}")

    if retrieval_precision < RETRIEVAL_PRECISION_THRESHOLD:
        print(f"  ❌ FAIL: Retrieval Precision {retrieval_precision:.3f} < {RETRIEVAL_PRECISION_THRESHOLD}")
        passed = False
    else:
        print(f"  ✅ PASS: Retrieval Precision {retrieval_precision:.3f}")

    print("="*60 + "\n")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LitLens evaluation pipeline")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of test questions to evaluate (default: 10)")
    args = parser.parse_args()
    asyncio.run(main(question_sample=args.sample_size))
