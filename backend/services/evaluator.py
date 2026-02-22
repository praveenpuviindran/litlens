"""Faithfulness evaluation using Ragas and a custom LLM-based harness.

This module is used by the evaluation pipeline (eval/run_eval.py) and by CI
to gate merges when faithfulness drops below the threshold.
"""

import json
from typing import Optional

import structlog
from openai import AsyncOpenAI

from backend.config import settings
from backend.schemas import Paper, Synthesis

logger = structlog.get_logger(__name__)

FAITHFULNESS_THRESHOLD = 0.75
FAITHFULNESS_SYSTEM_PROMPT = """\
You are an expert evaluator of scientific evidence synthesis accuracy.

You will be given:
1. A set of source paper abstracts (numbered).
2. A synthesised summary with key findings and citations.

Your task: for each claim in the consensus_statement and each key finding, determine whether
it is directly supported by at least one of the provided abstracts.

Return a JSON object:
- "supported_claims": integer  -  number of claims directly supported by the abstracts.
- "total_claims": integer  -  total number of distinct claims evaluated.
- "faithfulness_score": float  -  supported_claims / total_claims (0.0 to 1.0).
- "unsupported_claims": array of strings  -  claims not found in the abstracts.

Return only the JSON object. No preamble.\
"""


async def evaluate_faithfulness(
    query: str,
    papers: list[Paper],
    synthesis: Synthesis,
) -> float:
    """Evaluate the faithfulness of a synthesis against its source abstracts.

    Uses gpt-4o-mini to check what fraction of claims in the synthesis are
    directly supported by the retrieved abstracts.

    Args:
        query: The original research question.
        papers: The top-10 papers that were used to generate the synthesis.
        synthesis: The generated evidence synthesis to evaluate.

    Returns:
        Faithfulness score in [0.0, 1.0]. Returns 0.0 on failure.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    abstracts_block = "\n\n".join(
        f"[{i+1}] {p.title}\n{p.abstract or 'No abstract.'}"
        for i, p in enumerate(papers)
    )
    synthesis_block = json.dumps(
        {
            "consensus_statement": synthesis.consensus_statement,
            "key_findings": [
                {"finding": f.finding, "citations": f.citations}
                for f in synthesis.key_findings
            ],
        },
        indent=2,
    )

    user_msg = (
        f"Research question: {query}\n\n"
        f"Source abstracts:\n{abstracts_block}\n\n"
        f"Synthesis to evaluate:\n{synthesis_block}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FAITHFULNESS_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        score = float(parsed.get("faithfulness_score", 0.0))
        logger.info("faithfulness evaluation complete", score=round(score, 3))
        return score

    except (json.JSONDecodeError, Exception) as exc:
        logger.error("faithfulness evaluation failed", error=str(exc))
        return 0.0
