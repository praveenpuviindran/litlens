"""BioASQ test set loader.

Loads questions from eval/test_set.json. In production, this can be replaced
with a live BioASQ API loader once credentials are available.
"""

import json
from pathlib import Path
from typing import Any

TEST_SET_PATH = Path(__file__).parent / "test_set.json"


def load_test_set() -> list[dict[str, Any]]:
    """Load the evaluation test set from disk.

    Returns:
        List of question dicts with id, question, expected_answer_keywords, and category.
    """
    return json.loads(TEST_SET_PATH.read_text())


def load_by_category(category: str) -> list[dict[str, Any]]:
    """Load test questions filtered by category.

    Args:
        category: One of 'drug_efficacy', 'disease_mechanism', 'diagnostic_criteria',
                  'surgical_intervention', 'gene_disease'.

    Returns:
        Filtered list of question dicts.
    """
    return [q for q in load_test_set() if q.get("category") == category]
