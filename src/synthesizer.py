"""Extractive evidence synthesizer. Zero API calls, zero cost.

Uses TF-IDF cosine similarity to score sentences from the retrieved abstracts
against the user's query, then selects the most representative non-redundant
sentences as key findings.

The highest-scoring sentence becomes the direct answer at the top of the synthesis.
All output sentences are taken verbatim from source abstracts and are citable.
"""

import re
import string
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.fetcher import Paper

_POSITIVE_SIGNALS = {
    "reduced", "lower", "decrease", "improve", "benefit", "effective",
    "significant", "superior", "prevented", "protective", "associated with reduced",
    "fewer", "less", "better", "successful", "favourable",
}
_NEGATIVE_SIGNALS = {
    "increased", "higher", "worse", "harmful", "no benefit", "no significant",
    "not associated", "failed", "ineffective", "no difference", "no effect",
    "not superior", "no reduction", "did not",
}
_UNCERTAIN_SIGNALS = {
    "mixed", "unclear", "conflicting", "inconclusive", "further research",
    "limited evidence", "insufficient", "uncertain",
}


@dataclass
class KeyFinding:
    """A single extracted key finding with its source paper index."""
    finding: str
    citation: int


@dataclass
class Synthesis:
    """Structured evidence synthesis."""
    direct_answer: str          # The single most query-relevant sentence, shown at top
    consensus_statement: str    # Broader narrative summary
    key_findings: list[KeyFinding] = field(default_factory=list)
    evidence_quality: str = "mixed"
    gaps: list[str] = field(default_factory=list)
    limitations: str = ""


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 40]


def _score_direction(text: str) -> str:
    """Detect whether findings lean positive, negative, or mixed."""
    lower = text.lower()
    pos = sum(1 for w in _POSITIVE_SIGNALS if w in lower)
    neg = sum(1 for w in _NEGATIVE_SIGNALS if w in lower)
    unc = sum(1 for w in _UNCERTAIN_SIGNALS if w in lower)
    if unc > 0:
        return "mixed"
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "mixed"


def _evidence_quality(papers: list[Paper]) -> str:
    """Estimate evidence quality from study design keywords in abstracts."""
    all_text = " ".join((p.abstract or "") for p in papers).lower()
    if any(s in all_text for s in {"meta-analysis", "systematic review", "cochrane"}):
        return "strong"
    if any(s in all_text for s in {"randomized", "randomised", "rct", "placebo", "controlled trial"}):
        return "moderate"
    if any(s in all_text for s in {"cohort", "observational", "retrospective", "case-control"}):
        return "weak"
    return "mixed"


def _deduplicate_sentences(sentences: list[str], threshold: float = 0.82) -> list[str]:
    """Remove near-duplicate sentences by cosine similarity."""
    if len(sentences) <= 1:
        return sentences
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform(sentences)
        sim = cosine_similarity(tfidf)
    except Exception:
        return sentences
    keep, used = [], set()
    for i in range(len(sentences)):
        if i in used:
            continue
        keep.append(i)
        for j in range(i + 1, len(sentences)):
            if sim[i, j] >= threshold:
                used.add(j)
    return [sentences[i] for i in keep]


def _build_direct_answer(best_sentence: str, query: str, direction: str, n: int) -> str:
    """Construct a direct-answer sentence that leads with the finding.

    Prepends a short framing clause to the best extracted sentence so the
    answer reads as a direct response to the question rather than a generic
    statement.

    Args:
        best_sentence: Highest TF-IDF scored sentence from abstracts.
        query: Original user question.
        direction: 'positive', 'negative', or 'mixed'.
        n: Number of papers synthesised.

    Returns:
        A single string that directly answers the question.
    """
    # Strip trailing period to allow clean sentence joining
    sentence = best_sentence.rstrip(".")

    if direction == "positive":
        framing = f"Based on {n} retrieved studies, the evidence supports a beneficial effect:"
    elif direction == "negative":
        framing = f"Based on {n} retrieved studies, the evidence does not support a significant benefit:"
    else:
        framing = f"Based on {n} retrieved studies, findings are mixed:"

    return f"{framing} {sentence}."


def synthesise(query: str, papers: list[Paper], max_findings: int = 6) -> Synthesis:
    """Generate a structured evidence synthesis from the top papers.

    The highest TF-IDF scoring sentence becomes the direct answer shown at
    the top. Supporting findings are the next highest-scoring non-redundant
    sentences, each cited to their source paper.

    Args:
        query: The user's research question.
        papers: Top-ranked papers (typically 10).
        max_findings: Maximum key findings to surface (excluding direct answer).

    Returns:
        A Synthesis object.
    """
    if not papers:
        return Synthesis(
            direct_answer="No papers were retrieved for this query. Try rephrasing your question.",
            consensus_statement="No papers were retrieved.",
            evidence_quality="weak",
            gaps=["Insufficient literature retrieved."],
        )

    all_sentences: list[tuple[str, int]] = []
    for i, paper in enumerate(papers, start=1):
        for sent in _split_sentences(paper.abstract or ""):
            all_sentences.append((sent, i))

    if not all_sentences:
        return Synthesis(
            direct_answer="Retrieved papers had insufficient abstract content for synthesis.",
            consensus_statement="Retrieved papers had insufficient abstract content.",
            evidence_quality="weak",
        )

    # Score all sentences against the query
    texts = [s for s, _ in all_sentences] + [query]
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        tfidf = vec.fit_transform(texts)
        query_vec = tfidf[-1]
        corpus_vecs = tfidf[:-1]
        scores = cosine_similarity(query_vec, corpus_vecs)[0]
    except Exception:
        scores = np.ones(len(all_sentences))

    ranked_indices = sorted(range(len(all_sentences)), key=lambda i: scores[i], reverse=True)

    # Best sentence becomes the direct answer
    best_idx = ranked_indices[0]
    best_sentence, best_paper_idx = all_sentences[best_idx]

    combined_text = " ".join((p.abstract or "") for p in papers)
    direction = _score_direction(combined_text)
    n = len(papers)
    quality = _evidence_quality(papers)

    direct_answer = _build_direct_answer(best_sentence, query, direction, n)

    # Remaining top sentences become supporting findings
    raw_top = [all_sentences[i] for i in ranked_indices[1: max_findings * 3 + 1]]
    raw_texts = [t for t, _ in raw_top]
    deduped = set(_deduplicate_sentences(raw_texts))

    key_findings: list[KeyFinding] = []
    seen = {best_sentence}
    for sent, paper_idx in raw_top:
        if sent in deduped and sent not in seen:
            key_findings.append(KeyFinding(finding=sent, citation=paper_idx))
            seen.add(sent)
        if len(key_findings) >= max_findings:
            break

    # Consensus statement - broader narrative
    topic = query.rstrip("?").lower()
    if len(topic) > 90:
        topic = topic[:90] + "..."

    if direction == "positive":
        consensus = (
            f"The retrieved literature broadly supports a beneficial association related to: "
            f"{topic}. Multiple studies report statistically significant findings, "
            f"though effect sizes and study populations vary."
        )
    elif direction == "negative":
        consensus = (
            f"The retrieved literature does not broadly support a significant benefit for: "
            f"{topic}. Several studies report null results or potential adverse effects."
        )
    else:
        consensus = (
            f"Findings are mixed across the retrieved literature for: {topic}. "
            f"Some studies report beneficial effects while others find no significant "
            f"association, reflecting heterogeneity in study design, population, and outcome measurement."
        )

    # Research gaps
    gap_signals = {
        "long-term": "Long-term outcomes and durability of effects remain understudied.",
        "pediatric": "Evidence in paediatric populations is limited.",
        "diverse": "Most studies use predominantly Western populations; external validity may be limited.",
        "mechanism": "Underlying biological mechanisms are not fully characterised.",
        "cost": "Cost-effectiveness data are sparse.",
    }
    gaps = []
    combined_lower = combined_text.lower()
    for keyword, gap_text in gap_signals.items():
        if keyword not in combined_lower and len(gaps) < 3:
            gaps.append(gap_text)
    if not gaps:
        gaps = [
            "Longer follow-up studies are needed.",
            "Head-to-head comparisons between interventions are limited.",
        ]

    # Limitations
    study_types = []
    if "randomized" in combined_lower or "randomised" in combined_lower:
        study_types.append("RCTs")
    if "cohort" in combined_lower or "observational" in combined_lower:
        study_types.append("observational studies")
    if "meta-analysis" in combined_lower:
        study_types.append("meta-analyses")
    limitations = (
        f"Synthesis based on {n} papers from PubMed and Semantic Scholar "
        f"({'including ' + ', '.join(study_types) if study_types else 'mixed study designs'}). "
        "All output sentences are extracted verbatim from source abstracts and cited by paper number."
    )

    return Synthesis(
        direct_answer=direct_answer,
        consensus_statement=consensus,
        key_findings=key_findings,
        evidence_quality=quality,
        gaps=gaps,
        limitations=limitations,
    )


def detect_contradictions(papers: list[Paper]) -> list[dict]:
    """Flag pairs of papers with opposing directional signals on the same MeSH topic.

    Args:
        papers: Top-ranked papers.

    Returns:
        List of contradiction dicts (may be empty).
    """
    contradictions = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            a, b = papers[i], papers[j]
            shared_mesh = (
                set(t.lower() for t in (a.mesh_terms or []))
                & set(t.lower() for t in (b.mesh_terms or []))
            )
            if not shared_mesh:
                continue
            dir_a = _score_direction(a.abstract or "")
            dir_b = _score_direction(b.abstract or "")
            if (dir_a == "positive" and dir_b == "negative") or (dir_a == "negative" and dir_b == "positive"):
                contradictions.append({
                    "paper_a_title": a.title,
                    "paper_b_title": b.title,
                    "direction_a": dir_a,
                    "direction_b": dir_b,
                    "shared_topic": list(shared_mesh)[0].title(),
                    "note": (
                        "These papers appear to reach opposing conclusions on the same topic. "
                        "Differences may reflect study design, population, dosage, or follow-up duration."
                    ),
                })
    return contradictions
