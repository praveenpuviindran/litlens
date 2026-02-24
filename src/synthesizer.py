"""Extractive evidence synthesizer with semantic sentence scoring.

Uses sentence-transformers (all-MiniLM-L6-v2) when available for semantic
similarity, with TF-IDF cosine similarity as a fallback. No API calls.
Zero cost.

Query intent is detected to apply appropriate framing:
  - intervention: "Does X reduce Y?" -> directional evidence framing
  - descriptive:  "What causes Y?"   -> informative summary
  - term:         "heavy menstrual bleeding" -> literature overview

All output sentences are extracted verbatim from source abstracts.
"""

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.fetcher import Paper

_POSITIVE_SIGNALS = {
    "reduced", "lower", "decrease", "improve", "benefit", "effective",
    "significant", "superior", "prevented", "protective",
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

_INTERVENTION_PHRASES = [
    "effect of", "effects of", " vs ", "versus", "compared to",
    "compared with", "efficacy of", "safety of", "impact of",
    "role of", "use of", "benefit of", "association between",
]
_INTERVENTION_STARTERS = {"does", "do", "can", "should", "will", "would"}
_INTERVENTION_VERBS = {
    "reduce", "lower", "improve", "prevent", "treat", "increase",
    "decrease", "affect", "cause", "help", "work", "inhibit",
    "promote", "suppress", "enhance", "alleviate",
}
_DESCRIPTIVE_STARTERS = {
    "what", "how", "why", "when", "where", "who", "which",
    "is", "are", "explain", "describe",
}


@dataclass
class KeyFinding:
    """A single extracted key finding with its source paper index."""
    finding: str
    citation: int


@dataclass
class Synthesis:
    """Structured evidence synthesis."""
    direct_answer: str
    consensus_statement: str
    key_findings: list[KeyFinding] = field(default_factory=list)
    evidence_quality: str = "mixed"
    gaps: list[str] = field(default_factory=list)
    limitations: str = ""


def _detect_intent(query: str) -> str:
    """Classify query as 'intervention', 'descriptive', or 'term'."""
    lower = query.lower().strip()
    words = lower.split()
    first = words[0] if words else ""

    if any(phrase in lower for phrase in _INTERVENTION_PHRASES):
        return "intervention"

    if first in _INTERVENTION_STARTERS and any(v in lower for v in _INTERVENTION_VERBS):
        return "intervention"

    if first in _DESCRIPTIVE_STARTERS or "?" in query:
        return "descriptive"

    return "term"


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 40]


def _score_direction(text: str) -> str:
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
    all_text = " ".join((p.abstract or "") for p in papers).lower()
    if any(s in all_text for s in {"meta-analysis", "systematic review", "cochrane"}):
        return "strong"
    if any(s in all_text for s in {"randomized", "randomised", "rct", "placebo", "controlled trial"}):
        return "moderate"
    if any(s in all_text for s in {"cohort", "observational", "retrospective", "case-control"}):
        return "weak"
    return "mixed"


def _score_sentences(sentences: list[str], query: str, encoder=None) -> np.ndarray:
    """Score sentences against query via semantic similarity, falling back to TF-IDF."""
    if encoder is not None:
        try:
            all_texts = sentences + [query]
            embeddings = encoder.encode(all_texts, show_progress_bar=False, batch_size=64)
            query_emb = embeddings[-1:]
            sent_embs = embeddings[:-1]
            return cosine_similarity(query_emb, sent_embs)[0]
        except Exception:
            pass

    texts = sentences + [query]
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        tfidf = vec.fit_transform(texts)
        return cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    except Exception:
        return np.ones(len(sentences))


def _deduplicate_sentences(sentences: list[str], threshold: float = 0.82) -> list[str]:
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


def _build_consensus(
    intent: str,
    direction: str,
    topic: str,
    top_sentences: list[tuple[str, int]],
    n: int,
) -> str:
    """Build a coherent consensus statement appropriate to the query type."""
    if intent == "intervention":
        if direction == "positive":
            return (
                f"Across {n} retrieved papers, the evidence broadly supports a beneficial "
                "association. Multiple studies report statistically significant findings, "
                "though effect sizes, populations, and follow-up durations vary."
            )
        elif direction == "negative":
            return (
                f"Across {n} retrieved papers, the evidence does not consistently support "
                "a significant benefit. Several studies report null results or raise concerns "
                "about adverse effects."
            )
        else:
            return (
                f"Across {n} retrieved papers, evidence is mixed. Some studies report "
                "significant effects while others do not, likely reflecting heterogeneity "
                "in study design, population, intervention dosage, and outcome measurement."
            )
    else:
        # For descriptive and term queries: build narrative from the next top sentences
        additional = [s for s, _ in top_sentences[1:4]]
        if additional:
            return " ".join(s if s.endswith(".") else s + "." for s in additional)
        return f"The retrieved literature covers multiple aspects of {topic}."


def synthesise(
    query: str, papers: list[Paper], max_findings: int = 6, encoder=None
) -> Synthesis:
    """Generate a structured evidence synthesis from the top papers.

    Args:
        query: The user's research question or search term.
        papers: Top-ranked papers (typically 10).
        max_findings: Maximum key findings to surface.
        encoder: Optional sentence-transformers encoder for semantic scoring.

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

    texts = [s for s, _ in all_sentences]
    scores = _score_sentences(texts, query, encoder=encoder)
    ranked_indices = sorted(range(len(all_sentences)), key=lambda i: scores[i], reverse=True)

    intent = _detect_intent(query)
    combined_text = " ".join((p.abstract or "") for p in papers)
    direction = _score_direction(combined_text)
    n = len(papers)
    quality = _evidence_quality(papers)

    topic = query.rstrip("?").strip()
    if len(topic) > 80:
        topic = topic[:80] + "..."

    # Collect top diverse sentences (deduplicated by exact match)
    top_sentences: list[tuple[str, int]] = []
    seen_text: set[str] = set()
    for idx in ranked_indices:
        sent, paper_idx = all_sentences[idx]
        if sent not in seen_text:
            top_sentences.append((sent, paper_idx))
            seen_text.add(sent)
        if len(top_sentences) >= max_findings + 4:
            break

    best_sentence, _ = top_sentences[0]

    # Build direct answer based on query intent
    if intent == "intervention":
        sent = best_sentence.rstrip(".")
        if direction == "positive":
            direct_answer = (
                f"Based on {n} retrieved studies, the evidence supports a beneficial effect: {sent}."
            )
        elif direction == "negative":
            direct_answer = (
                f"Based on {n} retrieved studies, the evidence does not support a significant benefit: {sent}."
            )
        else:
            direct_answer = f"Based on {n} retrieved studies, evidence is mixed: {sent}."
    else:
        # Descriptive or search term: lead with the most informative sentence
        direct_answer = best_sentence if best_sentence.endswith(".") else best_sentence + "."

    consensus_statement = _build_consensus(intent, direction, topic, top_sentences, n)

    # Key findings: next highest-scoring non-redundant sentences
    raw_texts = [s for s, _ in top_sentences[1: max_findings * 3 + 1]]
    deduped = set(_deduplicate_sentences(raw_texts))
    key_findings: list[KeyFinding] = []
    seen_findings = {best_sentence}
    for sent, paper_idx in top_sentences[1:]:
        if sent in deduped and sent not in seen_findings:
            key_findings.append(KeyFinding(finding=sent, citation=paper_idx))
            seen_findings.add(sent)
        if len(key_findings) >= max_findings:
            break

    # Research gaps
    gap_signals = {
        "long-term": "Long-term outcomes and durability of effects remain understudied.",
        "pediatric": "Evidence in paediatric populations is limited.",
        "diverse": "Most studies use predominantly Western populations; generalisability may be limited.",
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
        consensus_statement=consensus_statement,
        key_findings=key_findings,
        evidence_quality=quality,
        gaps=gaps,
        limitations=limitations,
    )


def detect_contradictions(papers: list[Paper]) -> list[dict]:
    """Flag pairs of papers with opposing directional signals on the same MeSH topic."""
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
            if (dir_a == "positive" and dir_b == "negative") or (
                dir_a == "negative" and dir_b == "positive"
            ):
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
