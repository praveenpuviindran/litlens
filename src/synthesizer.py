"""Extractive evidence synthesizer. Zero API calls, zero cost.

Uses TF-IDF cosine similarity to score sentences from the retrieved abstracts
against the user's query, then selects the most representative non-redundant
sentences as key findings.

This approach is used in production literature review tools where LLM API
calls are not available or too expensive.
"""

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.fetcher import Paper

# Keywords indicating positive / negative / uncertain findings.
_POSITIVE_SIGNALS = {"reduced", "lower", "decrease", "improve", "benefit", "effective", "significant", "superior", "associated with reduced", "prevented", "protective"}
_NEGATIVE_SIGNALS = {"increased", "higher", "worse", "harmful", "no benefit", "no significant", "not associated", "failed", "ineffective", "no difference"}
_UNCERTAIN_SIGNALS = {"mixed", "unclear", "conflicting", "inconclusive", "further research", "limited evidence", "insufficient"}


@dataclass
class KeyFinding:
    """A single extracted key finding with its source paper index."""
    finding: str
    citation: int  # 1-based index into the papers list


@dataclass
class Synthesis:
    """Structured evidence synthesis produced without any LLM."""
    consensus_statement: str
    key_findings: list[KeyFinding] = field(default_factory=list)
    evidence_quality: str = "mixed"  # strong | moderate | weak | mixed
    gaps: list[str] = field(default_factory=list)
    limitations: str = ""


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex."""
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
    rct_signals = {"randomized", "randomised", "rct", "placebo", "double-blind", "controlled trial"}
    meta_signals = {"meta-analysis", "systematic review", "cochrane"}
    obs_signals = {"cohort", "observational", "retrospective", "case-control"}

    all_text = " ".join((p.abstract or "") for p in papers).lower()
    if any(s in all_text for s in meta_signals):
        return "strong"
    if any(s in all_text for s in rct_signals):
        return "moderate"
    if any(s in all_text for s in obs_signals):
        return "weak"
    return "mixed"


def _deduplicate_sentences(sentences: list[str], threshold: float = 0.85) -> list[str]:
    """Remove near-duplicate sentences using cosine similarity."""
    if len(sentences) <= 1:
        return sentences

    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform(sentences)
        sim = cosine_similarity(tfidf)
    except Exception:
        return sentences

    keep = []
    used = set()
    for i in range(len(sentences)):
        if i in used:
            continue
        keep.append(i)
        for j in range(i + 1, len(sentences)):
            if sim[i, j] >= threshold:
                used.add(j)
    return [sentences[i] for i in keep]


def synthesise(query: str, papers: list[Paper], max_findings: int = 6) -> Synthesis:
    """Generate a structured evidence synthesis from the top papers.

    Uses TF-IDF to extract the most query-relevant, non-redundant sentences
    from the paper abstracts. No API calls required.

    Args:
        query: The user's research question.
        papers: Top-ranked papers (typically 10).
        max_findings: Maximum key findings to surface.

    Returns:
        A Synthesis object with consensus statement, key findings, and gaps.
    """
    if not papers:
        return Synthesis(
            consensus_statement="No papers were retrieved for this query. Try rephrasing your question.",
            evidence_quality="weak",
            gaps=["Insufficient literature retrieved to synthesise."],
        )

    # Collect all sentences with their source paper index (1-based)
    all_sentences: list[tuple[str, int]] = []
    for i, paper in enumerate(papers, start=1):
        for sent in _split_sentences(paper.abstract or ""):
            all_sentences.append((sent, i))

    if not all_sentences:
        return Synthesis(
            consensus_statement="Retrieved papers had insufficient abstract content for synthesis.",
            evidence_quality="weak",
        )

    # Score all sentences against the query with TF-IDF cosine similarity
    texts = [s for s, _ in all_sentences] + [query]
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        tfidf = vec.fit_transform(texts)
        query_vec = tfidf[-1]
        corpus_vecs = tfidf[:-1]
        scores = cosine_similarity(query_vec, corpus_vecs)[0]
    except Exception:
        scores = np.ones(len(all_sentences))

    # Rank sentences by score
    ranked = sorted(
        range(len(all_sentences)),
        key=lambda i: scores[i],
        reverse=True,
    )

    # Pick top sentences, deduplicate, and assign citations
    raw_top = [all_sentences[i] for i in ranked[:max_findings * 3]]
    raw_texts = [t for t, _ in raw_top]
    deduped = _deduplicate_sentences(raw_texts)

    key_findings: list[KeyFinding] = []
    seen_texts = set()
    for i, (sent, paper_idx) in enumerate(raw_top):
        if sent in deduped and sent not in seen_texts:
            key_findings.append(KeyFinding(finding=sent, citation=paper_idx))
            seen_texts.add(sent)
        if len(key_findings) >= max_findings:
            break

    # Consensus statement — aggregate direction across all abstracts
    combined_text = " ".join((p.abstract or "") for p in papers)
    direction = _score_direction(combined_text)
    n = len(papers)

    # Build the topic from the query (simple heuristic: first noun phrase / keywords)
    topic = query.rstrip("?").lower()
    if len(topic) > 80:
        topic = topic[:80] + "…"

    if direction == "positive":
        consensus = (
            f"Across {n} retrieved papers, the evidence broadly supports a beneficial effect "
            f"related to: {topic}. Multiple studies report statistically significant findings, "
            f"though effect sizes and populations vary."
        )
    elif direction == "negative":
        consensus = (
            f"Across {n} retrieved papers, the evidence is largely negative or null for "
            f"the query: {topic}. Several studies report no significant benefit or potential harm."
        )
    else:
        consensus = (
            f"Across {n} retrieved papers, findings are mixed regarding: {topic}. "
            f"Some studies report beneficial effects while others find no significant association, "
            f"reflecting heterogeneity in study design, population, and outcomes measured."
        )

    # Quality estimate
    quality = _evidence_quality(papers)

    # Identify research gaps (simple heuristics)
    gap_signals = {
        "long-term": "Long-term outcomes and durability of effects remain understudied.",
        "pediatric": "Evidence in paediatric and adolescent populations is limited.",
        "diverse": "Most studies are conducted in predominantly Western/European populations; generalisability may be limited.",
        "mechanism": "The underlying biological mechanisms are not fully elucidated.",
        "cost": "Cost-effectiveness and real-world implementation have not been well characterised.",
    }
    gaps = []
    combined_lower = combined_text.lower()
    for keyword, gap_text in gap_signals.items():
        if keyword not in combined_lower and len(gaps) < 3:
            gaps.append(gap_text)
    if not gaps:
        gaps = [
            "Longer follow-up studies are needed to assess durability of effects.",
            "Head-to-head comparisons between interventions are limited.",
        ]

    # Limitations note
    study_types = []
    if "randomized" in combined_lower or "randomised" in combined_lower:
        study_types.append("RCTs")
    if "cohort" in combined_lower or "observational" in combined_lower:
        study_types.append("observational studies")
    if "meta-analysis" in combined_lower:
        study_types.append("meta-analyses")
    limitations = (
        f"This synthesis is based on {n} papers retrieved from PubMed and Semantic Scholar "
        f"({'including ' + ', '.join(study_types) if study_types else 'with mixed study designs'}). "
        "Extractive summarisation is used — results represent the most query-relevant sentences "
        "from the abstracts rather than a full expert narrative review."
    )

    return Synthesis(
        consensus_statement=consensus,
        key_findings=key_findings,
        evidence_quality=quality,
        gaps=gaps,
        limitations=limitations,
    )


def detect_contradictions(papers: list[Paper]) -> list[dict]:
    """Detect potential contradictions using simple direction-signal analysis.

    Flags pairs of papers where one shows positive signals and the other shows
    negative signals on the same topic. This is a heuristic, not LLM-based.

    Args:
        papers: Top-ranked papers.

    Returns:
        List of contradiction dicts (may be empty).
    """
    contradictions = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            a, b = papers[i], papers[j]

            # Only compare papers sharing a MeSH term (same topic)
            shared_mesh = set(t.lower() for t in (a.mesh_terms or [])) & set(t.lower() for t in (b.mesh_terms or []))
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
                        "These papers appear to reach opposite conclusions on the same topic. "
                        "Differences may reflect study design, population, dose, or follow-up duration."
                    ),
                })

    return contradictions
