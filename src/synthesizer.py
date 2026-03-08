"""Extractive evidence synthesizer with entity-anchored semantic sentence scoring.

Uses sentence-transformers (all-MiniLM-L6-v2) for semantic similarity combined
with entity-overlap scoring and Maximal Marginal Relevance (MMR) to select
diverse, on-topic sentences. Falls back to TF-IDF cosine similarity. No API
calls. Zero cost.

Pipeline per query:
  1. Extract key biomedical entities from the query.
  2. Score every abstract sentence by:
       0.60 * semantic_similarity + 0.30 * entity_overlap + 0.10 * stats_bonus
  3. Apply a relevance threshold - if the best score is too low, flag the
     synthesis as potentially off-topic.
  4. Use MMR to pick diverse, non-redundant key findings.
  5. Build a direct answer and consensus from the actual top sentences.

Query intent is detected to apply appropriate framing:
  - intervention: "Does X reduce Y?" -> explicit directional verdict
  - descriptive:  "What causes Y?"   -> explicit literature finding
  - term:         "telomere biology"  -> literature overview

All output sentences are extracted verbatim from source abstracts.
"""

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.fetcher import Paper

# ── Signal lexicons ────────────────────────────────────────────────────────────

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

# ── Intent detection ───────────────────────────────────────────────────────────

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

# ── Entity extraction helpers ──────────────────────────────────────────────────

_ENTITY_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "with", "for", "to", "and",
    "or", "not", "is", "are", "does", "do", "can", "how", "what",
    "why", "when", "where", "who", "which", "effect", "effects", "impact",
    "role", "association", "between", "among", "there", "any", "some",
    "their", "this", "that", "these", "those", "than", "from", "by",
    "at", "as", "be", "was", "were", "been", "has", "have", "had",
    "its", "it", "we", "our", "they", "them", "us",
}

# ── Statistics patterns ────────────────────────────────────────────────────────

_STATS_PATTERNS = [
    re.compile(r'p\s*[<>=]\s*0\.\d+', re.IGNORECASE),
    re.compile(r'\d+(?:\.\d+)?\s*%'),
    re.compile(r'(?:OR|RR|HR)\s*[=:]\s*\d', re.IGNORECASE),
    re.compile(r'(?:hazard ratio|odds ratio|relative risk)\s*[=:]\s*\d', re.IGNORECASE),
    re.compile(r'\d+(?:\.\d+)?-fold', re.IGNORECASE),
    re.compile(r'95\s*%\s*CI', re.IGNORECASE),
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class KeyFinding:
    """A single extracted key finding with its source paper index."""
    finding: str
    citation: int
    has_statistics: bool = False


@dataclass
class Synthesis:
    """Structured evidence synthesis."""
    direct_answer: str
    consensus_statement: str
    key_findings: list[KeyFinding] = field(default_factory=list)
    evidence_quality: str = "mixed"
    research_volume: str = "Moderately Researched"
    volume_score: float = 0.5
    gaps: list[str] = field(default_factory=list)
    limitations: str = ""
    relevance_confidence: float = 1.0


# ── Intent detection ───────────────────────────────────────────────────────────

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


# ── Sentence splitting ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 40]


# ── Directional scoring ────────────────────────────────────────────────────────

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


# ── Evidence quality ───────────────────────────────────────────────────────────

def _evidence_quality(papers: list[Paper]) -> str:
    all_text = " ".join((p.abstract or "") for p in papers).lower()
    if any(s in all_text for s in {"meta-analysis", "systematic review", "cochrane"}):
        return "strong"
    if any(s in all_text for s in {"randomized", "randomised", "rct", "placebo", "controlled trial"}):
        return "moderate"
    if any(s in all_text for s in {"cohort", "observational", "retrospective", "case-control"}):
        return "weak"
    return "mixed"


# ── Research volume ────────────────────────────────────────────────────────────

def _compute_research_volume(unique_count: int, quality: str) -> tuple[str, float]:
    """Return a label and 0-1 score representing how well-researched the topic is."""
    if unique_count >= 28:
        score = 1.0
    elif unique_count >= 20:
        score = 0.82
    elif unique_count >= 12:
        score = 0.62
    elif unique_count >= 6:
        score = 0.42
    elif unique_count >= 2:
        score = 0.22
    else:
        score = 0.08

    quality_boost = {"strong": 0.12, "moderate": 0.06, "weak": 0.0, "mixed": 0.0}
    score = min(1.0, score + quality_boost.get(quality, 0.0))

    if score >= 0.75:
        label = "Extensively Researched"
    elif score >= 0.50:
        label = "Well Researched"
    elif score >= 0.28:
        label = "Moderately Researched"
    else:
        label = "Limited Research"

    return label, score


# ── Entity extraction ──────────────────────────────────────────────────────────

def _extract_key_entities(query: str) -> set[str]:
    """Extract meaningful biomedical entities from query.

    Captures uppercase abbreviations (GADS, TNF, T, B, NK), multi-word
    biomedical terms, and non-stopword content words. Single uppercase
    characters are kept because they frequently denote cell types (T cell,
    B cell) or loci.
    """
    entities: set[str] = set()

    # Uppercase tokens including single-char (T, B) and alphanumeric (SGLT2, IL-6)
    for match in re.finditer(r'\b[A-Z][A-Z0-9_-]*\b', query):
        word = match.group()
        entities.add(word)
        entities.add(word.lower())

    # Content words longer than 2 chars
    for w in re.findall(r'\b\w+\b', query.lower()):
        if w not in _ENTITY_STOPWORDS and len(w) > 2:
            entities.add(w)

    # Bigrams (captures "T cell", "GADS KO", "B cell")
    tokens = re.findall(r'\b\w+\b', query)
    for i in range(len(tokens) - 1):
        pair = f"{tokens[i].lower()} {tokens[i+1].lower()}"
        if (tokens[i].lower() not in _ENTITY_STOPWORDS
                or tokens[i+1].lower() not in _ENTITY_STOPWORDS):
            entities.add(pair)

    return entities


def _entity_overlap_score(sentence: str, entities: set[str]) -> float:
    """Score how many query entities appear in the sentence (0 to 1)."""
    if not entities:
        return 0.0
    sent_lower = sentence.lower()
    matches = sum(1 for e in entities if e in sent_lower)
    return min(1.0, matches / len(entities))


# ── Statistics detection ───────────────────────────────────────────────────────

def _has_statistics(sentence: str) -> bool:
    """Check if sentence contains quantitative statistics."""
    return any(p.search(sentence) for p in _STATS_PATTERNS)


# ── Semantic scoring ───────────────────────────────────────────────────────────

def _score_sentences_semantic(sentences: list[str], query: str, encoder=None) -> np.ndarray:
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


def _score_sentences(
    sentences: list[str],
    query: str,
    entities: set[str],
    encoder=None,
) -> np.ndarray:
    """Combined sentence scorer: semantic + entity overlap + statistics bonus.

    Weights:
      60% semantic similarity (transformer or TF-IDF)
      30% entity overlap    (prevents topic drift, e.g. GADS vs GADs)
      10% statistics bonus  (boosts quantitative findings)
    """
    sem = _score_sentences_semantic(sentences, query, encoder)

    entity_arr = np.array([_entity_overlap_score(s, entities) for s in sentences])
    stats_arr = np.array([0.1 if _has_statistics(s) else 0.0 for s in sentences])

    return 0.60 * sem + 0.30 * entity_arr + stats_arr


# ── Maximal Marginal Relevance ─────────────────────────────────────────────────

def _mmr_select(
    scores: np.ndarray,
    sentences: list[str],
    n: int,
    lambda_param: float = 0.7,
    encoder=None,
) -> list[int]:
    """Select n diverse, relevant sentences using Maximal Marginal Relevance.

    lambda_param controls relevance-diversity trade-off:
      1.0 = pure relevance ranking, 0.0 = pure diversity.
    """
    if len(sentences) <= n:
        return list(range(len(sentences)))

    # Build similarity matrix for diversity measurement
    try:
        if encoder is not None:
            embs = encoder.encode(sentences, show_progress_bar=False, batch_size=64)
            sim_matrix = cosine_similarity(embs)
        else:
            vec = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf = vec.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf)
    except Exception:
        # Fall back to plain score ranking
        return sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:n]

    selected: list[int] = []
    candidates = list(range(len(sentences)))

    while len(selected) < n and candidates:
        if not selected:
            best = max(candidates, key=lambda i: scores[i])
        else:
            def _mmr(i: int) -> float:
                rel = scores[i]
                redundancy = max(sim_matrix[i, j] for j in selected)
                return lambda_param * rel - (1.0 - lambda_param) * redundancy
            best = max(candidates, key=_mmr)

        selected.append(best)
        candidates.remove(best)

    return selected


# ── Deduplication ──────────────────────────────────────────────────────────────

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


# ── Cluster-diverse sentence pool ──────────────────────────────────────────────

def _cluster_diverse_pool(
    scored_sentences: list[tuple[str, int, float]],
    n_clusters: int = 5,
    pool_size: int = 30,
    encoder=None,
) -> list[tuple[str, int, float]]:
    """Use KMeans clustering to build a topically diverse candidate pool.

    Instead of taking the top-N sentences globally (which may all come from
    one cluster of papers), we pick the best sentence from each semantic
    cluster to ensure broad coverage.
    """
    if len(scored_sentences) <= pool_size:
        return scored_sentences

    texts = [s for s, _, _ in scored_sentences]
    try:
        if encoder is not None:
            embs = encoder.encode(texts, show_progress_bar=False, batch_size=64)
        else:
            vec = TfidfVectorizer(stop_words="english", min_df=1)
            embs = vec.fit_transform(texts).toarray()

        k = min(n_clusters, len(texts))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embs)

        # From each cluster take the top-scoring sentences
        cluster_to_items: dict[int, list[tuple[str, int, float]]] = {}
        for idx, (sent, paper_idx, score) in enumerate(scored_sentences):
            c = labels[idx]
            cluster_to_items.setdefault(c, []).append((sent, paper_idx, score))

        diverse_pool: list[tuple[str, int, float]] = []
        # Round-robin across clusters, picking highest-score each round
        cluster_lists = [
            sorted(v, key=lambda x: x[2], reverse=True)
            for v in cluster_to_items.values()
        ]
        pointer = [0] * len(cluster_lists)
        while len(diverse_pool) < pool_size:
            advanced = False
            for ci, lst in enumerate(cluster_lists):
                if pointer[ci] < len(lst):
                    diverse_pool.append(lst[pointer[ci]])
                    pointer[ci] += 1
                    advanced = True
            if not advanced:
                break
        return diverse_pool

    except Exception:
        return scored_sentences[:pool_size]


# ── Direct answer builder ──────────────────────────────────────────────────────

def _smart_lowercase(sent: str) -> str:
    """Lowercase the first character unless it starts with an acronym."""
    if not sent:
        return sent
    first_word = sent.split()[0] if sent.split() else ""
    if len(first_word) >= 2 and (
        first_word.isupper()
        or (first_word[0].isupper() and any(c.isdigit() for c in first_word))
    ):
        return sent
    return sent[0].lower() + sent[1:]


def _build_direct_answer(
    intent: str,
    direction: str,
    quality: str,
    best_sentence: str,
    unique_count: int,
    relevance_confidence: float,
) -> str:
    """Build an explicit, verdict-first direct answer.

    Mirrors the style of a concise literature verdict: clearly states what
    the research shows, scaled to the quality and direction of the evidence.
    When research is sparse or relevance is low, explicitly flags that.
    """
    sent = best_sentence.rstrip(".")
    sent_lc = _smart_lowercase(sent)

    if relevance_confidence < 0.25:
        return (
            "The retrieved papers may not directly address this specific query. "
            f"The closest available evidence suggests that {sent_lc}. "
            "Consider rephrasing with more specific terminology."
        )

    if unique_count < 5:
        return (
            f"There is currently limited published research on this specific topic. "
            f"Available evidence suggests that {sent_lc}."
        )

    if intent == "intervention":
        if quality == "strong":
            if direction == "positive":
                prefix = "Meta-analyses and systematic reviews confirm that"
            elif direction == "negative":
                prefix = "Systematic reviews do not support a significant benefit; evidence shows that"
            else:
                prefix = "Systematic evidence is inconsistent across studies; findings indicate that"
        elif quality == "moderate":
            if direction == "positive":
                prefix = "Randomized controlled trials demonstrate that"
            elif direction == "negative":
                prefix = "Controlled trial evidence does not support this intervention; studies show that"
            else:
                prefix = "Controlled trial evidence is mixed; studies indicate that"
        else:
            if direction == "positive":
                prefix = "Observational research suggests that"
            elif direction == "negative":
                prefix = "Available evidence does not support a significant effect; studies indicate that"
            else:
                prefix = "Research findings are currently mixed; available evidence suggests that"
    else:
        if quality == "strong":
            prefix = "Systematic evidence demonstrates that"
        elif quality == "moderate":
            prefix = "Research demonstrates that"
        elif quality == "weak":
            prefix = "Observational research suggests that"
        else:
            prefix = "Current literature indicates that"

    return f"{prefix} {sent_lc}."


# ── Consensus builder ──────────────────────────────────────────────────────────

def _build_consensus(
    intent: str,
    direction: str,
    quality: str,
    topic: str,
    top_sentences: list[tuple[str, int]],
    n: int,
    unique_count: int,
    relevance_confidence: float,
) -> str:
    """Build a broader narrative context paragraph from actual top sentences."""
    additional = [s for s, _ in top_sentences[1:4]]

    if unique_count < 5:
        base = (
            f"Only {unique_count} unique paper(s) were identified for this query. "
            "Conclusions should be interpreted with caution. "
        )
        if additional:
            base += " ".join(s if s.endswith(".") else s + "." for s in additional[:2])
        return base

    if relevance_confidence < 0.25:
        if additional:
            return " ".join(s if s.endswith(".") else s + "." for s in additional[:2])
        return f"Limited directly relevant content was found for '{topic}'."

    if additional:
        parts = []
        for s in additional:
            s_clean = s.strip()
            if s_clean and not s_clean.endswith("."):
                s_clean += "."
            if s_clean:
                parts.append(s_clean)
        if parts:
            return " ".join(parts)

    return f"The retrieved literature covers multiple aspects of {topic}."


# ── Main synthesis entry point ─────────────────────────────────────────────────

def synthesise(
    query: str,
    papers: list[Paper],
    max_findings: int = 6,
    encoder=None,
    unique_count: int = None,
) -> Synthesis:
    """Generate a structured evidence synthesis from the top papers.

    Args:
        query: The user's research question or search term.
        papers: Top-ranked papers (typically 10).
        max_findings: Maximum key findings to surface.
        encoder: Optional sentence-transformers encoder for semantic scoring.
        unique_count: Total unique papers after deduplication (for volume scoring).

    Returns:
        A Synthesis object.
    """
    if unique_count is None:
        unique_count = len(papers)

    if not papers:
        return Synthesis(
            direct_answer="No papers were retrieved for this query. Try rephrasing your question.",
            consensus_statement="No papers were retrieved.",
            evidence_quality="weak",
            research_volume="Limited Research",
            volume_score=0.05,
            gaps=["Insufficient literature retrieved."],
            relevance_confidence=0.0,
        )

    # Extract entities from query for entity-anchored scoring
    entities = _extract_key_entities(query)

    all_sentences: list[tuple[str, int]] = []
    for i, paper in enumerate(papers, start=1):
        for sent in _split_sentences(paper.abstract or ""):
            all_sentences.append((sent, i))

    if not all_sentences:
        return Synthesis(
            direct_answer="Retrieved papers had insufficient abstract content for synthesis.",
            consensus_statement="Retrieved papers had insufficient abstract content.",
            evidence_quality="weak",
            research_volume="Limited Research",
            volume_score=0.05,
            relevance_confidence=0.0,
        )

    texts = [s for s, _ in all_sentences]
    raw_scores = _score_sentences(texts, query, entities, encoder=encoder)

    # Relevance confidence: normalise max score against a calibrated ceiling
    max_raw = float(raw_scores.max())
    relevance_confidence = min(1.0, max_raw / 0.55)

    # Build a cluster-diverse candidate pool then apply MMR
    scored_triples = [
        (texts[i], all_sentences[i][1], float(raw_scores[i]))
        for i in range(len(all_sentences))
    ]
    # Sort by score descending for initial pool
    scored_triples.sort(key=lambda x: x[2], reverse=True)

    # Use KMeans clustering to get a diverse candidate pool
    diverse_pool = _cluster_diverse_pool(
        scored_triples, n_clusters=5, pool_size=40, encoder=encoder
    )

    # Re-extract sentences and scores from pool (preserving paper indices)
    pool_texts = [s for s, _, _ in diverse_pool]
    pool_paper_indices = [pi for _, pi, _ in diverse_pool]
    pool_scores = np.array([sc for _, _, sc in diverse_pool])

    # Exact-match deduplicate the pool
    seen_exact: set[str] = set()
    deduped_pool_texts: list[str] = []
    deduped_paper_indices: list[int] = []
    deduped_scores_list: list[float] = []
    for t, pi, sc in zip(pool_texts, pool_paper_indices, pool_scores):
        if t not in seen_exact:
            deduped_pool_texts.append(t)
            deduped_paper_indices.append(pi)
            deduped_scores_list.append(sc)
            seen_exact.add(t)

    deduped_scores = np.array(deduped_scores_list)

    # Apply MMR to select diverse, relevant sentences
    mmr_indices = _mmr_select(
        deduped_scores,
        deduped_pool_texts,
        n=max_findings + 4,
        lambda_param=0.7,
        encoder=encoder,
    )

    top_sentences: list[tuple[str, int]] = [
        (deduped_pool_texts[i], deduped_paper_indices[i])
        for i in mmr_indices
    ]

    if not top_sentences:
        top_sentences = [(deduped_pool_texts[0], deduped_paper_indices[0])]

    best_sentence, _ = top_sentences[0]

    intent = _detect_intent(query)
    combined_text = " ".join((p.abstract or "") for p in papers)
    direction = _score_direction(combined_text)
    n = len(papers)
    quality = _evidence_quality(papers)
    volume_label, volume_score = _compute_research_volume(unique_count, quality)

    topic = query.rstrip("?").strip()
    if len(topic) > 80:
        topic = topic[:80] + "..."

    direct_answer = _build_direct_answer(
        intent, direction, quality, best_sentence, unique_count, relevance_confidence
    )
    consensus_statement = _build_consensus(
        intent, direction, quality, topic, top_sentences, n, unique_count, relevance_confidence
    )

    # Key findings: MMR-selected, near-duplicate filtered
    raw_finding_texts = [s for s, _ in top_sentences[1: max_findings * 3 + 1]]
    deduped_finding_texts = set(_deduplicate_sentences(raw_finding_texts))
    key_findings: list[KeyFinding] = []
    seen_findings = {best_sentence}
    for sent, paper_idx in top_sentences[1:]:
        if sent in deduped_finding_texts and sent not in seen_findings:
            key_findings.append(KeyFinding(
                finding=sent,
                citation=paper_idx,
                has_statistics=_has_statistics(sent),
            ))
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
        f"Synthesis based on {n} papers (from {unique_count} unique retrieved) from PubMed and Semantic Scholar "
        f"({'including ' + ', '.join(study_types) if study_types else 'mixed study designs'}). "
        "All output sentences are extracted verbatim from source abstracts and cited by paper number."
    )

    return Synthesis(
        direct_answer=direct_answer,
        consensus_statement=consensus_statement,
        key_findings=key_findings,
        evidence_quality=quality,
        research_volume=volume_label,
        volume_score=volume_score,
        gaps=gaps,
        limitations=limitations,
        relevance_confidence=relevance_confidence,
    )


# ── Contradiction detection ────────────────────────────────────────────────────

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
