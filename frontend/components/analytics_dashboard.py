"""Analytics dashboard component — usage metrics, topic trends, performance by intent."""

from typing import Any

import streamlit as st

from frontend.utils import api_client


# ── Intent colour mapping ─────────────────────────────────────────────────────
_INTENT_COLOURS: dict[str, str] = {
    "definitional":    "#2980B9",
    "comparative":     "#16A085",
    "search":          "#8E44AD",
    "mechanistic":     "#D35400",
    "epidemiological": "#C0392B",
}


def _intent_badge(intent: str | None) -> str:
    colour = _INTENT_COLOURS.get((intent or "").lower(), "#7F8C8D")
    label = (intent or "unknown").capitalize()
    return f'<span style="background:{colour};color:white;border-radius:4px;padding:2px 10px;font-size:0.8rem;font-weight:600;">{label}</span>'


def render_analytics_dashboard() -> None:
    """Render the analytics dashboard page."""
    st.markdown("## 📊 Analytics Dashboard")
    st.caption("Query patterns, synthesis quality, and usage metrics across all searches.")

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        data = api_client.get_analytics_summary()
    except Exception as exc:
        st.error(f"Could not load analytics: {exc}")
        st.info("Make sure the backend is running and the database is populated.")
        return

    # ── Section 1: Usage overview ─────────────────────────────────────────────
    st.markdown("### Usage Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", f"{data.get('total_queries', 0):,}")
    c2.metric("Last 7 Days", f"{data.get('queries_last_7_days', 0):,}")
    avg_lat = data.get("avg_latency_ms", 0)
    c3.metric("Avg Latency", f"{avg_lat:,.0f} ms" if avg_lat else "—")
    contradiction_rate = data.get("contradiction_rate", 0)
    c4.metric("Contradiction Rate", f"{contradiction_rate*100:.1f}%")

    st.divider()

    # ── Section 2: Query volume over time ─────────────────────────────────────
    queries_by_day = data.get("queries_by_day", [])
    if queries_by_day:
        st.markdown("### Query Volume — Last 30 Days")
        import pandas as pd
        df_day = pd.DataFrame(queries_by_day).set_index("day")
        df_day.index = pd.to_datetime(df_day.index)
        st.line_chart(df_day["count"])
    else:
        st.info("No daily query data yet.")

    # ── Section 3: Top topics ─────────────────────────────────────────────────
    top_topics = data.get("top_topics", [])
    if top_topics:
        st.markdown("### Top Query Topics")
        import pandas as pd
        df_topics = pd.DataFrame(top_topics).head(20)
        st.bar_chart(df_topics.set_index("word")["count"])

    # ── Section 4: Performance by intent ─────────────────────────────────────
    latency_by_intent = data.get("latency_by_intent", {})
    faithfulness_by_intent = data.get("faithfulness_by_intent", {})

    if latency_by_intent or faithfulness_by_intent:
        st.markdown("### Performance by Query Intent")
        import pandas as pd

        intents = sorted(set(list(latency_by_intent) + list(faithfulness_by_intent)))
        rows = []
        for intent in intents:
            lat = latency_by_intent.get(intent, {})
            faith = faithfulness_by_intent.get(intent, {})
            rows.append({
                "Intent": intent.capitalize(),
                "Queries": lat.get("n", 0) or faith.get("total", 0),
                "Avg Latency (ms)": lat.get("avg_ms", "—"),
                "P95 Latency (ms)": lat.get("p95_ms", "—"),
                "Avg Faithfulness": faith.get("avg_faithfulness", "—"),
                "% Above Threshold": (
                    f"{100 * faith['above_threshold'] / faith['total']:.0f}%"
                    if faith.get("total") else "—"
                ),
            })

        if rows:
            df_intent = pd.DataFrame(rows).set_index("Intent")
            st.dataframe(df_intent, use_container_width=True)

    st.divider()

    # ── Section 5: Recent queries ─────────────────────────────────────────────
    st.markdown("### Recent Queries")
    try:
        history = api_client.get_query_history(page=1, page_size=20)
        queries = history.get("queries", [])
        if queries:
            import pandas as pd
            rows = [
                {
                    "Query": (q.get("raw_query") or "")[:60] + ("…" if len(q.get("raw_query", "")) > 60 else ""),
                    "Intent": (q.get("intent") or "—").capitalize(),
                    "Papers": q.get("papers_retrieved") or "—",
                    "Latency (ms)": q.get("latency_ms") or "—",
                    "Contradictions": q.get("contradictions_found") or 0,
                    "Time": (q.get("created_at") or "")[:16].replace("T", " "),
                }
                for q in queries
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No query history yet.")
    except Exception as exc:
        st.warning(f"Could not load query history: {exc}")

    # ── Section 6: Metric explanations ───────────────────────────────────────
    with st.expander("What do these metrics mean?"):
        st.markdown("""
**Faithfulness** — measures whether the LLM's synthesis claims are supported by the retrieved paper abstracts.
A score of 1.0 means every claim cites a source; 0.0 means the synthesis was hallucinated.
Threshold: ≥ 0.75.

**Contradiction Rate** — the percentage of queries where at least one pair of retrieved papers
was detected to reach opposing conclusions on the same intervention and outcome.
Higher rates may indicate genuinely contested evidence in the literature.

**Average Latency** — end-to-end time in milliseconds from query submission to full response,
including PubMed/S2 fetch, embedding, reranking, and synthesis.

**Query Intent** — the classified intent of each query:
- *Definitional*: "What is X?" → focuses on review articles.
- *Comparative*: "X vs Y" → requires RCT evidence.
- *Mechanistic*: "Why does X cause Y?" → prefers basic science papers.
- *Epidemiological*: "How common is X?" → favours cohort studies.
- *Search*: "Find recent papers on X" → sorted by recency.
        """)
