"""Eval dashboard component — displays metric history charts and latest scores."""

import json
from pathlib import Path
from typing import Any

import streamlit as st

EVAL_HISTORY_PATH = Path(__file__).parent.parent.parent / "eval" / "eval_history.json"

THRESHOLDS = {
    "faithfulness": 0.75,
    "contradiction_precision": 0.65,
    "retrieval_precision": 0.55,
}


def _status_badge(score: float, threshold: float) -> str:
    """Return a coloured emoji badge based on whether score meets the threshold.

    Args:
        score: Observed metric value.
        threshold: Required minimum value.

    Returns:
        A green checkmark or red cross emoji string.
    """
    return "✅" if score >= threshold else "❌"


def render_eval_dashboard() -> None:
    """Render the evaluation dashboard page."""
    st.markdown("## 📊 Evaluation Dashboard")

    if not EVAL_HISTORY_PATH.exists():
        st.info(
            "No evaluation history found.\n\n"
            "Run `python eval/run_eval.py` to populate evaluation history."
        )
        return

    try:
        history: list[dict[str, Any]] = json.loads(EVAL_HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        st.error(f"Failed to load evaluation history: {exc}")
        return

    if not history:
        st.info("Evaluation history is empty. Run `python eval/run_eval.py` to populate it.")
        return

    latest = history[-1]

    # ── Metric cards (2×2 grid) ───────────────────────────────────────────────
    st.markdown("### Latest Run Results")
    col1, col2 = st.columns(2)

    with col1:
        faith = latest.get("faithfulness", 0.0)
        st.metric(
            label=f"Faithfulness {_status_badge(faith, THRESHOLDS['faithfulness'])}",
            value=f"{faith:.3f}",
            delta=f"Threshold: {THRESHOLDS['faithfulness']}",
        )

        queries_eval = latest.get("queries_evaluated", 0)
        st.metric(label="Queries Evaluated", value=queries_eval)

    with col2:
        contrad = latest.get("contradiction_precision", 0.0)
        st.metric(
            label=f"Contradiction Precision {_status_badge(contrad, THRESHOLDS['contradiction_precision'])}",
            value=f"{contrad:.3f}",
            delta=f"Threshold: {THRESHOLDS['contradiction_precision']}",
        )

        retrieval = latest.get("retrieval_precision", 0.0)
        st.metric(
            label=f"Retrieval Precision {_status_badge(retrieval, THRESHOLDS['retrieval_precision'])}",
            value=f"{retrieval:.3f}",
            delta=f"Threshold: {THRESHOLDS['retrieval_precision']}",
        )

    # ── Time-series chart ─────────────────────────────────────────────────────
    if len(history) > 1:
        st.markdown("### Metric History")
        import pandas as pd

        df = pd.DataFrame([
            {
                "timestamp": entry.get("timestamp", f"Run {i+1}"),
                "Faithfulness": entry.get("faithfulness", 0.0),
                "Contradiction Precision": entry.get("contradiction_precision", 0.0),
                "Retrieval Precision": entry.get("retrieval_precision", 0.0),
            }
            for i, entry in enumerate(history)
        ]).set_index("timestamp")

        st.line_chart(df)
    else:
        st.caption("Run the eval pipeline multiple times to see trend charts.")

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("Raw eval history JSON", expanded=False):
        st.json(history)
