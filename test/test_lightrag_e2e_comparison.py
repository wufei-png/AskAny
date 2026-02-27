#!/usr/bin/env python3
"""End-to-end comparison test: min_langchain_agent with and without LightRAG.

Runs a subset of lightrag_questions through the agent twice:
  1. With ``enable_lightrag=False`` (baseline — vector + keyword only)
  2. With ``enable_lightrag=True``  (augmented — vector + keyword + KG)

Results are written to ``test/e2e_comparison_results.json`` for manual inspection.

Prerequisites
-------------
* PostgreSQL running with both LlamaIndex vector tables AND LightRAG tables.
* LLM endpoint reachable (see .env / config.py).
* viper-v5.5 data ingested into BOTH pipelines.

Usage
-----
    python test/test_lightrag_e2e_comparison.py
    # or via pytest (marks as slow):
    python -m pytest test/test_lightrag_e2e_comparison.py -v -s
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.config import settings
from askany.workflow.question import lightrag_questions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use first N questions for the comparison (full list can take a long time)
MAX_QUESTIONS = 5
RESULT_FILE = Path(__file__).parent / "e2e_comparison_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_agent_on_questions(
    questions: list[str],
    *,
    enable_lightrag: bool,
) -> list[dict]:
    """Run the agent on *questions* and return a list of result dicts.

    Each dict: {"question": str, "answer": str, "duration_s": float, "enable_lightrag": bool}
    """
    # Patch the setting at runtime
    original_value = settings.enable_lightrag
    settings.enable_lightrag = enable_lightrag

    # Import here so the agent picks up current settings
    from askany.workflow.min_langchain_agent import (
        create_agent_with_tools,
        extract_and_format_response,
        invoke_with_retry,
    )

    agent = create_agent_with_tools()
    results = []

    for i, question in enumerate(questions, 1):
        mode_label = "LightRAG ON" if enable_lightrag else "LightRAG OFF"
        print(f"\n{'=' * 80}")
        print(f"[{mode_label}] Question {i}/{len(questions)}: {question}")
        print("=" * 80)

        t0 = time.time()
        try:
            raw_result = invoke_with_retry(
                agent,
                {"messages": [{"role": "user", "content": question}]},
            )
            answer = extract_and_format_response(raw_result)
        except Exception as exc:
            answer = f"ERROR: {type(exc).__name__}: {exc}"
            logger.error("Agent failed on question %d: %s", i, exc, exc_info=True)
        duration = time.time() - t0

        print(f"  Duration: {duration:.1f}s")
        print(f"  Answer preview: {answer[:300]}...")

        results.append(
            {
                "question": question,
                "answer": answer,
                "duration_s": round(duration, 2),
                "enable_lightrag": enable_lightrag,
            }
        )

    # Restore original setting
    settings.enable_lightrag = original_value
    return results


def _write_comparison(
    baseline: list[dict],
    augmented: list[dict],
    output_path: Path,
) -> None:
    """Write side-by-side comparison to JSON file."""
    comparison = []
    for b, a in zip(baseline, augmented):
        comparison.append(
            {
                "question": b["question"],
                "baseline": {
                    "answer": b["answer"],
                    "duration_s": b["duration_s"],
                },
                "lightrag": {
                    "answer": a["answer"],
                    "duration_s": a["duration_s"],
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Comparison results written to: {output_path}")


def _print_summary(baseline: list[dict], augmented: list[dict]) -> None:
    """Print a compact summary table."""
    print(f"\n{'=' * 100}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 100}")
    print(f"{'Question':<60} {'Baseline':>10} {'LightRAG':>10} {'Δ':>8}")
    print(f"{'-' * 60} {'-' * 10} {'-' * 10} {'-' * 8}")

    total_base = 0.0
    total_aug = 0.0

    for b, a in zip(baseline, augmented):
        q_short = (
            b["question"][:57] + "..." if len(b["question"]) > 57 else b["question"]
        )
        delta = a["duration_s"] - b["duration_s"]
        sign = "+" if delta >= 0 else ""
        print(
            f"{q_short:<60} {b['duration_s']:>8.1f}s {a['duration_s']:>8.1f}s {sign}{delta:>6.1f}s"
        )
        total_base += b["duration_s"]
        total_aug += a["duration_s"]

    print(f"{'-' * 60} {'-' * 10} {'-' * 10} {'-' * 8}")
    delta_total = total_aug - total_base
    sign = "+" if delta_total >= 0 else ""
    print(
        f"{'TOTAL':<60} {total_base:>8.1f}s {total_aug:>8.1f}s {sign}{delta_total:>6.1f}s"
    )

    # Check for error results
    base_errors = sum(1 for r in baseline if r["answer"].startswith("ERROR:"))
    aug_errors = sum(1 for r in augmented if r["answer"].startswith("ERROR:"))
    if base_errors or aug_errors:
        print(f"\n⚠ Errors: baseline={base_errors}, lightrag={aug_errors}")

    print(f"\nDetailed results: {RESULT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    questions = lightrag_questions[:MAX_QUESTIONS]

    print("=" * 80)
    print("LightRAG End-to-End Comparison Test")
    print(f"Questions: {len(questions)} (from lightrag_questions)")
    print(f"LLM model: {settings.openai_model}")
    print(f"Embedding: {settings.embedding_model}")
    print("=" * 80)

    # Phase 1: Baseline (LightRAG OFF)
    print(f"\n{'#' * 80}")
    print("PHASE 1: Running with LightRAG DISABLED (baseline)")
    print(f"{'#' * 80}")
    baseline = _run_agent_on_questions(questions, enable_lightrag=False)

    # Phase 2: Augmented (LightRAG ON)
    print(f"\n{'#' * 80}")
    print("PHASE 2: Running with LightRAG ENABLED")
    print(f"{'#' * 80}")
    augmented = _run_agent_on_questions(questions, enable_lightrag=True)

    # Write results
    _write_comparison(baseline, augmented, RESULT_FILE)

    # Print summary
    _print_summary(baseline, augmented)

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
