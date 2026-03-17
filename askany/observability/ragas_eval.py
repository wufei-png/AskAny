"""RAGAS evaluation module for AskAny.

Provides async, non-blocking RAG quality scoring using RAGAS metrics.
Scores are pushed into Langfuse traces when both systems are enabled.

Usage (fire-and-forget from server.py):

    from askany.observability.ragas_eval import evaluate_rag_response

    asyncio.create_task(evaluate_rag_response(
        trace_id="langfuse-trace-id",
        user_input="user question",
        response="llm answer",
        retrieved_contexts=["ctx1", "ctx2"],
    ))
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from askany.config import Settings

logger = logging.getLogger(__name__)

# ── Module-level state ───────────────────────────────────────────────────────
_metrics: Dict[str, object] = {}
_evaluator_llm: Optional[object] = None
_initialized: bool = False
_sample_rate: float = 1.0


# ── Public API ───────────────────────────────────────────────────────────────


def initialize_ragas(settings: "Settings") -> bool:
    """Initialize RAGAS metrics and evaluator LLM.

    Must be called after ``initialize_langfuse`` (scores are pushed to Langfuse).
    Returns ``True`` on success, ``False`` when RAGAS is disabled or deps are
    missing.
    """
    global _metrics, _evaluator_llm, _initialized, _sample_rate

    if _initialized:
        logger.debug("RAGAS already initialized, skipping")
        return True

    if not settings.enable_ragas:
        logger.info("RAGAS evaluation disabled (enable_ragas=False)")
        return False

    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        logger.warning(
            "ragas or langchain_openai not installed — RAGAS disabled. "
            "Install with: pip install ragas langchain-openai"
        )
        return False

    _sample_rate = settings.ragas_sample_rate

    # ── Create evaluator LLM ─────────────────────────────────────────────
    eval_model = settings.ragas_eval_llm_model or settings.openai_model
    eval_api_base = settings.ragas_eval_llm_api_base or settings.openai_api_base
    eval_api_key = settings.ragas_eval_llm_api_key or settings.openai_api_key or ""

    try:
        base_llm = ChatOpenAI(
            model=eval_model,
            api_key=eval_api_key,
            base_url=eval_api_base,
            temperature=0.0,  # Deterministic for evaluation
            max_tokens=4096,
        )
        _evaluator_llm = LangchainLLMWrapper(base_llm)
        logger.info("RAGAS evaluator LLM: model=%s base=%s", eval_model, eval_api_base)
    except Exception:
        logger.exception("Failed to create RAGAS evaluator LLM")
        return False

    # ── Instantiate metrics ──────────────────────────────────────────────
    metric_name_to_class = _get_metric_classes()
    for name in settings.ragas_metrics:
        normalized = name.lower().strip()
        if normalized in metric_name_to_class:
            try:
                metric_cls = metric_name_to_class[normalized]
                _metrics[normalized] = metric_cls(llm=_evaluator_llm)
                logger.info("RAGAS metric loaded: %s", normalized)
            except Exception:
                logger.exception("Failed to instantiate RAGAS metric: %s", normalized)
        else:
            logger.warning(
                "Unknown RAGAS metric '%s', available: %s",
                normalized,
                list(metric_name_to_class.keys()),
            )

    if not _metrics:
        logger.warning("No RAGAS metrics loaded — evaluation disabled")
        return False

    _initialized = True
    logger.info(
        "RAGAS initialization complete (metrics=%s, sample_rate=%.2f)",
        list(_metrics.keys()),
        _sample_rate,
    )
    return True


async def evaluate_rag_response(
    *,
    trace_id: Optional[str] = None,
    user_input: str,
    response: str,
    retrieved_contexts: List[str],
) -> Dict[str, float]:
    """Score a single RAG response and push results to Langfuse.

    This is designed to be called as a fire-and-forget background task:
    it catches all exceptions internally and never raises.

    Args:
        trace_id: Langfuse trace ID to attach scores to (optional).
        user_input: The original user query.
        response: The generated answer.
        retrieved_contexts: List of retrieved context strings.

    Returns:
        Dict of metric_name → score (float). Empty on error or skip.
    """
    if not _initialized or not _metrics:
        return {}

    # ── Sampling ─────────────────────────────────────────────────────────
    if _sample_rate < 1.0 and random.random() > _sample_rate:
        logger.debug("RAGAS: skipped by sampling (rate=%.2f)", _sample_rate)
        return {}

    scores: Dict[str, float] = {}

    try:
        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )

        # Score each metric independently (one failure doesn't block others)
        for name, metric in _metrics.items():
            try:
                score = await metric.single_turn_ascore(sample)
                scores[name] = float(score)
                logger.debug("RAGAS %s = %.4f", name, scores[name])
            except Exception:
                logger.exception("RAGAS metric '%s' failed", name)

    except ImportError:
        logger.warning("ragas package not available for evaluation")
        return {}
    except Exception:
        logger.exception("RAGAS evaluation failed")
        return {}

    # ── Push scores to Langfuse ──────────────────────────────────────────
    if trace_id and scores:
        try:
            from askany.observability.langfuse_setup import get_langfuse_client

            client = get_langfuse_client()
            if client is not None:
                for metric_name, value in scores.items():
                    client.score(
                        trace_id=trace_id,
                        name=f"ragas_{metric_name}",
                        value=value,
                    )
                logger.debug(
                    "Pushed %d RAGAS scores to Langfuse (trace=%s)",
                    len(scores),
                    trace_id,
                )
        except Exception:
            logger.exception("Failed to push RAGAS scores to Langfuse")

    return scores


def shutdown_ragas() -> None:
    """Reset RAGAS module-level state.

    Call this during application shutdown to clean up resources.
    """
    global _metrics, _evaluator_llm, _initialized, _sample_rate
    _metrics = {}
    _evaluator_llm = None
    _initialized = False
    _sample_rate = 1.0
    logger.info("RAGAS shutdown complete")


# ── Internal helpers ─────────────────────────────────────────────────────────


def _get_metric_classes() -> Dict[str, type]:
    """Return a mapping of normalised metric name → RAGAS metric class.

    Supports both the v0.4 ``collections`` API and the legacy ``metrics`` API.
    """
    classes: Dict[str, type] = {}

    # Try v0.4+ collections API first (preferred – avoids deprecation warnings)
    try:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            Faithfulness,
        )

        classes["faithfulness"] = Faithfulness
        classes["response_relevancy"] = AnswerRelevancy
        classes["context_precision"] = ContextPrecisionWithoutReference
        return classes
    except ImportError:
        pass

    # Fallback: ragas.metrics (pre-collections, still available but deprecated)
    try:
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            ResponseRelevancy,
        )

        classes["faithfulness"] = Faithfulness
        classes["response_relevancy"] = ResponseRelevancy
        classes["context_precision"] = LLMContextPrecisionWithoutReference
        return classes
    except ImportError:
        pass

    # Fallback: legacy API
    try:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        classes["faithfulness"] = type(faithfulness)
        classes["response_relevancy"] = type(answer_relevancy)
        classes["context_precision"] = type(context_precision)
    except ImportError:
        logger.warning("Could not import RAGAS metrics from either v0.4 or legacy API")

    return classes
