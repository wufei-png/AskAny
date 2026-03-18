#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.config import settings
from askany.observability import ragas_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TEST_QUERIES = [
    "cassandra组件的concurrent_reads有什么用？",
    "如何解决Cassandra OOM问题？",
    "Viper大数据服务是什么？",
]

skip_reason: str | None = None


def _check_prerequisites() -> bool:
    global skip_reason

    if not settings.postgres_host:
        skip_reason = "PostgreSQL not configured"
        return False

    if not settings.openai_api_key and not settings.openai_api_base:
        skip_reason = "LLM not configured"
        return False

    if not settings.enable_ragas:
        logger.warning(
            "RAGAS not enabled in settings - tests will verify enable/disable flow"
        )

    return True


def _make_mock_metric(name: str, score: float = 0.8):
    mock = MagicMock()

    async def mock_score(*args, **kwargs):
        return score

    mock.single_turn_ascore = mock_score
    return mock


class TestRAGASE2EIntegration:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        ragas_eval._initialized = False
        ragas_eval._metrics = {}
        ragas_eval._sample_rate = 1.0
        yield
        ragas_eval._initialized = False
        ragas_eval._metrics = {}
        ragas_eval._sample_rate = 1.0

    def test_simple_agent_extracts_references_from_response(self):
        from askany.workflow.min_langchain_agent import (
            FinalSummaryResponse,
            extract_references_from_result,
        )

        refs = [
            "Reference 1: Cassandra concurrent_reads controls read parallelism",
            "Reference 2: Default value is 64, can be increased to 128",
        ]

        structured_response = FinalSummaryResponse(
            summary_answer="concurrent_reads用于控制读并发，默认64",
            references=refs,
        )

        result = {"structured_response": structured_response}
        extracted = extract_references_from_result(result)

        assert extracted == refs
        assert len(extracted) == 2

    def test_workflow_nodes_can_produce_context_strings(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        nodes = [
            NodeWithScore(node=TextNode(text="Node 1 content: Cassandra config")),
            NodeWithScore(node=TextNode(text="Node 2 content: OOM solutions")),
        ]

        references = [node.node.get_content() for node in nodes]

        assert len(references) == 2
        assert "Cassandra config" in references[0]
        assert "OOM solutions" in references[1]

    def test_workflow_empty_nodes_produce_empty_contexts(self):
        nodes = []
        references = [node.node.get_content() for node in nodes]
        assert references == []

    @pytest.mark.asyncio
    async def test_ragas_evaluation_with_real_contexts(self):
        mock_faithfulness = _make_mock_metric("faithfulness", 0.85)
        mock_relevancy = _make_mock_metric("response_relevancy", 0.9)
        mock_precision = _make_mock_metric("context_precision", 0.75)

        ragas_eval._initialized = True
        ragas_eval._metrics = {
            "faithfulness": mock_faithfulness,
            "response_relevancy": mock_relevancy,
            "context_precision": mock_precision,
        }
        ragas_eval._sample_rate = 1.0

        retrieved_contexts = [
            "Context 1: Cassandra concurrent_reads default is 64",
            "Context 2: Increasing this value improves read throughput",
        ]

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            scores = await ragas_eval.evaluate_rag_response(
                user_input="concurrent_reads有什么用？",
                response="concurrent_reads用于控制读并发，默认64，可根据硬件配置调整",
                retrieved_contexts=retrieved_contexts,
            )

        assert "faithfulness" in scores
        assert "response_relevancy" in scores
        assert "context_precision" in scores
        assert scores["faithfulness"] == 0.85
        assert scores["response_relevancy"] == 0.9
        assert scores["context_precision"] == 0.75

    @pytest.mark.asyncio
    async def test_ragas_evaluation_skips_with_zero_sample_rate(self):
        ragas_eval._initialized = True
        ragas_eval._metrics = {"faithfulness": _make_mock_metric("faithfulness")}
        ragas_eval._sample_rate = 0.0

        scores = await ragas_eval.evaluate_rag_response(
            user_input="test query",
            response="test response",
            retrieved_contexts=["test context"],
        )

        assert scores == {}

    @pytest.mark.asyncio
    async def test_ragas_evaluation_with_empty_contexts(self):
        mock_relevancy = _make_mock_metric("response_relevancy", 0.7)

        ragas_eval._initialized = True
        ragas_eval._metrics = {"response_relevancy": mock_relevancy}
        ragas_eval._sample_rate = 1.0

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            scores = await ragas_eval.evaluate_rag_response(
                user_input="test query",
                response="test response",
                retrieved_contexts=[],
            )

        assert "response_relevancy" in scores
        assert scores["response_relevancy"] == 0.7

    @pytest.mark.asyncio
    async def test_ragas_pushes_scores_to_langfuse_when_enabled(self):
        from askany.observability import langfuse_setup

        mock_faithfulness = _make_mock_metric("faithfulness", 0.85)
        mock_langfuse = MagicMock()

        ragas_eval._initialized = True
        ragas_eval._metrics = {"faithfulness": mock_faithfulness}
        ragas_eval._sample_rate = 1.0

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            with patch.object(
                langfuse_setup, "get_langfuse_client", return_value=mock_langfuse
            ):
                scores = await ragas_eval.evaluate_rag_response(
                    trace_id="test-trace-123",
                    user_input="test query",
                    response="test response",
                    retrieved_contexts=["test context"],
                )

        assert scores["faithfulness"] == 0.85
        mock_langfuse.score.assert_called_once()
        call_kwargs = mock_langfuse.score.call_args[1]
        assert call_kwargs["trace_id"] == "test-trace-123"
        assert call_kwargs["name"] == "ragas_faithfulness"
        assert call_kwargs["value"] == 0.85


class TestRAGASInitialization:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        ragas_eval._initialized = False
        ragas_eval._metrics = {}
        ragas_eval._sample_rate = 1.0
        yield
        ragas_eval._initialized = False
        ragas_eval._metrics = {}
        ragas_eval._sample_rate = 1.0

    def test_initialize_with_all_defaults(self):
        from askany.observability.ragas_eval import initialize_ragas

        test_settings = MagicMock()
        test_settings.enable_ragas = True
        test_settings.ragas_sample_rate = 1.0
        test_settings.ragas_metrics = [
            "faithfulness",
            "response_relevancy",
            "context_precision",
        ]
        test_settings.ragas_eval_llm_model = None
        test_settings.ragas_eval_llm_api_base = None
        test_settings.ragas_eval_llm_api_key = None
        test_settings.openai_model = "gpt-4"
        test_settings.openai_api_base = "https://api.openai.com/v1"
        test_settings.openai_api_key = "test-key"

        with patch.dict(
            "sys.modules",
            {
                "ragas.llms": MagicMock(
                    LangchainLLMWrapper=MagicMock(return_value=MagicMock())
                ),
                "langchain_openai": MagicMock(
                    ChatOpenAI=MagicMock(return_value=MagicMock())
                ),
            },
        ):
            with patch(
                "askany.observability.ragas_eval._get_metric_classes",
                return_value={
                    "faithfulness": MagicMock(),
                    "response_relevancy": MagicMock(),
                    "context_precision": MagicMock(),
                },
            ):
                result = initialize_ragas(test_settings)

        assert result is True
        assert ragas_eval._initialized is True
        assert len(ragas_eval._metrics) == 3


class TestRAGASConfiguration:
    def test_ragas_config_defaults(self):
        assert settings.enable_ragas is False
        assert settings.ragas_sample_rate == 1.0
        assert "faithfulness" in settings.ragas_metrics
        assert "response_relevancy" in settings.ragas_metrics
        assert "context_precision" in settings.ragas_metrics

    def test_ragas_metrics_are_reference_free(self):
        reference_free_metrics = [
            "faithfulness",
            "response_relevancy",
            "context_precision",
        ]

        for metric in settings.ragas_metrics:
            assert metric in reference_free_metrics, (
                f"Metric {metric} may require reference"
            )


class TestSimpleAgentE2E:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        ragas_eval._initialized = False
        ragas_eval._metrics = {}
        ragas_eval._sample_rate = 1.0
        yield
        ragas_eval._initialized = False
        ragas_eval._metrics = {}
        ragas_eval._sample_rate = 1.0

    @pytest.mark.slow
    @pytest.mark.skipif(
        (not _check_prerequisites()) or (os.getenv("ASKANY_RUN_HEAVY_E2E") != "1"),
        reason=(
            f"Prerequisites not met: {skip_reason or 'unknown'} or "
            "ASKANY_RUN_HEAVY_E2E!=1"
        ),
    )
    def test_simple_agent_retrieves_and_evaluates(self):
        from askany.workflow.min_langchain_agent import (
            create_agent_with_tools,
            extract_and_format_response,
            extract_references_from_result,
            invoke_with_retry,
        )

        mock_faithfulness = _make_mock_metric("faithfulness", 0.85)
        ragas_eval._initialized = True
        ragas_eval._metrics = {"faithfulness": mock_faithfulness}
        ragas_eval._sample_rate = 1.0

        agent = create_agent_with_tools()
        query = "cassandra的concurrent_reads参数有什么用？"

        logger.info("Running query: %s", query)

        t0 = time.time()
        raw_result = invoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": query}]},
        )
        duration = time.time() - t0

        response_text = extract_and_format_response(raw_result)
        retrieved_contexts = extract_references_from_result(raw_result)

        logger.info("Response: %s...", response_text[:200])
        logger.info("Retrieved contexts: %d", len(retrieved_contexts))
        logger.info("Duration: %.1fs", duration)

        assert response_text, "Response should not be empty"
        assert isinstance(retrieved_contexts, list)

        for ctx in retrieved_contexts:
            logger.info("Context: %s...", ctx[:100])


class TestWorkflowAgentE2E:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        ragas_eval._initialized = False
        ragas_eval._metrics = {}
        ragas_eval._sample_rate = 1.0
        yield
        ragas_eval._initialized = False
        ragas_eval._metrics = {}
        ragas_eval._sample_rate = 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
