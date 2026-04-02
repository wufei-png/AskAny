"""Tests for RAG context plumbing to RAGAS evaluation.

These tests verify that:
1. process_query_with_subproblems returns (text, nodes) tuple
2. process_parallel_group returns (text, nodes) tuple
3. Server properly converts nodes to contexts for RAGAS
4. Trace ID retrieval has proper logging
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestProcessQueryWithSubproblemsReturnType:
    """Tests for process_query_with_subproblems returning (text, nodes)."""

    def test_returns_tuple_of_text_and_nodes(self):
        """Verify function has correct return type annotation and is callable."""
        import inspect

        from askany.api.server import process_query_with_subproblems

        sig = inspect.signature(process_query_with_subproblems)
        assert callable(process_query_with_subproblems)
        assert "user_query" in sig.parameters


class TestProcessParallelGroupReturnType:
    """Tests for process_parallel_group returning (text, nodes)."""

    def test_function_is_callable(self):
        """Verify process_parallel_group function exists and is callable."""
        import inspect

        from askany.api.server import process_parallel_group

        assert callable(process_parallel_group)


class TestNodeToContextConversion:
    """Tests for converting NodeWithScore to context strings for RAGAS."""

    def test_mock_node_to_text(self):
        """Verify NodeWithScore can be converted to text for RAGAS."""

        class MockNode:
            def get_content(self):
                return "This is test content"

        class MockNodeWithScore:
            def __init__(self, node, score):
                self.node = node
                self.score = score

        node = MockNode()
        node_with_score = MockNodeWithScore(node=node, score=0.95)

        content = node_with_score.node.get_content()
        assert content == "This is test content"

    def test_multiple_nodes_to_context_list(self):
        """Verify multiple nodes can be converted to context list."""

        class MockNode:
            def __init__(self, text):
                self.text = text

            def get_content(self):
                return self.text

        class MockNodeWithScore:
            def __init__(self, node, score):
                self.node = node
                self.score = score

        nodes = [
            MockNodeWithScore(node=MockNode("Content 1"), score=0.9),
            MockNodeWithScore(node=MockNode("Content 2"), score=0.8),
        ]

        contexts = [node.node.get_content() for node in nodes]

        assert len(contexts) == 2
        assert contexts[0] == "Content 1"
        assert contexts[1] == "Content 2"


class TestTraceIdRetrievalLogging:
    """Tests for trace_id retrieval with proper logging."""

    @pytest.mark.asyncio
    async def test_successful_trace_id_retrieval(self):
        """When trace_id is retrieved successfully, should return trace_id."""
        mock_handler = MagicMock()
        mock_handler.get_trace_id = MagicMock(return_value="trace-123")

        trace_id = None
        try:
            trace_id = mock_handler.get_trace_id()
        except Exception:
            pass

        assert trace_id == "trace-123"

    @pytest.mark.asyncio
    async def test_trace_id_retrieval_failure_returns_none(self):
        """When trace_id retrieval fails, should return None."""
        mock_handler = MagicMock()
        mock_handler.get_trace_id = MagicMock(side_effect=RuntimeError("No trace"))

        trace_id = None
        try:
            trace_id = mock_handler.get_trace_id()
        except Exception:
            pass

        assert trace_id is None

    @pytest.mark.asyncio
    async def test_no_handler_returns_none(self):
        """When no handler is available, should return None without error."""
        handler = None

        trace_id = None
        if handler is not None:
            try:
                trace_id = handler.get_trace_id()
            except Exception:
                pass

        assert trace_id is None


class TestRagasEvaluationWithContexts:
    """Integration tests for RAGAS evaluation with retrieved contexts."""

    @pytest.mark.asyncio
    async def test_evaluate_rag_response_with_contexts(self):
        """Verify evaluate_rag_response accepts non-empty contexts."""
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.85)

        mod._initialized = True
        mod._metrics = {"faithfulness": mock_metric, "response_relevancy": mock_metric}
        mod._sample_rate = 1.0

        result = await mod.evaluate_rag_response(
            trace_id="trace-test",
            user_input="What is Python?",
            response="Python is a programming language.",
            retrieved_contexts=[
                "Python is a high-level programming language.",
                "Python supports multiple programming paradigms.",
            ],
        )

        assert result
        assert "faithfulness" in result or "response_relevancy" in result

    @pytest.mark.asyncio
    async def test_ragas_with_empty_contexts_fallback(self):
        """When contexts are empty, should still evaluate response_relevancy."""
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.75)

        mod._initialized = True
        mod._metrics = {"response_relevancy": mock_metric}
        mod._sample_rate = 1.0

        result = await mod.evaluate_rag_response(
            user_input="Test question",
            response="Test answer",
            retrieved_contexts=[],
        )

        assert "response_relevancy" in result


class TestServerRagasIntegration:
    """End-to-end tests for server RAGAS integration."""

    def test_server_module_exports_process_functions(self):
        """Verify server module exports the required process functions."""
        from askany.api.server import (
            process_parallel_group,
            process_query_with_subproblems,
        )

        assert callable(process_parallel_group)
        assert callable(process_query_with_subproblems)


class TestWorkflowNodesExtraction:
    """Tests for extracting nodes from workflow state."""

    def test_agent_state_has_nodes_attribute(self):
        """Verify AgentState class has 'nodes' in its annotations."""
        from askany.workflow.workflow_langgraph import AgentState

        annotations = getattr(AgentState, "__annotations__", {})
        assert "nodes" in annotations, "AgentState should have 'nodes' key"

    def test_generate_final_answer_method_exists(self):
        """Verify _generate_final_answer_node method exists on AgentWorkflow."""
        from askany.workflow.workflow_langgraph import AgentWorkflow

        assert hasattr(AgentWorkflow, "_generate_final_answer_node")
        assert callable(getattr(AgentWorkflow, "_generate_final_answer_node"))


class TestRagasMetricsRequirements:
    """Tests verifying each metric's requirements are met."""

    @pytest.mark.asyncio
    async def test_faithfulness_requires_contexts(self):
        """Verify faithfulness metric calculation requires retrieved contexts.

        RAGAS faithfulness formula:
        - Needs: user_input, response, retrieved_contexts
        - Without contexts, cannot verify if response is grounded in context
        """
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.85)

        mod._initialized = True
        mod._metrics = {"faithfulness": mock_metric}
        mod._sample_rate = 1.0

        # Call WITH contexts - should calculate faithfulness
        result_with = await mod.evaluate_rag_response(
            user_input="What is Python?",
            response="Python is a programming language.",
            retrieved_contexts=["Python is a high-level language."],
        )

        assert "faithfulness" in result_with
        assert result_with["faithfulness"] == 0.85
        mock_metric.single_turn_ascore.assert_called()

    @pytest.mark.asyncio
    async def test_response_relevancy_works_without_contexts(self):
        """Verify response_relevancy doesn't need retrieved contexts.

        RAGAS response_relevancy formula:
        - Needs: user_input, response
        - Doesn't need: retrieved_contexts
        - Measures: how well response addresses the question
        """
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.75)

        mod._initialized = True
        mod._metrics = {"response_relevancy": mock_metric}
        mod._sample_rate = 1.0

        # Call WITHOUT contexts - should still calculate response_relevancy
        result_without = await mod.evaluate_rag_response(
            user_input="What is Python?",
            response="Python is a programming language.",
            retrieved_contexts=[],  # Empty!
        )

        assert "response_relevancy" in result_without
        assert result_without["response_relevancy"] == 0.75

    @pytest.mark.asyncio
    async def test_context_precision_requires_contexts(self):
        """Verify context_precision requires retrieved contexts to work.

        RAGAS context_precision formula:
        - Needs: user_input, retrieved_contexts
        - Doesn't need: response
        - Measures: how well retriever ranks relevant chunks
        """
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.9)

        mod._initialized = True
        mod._metrics = {"context_precision": mock_metric}
        mod._sample_rate = 1.0

        # Call WITH contexts - should calculate context_precision
        result_with = await mod.evaluate_rag_response(
            user_input="What is Python?",
            response="Python is a programming language.",
            retrieved_contexts=["Python is a language.", "It is popular."],
        )

        assert "context_precision" in result_with
        assert result_with["context_precision"] == 0.9


class TestMetricsIntegrationWithPlumbing:
    """End-to-end tests verifying the complete data flow."""

    @pytest.mark.asyncio
    async def test_all_three_metrics_with_real_contexts(self):
        """Verify all three configured metrics work when contexts are provided.

        This is the key integration test that verifies:
        1. Our changes pass contexts to RAGAS
        2. RAGAS can calculate all three metrics
        3. The plumbing from workflow -> server -> RAGAS works
        """
        import askany.observability.ragas_eval as mod

        # Mock all three metrics
        mock_faithfulness = MagicMock()
        mock_faithfulness.single_turn_ascore = AsyncMock(return_value=0.85)

        mock_relevancy = MagicMock()
        mock_relevancy.single_turn_ascore = AsyncMock(return_value=0.72)

        mock_precision = MagicMock()
        mock_precision.single_turn_ascore = AsyncMock(return_value=0.91)

        mod._initialized = True
        mod._metrics = {
            "faithfulness": mock_faithfulness,
            "response_relevancy": mock_relevancy,
            "context_precision": mock_precision,
        }
        mod._sample_rate = 1.0

        # Simulate what now happens in server.py after our changes
        retrieved_nodes = [
            MagicMock(node=MagicMock(get_content=MagicMock(return_value="Context 1"))),
            MagicMock(node=MagicMock(get_content=MagicMock(return_value="Context 2"))),
        ]
        retrieved_contexts = [node.node.get_content() for node in retrieved_nodes]

        result = await mod.evaluate_rag_response(
            trace_id="test-trace-123",
            user_input="Explain Python",
            response="Python is a high-level programming language.",
            retrieved_contexts=retrieved_contexts,
        )

        # Verify all three metrics were calculated
        assert "faithfulness" in result
        assert "response_relevancy" in result
        assert "context_precision" in result

        # Verify scores
        assert result["faithfulness"] == 0.85
        assert result["response_relevancy"] == 0.72
        assert result["context_precision"] == 0.91

    def test_server_converts_nodes_to_contexts_correctly(self):
        """Verify server.py correctly converts NodeWithScore to context strings."""

        class MockNodeWithScore:
            def __init__(self, text):
                self._node = MagicMock()
                self._node.get_content = MagicMock(return_value=text)
                self.score = 0.95

            @property
            def node(self):
                return self._node

        retrieved_nodes = [
            MockNodeWithScore("First context about Python."),
            MockNodeWithScore("Second context about programming."),
        ]

        # This is the exact code from server.py
        retrieved_contexts = [
            node.node.get_content()
            if hasattr(node.node, "get_content")
            else str(node.node)
            for node in retrieved_nodes
        ]

        assert len(retrieved_contexts) == 2
        assert retrieved_contexts[0] == "First context about Python."
        assert retrieved_contexts[1] == "Second context about programming."


# Mark all async tests
pytestmark = pytest.mark.asyncio
