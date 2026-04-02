#!/usr/bin/env python3
"""Tests for QueryRouter - query routing logic for FAQ/DOCS/AUTO."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from askany.rag.router import QueryRouter, QueryType


class TestQueryType:
    """Tests for QueryType enum."""

    def test_query_type_values(self):
        assert QueryType.AUTO == "auto"
        assert QueryType.FAQ == "faq"
        assert QueryType.DOCS == "docs"
        assert QueryType.CODE == "code"

    def test_query_type_is_string(self):
        assert isinstance(QueryType.AUTO, str)
        assert isinstance(QueryType.FAQ, str)


class TestQueryRouterRoute:
    """Tests for QueryRouter.route() method."""

    def test_route_faq_query(self):
        mock_docs_engine = MagicMock()
        mock_faq_engine = MagicMock()
        mock_faq_engine.query.return_value = "FAQ answer"

        router = QueryRouter(
            docs_query_engine=mock_docs_engine,
            faq_query_engine=mock_faq_engine,
        )

        with patch("askany.rag.router.settings") as mock_settings:
            mock_settings.enable_qa_cache = False
            result = router.route("test query", query_type=QueryType.FAQ)

        assert result == "FAQ answer"
        mock_faq_engine.query.assert_called_once()

    def test_route_docs_query(self):
        mock_docs_engine = MagicMock()
        mock_faq_engine = MagicMock()
        mock_docs_engine.query.return_value = "DOCS answer"

        router = QueryRouter(
            docs_query_engine=mock_docs_engine,
            faq_query_engine=mock_faq_engine,
        )

        with patch("askany.rag.router.settings") as mock_settings:
            mock_settings.enable_qa_cache = False
            result = router.route("test query", query_type=QueryType.DOCS)

        assert result == "DOCS answer"
        mock_docs_engine.query.assert_called_once()

    def test_route_code_query_not_implemented(self):
        mock_docs_engine = MagicMock()
        mock_faq_engine = MagicMock()

        router = QueryRouter(
            docs_query_engine=mock_docs_engine,
            faq_query_engine=mock_faq_engine,
        )

        with patch("askany.rag.router.settings") as mock_settings:
            mock_settings.enable_qa_cache = False
            result = router.route("test code query", query_type=QueryType.CODE)

        assert "not yet implemented" in result

    def test_route_auto_fallback_to_docs_without_faq_engine(self):
        mock_docs_engine = MagicMock()
        mock_docs_engine.query.return_value = "DOCS fallback"

        router = QueryRouter(
            docs_query_engine=mock_docs_engine,
            faq_query_engine=None,
        )

        with patch.object(router, "_route_auto", wraps=router._route_auto):
            with patch("askany.rag.router.settings") as mock_settings:
                mock_settings.enable_qa_cache = False
                result = router.route("test query", query_type=QueryType.AUTO)

        assert result == "DOCS fallback"


class TestQueryRouterIsCodeQuery:
    """Tests for QueryRouter._is_code_query() method."""

    def test_is_code_query_with_code_keywords(self):
        router = QueryRouter(
            docs_query_engine=MagicMock(),
            faq_query_engine=MagicMock(),
        )

        code_queries = [
            "how to write this code",
            "python function example",
            "class inheritance in java",
            "import numpy as np",
        ]

        for query in code_queries:
            assert router._is_code_query(query) is True, f"Expected True for: {query}"

    def test_is_code_query_with_non_code_queries(self):
        router = QueryRouter(
            docs_query_engine=MagicMock(),
            faq_query_engine=MagicMock(),
        )

        non_code_queries = [
            "how to configure API",
            "what is the meaning of life",
            "如何配置API",
        ]

        for query in non_code_queries:
            assert router._is_code_query(query) is False, f"Expected False for: {query}"


class TestQueryRouterGetNodeId:
    """Tests for QueryRouter._get_node_id() method."""

    def test_get_node_id_prefers_node_id(self):
        router = QueryRouter(
            docs_query_engine=MagicMock(),
            faq_query_engine=MagicMock(),
        )

        mock_node = MagicMock()
        mock_node.node.node_id = "preferred-id"
        mock_node.node.id_ = "fallback-id"
        mock_node.score = 0.9

        node_id = router._get_node_id(mock_node)
        assert node_id == "preferred-id"

    def test_get_node_id_fallback_to_id(self):
        router = QueryRouter(
            docs_query_engine=MagicMock(),
            faq_query_engine=MagicMock(),
        )

        mock_node = MagicMock()
        mock_node.node.node_id = None
        mock_node.node.id_ = "fallback-id"
        mock_node.score = 0.9

        node_id = router._get_node_id(mock_node)
        assert node_id == "fallback-id"


class TestQueryRouterMarkFaqNodesWithLowReliability:
    """Tests for QueryRouter._mark_faq_nodes_with_low_reliability() method."""

    def test_mark_faq_nodes_adds_prefix_to_text(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        router = QueryRouter(
            docs_query_engine=MagicMock(),
            faq_query_engine=MagicMock(),
        )

        original_text = "Original FAQ content"
        node = NodeWithScore(
            node=TextNode(text=original_text),
            score=0.85,
        )

        with patch("askany.rag.router.settings") as mock_settings:
            mock_settings.faq_score_threshold = 0.90
            marked_nodes = router._mark_faq_nodes_with_low_reliability([node], 0.85)

        assert len(marked_nodes) == 1
        marked_node = marked_nodes[0]
        assert "[FAQ-低相关性" in marked_node.node.text
        assert original_text in marked_node.node.text

    def test_mark_faq_nodes_reduces_score(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        router = QueryRouter(
            docs_query_engine=MagicMock(),
            faq_query_engine=MagicMock(),
        )

        node = NodeWithScore(
            node=TextNode(text="FAQ content"),
            score=0.80,
        )

        with patch("askany.rag.router.settings") as mock_settings:
            mock_settings.faq_score_threshold = 0.90
            marked_nodes = router._mark_faq_nodes_with_low_reliability([node], 0.80)

        assert marked_nodes[0].score == 0.40

    def test_mark_faq_nodes_does_not_modify_original(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        router = QueryRouter(
            docs_query_engine=MagicMock(),
            faq_query_engine=MagicMock(),
        )

        original_text = "Original FAQ content"
        original_score = 0.80
        node = NodeWithScore(
            node=TextNode(text=original_text),
            score=original_score,
        )

        with patch("askany.rag.router.settings") as mock_settings:
            mock_settings.faq_score_threshold = 0.90
            router._mark_faq_nodes_with_low_reliability([node], 0.80)

        assert node.node.text == original_text
        assert node.score == original_score


class TestQueryRouterAutoRouting:
    """Tests for QueryRouter._route_auto() method."""

    def test_auto_route_uses_faq_when_score_sufficient(self):
        mock_docs_engine = MagicMock()
        mock_faq_engine = MagicMock()
        mock_faq_engine.retrieve_with_scores.return_value = (
            [MagicMock()],
            0.95,
        )
        mock_faq_engine.synthesize_from_nodes.return_value = "FAQ synthesis"

        router = QueryRouter(
            docs_query_engine=mock_docs_engine,
            faq_query_engine=mock_faq_engine,
        )

        with patch("askany.rag.router.settings") as mock_settings:
            mock_settings.enable_qa_cache = False
            mock_settings.faq_score_threshold = 0.90
            result = router._route_auto("test query")

        assert result == "FAQ synthesis"
        mock_faq_engine.synthesize_from_nodes.assert_called_once()

    def test_auto_route_enhances_with_docs_when_score_insufficient(self):
        mock_docs_engine = MagicMock()
        mock_faq_engine = MagicMock()
        mock_faq_engine.retrieve_with_scores.return_value = (
            [MagicMock()],
            0.70,
        )
        mock_docs_engine.retrieve.return_value = [MagicMock()]
        mock_docs_engine.synthesize_from_nodes.return_value = "Enhanced docs synthesis"

        router = QueryRouter(
            docs_query_engine=mock_docs_engine,
            faq_query_engine=mock_faq_engine,
        )

        with patch("askany.rag.router.settings") as mock_settings:
            mock_settings.enable_qa_cache = False
            mock_settings.faq_score_threshold = 0.90
            mock_settings.faq_second_score_threshold = 0.70
            result = router._route_auto("test query")

        assert result == "Enhanced docs synthesis"
        mock_docs_engine.retrieve.assert_called_once()


class TestGetDevice:
    """Tests for get_device() function."""

    def test_get_device_returns_cuda_when_available(self):
        with patch("askany.rag.router.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.AVAILABLE = True

            from askany.rag.router import get_device

            result = get_device()
            assert result == "cuda"

    def test_get_device_returns_cpu_when_cuda_unavailable(self):
        with patch("askany.rag.router.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.AVAILABLE = False

            from askany.rag.router import get_device

            result = get_device()
            assert result == "cpu"
