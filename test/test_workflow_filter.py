#!/usr/bin/env python3
"""Tests for the pre-processing workflow filter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from askany.workflow.workflow_filter import WorkflowFilter


def make_filter() -> tuple[WorkflowFilter, MagicMock, MagicMock, MagicMock, MagicMock]:
    direct_answer_generator = MagicMock()
    web_or_rag_generator = MagicMock()
    final_answer_generator = MagicMock()
    web_search_tool = MagicMock()
    reranker = MagicMock()

    final_answer_generator.generate_final_answer.return_value = (
        "answer",
        "reasoning",
    )
    web_search_tool.search.return_value = ["node"]
    reranker.postprocess_nodes.return_value = ["ranked node"]

    workflow_filter = WorkflowFilter(
        direct_answer_generator=direct_answer_generator,
        web_or_rag_generator=web_or_rag_generator,
        final_answer_generator=final_answer_generator,
        web_search_tool=web_search_tool,
        reranker=reranker,
    )
    return (
        workflow_filter,
        direct_answer_generator,
        web_or_rag_generator,
        final_answer_generator,
        web_search_tool,
    )


def test_url_query_uses_web_search_without_running_other_checks():
    (
        workflow_filter,
        direct_answer_generator,
        web_or_rag_generator,
        final_answer_generator,
        web_search_tool,
    ) = make_filter()

    result = workflow_filter.process("请总结 https://example.com/article")

    assert result.have_result is True
    assert result.need_web_search is True
    assert result.need_rag_search is False
    assert result.result == "answer\n\nreasoning"
    direct_answer_generator.generate.assert_not_called()
    web_or_rag_generator.generate.assert_not_called()
    web_search_tool.search.assert_called_once_with("请总结 https://example.com/article")
    final_answer_generator.generate_final_answer.assert_called_once_with(
        "请总结 https://example.com/article", ["ranked node"]
    )


def test_web_only_decision_uses_the_same_search_flow():
    (
        workflow_filter,
        direct_answer_generator,
        web_or_rag_generator,
        final_answer_generator,
        web_search_tool,
    ) = make_filter()
    direct_answer_generator.generate.return_value = SimpleNamespace(
        can_direct_answer=False
    )
    web_or_rag_generator.generate.return_value = SimpleNamespace(
        need_web_search=True,
        need_rag_search=False,
    )

    result = workflow_filter.process("latest news")

    assert result.have_result is True
    assert result.need_web_search is True
    assert result.need_rag_search is False
    assert result.result == "answer\n\nreasoning"
    web_search_tool.search.assert_called_once_with("latest news")
    final_answer_generator.generate_final_answer.assert_called_once_with(
        "latest news", ["ranked node"]
    )
