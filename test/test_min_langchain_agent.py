#!/usr/bin/env python3
"""Tests for min_langchain_agent - LangChain agent utilities and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.schema import NodeWithScore, TextNode

from askany.workflow.min_langchain_agent import (
    extract_all_tool_calls,
    extract_and_format_response,
    extract_references_from_result,
    should_retry_model,
    _search_results_to_nodes,
    _get_overlap_content,
    _merge_nodes,
)


class TestShouldRetryModel:
    """Tests for should_retry_model function."""

    def test_retry_on_json_invalid_message(self):
        exc = RuntimeError("json_invalid: some error")

        assert should_retry_model(exc) is True

    def test_retry_on_eof_parsing_message(self):
        exc = RuntimeError("EOF while parsing object")

        assert should_retry_model(exc) is True

    def test_no_retry_on_other_exceptions(self):
        exc = ValueError("some value error")

        assert should_retry_model(exc) is False


class TestExtractAllToolCalls:
    """Tests for extract_all_tool_calls function."""

    def test_extract_from_dict_tool_calls(self):
        mock_message = MagicMock()
        mock_message.tool_calls = [
            {"name": "rag_search", "args": {"query": "test"}, "type": "tool_call"}
        ]

        mock_result = {"messages": [mock_message]}

        tool_calls = extract_all_tool_calls(mock_result)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "rag_search"
        assert tool_calls[0]["args"] == {"query": "test"}

    def test_extract_skips_final_summary_response(self):
        mock_message = MagicMock()
        mock_message.tool_calls = [
            {"name": "FinalSummaryResponse", "args": {}, "type": "tool_call"}
        ]

        mock_result = {"messages": [mock_message]}

        tool_calls = extract_all_tool_calls(mock_result)

        assert len(tool_calls) == 0

    def test_extract_from_object_tool_calls(self):
        mock_tool_call = MagicMock()
        mock_tool_call.name = "web_search"
        mock_tool_call.args = {"query": "weather"}
        mock_tool_call.type = "tool_call"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]

        mock_result = {"messages": [mock_message]}

        tool_calls = extract_all_tool_calls(mock_result)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "web_search"

    def test_extract_empty_when_no_messages(self):
        mock_result = {"messages": []}

        tool_calls = extract_all_tool_calls(mock_result)

        assert len(tool_calls) == 0

    def test_extract_empty_when_no_tool_calls(self):
        mock_message = MagicMock()
        mock_message.tool_calls = []

        mock_result = {"messages": [mock_message]}

        tool_calls = extract_all_tool_calls(mock_result)

        assert len(tool_calls) == 0


class TestExtractAndFormatResponse:
    """Tests for extract_and_format_response function."""

    def test_format_string_input(self):
        result = "direct string response"

        response = extract_and_format_response(result)

        assert response == "direct string response"

    def test_format_structured_response(self):
        from askany.workflow.min_langchain_agent import FinalSummaryResponse

        mock_structured = FinalSummaryResponse(
            summary_answer="The answer is 42.",
            references=["doc1.md", "doc2.md"],
        )

        mock_result = {"structured_response": mock_structured}

        response = extract_and_format_response(mock_result)

        assert "The answer is 42." in response
        assert "doc1.md" in response
        assert "doc2.md" in response

    def test_format_response_without_references(self):
        from askany.workflow.min_langchain_agent import FinalSummaryResponse

        mock_structured = FinalSummaryResponse(
            summary_answer="Simple answer.",
            references=[],
        )

        mock_result = {"structured_response": mock_structured}

        response = extract_and_format_response(mock_result)

        assert "Simple answer." in response
        assert "参考数据来源" not in response

    def test_format_response_with_message_fallback(self):
        mock_message = MagicMock()
        mock_message.content = "Message content"

        mock_result = {"messages": [mock_message]}

        response = extract_and_format_response(mock_result)

        assert "Message content" in response

    def test_format_response_adds_tool_calls(self):
        from askany.workflow.min_langchain_agent import FinalSummaryResponse

        mock_message = MagicMock()
        mock_message.tool_calls = [
            {"name": "rag_search", "args": {"query": "test"}, "type": "tool_call"}
        ]
        mock_message.content = "Result"

        mock_structured = FinalSummaryResponse(
            summary_answer="The answer.",
            references=[],
        )

        mock_result = {
            "structured_response": mock_structured,
            "messages": [mock_message],
        }

        response = extract_and_format_response(mock_result)

        assert "rag_search" in response
        assert "工具调用" in response


class TestExtractReferencesFromResult:
    """Tests for extract_references_from_result function."""

    def test_extract_references_from_structured_response(self):
        from askany.workflow.min_langchain_agent import FinalSummaryResponse

        mock_structured = FinalSummaryResponse(
            summary_answer="Answer",
            references=["ref1.md", "ref2.md"],
        )

        mock_result = {"structured_response": mock_structured}

        refs = extract_references_from_result(mock_result)

        assert refs == ["ref1.md", "ref2.md"]

    def test_extract_empty_references(self):
        from askany.workflow.min_langchain_agent import FinalSummaryResponse

        mock_structured = FinalSummaryResponse(
            summary_answer="Answer",
            references=[],
        )

        mock_result = {"structured_response": mock_structured}

        refs = extract_references_from_result(mock_result)

        assert refs == []

    def test_extract_empty_on_invalid_input(self):
        refs = extract_references_from_result("string input")

        assert refs == []


class TestSearchResultsToNodes:
    """Tests for _search_results_to_nodes helper."""

    def test_convert_search_results_to_nodes(self):
        mock_local_file_search = MagicMock()

        search_results = {
            "API": [
                {
                    "content": "API content line",
                    "file_path": "/path/to/doc.md",
                    "start_line": 10,
                    "end_line": 15,
                }
            ]
        }

        nodes = _search_results_to_nodes(search_results, mock_local_file_search)

        assert len(nodes) == 1
        assert nodes[0].node.text == "API content line"
        assert nodes[0].node.metadata["file_path"] == "/path/to/doc.md"
        assert nodes[0].node.metadata["start_line"] == 10
        assert nodes[0].node.metadata["end_line"] == 15
        assert nodes[0].node.metadata["keyword"] == "API"
        assert nodes[0].score == 0.8

    def test_convert_multiple_keywords(self):
        mock_local_file_search = MagicMock()

        search_results = {
            "API": [
                {
                    "content": "API content",
                    "file_path": "/path/to/doc.md",
                    "start_line": 10,
                    "end_line": 15,
                }
            ],
            "config": [
                {
                    "content": "config content",
                    "file_path": "/path/to/doc2.md",
                    "start_line": 20,
                    "end_line": 25,
                }
            ],
        }

        nodes = _search_results_to_nodes(search_results, mock_local_file_search)

        assert len(nodes) == 2


class TestGetOverlapContent:
    """Tests for _get_overlap_content helper."""

    def test_get_overlap_from_file(self):
        mock_local_file_search = MagicMock()
        mock_local_file_search.get_file_content_by_lines.return_value = "file content"

        mock_node = MagicMock()
        mock_node.node.metadata = {}

        result = _get_overlap_content(
            "/path/to/file.md", 10, 20, mock_node, mock_local_file_search
        )

        assert result == "file content"
        mock_local_file_search.get_file_content_by_lines.assert_called_once()

    def test_get_overlap_returns_none_when_file_read_fails(self):
        mock_local_file_search = MagicMock()
        mock_local_file_search.get_file_content_by_lines.side_effect = Exception(
            "Read error"
        )

        mock_node = MagicMock()
        mock_node.node.get_content.return_value = None
        mock_node.node.metadata = {}

        result = _get_overlap_content(
            "/path/to/file.md", 10, 20, mock_node, mock_local_file_search
        )

        assert result is None


class TestMergeNodes:
    """Tests for _merge_nodes helper."""

    def test_merge_empty_lists(self):
        mock_local_file_search = MagicMock()

        result = _merge_nodes([], [], mock_local_file_search)

        assert result == []

    def test_merge_single_node(self):
        mock_local_file_search = MagicMock()

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="content",
                    metadata={"file_path": "/path/to/doc.md"},
                ),
                score=0.9,
            )
        ]

        result = _merge_nodes(nodes, [], mock_local_file_search)

        assert len(result) == 1

    def test_merge_no_overlap(self):
        mock_local_file_search = MagicMock()

        existing = [
            NodeWithScore(
                node=TextNode(
                    text="content1",
                    metadata={
                        "file_path": "/path/to/doc1.md",
                        "start_line": 10,
                        "end_line": 20,
                    },
                ),
                score=0.9,
            )
        ]

        new_nodes = [
            NodeWithScore(
                node=TextNode(
                    text="content2",
                    metadata={
                        "file_path": "/path/to/doc2.md",
                        "start_line": 30,
                        "end_line": 40,
                    },
                ),
                score=0.8,
            )
        ]

        result = _merge_nodes(existing, new_nodes, mock_local_file_search)

        assert len(result) == 2

    def test_merge_preserves_nodes_without_path(self):
        mock_local_file_search = MagicMock()

        existing = [
            NodeWithScore(node=TextNode(text="content without path"), score=0.9)
        ]

        result = _merge_nodes(existing, [], mock_local_file_search)

        assert len(result) == 1
        assert result[0].node.text == "content without path"

    def test_merge_combines_nodes_from_both_lists(self):
        mock_local_file_search = MagicMock()

        existing = [
            NodeWithScore(
                node=TextNode(
                    text="existing content",
                    metadata={"file_path": "/path/to/doc1.md"},
                ),
                score=0.9,
            )
        ]

        new_nodes = [
            NodeWithScore(
                node=TextNode(
                    text="new content",
                    metadata={"file_path": "/path/to/doc2.md"},
                ),
                score=0.8,
            )
        ]

        result = _merge_nodes(existing, new_nodes, mock_local_file_search)

        assert len(result) == 2
