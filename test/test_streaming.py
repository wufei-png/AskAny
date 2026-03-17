#!/usr/bin/env python3
"""Tests for SSE streaming support in AskAny.

Tests cover:
1. SSE chunk formatting (_make_sse_chunk)
2. SSE generator wrapper (_sse_generator)
3. Streaming endpoint integration via FastAPI TestClient
"""

import json
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.api.server import _make_sse_chunk, _sse_generator

# ─── Unit tests for SSE helpers ───


class TestMakeSseChunk:
    """Test _make_sse_chunk formatting."""

    def test_content_chunk(self):
        """Content chunk should include delta.content."""
        line = _make_sse_chunk("test-model", content="Hello")
        assert line.startswith("data: ")
        assert line.endswith("\n\n")

        payload = json.loads(line[len("data: ") :].strip())
        assert payload["object"] == "chat.completion.chunk"
        assert payload["model"] == "test-model"
        assert payload["choices"][0]["delta"]["content"] == "Hello"
        assert payload["choices"][0]["finish_reason"] is None

    def test_finish_reason_chunk(self):
        """Finish chunk should have finish_reason='stop' and empty delta."""
        line = _make_sse_chunk("test-model", finish_reason="stop")
        payload = json.loads(line[len("data: ") :].strip())
        assert payload["choices"][0]["finish_reason"] == "stop"
        assert payload["choices"][0]["delta"] == {}

    def test_custom_chunk_id(self):
        """Custom chunk_id should be used."""
        line = _make_sse_chunk("m", content="x", chunk_id="custom-123")
        payload = json.loads(line[len("data: ") :].strip())
        assert payload["id"] == "custom-123"

    def test_auto_generated_chunk_id(self):
        """Auto-generated chunk_id should start with chatcmpl-."""
        line = _make_sse_chunk("m", content="x")
        payload = json.loads(line[len("data: ") :].strip())
        assert payload["id"].startswith("chatcmpl-")

    def test_empty_content_is_included(self):
        """Empty string content should still be included in delta."""
        line = _make_sse_chunk("m", content="")
        payload = json.loads(line[len("data: ") :].strip())
        assert "content" in payload["choices"][0]["delta"]
        assert payload["choices"][0]["delta"]["content"] == ""

    def test_none_content_excluded(self):
        """None content should NOT appear in delta."""
        line = _make_sse_chunk("m")
        payload = json.loads(line[len("data: ") :].strip())
        assert "content" not in payload["choices"][0]["delta"]


# ─── Tests for _sse_generator ───


class TestSseGenerator:
    """Test _sse_generator wrapping logic."""

    @pytest.mark.asyncio
    async def test_basic_streaming(self):
        """Should yield role chunk, content chunks, stop chunk, and [DONE]."""

        async def content_gen() -> AsyncGenerator[str, None]:
            yield "Hello"
            yield " world"

        chunks = []
        async for chunk in _sse_generator("test-model", content_gen()):
            chunks.append(chunk)

        # Should be: role chunk, "Hello" chunk, " world" chunk, stop chunk, [DONE]
        assert len(chunks) == 5

        # First chunk: role
        role_payload = json.loads(chunks[0][len("data: ") :].strip())
        assert role_payload["choices"][0]["delta"]["role"] == "assistant"

        # Content chunks
        c1 = json.loads(chunks[1][len("data: ") :].strip())
        assert c1["choices"][0]["delta"]["content"] == "Hello"

        c2 = json.loads(chunks[2][len("data: ") :].strip())
        assert c2["choices"][0]["delta"]["content"] == " world"

        # Stop chunk
        stop_payload = json.loads(chunks[3][len("data: ") :].strip())
        assert stop_payload["choices"][0]["finish_reason"] == "stop"

        # [DONE]
        assert chunks[4] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_empty_chunks_skipped(self):
        """Empty/falsy content chunks should be skipped."""

        async def content_gen() -> AsyncGenerator[str, None]:
            yield ""
            yield "data"
            yield ""

        chunks = []
        async for chunk in _sse_generator("m", content_gen()):
            chunks.append(chunk)

        # role + "data" + stop + [DONE] = 4
        assert len(chunks) == 4

    @pytest.mark.asyncio
    async def test_consistent_chunk_ids(self):
        """All chunks from one stream should share the same id."""

        async def content_gen() -> AsyncGenerator[str, None]:
            yield "a"
            yield "b"

        ids = set()
        async for chunk in _sse_generator("m", content_gen()):
            if chunk.startswith("data: {"):
                payload = json.loads(chunk[len("data: ") :].strip())
                ids.add(payload["id"])

        assert len(ids) == 1, f"Expected consistent chunk ids, got: {ids}"

    @pytest.mark.asyncio
    async def test_error_in_stream_still_sends_done(self):
        """If the content generator raises, stop+[DONE] should still be sent."""

        async def failing_gen() -> AsyncGenerator[str, None]:
            yield "ok"
            raise RuntimeError("boom")

        chunks = []
        async for chunk in _sse_generator("m", failing_gen()):
            chunks.append(chunk)

        # role + "ok" + stop + [DONE] = 4 (error is caught, stop/DONE still sent)
        assert chunks[-1] == "data: [DONE]\n\n"
        stop_payload = json.loads(chunks[-2][len("data: ") :].strip())
        assert stop_payload["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_no_content_still_sends_stop_and_done(self):
        """Even with zero content chunks, stop+[DONE] should be sent."""

        async def empty_gen() -> AsyncGenerator[str, None]:
            return
            yield  # noqa: F811 - makes this an async generator

        chunks = []
        async for chunk in _sse_generator("m", empty_gen()):
            chunks.append(chunk)

        # stop + [DONE] = 2 (no role chunk since first is never True-set)
        assert len(chunks) == 2
        assert chunks[-1] == "data: [DONE]\n\n"


# ─── Integration tests for streaming endpoint ───


class TestStreamingEndpoint:
    """Test the /v1/chat/completions endpoint with stream=True.

    Uses FastAPI TestClient with mocked workflow/agent globals.
    """

    def _create_test_app(self):
        """Create a test app with mocked globals."""
        from askany.api.server import create_app
        from askany.rag.router import QueryRouter

        # Create minimal mocks
        mock_router = MagicMock(spec=QueryRouter)
        mock_workflow = MagicMock()
        mock_filter = MagicMock()
        mock_agent = MagicMock()

        app = create_app(
            query_router=mock_router,
            agent_workflow=mock_workflow,
            workflow_filter=mock_filter,
            simple_agent=mock_agent,
        )
        return app, mock_workflow, mock_agent

    @pytest.mark.asyncio
    async def test_stream_simple_agent(self):
        """Streaming with simple agent should return SSE events."""
        app, mock_workflow, mock_agent = self._create_test_app()

        async def mock_astream(*args, **kwargs):
            yield "Hello"
            yield " from"
            yield " agent"

        with (
            patch("askany.api.server.get_mem0_adapter", return_value=None),
            patch(
                "askany.workflow.min_langchain_agent.astream_agent_response",
                side_effect=mock_astream,
            ),
        ):
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

                # Parse SSE lines
                body = response.text
                lines = [line for line in body.split("\n") if line.startswith("data: ")]
                assert len(lines) >= 3  # at least: role + content + stop + [DONE]

                # Last data line should be [DONE]
                assert lines[-1] == "data: [DONE]"

                # Second-to-last should have finish_reason=stop
                stop_data = json.loads(lines[-2][len("data: ") :])
                assert stop_data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_stream_deepsearch(self):
        """Streaming with deepsearch model should use AgentWorkflow.astream_final_answer."""
        app, mock_workflow, mock_agent = self._create_test_app()

        async def mock_astream_final(*args, **kwargs):
            yield "Deep"
            yield " answer"

        mock_workflow.astream_final_answer = mock_astream_final

        with patch("askany.api.server.get_mem0_adapter", return_value=None):
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model-deepsearch",
                        "messages": [{"role": "user", "content": "deep question"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

                body = response.text
                lines = [line for line in body.split("\n") if line.startswith("data: ")]

                # Verify content appears
                content_parts = []
                for line in lines:
                    if line == "data: [DONE]":
                        continue
                    payload = json.loads(line[len("data: ") :])
                    delta = payload["choices"][0]["delta"]
                    if "content" in delta and delta["content"]:
                        content_parts.append(delta["content"])

                assert "Deep" in content_parts
                assert " answer" in content_parts

    @pytest.mark.asyncio
    async def test_non_stream_still_works(self):
        """Non-streaming request should still return normal JSON response."""
        app, mock_workflow, mock_agent = self._create_test_app()

        with (
            patch("askany.api.server.get_mem0_adapter", return_value=None),
            patch(
                "askany.workflow.min_langchain_agent.invoke_with_retry",
                return_value={"messages": [MagicMock(content="normal response")]},
            ),
            patch(
                "askany.workflow.min_langchain_agent.extract_and_format_response",
                return_value="normal response",
            ),
        ):
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert data["object"] == "chat.completion"
                assert data["choices"][0]["message"]["content"] == "normal response"


# ─── OpenAI-compatibility format tests ───


class TestOpenAICompatibility:
    """Verify SSE output matches the OpenAI streaming specification."""

    @pytest.mark.asyncio
    async def test_chunk_format_matches_openai_spec(self):
        """Each chunk must have: id, object, created, model, choices."""

        async def gen() -> AsyncGenerator[str, None]:
            yield "test"

        async for chunk in _sse_generator("gpt-4", gen()):
            if chunk.startswith("data: {"):
                payload = json.loads(chunk[len("data: ") :].strip())
                assert "id" in payload
                assert payload["object"] == "chat.completion.chunk"
                assert "created" in payload
                assert payload["model"] == "gpt-4"
                assert "choices" in payload
                assert len(payload["choices"]) == 1
                assert "index" in payload["choices"][0]
                assert "delta" in payload["choices"][0]
                assert "finish_reason" in payload["choices"][0]

    @pytest.mark.asyncio
    async def test_done_sentinel(self):
        """Stream must end with 'data: [DONE]\\n\\n'."""

        async def gen() -> AsyncGenerator[str, None]:
            yield "x"

        chunks = []
        async for chunk in _sse_generator("m", gen()):
            chunks.append(chunk)

        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_first_chunk_has_role(self):
        """First SSE chunk should include delta.role='assistant'."""

        async def gen() -> AsyncGenerator[str, None]:
            yield "hi"

        chunks = []
        async for chunk in _sse_generator("m", gen()):
            chunks.append(chunk)

        first = json.loads(chunks[0][len("data: ") :].strip())
        assert first["choices"][0]["delta"]["role"] == "assistant"


# ─── Additional edge case tests ───


class TestSseEdgeCases:
    """Additional edge case tests for SSE streaming."""

    @pytest.mark.asyncio
    async def test_large_content_chunk(self):
        """Should handle large content chunks correctly."""
        large_content = "x" * 10000  # 10KB content

        async def gen() -> AsyncGenerator[str, None]:
            yield large_content

        chunks = []
        async for chunk in _sse_generator("m", gen()):
            chunks.append(chunk)

        # role + content + stop + [DONE]
        assert len(chunks) == 4

        content_chunk = json.loads(chunks[1][len("data: ") :].strip())
        assert content_chunk["choices"][0]["delta"]["content"] == large_content

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self):
        """Should handle special characters correctly."""
        special_content = "Hello 世界 🌍\n\t\r\"'{}[]"

        async def gen() -> AsyncGenerator[str, None]:
            yield special_content

        chunks = []
        async for chunk in _sse_generator("m", gen()):
            chunks.append(chunk)

        content_chunk = json.loads(chunks[1][len("data: ") :].strip())
        assert content_chunk["choices"][0]["delta"]["content"] == special_content

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Should handle unicode content correctly."""
        unicode_content = "中文内容 🎉 émojis and Ünicode"

        async def gen() -> AsyncGenerator[str, None]:
            yield unicode_content

        chunks = []
        async for chunk in _sse_generator("m", gen()):
            chunks.append(chunk)

        content_chunk = json.loads(chunks[1][len("data: ") :].strip())
        assert content_chunk["choices"][0]["delta"]["content"] == unicode_content

    @pytest.mark.asyncio
    async def test_multiple_rapid_chunks(self):
        """Should handle many rapid sequential chunks."""

        async def gen() -> AsyncGenerator[str, None]:
            for i in range(100):
                yield f"chunk{i},"

        chunks = []
        async for chunk in _sse_generator("m", gen()):
            chunks.append(chunk)

        # role + 100 content + stop + [DONE] = 103
        assert len(chunks) == 103

    @pytest.mark.asyncio
    async def test_created_timestamp_consistency(self):
        """All chunks should have consistent created timestamp."""

        async def gen() -> AsyncGenerator[str, None]:
            yield "a"
            yield "b"

        chunks = []
        async for chunk in _sse_generator("m", gen()):
            chunks.append(chunk)

        timestamps = set()
        for chunk in chunks:
            if chunk.startswith("data: {"):
                payload = json.loads(chunk[len("data: ") :].strip())
                timestamps.add(payload["created"])

        assert len(timestamps) == 1, "All chunks should have same created timestamp"

    @pytest.mark.asyncio
    async def test_model_name_echo(self):
        """Model name should be echoed in all chunks."""
        test_model = "custom-model-name"

        async def gen() -> AsyncGenerator[str, None]:
            yield "test"

        chunks = []
        async for chunk in _sse_generator(test_model, gen()):
            chunks.append(chunk)

        for chunk in chunks:
            if chunk.startswith("data: {"):
                payload = json.loads(chunk[len("data: ") :].strip())
                assert payload["model"] == test_model

    @pytest.mark.asyncio
    async def test_chunk_index_always_zero(self):
        """Choice index should always be 0."""

        async def gen() -> AsyncGenerator[str, None]:
            yield "content"

        chunks = []
        async for chunk in _sse_generator("m", gen()):
            chunks.append(chunk)

        for chunk in chunks:
            if chunk.startswith("data: {"):
                payload = json.loads(chunk[len("data: ") :].strip())
                assert payload["choices"][0]["index"] == 0


class TestStreamingEndpointEdgeCases:
    """Additional edge case tests for streaming endpoint."""

    @pytest.mark.asyncio
    async def test_stream_with_empty_messages(self):
        """Should handle empty messages list."""
        app, _, _ = self._create_test_app()

        with patch("askany.api.server.get_mem0_adapter", return_value=None):
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [],
                        "stream": True,
                    },
                )
                # Server returns 400 for empty messages (no user message found)
                assert response.status_code == 400

    def _create_test_app(self):
        """Create a test app with mocked globals."""
        from askany.api.server import create_app
        from askany.rag.router import QueryRouter

        mock_router = MagicMock(spec=QueryRouter)
        mock_workflow = MagicMock()
        mock_filter = MagicMock()
        mock_agent = MagicMock()

        app = create_app(
            query_router=mock_router,
            agent_workflow=mock_workflow,
            workflow_filter=mock_filter,
            simple_agent=mock_agent,
        )
        return app, mock_workflow, mock_agent
