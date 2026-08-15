"""Unit tests for LightRAG retrieval error handling and diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lightrag_test_support import (
    is_lightrag_prerequisite_error,
    skip_if_no_lightrag_data,
)

import askany.rag.lightrag_adapter as lightrag_adapter
from askany.rag.lightrag_adapter import LightRAGAdapter


class _FakeMetric:
    def labels(self, **_labels: object) -> _FakeMetric:
        return self

    def inc(self) -> None:
        return None

    def observe(self, _value: float) -> None:
        return None


class _FakeMetrics:
    askany_lightrag_retrieval_total = _FakeMetric()
    askany_lightrag_retrieval_duration_seconds = _FakeMetric()


class _FakeQueryParam:
    def __init__(self, **_kwargs: object) -> None:
        pass


class _FakeRag:
    def __init__(
        self, *, result: object = None, error: BaseException | None = None
    ) -> None:
        self.result = result
        self.error = error

    async def aquery_data(self, _query: str, *, param: Any) -> object:
        if self.error is not None:
            raise self.error
        return self.result


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: object = None,
    error: BaseException | None = None,
) -> LightRAGAdapter:
    monkeypatch.setattr(lightrag_adapter, "_LIGHTRAG_AVAILABLE", True)
    monkeypatch.setattr(lightrag_adapter, "QueryParam", _FakeQueryParam)
    monkeypatch.setattr(lightrag_adapter, "_LANGFUSE_ENABLED", False)
    monkeypatch.setattr(lightrag_adapter, "_langfuse_client", None)
    monkeypatch.setattr(lightrag_adapter, "get_metrics", _FakeMetrics)

    adapter = object.__new__(LightRAGAdapter)
    adapter._rag = _FakeRag(result=result, error=error)
    adapter._initialized = True
    adapter._default_mode = "mix"
    adapter._top_k = 60
    adapter._chunk_top_k = 10
    adapter._max_total_tokens = 16000
    adapter._convert_to_nodes = lambda _result, _query: []
    return adapter


@pytest.mark.asyncio
async def test_query_exception_returns_empty_list_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch, error=TypeError("query regression"))

    assert await adapter.retrieve_async("question") == []


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("provider is unreachable"),
        openai.APIConnectionError(request=None),
    ],
)
@pytest.mark.asyncio
async def test_diagnostic_query_external_error_is_re_raised_for_classification(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    adapter = _make_adapter(monkeypatch, error=error)

    with pytest.raises(type(error)):
        await adapter.retrieve_async("question", raise_on_error=True)
    assert is_lightrag_prerequisite_error(error) is True


@pytest.mark.parametrize(
    "error",
    [
        TypeError("unexpected provider API"),
        AttributeError("missing query method"),
    ],
)
@pytest.mark.asyncio
async def test_diagnostic_query_programming_error_is_re_raised(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    adapter = _make_adapter(monkeypatch, error=error)

    with pytest.raises(type(error)):
        await adapter.retrieve_async("question", raise_on_error=True)
    assert is_lightrag_prerequisite_error(error) is False


@pytest.mark.asyncio
async def test_diagnostic_empty_result_is_explicit_no_data_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch, result={})

    nodes = await adapter.retrieve_async("question", raise_on_error=True)

    assert nodes == []
    with pytest.raises(pytest.skip.Exception, match="no ingested LightRAG data"):
        skip_if_no_lightrag_data(nodes)
