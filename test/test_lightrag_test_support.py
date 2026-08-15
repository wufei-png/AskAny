"""Unit tests for LightRAG integration prerequisite classification."""

from __future__ import annotations

import httpx
import openai
import psycopg2
import pytest
from lightrag_test_support import is_lightrag_prerequisite_error
from psycopg2 import errors as psycopg2_errors


def _openai_response(status_code: int) -> httpx.Response:
    request = httpx.Request("GET", "https://provider.invalid")
    return httpx.Response(status_code, request=request)


@pytest.mark.parametrize(
    "error",
    [
        ImportError("lightrag is not installed"),
        OSError("model file is unavailable"),
        httpx.ConnectError("provider is unreachable"),
        httpx.ReadTimeout("provider timed out"),
        openai.APIConnectionError(request=None),
        openai.APITimeoutError(request=None),
        openai.AuthenticationError(
            message="missing credentials",
            response=_openai_response(401),
            body=None,
        ),
        openai.NotFoundError(
            message="model is unavailable", response=_openai_response(404), body=None
        ),
        psycopg2.OperationalError("database is unavailable"),
        psycopg2_errors.UndefinedTable("LightRAG table is absent"),
    ],
)
def test_external_prerequisites_are_classified_as_skips(
    error: BaseException,
) -> None:
    assert is_lightrag_prerequisite_error(error) is True


@pytest.mark.parametrize(
    "error",
    [
        TypeError("unexpected constructor argument"),
        AttributeError("incompatible adapter API"),
        AssertionError("broken invariant"),
        RuntimeError("unclassified runtime regression"),
        RuntimeError("wrapped regression"),
        httpx.HTTPStatusError(
            "provider rejected the request",
            request=httpx.Request("GET", "https://provider.invalid"),
            response=_openai_response(422),
        ),
    ],
)
def test_programming_and_compatibility_errors_are_not_skips(
    error: BaseException,
) -> None:
    assert is_lightrag_prerequisite_error(error) is False


def test_wrapped_external_error_is_classified() -> None:
    try:
        raise httpx.ConnectTimeout("provider timed out")
    except httpx.ConnectTimeout as cause:
        wrapped = RuntimeError("adapter initialization failed")
        wrapped.__cause__ = cause

    assert is_lightrag_prerequisite_error(wrapped) is True


def test_unclassified_runtime_error_with_no_cause_is_not_a_skip() -> None:
    error = RuntimeError("adapter initialization failed")

    assert error.__cause__ is None
    assert error.__context__ is None
    assert is_lightrag_prerequisite_error(error) is False
