"""Shared classification helpers for opt-in LightRAG integration tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import httpx
import openai
import psycopg2
import pytest
from psycopg2 import errors as psycopg2_errors

_PREREQUISITE_ERROR_TYPES: Final[tuple[type[BaseException], ...]] = (
    ImportError,
    OSError,
    TimeoutError,
    ConnectionError,
    psycopg2.OperationalError,
    psycopg2.InterfaceError,
    psycopg2_errors.UndefinedTable,
    psycopg2_errors.InvalidCatalogName,
    psycopg2_errors.ConnectionException,
    psycopg2_errors.ConnectionDoesNotExist,
    httpx.TransportError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.AuthenticationError,
    openai.NotFoundError,
    openai.RateLimitError,
)


def is_lightrag_prerequisite_error(error: BaseException) -> bool:
    """Return whether *error* identifies an unavailable external prerequisite.

    Missing packages, model files, network resources, and PostgreSQL connection
    failures are legitimate opt-in integration skips.  Programming errors and
    compatibility failures (for example ``TypeError`` or ``AttributeError``)
    deliberately return ``False`` so the integration test fails visibly.
    """

    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, _PREREQUISITE_ERROR_TYPES):
            return True
        if candidate.__cause__ is not None:
            pending.append(candidate.__cause__)
        if candidate.__context__ is not None:
            pending.append(candidate.__context__)
    return False


def prerequisite_error_reason(error: BaseException) -> str:
    """Format a concise, non-sensitive prerequisite failure reason."""

    return type(error).__name__


def skip_if_no_lightrag_data(nodes: Sequence[object]) -> None:
    """Skip the opt-in retrieval test only when the probe returned no nodes."""
    if not nodes:
        pytest.skip(
            "LightRAG retrieval skipped: no ingested LightRAG data returned a "
            "result for the first configured question."
        )
