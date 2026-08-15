"""Tests for workflow diagnostic logging defaults and timestamp semantics."""

from __future__ import annotations

from datetime import UTC, datetime
from logging import FileHandler

from askany.workflow import workflow_langgraph


def test_workflow_debug_logging_is_enabled_by_default() -> None:
    assert workflow_langgraph.debug is True
    assert any(
        isinstance(handler, FileHandler)
        for handler in workflow_langgraph.debug_logger.handlers
    )


def test_workflow_log_timestamp_contains_explicit_utc_offset() -> None:
    timestamp = workflow_langgraph._utc_timestamp()
    parsed = datetime.fromisoformat(timestamp)

    assert timestamp.endswith("+00:00")
    assert parsed.tzinfo is UTC
