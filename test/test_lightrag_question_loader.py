"""Unit tests for the operator-supplied LightRAG question file contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from lightrag_question_loader import (
    DEFAULT_QUESTIONS_PATH,
    QUESTIONS_ENV_VAR,
    get_lightrag_questions_path,
    load_lightrag_questions,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_default_path_and_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(QUESTIONS_ENV_VAR, raising=False)
    assert get_lightrag_questions_path() == DEFAULT_QUESTIONS_PATH

    configured_path = tmp_path / "questions.json"
    _write_json(configured_path, ["injected question"])
    monkeypatch.setenv(QUESTIONS_ENV_VAR, str(configured_path))
    assert get_lightrag_questions_path() == configured_path
    assert load_lightrag_questions() == ["injected question"]


def test_loads_valid_list_of_strings(tmp_path: Path) -> None:
    question_path = tmp_path / "questions.json"
    _write_json(question_path, ["第一个问题", " second question "])

    assert load_lightrag_questions(question_path) == [
        "第一个问题",
        " second question ",
    ]


def test_missing_file_is_reported_for_caller_skip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lightrag_questions(tmp_path / "missing.json")


def test_empty_array_is_valid_and_left_to_caller_skip(tmp_path: Path) -> None:
    question_path = tmp_path / "empty.json"
    _write_json(question_path, [])

    assert load_lightrag_questions(question_path) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "not a list"},
        ["valid", 3],
        ["   "],
    ],
)
def test_invalid_question_shapes_raise_value_error(
    tmp_path: Path, payload: object
) -> None:
    question_path = tmp_path / "invalid.json"
    _write_json(question_path, payload)

    with pytest.raises(ValueError):
        load_lightrag_questions(question_path)


def test_malformed_json_raises_value_error(tmp_path: Path) -> None:
    question_path = tmp_path / "malformed.json"
    question_path.write_text('{"questions":', encoding="utf-8")

    with pytest.raises(ValueError):
        load_lightrag_questions(question_path)
