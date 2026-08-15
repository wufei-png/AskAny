"""Load the operator-supplied question set used by LightRAG checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

QUESTIONS_ENV_VAR = "ASKANY_LIGHTRAG_QUESTIONS_FILE"
DEFAULT_QUESTIONS_PATH = (
    Path(__file__).parent / "fixtures" / "lightrag_questions.local.json"
)


def get_lightrag_questions_path() -> Path:
    """Return the configured question file path.

    The local default is deliberately gitignored; CI or a developer may inject
    a different file through ``ASKANY_LIGHTRAG_QUESTIONS_FILE``.
    """
    configured_path = os.environ.get(QUESTIONS_ENV_VAR)
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_QUESTIONS_PATH


def load_lightrag_questions(path: Path | None = None) -> list[str]:
    """Load and validate a LightRAG question file.

    Missing files are reported as ``FileNotFoundError`` so callers can make a
    precise, explicit skip decision.  Invalid JSON, non-list documents,
    non-string items, and blank questions always raise ``ValueError``.
    """
    question_path = path or get_lightrag_questions_path()
    try:
        with question_path.open(encoding="utf-8") as question_file:
            payload: Any = json.load(question_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed LightRAG question JSON: {question_path}") from exc

    if not isinstance(payload, list):
        raise ValueError(
            f"LightRAG question file must contain a JSON list[str]: {question_path}"
        )

    questions: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str):
            raise ValueError(
                f"LightRAG question list item {index} must be a string: {question_path}"
            )
        if not item.strip():
            raise ValueError(
                f"LightRAG question list item {index} must not be blank: {question_path}"
            )
        questions.append(item)
    return questions
