"""Offline tests for manual LightRAG comparison control-flow branches."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import test_lightrag_e2e_comparison as comparison

from askany.config import settings


def test_run_agent_restores_setting_when_agent_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_value = settings.enable_lightrag
    monkeypatch.setattr(
        comparison,
        "_load_agent_functions",
        Mock(side_effect=RuntimeError("agent compatibility regression")),
    )

    with pytest.raises(RuntimeError, match="compatibility"):
        comparison._run_agent_on_questions(["question"], enable_lightrag=True)

    assert settings.enable_lightrag is original_value


def test_main_reports_failed_when_question_execution_returns_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(comparison, "load_lightrag_questions", lambda: ["question"])
    monkeypatch.setattr(
        comparison,
        "_run_agent_on_questions",
        Mock(
            side_effect=[
                [
                    {
                        "question": "question",
                        "answer": "ERROR: offline",
                        "duration_s": 0.1,
                    }
                ],
                [{"question": "question", "answer": "ok", "duration_s": 0.1}],
            ]
        ),
    )
    result_path = tmp_path / "comparison.json"
    monkeypatch.setattr(comparison, "RESULT_FILE", result_path)

    assert comparison.main() == 1
    assert result_path.exists()
    assert "FAILED" in capsys.readouterr().out


def test_main_reports_passed_when_all_questions_succeed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(comparison, "load_lightrag_questions", lambda: ["question"])
    successful_results = [
        {"question": "question", "answer": "ok", "duration_s": 0.1},
    ]
    monkeypatch.setattr(
        comparison,
        "_run_agent_on_questions",
        Mock(side_effect=[successful_results, successful_results]),
    )
    result_path = tmp_path / "comparison.json"
    monkeypatch.setattr(comparison, "RESULT_FILE", result_path)

    assert comparison.main() == 0
    assert result_path.exists()
    assert "PASSED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "questions",
    [
        [],
        FileNotFoundError("questions.json"),
    ],
)
def test_main_skips_without_questions_and_does_not_write_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    questions: object,
) -> None:
    if isinstance(questions, BaseException):
        loader = Mock(side_effect=questions)
    else:
        loader = Mock(return_value=questions)
    monkeypatch.setattr(comparison, "load_lightrag_questions", loader)
    result_path = tmp_path / "comparison.json"
    monkeypatch.setattr(comparison, "RESULT_FILE", result_path)

    assert comparison.main() == 0
    assert not result_path.exists()
    assert "SKIPPED" in capsys.readouterr().out
