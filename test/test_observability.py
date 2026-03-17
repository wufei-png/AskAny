"""Tests for askany.observability – Langfuse setup, RAGAS evaluation, and integrations.

All tests use mocks to avoid requiring real Langfuse/RAGAS credentials.
"""

from __future__ import annotations

import base64
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: create a minimal Settings-like object for testing
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> SimpleNamespace:
    """Build a fake Settings namespace with observability defaults."""
    defaults = dict(
        enable_langfuse=True,
        langfuse_public_key="pk-test-123",
        langfuse_secret_key="sk-test-456",
        langfuse_host="http://localhost:3000",
        langfuse_release="test-v1",
        langfuse_debug=False,
        # RAGAS
        enable_ragas=True,
        ragas_sample_rate=1.0,
        ragas_metrics=["faithfulness", "response_relevancy", "context_precision"],
        ragas_eval_llm_model="test-model",
        ragas_eval_llm_api_base="http://localhost:8080",
        ragas_eval_llm_api_key="test-key",
        # Fallbacks used by RAGAS when specific keys are None
        openai_model="fallback-model",
        openai_api_base="http://fallback",
        openai_api_key="fallback-key",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Fixtures – Reset module-level singletons between tests
# ---------------------------------------------------------------------------


def _reset_langfuse_module():
    """Reset all module-level singletons in langfuse_setup."""
    import askany.observability.langfuse_setup as mod

    mod._langfuse_client = None
    mod._langfuse_callback_handler = None
    mod._llamaindex_instrumentor = None
    mod._initialized = False


def _reset_ragas_module():
    """Reset all module-level singletons in ragas_eval."""
    import askany.observability.ragas_eval as mod

    mod._metrics = {}
    mod._evaluator_llm = None
    mod._initialized = False
    mod._sample_rate = 1.0


@pytest.fixture(autouse=True)
def _reset_modules():
    """Ensure every test starts with clean module state and env vars."""
    _reset_langfuse_module()
    _reset_ragas_module()
    # Snapshot and restore LANGFUSE_* env vars to prevent cross-test leakage
    langfuse_env_keys = [k for k in os.environ if k.startswith("LANGFUSE_")]
    saved_env = {k: os.environ[k] for k in langfuse_env_keys}
    yield
    _reset_langfuse_module()
    _reset_ragas_module()
    # Remove any LANGFUSE_* env vars set during the test
    for k in list(os.environ):
        if k.startswith("LANGFUSE_") and k not in saved_env:
            del os.environ[k]
    # Restore original values
    for k, v in saved_env.items():
        os.environ[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# Tests for askany.observability.langfuse_setup
# ═══════════════════════════════════════════════════════════════════════════


class TestEncodingBasicAuth:
    """Tests for _encode_basic_auth helper."""

    def test_encoding_produces_valid_base64(self):
        from askany.observability.langfuse_setup import _encode_basic_auth

        result = _encode_basic_auth("pk-123", "sk-456")
        decoded = base64.b64decode(result).decode()
        assert decoded == "pk-123:sk-456"

    def test_encoding_with_empty_strings(self):
        from askany.observability.langfuse_setup import _encode_basic_auth

        result = _encode_basic_auth("", "")
        decoded = base64.b64decode(result).decode()
        assert decoded == ":"

    def test_encoding_with_special_characters(self):
        from askany.observability.langfuse_setup import _encode_basic_auth

        result = _encode_basic_auth("pk+special/key=", "sk@secret!")
        decoded = base64.b64decode(result).decode()
        assert decoded == "pk+special/key=:sk@secret!"


class TestInitializeLangfuse:
    """Tests for initialize_langfuse()."""

    def test_returns_false_when_disabled(self):
        from askany.observability.langfuse_setup import initialize_langfuse

        settings = _make_settings(enable_langfuse=False)
        assert initialize_langfuse(settings) is False

    def test_returns_false_when_keys_missing(self):
        from askany.observability.langfuse_setup import initialize_langfuse

        settings = _make_settings(langfuse_public_key=None, langfuse_secret_key=None)
        assert initialize_langfuse(settings) is False

    def test_returns_false_when_public_key_only(self):
        from askany.observability.langfuse_setup import initialize_langfuse

        settings = _make_settings(langfuse_secret_key=None)
        assert initialize_langfuse(settings) is False

    @patch("askany.observability.langfuse_setup.Langfuse", create=True)
    def test_sets_env_vars(self, mock_langfuse_cls):
        """Verify env vars are set for LightRAG compatibility."""
        from askany.observability.langfuse_setup import initialize_langfuse

        # Patch imports inside the function
        with patch.dict(
            "sys.modules", {"langfuse": MagicMock(), "langfuse.langchain": MagicMock()}
        ):
            with patch("builtins.__import__", side_effect=_import_side_effect):
                settings = _make_settings()
                # We need to mock the actual imports inside the function
                mock_langfuse_mod = MagicMock()
                mock_langfuse_mod.Langfuse.return_value = MagicMock()
                mock_langchain_mod = MagicMock()
                mock_langchain_mod.CallbackHandler.return_value = MagicMock()

                with patch.dict(
                    "sys.modules",
                    {
                        "langfuse": mock_langfuse_mod,
                        "langfuse.langchain": mock_langchain_mod,
                        "opentelemetry": MagicMock(),
                        "opentelemetry.trace": MagicMock(),
                        "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(),
                        "opentelemetry.sdk.trace": MagicMock(),
                        "opentelemetry.sdk.trace.export": MagicMock(),
                        "openinference": MagicMock(),
                        "openinference.instrumentation": MagicMock(),
                        "openinference.instrumentation.llama_index": MagicMock(),
                    },
                ):
                    # Re-import to use patched modules
                    initialize_langfuse(settings)

                    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "pk-test-123"
                    assert os.environ.get("LANGFUSE_SECRET_KEY") == "sk-test-456"
                    assert os.environ.get("LANGFUSE_HOST") == "http://localhost:3000"

    def test_idempotent_returns_true_on_second_call(self):
        """Once initialized, subsequent calls return True without re-initializing."""
        import askany.observability.langfuse_setup as mod

        # Simulate already initialized
        mod._initialized = True
        settings = _make_settings()
        assert mod.initialize_langfuse(settings) is True

    def test_returns_false_when_langfuse_import_fails(self):
        """Gracefully handles missing langfuse package."""
        from askany.observability.langfuse_setup import initialize_langfuse

        settings = _make_settings()

        with patch.dict("sys.modules", {"langfuse": None}):
            # Force ImportError on `from langfuse import Langfuse`
            original_import = (
                __builtins__.__import__
                if hasattr(__builtins__, "__import__")
                else __import__
            )

            def failing_import(name, *args, **kwargs):
                if name == "langfuse":
                    raise ImportError("No module named 'langfuse'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=failing_import):
                result = initialize_langfuse(settings)
                assert result is False


class TestGetters:
    """Tests for get_langfuse_callback_handler / get_langfuse_client."""

    def test_returns_none_before_init(self):
        from askany.observability.langfuse_setup import (
            get_langfuse_callback_handler,
            get_langfuse_client,
        )

        assert get_langfuse_callback_handler() is None
        assert get_langfuse_client() is None

    def test_returns_objects_after_init(self):
        import askany.observability.langfuse_setup as mod

        sentinel_client = object()
        sentinel_handler = object()
        mod._langfuse_client = sentinel_client
        mod._langfuse_callback_handler = sentinel_handler

        assert mod.get_langfuse_client() is sentinel_client
        assert mod.get_langfuse_callback_handler() is sentinel_handler


class TestShutdownLangfuse:
    """Tests for shutdown_langfuse()."""

    def test_shutdown_resets_initialized_flag(self):
        import askany.observability.langfuse_setup as mod

        mod._initialized = True
        mod.shutdown_langfuse()
        assert mod._initialized is False

    def test_shutdown_flushes_client(self):
        import askany.observability.langfuse_setup as mod

        mock_client = MagicMock()
        mock_handler = MagicMock()
        mod._langfuse_client = mock_client
        mod._langfuse_callback_handler = mock_handler
        mod._initialized = True

        mod.shutdown_langfuse()

        mock_handler.flush.assert_called_once()
        mock_client.flush.assert_called_once()
        mock_client.shutdown.assert_called_once()

    def test_shutdown_handles_flush_errors_gracefully(self):
        import askany.observability.langfuse_setup as mod

        mock_client = MagicMock()
        mock_client.flush.side_effect = RuntimeError("flush failed")
        mock_handler = MagicMock()
        mock_handler.flush.side_effect = RuntimeError("handler flush failed")
        mod._langfuse_client = mock_client
        mod._langfuse_callback_handler = mock_handler
        mod._initialized = True

        # Should NOT raise
        mod.shutdown_langfuse()
        assert mod._initialized is False

    def test_shutdown_with_no_client_or_handler(self):
        import askany.observability.langfuse_setup as mod

        mod._initialized = True
        # Should NOT raise when both are None
        mod.shutdown_langfuse()
        assert mod._initialized is False

    def test_shutdown_resets_singletons_to_none(self):
        """Verify singletons are reset to None after shutdown."""
        import askany.observability.langfuse_setup as mod

        mock_client = MagicMock()
        mock_handler = MagicMock()
        mock_instrumentor = MagicMock()
        mod._langfuse_client = mock_client
        mod._langfuse_callback_handler = mock_handler
        mod._llamaindex_instrumentor = mock_instrumentor
        mod._initialized = True

        mod.shutdown_langfuse()

        assert mod._langfuse_client is None
        assert mod._langfuse_callback_handler is None
        assert mod._llamaindex_instrumentor is None
        assert mod._initialized is False

    def test_shutdown_calls_uninstrument(self):
        """Verify uninstrument is called on LlamaIndex instrumentor during shutdown."""
        import askany.observability.langfuse_setup as mod

        mock_client = MagicMock()
        mock_handler = MagicMock()
        mock_instrumentor = MagicMock()
        mod._langfuse_client = mock_client
        mod._langfuse_callback_handler = mock_handler
        mod._llamaindex_instrumentor = mock_instrumentor
        mod._initialized = True

        mod.shutdown_langfuse()

        mock_instrumentor.uninstrument.assert_called_once()
        assert mod._llamaindex_instrumentor is None

    def test_getters_return_none_after_shutdown(self):
        """Verify getters return None after shutdown."""
        import askany.observability.langfuse_setup as mod

        mock_client = MagicMock()
        mock_handler = MagicMock()
        mod._langfuse_client = mock_client
        mod._langfuse_callback_handler = mock_handler
        mod._initialized = True

        mod.shutdown_langfuse()

        # After shutdown, getters should return None
        assert mod.get_langfuse_client() is None
        assert mod.get_langfuse_callback_handler() is None


class TestInitializeLangfuseEnvVars:
    """Tests for Langfuse env var handling."""

    def test_does_not_overwrite_existing_env_vars(self):
        """If env vars already set, initialize_langfuse should NOT overwrite them."""
        import askany.observability.langfuse_setup as mod

        env_backup = {}
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
            env_backup[key] = os.environ.pop(key, None)

        try:
            os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-existing"
            os.environ["LANGFUSE_SECRET_KEY"] = "sk-existing"
            os.environ["LANGFUSE_HOST"] = "http://existing-host"

            # Need to re-import to reset _initialized
            mod._initialized = False

            settings = _make_settings(
                langfuse_public_key="pk-new",
                langfuse_secret_key="sk-new",
                langfuse_host="http://new-host",
            )

            with patch.dict(
                "sys.modules",
                {
                    "langfuse": MagicMock(),
                    "langfuse.langchain": MagicMock(),
                    "opentelemetry": MagicMock(),
                    "opentelemetry.trace": MagicMock(),
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(),
                    "opentelemetry.sdk.trace": MagicMock(),
                    "opentelemetry.sdk.trace.export": MagicMock(),
                    "openinference": MagicMock(),
                    "openinference.instrumentation": MagicMock(),
                    "openinference.instrumentation.llama_index": MagicMock(),
                },
            ):
                mod.initialize_langfuse(settings)

            assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-existing"
            assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-existing"
            assert os.environ["LANGFUSE_HOST"] == "http://existing-host"
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)


# ═══════════════════════════════════════════════════════════════════════════
# Tests for askany.observability.ragas_eval
# ═══════════════════════════════════════════════════════════════════════════


class TestGetMetricClasses:
    """Tests for _get_metric_classes()."""

    def test_returns_dict_with_expected_keys(self):
        from askany.observability.ragas_eval import _get_metric_classes

        classes = _get_metric_classes()
        assert isinstance(classes, dict)
        # Depending on ragas version, should have these keys
        if classes:  # If ragas is installed
            assert "faithfulness" in classes
            assert "response_relevancy" in classes
            assert "context_precision" in classes

    def test_returns_empty_when_ragas_not_available(self):
        """Gracefully returns empty dict when ragas not importable."""
        from askany.observability.ragas_eval import _get_metric_classes

        with patch.dict(
            "sys.modules",
            {
                "ragas": MagicMock(),
                "ragas.metrics": MagicMock(spec=[]),  # Empty spec → missing attrs
            },
        ):
            # Force ImportError on both import paths
            original_import = __import__

            def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "ragas.metrics" and fromlist:
                    raise ImportError("mocked")
                return original_import(name, globals, locals, fromlist, level)

            with patch("builtins.__import__", side_effect=failing_import):
                classes = _get_metric_classes()
                assert isinstance(classes, dict)


class TestInitializeRagas:
    """Tests for initialize_ragas()."""

    def test_returns_false_when_disabled(self):
        from askany.observability.ragas_eval import initialize_ragas

        settings = _make_settings(enable_ragas=False)
        assert initialize_ragas(settings) is False

    def test_returns_false_when_ragas_not_installed(self):
        from askany.observability.ragas_eval import initialize_ragas

        settings = _make_settings()

        original_import = __import__

        def failing_import(name, *args, **kwargs):
            if name in ("ragas.llms", "ragas"):
                raise ImportError("No ragas")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = initialize_ragas(settings)
            assert result is False

    def test_idempotent_returns_true_on_second_call(self):
        import askany.observability.ragas_eval as mod

        mod._initialized = True
        settings = _make_settings()
        assert mod.initialize_ragas(settings) is True

    def test_returns_false_when_no_metrics_loaded(self):
        """If all metric names are unknown, returns False."""
        from askany.observability.ragas_eval import initialize_ragas

        settings = _make_settings(ragas_metrics=["nonexistent_metric_xyz"])

        # Mock the LLM creation to succeed but metrics to fail
        mock_chat = MagicMock()
        mock_wrapper = MagicMock()

        with (
            patch(
                "askany.observability.ragas_eval.ChatOpenAI", create=True
            ) as mock_cls,
            patch(
                "askany.observability.ragas_eval.LangchainLLMWrapper", create=True
            ) as mock_wrap,
        ):
            # Fake the import check
            # (mod imported below within nested context)

            # Need to handle the import inside the function
            with patch.dict(
                "sys.modules",
                {
                    "ragas.llms": MagicMock(LangchainLLMWrapper=mock_wrap),
                    "langchain_openai": MagicMock(ChatOpenAI=mock_cls),
                },
            ):
                mock_cls.return_value = mock_chat
                mock_wrap.return_value = mock_wrapper
                result = initialize_ragas(settings)
                assert result is False

    def test_successful_initialization(self):
        """Verify successful init with mocked dependencies."""
        # initialize_ragas accessed via mod below

        settings = _make_settings()

        mock_chat = MagicMock()
        mock_wrapper = MagicMock()
        mock_faithfulness_cls = MagicMock()
        mock_relevancy_cls = MagicMock()
        mock_precision_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "ragas.llms": MagicMock(
                    LangchainLLMWrapper=MagicMock(return_value=mock_wrapper)
                ),
                "langchain_openai": MagicMock(
                    ChatOpenAI=MagicMock(return_value=mock_chat)
                ),
            },
        ):
            with patch(
                "askany.observability.ragas_eval._get_metric_classes",
                return_value={
                    "faithfulness": mock_faithfulness_cls,
                    "response_relevancy": mock_relevancy_cls,
                    "context_precision": mock_precision_cls,
                },
            ):
                import askany.observability.ragas_eval as mod

                result = mod.initialize_ragas(settings)
                assert result is True
                assert mod._initialized is True
                assert len(mod._metrics) == 3

    def test_sample_rate_is_set(self):
        """Verify sample_rate is propagated from settings."""
        import askany.observability.ragas_eval as mod

        settings = _make_settings(ragas_sample_rate=0.5)

        mock_metric_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "ragas.llms": MagicMock(
                    LangchainLLMWrapper=MagicMock(return_value=MagicMock())
                ),
                "langchain_openai": MagicMock(
                    ChatOpenAI=MagicMock(return_value=MagicMock())
                ),
            },
        ):
            with patch(
                "askany.observability.ragas_eval._get_metric_classes",
                return_value={"faithfulness": mock_metric_cls},
            ):
                mod.initialize_ragas(settings)
                assert mod._sample_rate == 0.5

    def test_uses_fallback_llm_settings(self):
        """When ragas_eval_llm_* are None, falls back to openai_* settings."""
        from askany.observability.ragas_eval import initialize_ragas

        settings = _make_settings(
            ragas_eval_llm_model=None,
            ragas_eval_llm_api_base=None,
            ragas_eval_llm_api_key=None,
        )

        captured_kwargs = {}
        mock_chat_cls = MagicMock()

        def capture_chat(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        mock_chat_cls.side_effect = capture_chat

        with patch.dict(
            "sys.modules",
            {
                "ragas.llms": MagicMock(
                    LangchainLLMWrapper=MagicMock(return_value=MagicMock())
                ),
                "langchain_openai": MagicMock(ChatOpenAI=mock_chat_cls),
            },
        ):
            with patch(
                "askany.observability.ragas_eval._get_metric_classes",
                return_value={"faithfulness": MagicMock()},
            ):
                initialize_ragas(settings)
                assert captured_kwargs.get("model") == "fallback-model"
                assert captured_kwargs.get("base_url") == "http://fallback"


class TestEvaluateRagResponse:
    """Tests for evaluate_rag_response()."""

    @pytest.mark.asyncio
    async def test_works_with_empty_retrieved_contexts(self):
        """RAGAS should still evaluate even when retrieved_contexts is empty.

        Note: This tests the current behavior where context-dependent metrics
        (faithfulness, context_precision) will fail but response_relevancy should work.
        """
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.85)

        mod._initialized = True
        mod._metrics = {"response_relevancy": mock_metric}
        mod._sample_rate = 1.0

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            result = await mod.evaluate_rag_response(
                user_input="test question",
                response="test answer",
                retrieved_contexts=[],  # Empty contexts - should still work for response_relevancy
            )
            assert "response_relevancy" in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_initialized(self):
        from askany.observability.ragas_eval import evaluate_rag_response

        result = await evaluate_rag_response(
            user_input="test",
            response="answer",
            retrieved_contexts=["ctx1"],
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_metrics(self):
        import askany.observability.ragas_eval as mod

        mod._initialized = True
        mod._metrics = {}  # No metrics loaded

        result = await mod.evaluate_rag_response(
            user_input="test",
            response="answer",
            retrieved_contexts=["ctx1"],
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_sampling_skips_when_rate_zero(self):
        """When sample_rate=0, evaluation should almost always be skipped."""
        import askany.observability.ragas_eval as mod

        mod._initialized = True
        mod._metrics = {"faithfulness": MagicMock()}
        mod._sample_rate = 0.0

        # random.random() returns [0, 1), so > 0.0 is always True → always skip
        result = await mod.evaluate_rag_response(
            user_input="test",
            response="answer",
            retrieved_contexts=["ctx1"],
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_sampling_always_runs_at_rate_one(self):
        """When sample_rate=1.0, sampling check is bypassed."""
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.85)

        mod._initialized = True
        mod._metrics = {"faithfulness": mock_metric}
        mod._sample_rate = 1.0

        with patch.dict(
            "sys.modules",
            {
                "ragas": MagicMock(SingleTurnSample=MagicMock()),
            },
        ):
            with patch(
                "askany.observability.ragas_eval.SingleTurnSample", create=True
            ) as mock_sample_cls:
                mock_sample_cls.return_value = MagicMock()
                result = await mod.evaluate_rag_response(
                    user_input="test question",
                    response="test answer",
                    retrieved_contexts=["context 1"],
                )
                assert "faithfulness" in result
                assert result["faithfulness"] == 0.85

    @pytest.mark.asyncio
    async def test_individual_metric_failure_does_not_block_others(self):
        """One metric failing shouldn't prevent other metrics from running."""
        import askany.observability.ragas_eval as mod

        failing_metric = MagicMock()
        failing_metric.single_turn_ascore = AsyncMock(side_effect=RuntimeError("boom"))

        passing_metric = MagicMock()
        passing_metric.single_turn_ascore = AsyncMock(return_value=0.9)

        mod._initialized = True
        mod._metrics = {
            "failing_one": failing_metric,
            "passing_one": passing_metric,
        }
        mod._sample_rate = 1.0

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            result = await mod.evaluate_rag_response(
                user_input="test",
                response="answer",
                retrieved_contexts=["ctx"],
            )

        assert "passing_one" in result
        assert result["passing_one"] == 0.9
        assert "failing_one" not in result

    @pytest.mark.asyncio
    async def test_pushes_scores_to_langfuse(self):
        """Scores should be pushed to Langfuse when trace_id is provided."""
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.75)

        mod._initialized = True
        mod._metrics = {"faithfulness": mock_metric}
        mod._sample_rate = 1.0

        mock_langfuse_client = MagicMock()

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            with patch(
                "askany.observability.langfuse_setup.get_langfuse_client",
                return_value=mock_langfuse_client,
            ):
                result = await mod.evaluate_rag_response(
                    trace_id="trace-abc",
                    user_input="test",
                    response="answer",
                    retrieved_contexts=["ctx"],
                )

        assert result["faithfulness"] == 0.75
        mock_langfuse_client.score.assert_called_once_with(
            trace_id="trace-abc",
            name="ragas_faithfulness",
            value=0.75,
        )

    @pytest.mark.asyncio
    async def test_no_langfuse_push_without_trace_id(self):
        """Without trace_id, scores should NOT be pushed to Langfuse."""
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.8)

        mod._initialized = True
        mod._metrics = {"faithfulness": mock_metric}
        mod._sample_rate = 1.0

        mock_langfuse_client = MagicMock()

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            with patch(
                "askany.observability.langfuse_setup.get_langfuse_client",
                return_value=mock_langfuse_client,
            ):
                result = await mod.evaluate_rag_response(
                    trace_id=None,
                    user_input="test",
                    response="answer",
                    retrieved_contexts=["ctx"],
                )

        assert result["faithfulness"] == 0.8
        mock_langfuse_client.score.assert_not_called()

    @pytest.mark.asyncio
    async def test_langfuse_push_failure_does_not_raise(self):
        """Langfuse push errors should be caught silently."""
        import askany.observability.ragas_eval as mod

        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.7)

        mod._initialized = True
        mod._metrics = {"faithfulness": mock_metric}
        mod._sample_rate = 1.0

        mock_langfuse_client = MagicMock()
        mock_langfuse_client.score.side_effect = RuntimeError("network error")

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            with patch(
                "askany.observability.langfuse_setup.get_langfuse_client",
                return_value=mock_langfuse_client,
            ):
                # Should NOT raise
                result = await mod.evaluate_rag_response(
                    trace_id="trace-abc",
                    user_input="test",
                    response="answer",
                    retrieved_contexts=["ctx"],
                )

        # Scores should still be returned even if push failed
        assert result["faithfulness"] == 0.7

    @pytest.mark.asyncio
    async def test_multiple_metrics_scored_in_parallel(self):
        """All metrics should be scored even if one temporarily fails."""
        import askany.observability.ragas_eval as mod

        call_count = {"faithfulness": 0, "response_relevancy": 0}

        async def counting_score(sample):
            call_count[
                sample.__class__.__name__ if hasattr(sample, "__class__") else "unknown"
            ] += 1
            return 0.8

        mock_faithfulness = MagicMock()
        mock_faithfulness.single_turn_ascore = AsyncMock(
            side_effect=[
                RuntimeError("temp failure"),
                0.9,
            ]
        )

        mock_relevancy = MagicMock()
        mock_relevancy.single_turn_ascore = AsyncMock(return_value=0.85)

        mod._initialized = True
        mod._metrics = {
            "faithfulness": mock_faithfulness,
            "response_relevancy": mock_relevancy,
        }
        mod._sample_rate = 1.0

        with patch("askany.observability.ragas_eval.SingleTurnSample", create=True):
            result = await mod.evaluate_rag_response(
                user_input="test",
                response="answer",
                retrieved_contexts=["ctx1", "ctx2"],
            )

        assert "response_relevancy" in result
        assert result["response_relevancy"] == 0.85


class TestShutdownRagas:
    """Tests for shutdown_ragas()."""

    def test_shutdown_resets_initialized_flag(self):
        import askany.observability.ragas_eval as mod

        mod._initialized = True
        mod.shutdown_ragas()
        assert mod._initialized is False

    def test_shutdown_resets_metrics_and_evaluator(self):
        import askany.observability.ragas_eval as mod

        mock_metric = object()
        mock_evaluator = object()
        mod._metrics = {"faithfulness": mock_metric}
        mod._evaluator_llm = mock_evaluator
        mod._initialized = True
        mod._sample_rate = 0.5

        mod.shutdown_ragas()

        assert mod._metrics == {}
        assert mod._evaluator_llm is None
        assert mod._sample_rate == 1.0

    def test_shutdown_is_idempotent(self):
        """shutdown_ragas can be called multiple times without error."""
        import askany.observability.ragas_eval as mod

        mod.shutdown_ragas()
        mod.shutdown_ragas()
        # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# Tests for askany.observability.__init__ (re-exports)
# ═══════════════════════════════════════════════════════════════════════════


class TestObservabilityPackageExports:
    """Verify the __init__.py re-exports the expected public API."""

    def test_all_exports_are_importable(self):
        from askany.observability import (
            evaluate_rag_response,
            get_langfuse_callback_handler,
            get_langfuse_client,
            initialize_langfuse,
            initialize_ragas,
            shutdown_langfuse,
        )

        # All should be callable
        assert callable(get_langfuse_callback_handler)
        assert callable(get_langfuse_client)
        assert callable(initialize_langfuse)
        assert callable(shutdown_langfuse)
        assert callable(evaluate_rag_response)
        assert callable(initialize_ragas)

    def test_dunder_all_matches_exports(self):
        import askany.observability as obs

        expected = {
            "get_langfuse_callback_handler",
            "get_langfuse_client",
            "initialize_langfuse",
            "shutdown_langfuse",
            "evaluate_rag_response",
            "initialize_ragas",
            "shutdown_ragas",
        }
        assert set(obs.__all__) == expected


# ═══════════════════════════════════════════════════════════════════════════
# Tests for config.py – new Langfuse/RAGAS settings
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigSettings:
    """Verify new Langfuse and RAGAS config fields."""

    def test_langfuse_defaults(self):
        from askany.config import Settings

        s = Settings()
        assert s.enable_langfuse is False
        assert s.langfuse_public_key is None
        assert s.langfuse_secret_key is None
        assert s.langfuse_host == "https://cloud.langfuse.com"
        assert s.langfuse_release is None
        assert s.langfuse_debug is False

    def test_ragas_defaults(self):
        from askany.config import Settings

        s = Settings()
        assert s.enable_ragas is False
        assert s.ragas_sample_rate == 1.0
        assert s.ragas_metrics == [
            "faithfulness",
            "response_relevancy",
            "context_precision",
        ]
        assert s.ragas_eval_llm_model is None
        assert s.ragas_eval_llm_api_base is None
        assert s.ragas_eval_llm_api_key is None

    def test_langfuse_settings_from_env(self):
        """Settings should pick up env vars."""
        from askany.config import Settings

        env_overrides = {
            "ENABLE_LANGFUSE": "true",
            "LANGFUSE_PUBLIC_KEY": "pk-env",
            "LANGFUSE_SECRET_KEY": "sk-env",
            "LANGFUSE_HOST": "http://my-host:3000",
            "LANGFUSE_RELEASE": "v2.0",
            "LANGFUSE_DEBUG": "true",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            s = Settings()
            assert s.enable_langfuse is True
            assert s.langfuse_public_key == "pk-env"
            assert s.langfuse_secret_key == "sk-env"
            assert s.langfuse_host == "http://my-host:3000"
            assert s.langfuse_release == "v2.0"
            assert s.langfuse_debug is True

    def test_ragas_settings_from_env(self):
        from askany.config import Settings

        env_overrides = {
            "ENABLE_RAGAS": "true",
            "RAGAS_SAMPLE_RATE": "0.5",
            "RAGAS_EVAL_LLM_MODEL": "gpt-4",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            s = Settings()
            assert s.enable_ragas is True
            assert s.ragas_sample_rate == 0.5
            assert s.ragas_eval_llm_model == "gpt-4"


# ═══════════════════════════════════════════════════════════════════════════
# Tests for lightrag_adapter.py Langfuse integration
# ═══════════════════════════════════════════════════════════════════════════


class TestPropagateLightragAttributes:
    """Tests for the propagate_lightrag_attributes context manager."""

    def test_noop_when_langfuse_disabled(self):
        """Should be a no-op context manager when Langfuse is not enabled."""
        import askany.rag.lightrag_adapter as mod

        # Save and override module state
        orig_enabled = mod._LANGFUSE_ENABLED
        try:
            mod._LANGFUSE_ENABLED = False

            # Should execute without error and yield
            with mod.propagate_lightrag_attributes(user_id="u1", session_id="s1"):
                pass  # no-op
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled

    def test_calls_propagate_when_enabled(self):
        """When Langfuse is enabled, should call langfuse.propagate_attributes."""
        import askany.rag.lightrag_adapter as mod

        orig_enabled = mod._LANGFUSE_ENABLED
        orig_propagate = getattr(mod, "_lf_propagate", None)

        mock_propagate = MagicMock()
        mock_propagate.return_value.__enter__ = MagicMock(return_value=None)
        mock_propagate.return_value.__exit__ = MagicMock(return_value=False)

        try:
            mod._LANGFUSE_ENABLED = True
            mod._lf_propagate = mock_propagate

            with mod.propagate_lightrag_attributes(
                user_id="u1", session_id="s1", tags=["test"]
            ):
                pass

            mock_propagate.assert_called_once_with(
                user_id="u1",
                session_id="s1",
                tags=["test"],
                metadata={},
            )
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled
            mod._lf_propagate = orig_propagate


class TestLightRAGAdapterLangfuseMethods:
    """Tests for LightRAGAdapter.langfuse_flush() and langfuse_shutdown()."""

    def test_flush_calls_client_flush_when_enabled(self):
        import askany.rag.lightrag_adapter as mod

        orig_enabled = mod._LANGFUSE_ENABLED
        orig_client = mod._langfuse_client

        mock_client = MagicMock()
        try:
            mod._LANGFUSE_ENABLED = True
            mod._langfuse_client = mock_client

            # Create a minimal adapter instance (skip __init__ by using __new__)
            adapter = object.__new__(mod.LightRAGAdapter)
            adapter.langfuse_flush()

            mock_client.flush.assert_called_once()
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled
            mod._langfuse_client = orig_client

    def test_flush_noop_when_disabled(self):
        import askany.rag.lightrag_adapter as mod

        orig_enabled = mod._LANGFUSE_ENABLED
        try:
            mod._LANGFUSE_ENABLED = False
            adapter = object.__new__(mod.LightRAGAdapter)
            adapter.langfuse_flush()  # Should not raise
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled

    def test_shutdown_calls_client_shutdown(self):
        import askany.rag.lightrag_adapter as mod

        orig_enabled = mod._LANGFUSE_ENABLED
        orig_client = mod._langfuse_client

        mock_client = MagicMock()
        try:
            mod._LANGFUSE_ENABLED = True
            mod._langfuse_client = mock_client

            adapter = object.__new__(mod.LightRAGAdapter)
            adapter.langfuse_shutdown()

            mock_client.shutdown.assert_called_once()
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled
            mod._langfuse_client = orig_client

    def test_shutdown_noop_when_disabled(self):
        import askany.rag.lightrag_adapter as mod

        orig_enabled = mod._LANGFUSE_ENABLED
        try:
            mod._LANGFUSE_ENABLED = False
            adapter = object.__new__(mod.LightRAGAdapter)
            adapter.langfuse_shutdown()  # Should not raise
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled


class TestLightRAGSetupLangfuse:
    """Tests for LightRAGAdapter._setup_langfuse() static method."""

    def test_noop_when_enable_langfuse_false(self):
        import askany.rag.lightrag_adapter as mod

        orig_enabled = mod._LANGFUSE_ENABLED
        settings = SimpleNamespace(enable_langfuse=False)

        try:
            mod.LightRAGAdapter._setup_langfuse(settings)
            # Nothing should change
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled

    def test_sets_env_vars_from_settings(self):
        import askany.rag.lightrag_adapter as mod

        orig_enabled = mod._LANGFUSE_ENABLED

        settings = SimpleNamespace(
            enable_langfuse=True,
            langfuse_public_key="pk-from-settings",
            langfuse_secret_key="sk-from-settings",
            langfuse_host="http://custom-host:3000",
        )

        # Remove env vars so _setup_langfuse will set them
        env_backup = {}
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
            env_backup[key] = os.environ.pop(key, None)

        try:
            mod.LightRAGAdapter._setup_langfuse(settings)

            assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "pk-from-settings"
            assert os.environ.get("LANGFUSE_SECRET_KEY") == "sk-from-settings"
            assert os.environ.get("LANGFUSE_HOST") == "http://custom-host:3000"
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled
            # Restore env
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)

    def test_does_not_overwrite_existing_env_vars(self):
        """If env vars already set, _setup_langfuse should NOT overwrite them."""
        import askany.rag.lightrag_adapter as mod

        orig_enabled = mod._LANGFUSE_ENABLED

        settings = SimpleNamespace(
            enable_langfuse=True,
            langfuse_public_key="pk-new",
            langfuse_secret_key="sk-new",
            langfuse_host="http://new-host",
        )

        env_backup = {}
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
            env_backup[key] = os.environ.get(key)

        try:
            os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-existing"
            os.environ["LANGFUSE_SECRET_KEY"] = "sk-existing"
            os.environ["LANGFUSE_HOST"] = "http://existing-host"

            mod.LightRAGAdapter._setup_langfuse(settings)

            # Should keep existing values
            assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-existing"
            assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-existing"
            assert os.environ["LANGFUSE_HOST"] == "http://existing-host"
        finally:
            mod._LANGFUSE_ENABLED = orig_enabled
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)


# ═══════════════════════════════════════════════════════════════════════════
# Tests for workflow module Langfuse callback injection
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkflowLangfuseCallbackInjection:
    """Verify that workflow modules import get_langfuse_callback_handler gracefully."""

    def test_workflow_langgraph_has_fallback_import(self):
        """workflow_langgraph.py should have get_langfuse_callback_handler available."""
        # This just tests that the import at module top doesn't fail
        import askany.workflow.workflow_langgraph as mod

        assert hasattr(mod, "get_langfuse_callback_handler")
        assert callable(mod.get_langfuse_callback_handler)

    def test_min_langchain_agent_has_fallback_import(self):
        import askany.workflow.min_langchain_agent as mod

        assert hasattr(mod, "get_langfuse_callback_handler")
        assert callable(mod.get_langfuse_callback_handler)

    def test_analysis_related_has_fallback_import(self):
        import askany.workflow.AnalysisRelated_langchain as mod

        assert hasattr(mod, "get_langfuse_callback_handler")
        assert callable(mod.get_langfuse_callback_handler)

    def test_callback_handler_returns_none_by_default(self):
        """When Langfuse is not initialized, handler should be None."""
        from askany.observability.langfuse_setup import get_langfuse_callback_handler

        assert get_langfuse_callback_handler() is None


# ═══════════════════════════════════════════════════════════════════════════
# Helper for import mocking
# ═══════════════════════════════════════════════════════════════════════════


def _import_side_effect(name, *args, **kwargs):
    """Default side effect that falls through to real import."""
    return __import__(name, *args, **kwargs)
