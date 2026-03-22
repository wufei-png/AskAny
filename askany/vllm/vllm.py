"""vLLM integration for OpenAI-compatible LLM endpoints."""

import time
from logging import getLogger

import httpx
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI

from askany.config import settings
from askany.metrics import get_metrics

logger = getLogger(__name__)


def get_first_available_model(api_base: str) -> str | None:
    """Get the first available model from vLLM API.

    Args:
        api_base: The base URL of the vLLM API (e.g., "http://localhost:8001/v1")

    Returns:
        The first model ID, or None if failed to fetch
    """
    try:
        # Remove /v1 suffix if present, then add /v1/models
        base_url = api_base.rstrip("/v1").rstrip("/")
        models_url = f"{base_url}/v1/models"

        logger.info("Fetching available models from %s", models_url)
        with httpx.Client(timeout=10.0, proxies=None, trust_env=False) as client:
            response = client.get(models_url)
            response.raise_for_status()
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                first_model = data["data"][0]["id"]
                logger.info("Found first available model: %s", first_model)
                return first_model
            else:
                logger.warning("No models found in API response")
                return None
    except Exception as e:
        logger.error("Failed to fetch models from vLLM API: %s", e)
        return None


class VLLMOpenAI(OpenAI):
    """OpenAI wrapper for vLLM with custom model paths.

    This class handles unknown model names by providing default metadata
    when the model name is not in the known OpenAI model list.
    Also disables proxy for local vLLM endpoints.
    """

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata, with fallback for unknown models."""
        try:
            return super().metadata
        except ValueError as e:
            if "Unknown model" in str(e):
                # For vLLM with custom model paths, provide default metadata
                # Use a reasonable default context window (32k for qwen2.5-7b)
                logger.warning(
                    "Model %s not in known OpenAI models, using default metadata for vLLM",
                    self.model,
                )
                return LLMMetadata(
                    context_window=32768,  # Default for qwen2.5-7b-instruct
                    num_output=self.max_tokens or -1,
                    is_chat_model=True,  # Assume chat model for vLLM
                    is_function_calling_model=True,  # Assume function calling support
                    model_name=self.model,
                    system_role=MessageRole.SYSTEM,
                )
            raise


class AutoRetryVLLM:
    """Wrapper that auto-retries with first available model on 404 errors."""

    def __init__(self, initial_model, api_base, api_key):
        self._model_name = initial_model
        self._api_base = api_base
        self._api_key = api_key
        self._llm = VLLMOpenAI(
            api_key=api_key,
            api_base=api_base,
            model=initial_model,
            temperature=settings.temperature,
        )
        self._retried = False

    def _handle_404_error(self, error):
        """Handle 404 error by getting first available model and retrying."""
        if self._retried:
            return False  # Already retried, don't retry again

        if not (
            "404" in str(error)
            or "does not exist" in str(error).lower()
            or "NotFoundError" in str(type(error).__name__)
        ):
            return False  # Not a model not found error

        logger.warning(
            "Model %s not found, attempting to get first available model from vLLM API",
            self._model_name,
        )
        metrics = get_metrics()
        metrics.askany_llm_404_retries_total.labels(model=self._model_name).inc()
        first_model = get_first_available_model(self._api_base)
        if first_model:
            logger.info("Retrying with first available model: %s", first_model)
            self._model_name = first_model
            self._llm = VLLMOpenAI(
                api_key=self._api_key,
                api_base=self._api_base,
                model=first_model,
                temperature=settings.temperature,
            )
            self._retried = True
            return True  # Successfully retried
        return False  # Failed to get first model

    def __getattr__(self, name):
        """Delegate attributes to wrapped LLM, with auto-retry on 404 errors."""
        attr = getattr(self._llm, name)

        # If it's a callable method, wrap it to catch 404 errors
        if callable(attr):

            def wrapper(*args, **kwargs):
                metrics = get_metrics()
                model = self._model_name
                operation = "chat"  # default operation type
                start_time = time.perf_counter()

                # Determine if streaming
                is_streaming = kwargs.get("stream", False) or name == "stream"

                current_attr = attr
                try:
                    result = current_attr(*args, **kwargs)
                    duration = time.perf_counter() - start_time

                    # Record duration and success
                    metrics.askany_llm_request_duration_seconds.labels(
                        model=model, operation=operation
                    ).observe(duration)
                    metrics.askany_llm_requests_total.labels(
                        model=model, status="success", error_type="none"
                    ).inc()

                    # Handle token counting for streaming
                    if is_streaming:
                        # Wrap streaming generator to count chunks and tokens
                        prompt_tokens = (
                            kwargs.get("prompt", "").__len__()
                            if kwargs.get("prompt")
                            else 0
                        )

                        def token_counter():
                            chunk_count = 0
                            completion_tokens = 0
                            for chunk in result:
                                chunk_count += 1
                                yield chunk
                            # Record streaming metrics after iteration completes
                            metrics.askany_llm_stream_chunks_total.labels(
                                model=model
                            ).inc(chunk_count)
                            metrics.askany_llm_tokens_total.labels(
                                model=model, token_type="prompt"
                            ).inc(prompt_tokens)
                            metrics.askany_llm_tokens_total.labels(
                                model=model, token_type="completion"
                            ).inc(completion_tokens)
                            metrics.askany_llm_tokens_total.labels(
                                model=model, token_type="total"
                            ).inc(prompt_tokens + completion_tokens)

                        return token_counter()

                    # Handle token counting for non-streaming responses
                    if hasattr(result, "raw") and result.raw:
                        raw = result.raw
                        # Try to extract token usage from OpenAI-compatible response
                        if hasattr(raw, "usage") and raw.usage:
                            prompt_tokens = raw.usage.prompt_tokens or 0
                            completion_tokens = raw.usage.completion_tokens or 0
                            metrics.askany_llm_tokens_total.labels(
                                model=model, token_type="prompt"
                            ).inc(prompt_tokens)
                            metrics.askany_llm_tokens_total.labels(
                                model=model, token_type="completion"
                            ).inc(completion_tokens)
                            metrics.askany_llm_tokens_total.labels(
                                model=model, token_type="total"
                            ).inc(prompt_tokens + completion_tokens)

                    return result

                except TimeoutError:
                    duration = time.perf_counter() - start_time
                    metrics.askany_llm_request_duration_seconds.labels(
                        model=model, operation=operation
                    ).observe(duration)
                    metrics.askany_llm_timeout_total.labels(model=model).inc()
                    metrics.askany_llm_requests_total.labels(
                        model=model, status="error", error_type="timeout"
                    ).inc()
                    raise

                except Exception as e:
                    duration = time.perf_counter() - start_time
                    metrics.askany_llm_request_duration_seconds.labels(
                        model=model, operation=operation
                    ).observe(duration)
                    error_type = type(e).__name__
                    metrics.askany_llm_connection_errors_total.labels(
                        model=model, error_type=error_type
                    ).inc()
                    metrics.askany_llm_requests_total.labels(
                        model=model, status="error", error_type=error_type
                    ).inc()

                    # Handle 404 retry on error
                    if self._handle_404_error(e):
                        # Retry the original call with new LLM
                        current_attr = getattr(self._llm, name)
                        retry_start_time = time.perf_counter()
                        try:
                            result = current_attr(*args, **kwargs)
                            retry_duration = time.perf_counter() - retry_start_time
                            metrics.askany_llm_request_duration_seconds.labels(
                                model=model, operation=operation
                            ).observe(retry_duration)
                            metrics.askany_llm_requests_total.labels(
                                model=model, status="success", error_type="none"
                            ).inc()
                            return result
                        except TimeoutError:
                            retry_duration = time.perf_counter() - retry_start_time
                            metrics.askany_llm_request_duration_seconds.labels(
                                model=model, operation=operation
                            ).observe(retry_duration)
                            metrics.askany_llm_timeout_total.labels(model=model).inc()
                            metrics.askany_llm_requests_total.labels(
                                model=model, status="error", error_type="timeout"
                            ).inc()
                            raise
                        except Exception as retry_e:
                            retry_duration = time.perf_counter() - retry_start_time
                            metrics.askany_llm_request_duration_seconds.labels(
                                model=model, operation=operation
                            ).observe(retry_duration)
                            retry_error_type = type(retry_e).__name__
                            metrics.askany_llm_connection_errors_total.labels(
                                model=model, error_type=retry_error_type
                            ).inc()
                            metrics.askany_llm_requests_total.labels(
                                model=model, status="error", error_type=retry_error_type
                            ).inc()
                            raise

                    raise

            return wrapper
        return attr
