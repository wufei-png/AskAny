"""vLLM integration for OpenAI-compatible LLM endpoints."""

from logging import getLogger
from typing import Optional

import httpx
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI
from askany.config import settings

logger = getLogger(__name__)


def get_first_available_model(api_base: str) -> Optional[str]:
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
                current_attr = attr
                try:
                    return current_attr(*args, **kwargs)
                except Exception as e:
                    if self._handle_404_error(e):
                        # Retry the original call with new LLM
                        current_attr = getattr(self._llm, name)
                        return current_attr(*args, **kwargs)
                    raise

            return wrapper
        return attr
