"""Central Langfuse bootstrap for AskAny.

Responsibilities:
1. Propagate config settings → environment variables (for LightRAG native support)
2. Initialise OpenTelemetry + LlamaIndex instrumentation
3. Provide a LangChain ``CallbackHandler`` factory
4. Expose a ``get_langfuse_client()`` helper (for RAGAS score pushing)

Call ``initialize_langfuse(settings)`` once at server start-up.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from askany.config import Settings

logger = logging.getLogger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────────
_langfuse_client: Optional[object] = None
_langfuse_callback_handler: Optional[object] = None
_llamaindex_instrumentor: Optional[object] = None
_initialized: bool = False


# ── Public API ───────────────────────────────────────────────────────────────


def initialize_langfuse(settings: "Settings") -> bool:
    """Bootstrap Langfuse tracing for all three frameworks.

    1. Sets ``LANGFUSE_*`` env vars so LightRAG picks them up natively.
    2. Creates a shared :class:`langfuse.Langfuse` client for score pushing.
    3. Creates a LangChain ``CallbackHandler`` (shared across all ChatOpenAI
       instances — injected via ``callbacks=[handler]`` at constructor time).
    4. Instruments LlamaIndex via OpenTelemetry (``LlamaIndexInstrumentor``).

    Returns ``True`` on success, ``False`` when Langfuse is disabled or deps
    are missing.
    """
    global \
        _langfuse_client, \
        _langfuse_callback_handler, \
        _llamaindex_instrumentor, \
        _initialized

    if _initialized:
        logger.debug("Langfuse already initialized, skipping")
        return True

    if not settings.enable_langfuse:
        logger.info("Langfuse tracing disabled (enable_langfuse=False)")
        return False

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("Langfuse enabled but public/secret key not set — skipping init")
        return False

    # ── 1. Propagate to env vars (LightRAG reads these directly) ──────────
    # Only set if not already present (allow env var overrides)
    if "LANGFUSE_PUBLIC_KEY" not in os.environ:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    if "LANGFUSE_SECRET_KEY" not in os.environ:
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    if "LANGFUSE_HOST" not in os.environ:
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    if settings.langfuse_release and "LANGFUSE_RELEASE" not in os.environ:
        os.environ["LANGFUSE_RELEASE"] = settings.langfuse_release
    if settings.langfuse_debug and "LANGFUSE_DEBUG" not in os.environ:
        os.environ["LANGFUSE_DEBUG"] = "true"

    logger.info(
        "Langfuse env vars set (host=%s, release=%s)",
        settings.langfuse_host,
        settings.langfuse_release or "<none>",
    )

    # ── 2. Create shared Langfuse client ──────────────────────────────────
    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            release=settings.langfuse_release,
            debug=settings.langfuse_debug,
        )
        logger.info("Langfuse client created")
    except ImportError:
        logger.warning("langfuse package not installed — pip install langfuse")
        return False
    except Exception:
        logger.exception("Failed to create Langfuse client")
        return False

    # ── 3. LangChain CallbackHandler ──────────────────────────────────────
    try:
        from langfuse.langchain import CallbackHandler

        _langfuse_callback_handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            release=settings.langfuse_release,
            debug=settings.langfuse_debug,
        )
        logger.info("LangChain Langfuse CallbackHandler created")
    except ImportError:
        logger.warning("langfuse.langchain not available — LangChain tracing disabled")
    except Exception:
        logger.exception("Failed to create LangChain CallbackHandler")

    # ── 4. LlamaIndex OpenTelemetry instrumentation ───────────────────────
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        # Langfuse accepts OTLP traces on /api/public/otel/v1/traces
        otlp_endpoint = settings.langfuse_host.rstrip("/") + "/api/public/otel"

        # Create OTLP exporter with Langfuse auth header
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint + "/v1/traces",
            headers={
                "Authorization": "Basic "
                + _encode_basic_auth(
                    settings.langfuse_public_key, settings.langfuse_secret_key
                ),
            },
        )

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        _llamaindex_instrumentor = LlamaIndexInstrumentor()
        _llamaindex_instrumentor.instrument()
        logger.info(
            "LlamaIndex OTel instrumentation active (endpoint=%s)", otlp_endpoint
        )
    except ImportError as e:
        logger.warning(
            "LlamaIndex OTel deps missing (%s) — LlamaIndex tracing disabled. "
            "Install with: pip install openinference-instrumentation-llama-index "
            "opentelemetry-sdk opentelemetry-exporter-otlp",
            e,
        )
    except Exception:
        logger.exception("Failed to instrument LlamaIndex")

    _initialized = True
    logger.info("Langfuse initialization complete")
    return True


def get_langfuse_callback_handler():
    """Return the shared LangChain CallbackHandler, or ``None``."""
    return _langfuse_callback_handler


def get_langfuse_client():
    """Return the shared :class:`langfuse.Langfuse` client, or ``None``."""
    return _langfuse_client


def shutdown_langfuse() -> None:
    """Flush pending events and shut down cleanly.

    Call this during application shutdown (e.g. in a FastAPI lifespan handler).
    """
    global \
        _langfuse_client, \
        _langfuse_callback_handler, \
        _llamaindex_instrumentor, \
        _initialized

    if _langfuse_callback_handler is not None:
        try:
            _langfuse_callback_handler.flush()
        except Exception:
            logger.exception("Error flushing LangChain CallbackHandler")

    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
            _langfuse_client.shutdown()
            logger.info("Langfuse client shut down")
        except Exception:
            logger.exception("Error shutting down Langfuse client")

    _langfuse_client = None
    _langfuse_callback_handler = None
    _llamaindex_instrumentor = None
    _initialized = False


# ── Helpers ──────────────────────────────────────────────────────────────────


def _encode_basic_auth(public_key: str, secret_key: str) -> str:
    """Encode Langfuse keys as HTTP Basic auth value."""
    import base64

    credentials = f"{public_key}:{secret_key}"
    return base64.b64encode(credentials.encode()).decode()
