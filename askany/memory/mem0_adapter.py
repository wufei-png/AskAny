"""Thin adapter around Mem0 OSS SDK for persistent user memory.

Usage:
    adapter = Mem0Adapter()          # reads config from askany.config.settings
    memories = adapter.search("what is X?", user_id="alice")
    adapter.save_turn("what is X?", "X is ...", user_id="alice")  # fire-and-forget
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from askany.config import settings

logger = logging.getLogger(__name__)

# Lazy singleton — initialised on first access via get_mem0_adapter()
_adapter_instance: Optional["Mem0Adapter"] = None


def get_mem0_adapter() -> Optional["Mem0Adapter"]:
    """Return the global Mem0Adapter singleton, or None if mem0 is disabled."""
    global _adapter_instance
    if not settings.enable_mem0:
        return None
    if _adapter_instance is None:
        _adapter_instance = Mem0Adapter()
    return _adapter_instance


class Mem0Adapter:
    """Wraps ``mem0.Memory`` configured with pgvector + the project's own LLM."""

    def __init__(self) -> None:
        try:
            from mem0 import Memory  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "mem0ai is required for user memory. "
                "Install it with:  pip install mem0ai"
            ) from exc

        config = self._build_config()
        logger.info("Initialising Mem0 Memory with pgvector backend …")
        self._memory = Memory.from_config(config)
        logger.info("Mem0 Memory ready (collection=%s)", settings.mem0_collection_name)

    # ── public API ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        user_id: str,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories relevant to *query* for a specific user.

        Returns a list of dicts, each containing at least ``{"memory": str, "score": float}``.
        """
        top_k = top_k or settings.mem0_top_k
        threshold = score_threshold or settings.mem0_score_threshold

        try:
            raw = self._memory.search(query, user_id=user_id, limit=top_k)
        except Exception:
            logger.exception("Mem0 search failed for user_id=%s", user_id)
            return []

        if not raw:
            return []

        raw_results = raw.get("results") if isinstance(raw, dict) else raw
        if not raw_results:
            return []

        results: List[Dict[str, Any]] = raw_results  # type: ignore[assignment]
        # Filter by score threshold
        results = [r for r in results if r.get("score", 0) >= threshold]
        logger.debug(
            "Mem0 search user_id=%s top_k=%d threshold=%.2f → %d memories",
            user_id,
            top_k,
            threshold,
            len(results),
        )
        return results

    def save_turn(
        self,
        user_query: str,
        assistant_response: str,
        user_id: str,
    ) -> None:
        """Persist a single Q&A turn into the user's memory store.

        This is designed to be called in a fire-and-forget manner
        (wrapped in ``asyncio.create_task``).
        """
        messages = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_response},
        ]
        try:
            self._memory.add(messages, user_id=user_id)
            logger.debug("Mem0 saved turn for user_id=%s", user_id)
        except Exception:
            logger.exception("Mem0 save_turn failed for user_id=%s", user_id)

    async def save_turn_async(
        self,
        user_query: str,
        assistant_response: str,
        user_id: str,
    ) -> None:
        """Async wrapper that runs ``save_turn`` in a thread pool (non-blocking)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self.save_turn, user_query, assistant_response, user_id
        )

    # ── memory → prompt text ────────────────────────────────────────────────

    @staticmethod
    def format_memories_as_system_text(memories: List[Dict[str, Any]]) -> str:
        """Format a list of memory dicts into a system-message text block.

        Example output:
            [User Memory Context]
            - Prefers answers in Chinese.
            - Works on the DevOps team.
        """
        if not memories:
            return ""
        lines = ["[User Memory Context]"]
        for mem in memories:
            text = mem.get("memory", "")
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    # ── config builder ──────────────────────────────────────────────────────

    @staticmethod
    def _build_config() -> Dict[str, Any]:
        """Build the ``mem0.Memory.from_config`` dict from AskAny settings."""
        # --- Vector store: reuse existing PostgreSQL / pgvector ---
        vector_store_cfg: Dict[str, Any] = {
            "provider": "pgvector",
            "config": {
                "user": settings.postgres_user,
                "password": settings.postgres_password,
                "host": settings.postgres_host,
                "port": settings.postgres_port,
                "dbname": settings.postgres_db,
                "collection_name": settings.mem0_collection_name,
                "embedding_model_dims": settings.vector_dimension,
            },
        }

        # --- LLM for memory extraction / consolidation ---
        llm_model = settings.mem0_llm_model or settings.openai_model
        llm_api_base = settings.mem0_llm_api_base or settings.openai_api_base
        llm_api_key = settings.mem0_llm_api_key or settings.openai_api_key

        llm_cfg: Dict[str, Any] = {
            "provider": settings.mem0_llm_provider,
            "config": {
                "model": llm_model,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        }

        # For vllm / openai-compatible providers, inject the base URL under the
        # provider-specific key that mem0 expects.
        provider = settings.mem0_llm_provider.lower()
        if provider == "openai":
            llm_cfg["config"]["openai_base_url"] = llm_api_base
            if llm_api_key:
                llm_cfg["config"]["api_key"] = llm_api_key
        elif provider == "vllm":
            llm_cfg["config"]["vllm_base_url"] = llm_api_base
            if llm_api_key:
                llm_cfg["config"]["api_key"] = llm_api_key

        # --- Embedder ---
        embedder_model = settings.mem0_embedder_model or settings.embedding_model
        embedder_cfg: Dict[str, Any] = {
            "provider": settings.mem0_embedder_provider,
            "config": {
                "model": embedder_model,
            },
        }

        if settings.mem0_embedder_provider == "huggingface":
            embedder_cfg["config"]["model_kwargs"] = {"trust_remote_code": True}

        config: Dict[str, Any] = {
            "vector_store": vector_store_cfg,
            "llm": llm_cfg,
            "embedder": embedder_cfg,
            "version": "v1.1",
        }
        return config
