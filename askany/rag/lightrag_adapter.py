"""
LightRAG adapter: wraps LightRAG knowledge-graph retrieval and converts
results to LlamaIndex NodeWithScore objects so they can be merged with
the existing FAQ/DOCS pipeline inside workflow_langgraph.py.

Usage
-----
Initialisation (once, at server start-up):
    adapter = LightRAGAdapter()
    await adapter.initialize()           # opens DB connections

Retrieval (per query):
    nodes = await adapter.retrieve_async("your question", mode="mix")
    # or sync wrapper:
    nodes = adapter.retrieve("your question", mode="mix")

Langfuse tracing (optional):
    # Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in the environment
    # (or via settings.langfuse_public_key / langfuse_secret_key).
    # LightRAG auto-instruments per-OpenAI-call tracing when those keys exist.
    # The adapter adds query-level trace grouping and user/session context.
    #
    # To attach user context from a workflow:
    #   with propagate_lightrag_attributes(user_id="u123", session_id="s456"):
    #       nodes = await adapter.retrieve_async(query)
    #
    # Flush buffered events at shutdown:
    #   adapter.langfuse_flush()

Shutdown:
    await adapter.finalize()
    adapter.langfuse_shutdown()
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import psycopg2
from askany.config import settings as _settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports: LightRAG is an optional dependency.  If it is not installed
# the adapter simply returns an empty node list (graceful degradation).
# ---------------------------------------------------------------------------
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    _LIGHTRAG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTRAG_AVAILABLE = False
    logger.warning(
        "lightrag-hku package not installed – LightRAGAdapter will return empty results. "
        "Install with: uv add lightrag-hku"
    )

# LlamaIndex imports are already a hard dependency of the project
from llama_index.core.schema import NodeWithScore, TextNode
from askany.rag.provenance import (
    ProvenanceRepository,
    build_provenance_record,
    canonicalize_path,
    compute_source_doc_id,
)

# ---------------------------------------------------------------------------
# Langfuse observability – optional dependency.
#
# If LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are present in the environment
# AND the ``langfuse`` package is installed:
#   - LightRAG's own OpenAI client (lightrag/llm/openai.py) will already
#     auto-instrument every individual LLM call via langfuse.openai.AsyncOpenAI.
#   - This adapter additionally wraps each ``retrieve_async()`` call in a
#     parent Langfuse span so sub-calls are grouped under a single trace, and
#     exposes helpers to attach user_id / session_id context.
#
# Env vars checked at import time:
#   LANGFUSE_PUBLIC_KEY   (required for activation)
#   LANGFUSE_SECRET_KEY   (required for activation)
#   LANGFUSE_HOST         (optional, defaults to https://cloud.langfuse.com)
# ---------------------------------------------------------------------------
_LANGFUSE_ENABLED = False
_langfuse_client: Any = None  # langfuse.Langfuse singleton or None

try:
    _langfuse_pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    _langfuse_sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if _langfuse_pub and _langfuse_sec:
        from langfuse import get_client as _lf_get_client
        from langfuse import observe as _lf_observe  # noqa: F401 – used below
        from langfuse import propagate_attributes as _lf_propagate  # noqa: F401

        _langfuse_client = _lf_get_client()
        _LANGFUSE_ENABLED = True
        logger.info("Langfuse observability enabled for LightRAGAdapter")
    else:
        _lf_observe = None  # type: ignore[assignment]
        _lf_propagate = None  # type: ignore[assignment]
except ImportError:
    _lf_observe = None  # type: ignore[assignment]
    _lf_propagate = None  # type: ignore[assignment]


class LightRAGAdapter:
    """
    Thin adapter that drives LightRAG and converts its results to
    ``List[NodeWithScore]`` so they integrate transparently with
    the existing AskAny retrieval pipeline.

    Design decisions
    ~~~~~~~~~~~~~~~~
    * ``only_need_context=True`` – we skip LightRAG's own LLM synthesis
      step.  This avoids burning extra LLM tokens; the existing
      workflow_langgraph.py synthesiser will do the final generation.
    * We use ``raw_data["data"]["chunks"]`` (structured data) rather than
      parsing the formatted context string.  Each chunk becomes one
      ``NodeWithScore``.
    * Entity / relationship data from ``raw_data["data"]["entities"]`` and
      ``raw_data["data"]["relationships"]`` is appended as additional
      context nodes with a lower default score so the reranker can
      decide their relevance.
    * The adapter is async-native.  A sync ``retrieve()`` wrapper is
      provided for callers that cannot easily use ``await``.
    """

    def __init__(
        self,
        *,
        working_dir: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: int = 1024,
        embedding_max_tokens: int = 8192,
        kv_storage: str = "PGKVStorage",
        vector_storage: str = "PGVectorStorage",
        graph_storage: str = "NetworkXStorage",
        doc_status_storage: str = "PGDocStatusStorage",
        query_mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
        max_total_tokens: int = 16000,
        chunk_score: float = 0.75,
        entity_score: float = 0.5,
        relation_score: float = 0.45,
        # ── Chunking (controls entity extraction granularity) ──────────────
        # Smaller chunks → fewer mixed topics per chunk → cleaner entities.
        chunk_token_size: Optional[int] = None,
        chunk_overlap_token_size: Optional[int] = None,
        # ── Entity extraction ─────────────────────────────────────────────
        entity_extract_max_gleaning: Optional[int] = None,
        # ── Summary ───────────────────────────────────────────────────────
        summary_max_tokens: Optional[int] = None,
        # ── Addon params (language, entity_types) ─────────────────────────
        addon_params: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        working_dir:
            Directory used by LightRAG for temporary / cache files.
            Defaults to ``./lightrag_data`` (relative to cwd).
        llm_model / llm_api_base / llm_api_key:
            LLM endpoint.  Defaults fall back to the AskAny ``settings``
            values (``openai_model``, ``openai_api_base``, ``openai_api_key``).
        embedding_model:
            Name of the embedding model served by the same OpenAI-compatible
            endpoint.  Defaults to ``settings.embedding_model``.
        embedding_dim:
            Output dimension of the embedding model (default 1024 for bge-m3).
        kv_storage / vector_storage / graph_storage / doc_status_storage:
            LightRAG storage backend names.  kv, vector, and doc_status default
            to PostgreSQL backends.  graph_storage defaults to ``NetworkXStorage``
            (file-based) because ``PGGraphStorage`` requires the Apache AGE
            extension which may not be installed.
        query_mode:
            Default LightRAG query mode.  Can be overridden per call.
            Options: "local", "global", "hybrid", "naive", "mix", "bypass".
        top_k:
            Number of entities/relations to retrieve from the KG.
        chunk_top_k:
            Number of text chunks to retrieve.
        max_total_tokens:
            Token budget for the assembled context.
        chunk_score / entity_score / relation_score:
            Synthetic LlamaIndex scores assigned to the three node types so
            that downstream reranking works sensibly.
        chunk_token_size:
            Tokens per chunk for LightRAG's internal splitter.  Smaller chunks
            improve entity extraction on dense Chinese docs.  Defaults to
            ``settings.lightrag_chunk_token_size`` (800).
        chunk_overlap_token_size:
            Overlap between consecutive chunks to preserve context at
            boundaries.  Defaults to ``settings.lightrag_chunk_overlap_token_size`` (150).
        entity_extract_max_gleaning:
            Extra entity extraction passes (>0 = enabled).  Defaults to
            ``settings.lightrag_entity_extract_max_gleaning`` (1).
        summary_max_tokens:
            Max tokens for entity/community summaries.  Higher values avoid
            truncation of verbose Chinese descriptions.  Defaults to
            ``settings.lightrag_summary_max_tokens`` (1500).
        addon_params:
            Dict passed to ``LightRAG(addon_params=...)``.  Controls prompt
            language and custom entity types for extraction.  If ``None``,
            uses a Chinese-optimised default with DevOps entity types.
        """
        if not _LIGHTRAG_AVAILABLE:
            self._rag: Optional["LightRAG"] = None
            return

        # --- resolve settings ---------------------------------------------------
        from askany.config import settings as _settings  # local import avoids circular

        working_dir = working_dir or os.environ.get(
            "LIGHTRAG_WORKING_DIR",
            getattr(_settings, "lightrag_working_dir", "./lightrag_data"),
        )
        llm_model = (
            llm_model
            or getattr(_settings, "lightrag_llm_model", None)
            or _settings.openai_model
        )
        llm_api_base = (
            llm_api_base
            or getattr(_settings, "lightrag_api_base", None)
            or _settings.openai_api_base
        )
        llm_api_key = (
            llm_api_key
            or getattr(_settings, "lightrag_api_key", None)
            or _settings.openai_api_key
            or "EMPTY"
        )
        embedding_model = (
            embedding_model
            or getattr(_settings, "lightrag_embedding_model", None)
            or _settings.embedding_model
        )

        # --- resolve new chunking / extraction params from settings ----------------
        chunk_token_size = chunk_token_size or getattr(
            _settings, "lightrag_chunk_token_size", 800
        )
        chunk_overlap_token_size = chunk_overlap_token_size or getattr(
            _settings, "lightrag_chunk_overlap_token_size", 150
        )
        entity_extract_max_gleaning = (
            entity_extract_max_gleaning
            if entity_extract_max_gleaning is not None
            else getattr(_settings, "lightrag_entity_extract_max_gleaning", 1)
        )
        summary_max_tokens = summary_max_tokens or getattr(
            _settings, "lightrag_summary_max_tokens", 1500
        )

        # Default addon_params: Chinese language + DevOps-specific entity types.
        # Custom entity_types help LightRAG's extraction prompt focus on domain
        # concepts (components, services, data flows) rather than generic NER.
        if addon_params is None:
            addon_params = {
                "language": "Chinese",
                "entity_types": [
                    "SystemComponent",  # e.g. TSFD, CJM, VPS, IPS, SFD
                    "Service",  # e.g. EntityService, BatchManager
                    "Infrastructure",  # e.g. Cassandra, ClickHouse, HDFS, Kafka
                    "Operation",  # e.g. 聚类, 扩容, 备份还原, 部署
                    "DataFlow",  # e.g. Kafka topic, oplog, ETL pipeline
                    "Configuration",  # e.g. concurrent_reads, ShardExpiredDays
                    "Metric",  # e.g. Grafana面板, Prometheus指标
                    "Concept",  # e.g. 类中心, 分片, 置信, 特征
                    "Method",  # e.g. ClusterSearch, EntityBatchGet
                    "Document",  # e.g. FAQ, 运维手册, 配置说明
                ],
            }
        os.makedirs(working_dir, exist_ok=True)

        # --- propagate PG credentials as env vars (LightRAG reads them itself) --
        self._set_pg_env(_settings)

        # --- build async LLM function -------------------------------------------
        _llm_model = llm_model
        _llm_api_base = llm_api_base
        _llm_api_key = llm_api_key

        async def _llm_func_inner(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: list | None = None,
            keyword_extraction: bool = False,
            **kwargs,
        ) -> str:
            if history_messages is None:
                history_messages = []
            return await openai_complete_if_cache(
                _llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=_llm_api_key,
                base_url=_llm_api_base,
                **kwargs,
            )

        # Wrap with Langfuse @observe so each LLM call becomes a child
        # "generation" span under the parent retrieve_async trace.
        # When Langfuse is not available the inner function is used directly.
        if _LANGFUSE_ENABLED and _lf_observe is not None:
            _llm_func = _lf_observe(
                name="lightrag-llm-call",
                as_type="generation",
            )(_llm_func_inner)
        else:
            _llm_func = _llm_func_inner

        # --- build async embedding function -------------------------------------
        # AskAny uses a local SentenceTransformer model (e.g. BAAI/bge-m3) for
        # embeddings, NOT an OpenAI-compatible embedding API.  We load the same
        # local model and wrap it for LightRAG's EmbeddingFunc interface.
        import numpy as np

        try:
            from sentence_transformers import SentenceTransformer as _ST
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for LightRAG embedding. "
                "Install with: uv add sentence-transformers"
            )

        _embed_model_name = embedding_model
        _device = getattr(_settings, "device", "cpu")
        logger.info(
            "LightRAGAdapter: loading SentenceTransformer model %s on %s",
            _embed_model_name,
            _device,
        )
        _st_model = _ST(_embed_model_name, device=_device)

        async def _embed_func(texts: List[str]) -> np.ndarray:
            """Embed texts using local SentenceTransformer model.

            LightRAG's EmbeddingFunc expects ``func(texts) -> np.ndarray``
            with shape ``(len(texts), embedding_dim)``.
            """
            # SentenceTransformer.encode() is synchronous; run in executor to
            # avoid blocking the event loop during large batch embedding.
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: _st_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
            )
            return np.array(embeddings, dtype=np.float32)

        # --- assemble LightRAG instance -----------------------------------------
        self._rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=embedding_max_tokens,
                func=_embed_func,
            ),
            kv_storage=kv_storage,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            doc_status_storage=doc_status_storage,
            # ── Chunking: smaller chunks improve entity extraction on dense
            # Chinese technical docs (default 1200 is too large for mixed-topic paragraphs).
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
            # ── Entity extraction: extra gleaning pass re-examines chunks to catch
            # entities missed on the first pass (important for abbreviated Chinese terms).
            entity_extract_max_gleaning=entity_extract_max_gleaning,
            # ── Summary: higher limit avoids truncation of verbose Chinese
            # entity / community descriptions.
            summary_max_tokens=summary_max_tokens,
            # ── Addon: language="Chinese" prompts Chinese output; custom entity_types
            # focus extraction on DevOps domain concepts instead of generic NER.
            addon_params=addon_params,
        )

        # --- Langfuse: propagate settings-defined keys to env vars ---------------
        self._setup_langfuse(_settings)

        # --- defaults for query calls -------------------------------------------
        self._default_mode = query_mode
        self._top_k = top_k
        self._chunk_top_k = chunk_top_k
        self._max_total_tokens = max_total_tokens
        self._chunk_score = chunk_score
        self._entity_score = entity_score
        self._relation_score = relation_score
        self._provenance_repo = ProvenanceRepository()

        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _set_pg_env(settings) -> None:
        """Mirror AskAny postgres_* settings into the env vars LightRAG expects.

        Only sets variables that are not already present in ``os.environ``
        so explicit overrides are preserved.
        """

        def _get_password(value):
            if hasattr(value, "get_secret_value"):
                return value.get_secret_value()
            return str(value)

        mapping = {
            "POSTGRES_HOST": str(getattr(settings, "postgres_host", "localhost")),
            "POSTGRES_PORT": str(getattr(settings, "postgres_port", 5432)),
            "POSTGRES_USER": str(getattr(settings, "postgres_user", "root")),
            "POSTGRES_PASSWORD": _get_password(
                getattr(settings, "postgres_password", "")
            ),
            "POSTGRES_DATABASE": str(getattr(settings, "postgres_db", "askany")),
        }
        changed: list[str] = []
        for key, value in mapping.items():
            if key not in os.environ:  # don't overwrite explicit overrides
                os.environ[key] = value
                changed.append(key)
        if changed:
            logger.debug("LightRAGAdapter: set PG env vars: %s", changed)

    @staticmethod
    def _setup_langfuse(settings) -> None:
        """Propagate Langfuse credentials from AskAny settings into env vars.

        LightRAG's ``lightrag/llm/openai.py`` reads ``LANGFUSE_PUBLIC_KEY`` and
        ``LANGFUSE_SECRET_KEY`` at *import time* to decide whether to swap in
        ``langfuse.openai.AsyncOpenAI``.  We cannot retroactively change that
        choice here, but we can ensure the keys are set before the first import
        of LightRAG's LLM module (which happens lazily on first query).

        For AskAny's own adapter-level instrumentation the module-level
        ``_LANGFUSE_ENABLED`` flag is checked; it was set at import time from
        the *existing* env vars, so this method is only relevant for runs where
        the keys are absent from the initial environment but present in
        ``settings``.
        """
        global _LANGFUSE_ENABLED, _langfuse_client
        if not getattr(settings, "enable_langfuse", False):
            return
        pairs = {
            "LANGFUSE_PUBLIC_KEY": getattr(settings, "langfuse_public_key", None),
            "LANGFUSE_SECRET_KEY": getattr(settings, "langfuse_secret_key", None),
            "LANGFUSE_HOST": getattr(settings, "langfuse_host", None),
        }
        changed: list[str] = []
        for key, value in pairs.items():
            if value and key not in os.environ:
                os.environ[key] = value
                changed.append(key)
        if changed:
            logger.debug("LightRAGAdapter: set Langfuse env vars: %s", changed)
        # Re-check activation in case keys were just added from settings
        if not _LANGFUSE_ENABLED:
            pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
            sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
            if pub and sec:
                try:
                    from langfuse import get_client as _lf_get_client2

                    _langfuse_client = _lf_get_client2()
                    _LANGFUSE_ENABLED = True
                    logger.info(
                        "Langfuse observability activated via settings for LightRAGAdapter"
                    )
                except ImportError:
                    logger.debug(
                        "Langfuse keys found but 'langfuse' package not installed; "
                        "install with: uv add langfuse"
                    )

    def langfuse_flush(self) -> None:
        """Block until all buffered Langfuse events have been sent.

        Call this after a batch of queries in a short-lived process (e.g. CLI
        tools, test harnesses) to ensure traces are not lost when the process
        exits.
        """
        if _LANGFUSE_ENABLED and _langfuse_client is not None:
            _langfuse_client.flush()
            logger.debug("LightRAGAdapter: Langfuse events flushed")

    def langfuse_shutdown(self) -> None:
        """Flush all buffered events and terminate the Langfuse background thread.

        Call this alongside ``finalize()`` at application shutdown.
        """
        if _LANGFUSE_ENABLED and _langfuse_client is not None:
            _langfuse_client.shutdown()
            logger.info("LightRAGAdapter: Langfuse client shut down")

    async def initialize(self) -> None:
        """Open DB connections and create tables.  Must be called before ``retrieve``."""
        if not _LIGHTRAG_AVAILABLE or self._rag is None:
            return
        if self._initialized:
            return
        self._provenance_repo.ensure_table()
        await self._rag.initialize_storages()
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()

        # Ensure NetworkX graph is populated from existing entities
        await self._ensure_graph_loaded()

        self._initialized = True
        logger.info("LightRAGAdapter: storages initialised")

    async def _ensure_graph_loaded(self) -> None:
        kg = self._rag.chunk_entity_relation_graph
        entities_vdb = self._rag.entities_vdb

        all_nodes = await kg.get_all_nodes()
        all_edges = await kg.get_all_edges()

        if len(all_nodes) > 0:
            logger.debug("NetworkX graph already has nodes, skipping load")
            return

        sample_entities = await entities_vdb.query("test", top_k=1)
        if not sample_entities:
            logger.debug("No entities in vector store, skipping graph load")
            return

        logger.info("Loading entities into NetworkX graph from PostgreSQL...")

        all_entities = await entities_vdb.query("", top_k=5000)

        if not all_entities:
            logger.debug("No entities found to load")
            return

        for entity in all_entities:
            entity_name = entity.get("entity_name", "")
            if entity_name:
                await kg.upsert_node(
                    entity_name,
                    {"content": entity_name, "entity_type": "unknown"},
                )

        logger.info("Loading relations into NetworkX graph...")

        relations_vdb = self._rag.relationships_vdb
        all_relations = await relations_vdb.query("", top_k=5000)

        loaded_edges = 0
        for rel in all_relations:
            src = rel.get("src_id", "")
            tgt = rel.get("tgt_id", "")
            if src and tgt:
                await kg.upsert_edge(
                    src,
                    tgt,
                    {"content": f"{src} -> {tgt}"},
                )
                loaded_edges += 1

        await kg.index_done_callback()

        final_nodes = len(await kg.get_all_nodes())
        final_edges = len(await kg.get_all_edges())
        logger.info(
            f"NetworkX graph populated with {final_nodes} nodes, {final_edges} edges"
        )

    async def finalize(self) -> None:
        """Close DB connections.  Call at application shutdown."""
        if not _LIGHTRAG_AVAILABLE or self._rag is None or not self._initialized:
            return
        await self._rag.finalize_storages()
        self._initialized = False
        logger.info("LightRAGAdapter: storages finalised")

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    async def insert_async(
        self,
        texts: str | List[str],
        file_paths: Optional[str | List[str]] = None,
        split_by_character: Optional[str] = None,
    ) -> None:
        """Insert one or more documents into the LightRAG knowledge graph.

        This is **incremental** – documents that have already been inserted
        (identified by content hash) are silently skipped.

        Parameters
        ----------
        split_by_character:
            If set, LightRAG splits each document on this string *before*
            its token-level chunker runs.  For Markdown docs use ``"\\n## "``
            so each H2 section becomes an independent chunk, improving entity
            extraction granularity.
        """
        if not _LIGHTRAG_AVAILABLE or self._rag is None:
            return
        if not self._initialized:
            await self.initialize()
        track_id = await self._rag.ainsert(
            texts, file_paths=file_paths, split_by_character=split_by_character
        )
        self._backfill_chunk_provenance(track_id, file_paths)

    def insert(
        self,
        texts: str | List[str],
        file_paths: Optional[str | List[str]] = None,
        split_by_character: Optional[str] = None,
    ) -> None:
        """Synchronous wrapper around :meth:`insert_async`."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            try:
                import nest_asyncio

                nest_asyncio.apply(loop)
            except ImportError:
                logger.warning(
                    "nest_asyncio not installed; sync insert() cannot run inside an "
                    "async context.  Install with: uv add nest-asyncio  — or call "
                    "insert_async() directly."
                )
                return

        loop = _get_or_create_loop()
        loop.run_until_complete(
            self.insert_async(
                texts, file_paths=file_paths, split_by_character=split_by_character
            )
        )

    def _get_db_connection(self):
        return psycopg2.connect(
            host=_settings.postgres_host,
            port=str(_settings.postgres_port),
            user=_settings.postgres_user,
            password=_settings.postgres_password.get_secret_value(),
            database=_settings.postgres_db,
        )

    def _backfill_chunk_provenance(
        self,
        track_id: Optional[str],
        file_paths: Optional[str | List[str]],
    ) -> None:
        if not track_id or not file_paths:
            return

        file_path_list = (
            [file_paths] if isinstance(file_paths, str) else list(file_paths)
        )
        if not file_path_list:
            return

        conn = self._get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT file_path, chunks_list
                    FROM lightrag_doc_status
                    WHERE track_id = %s AND file_path = ANY(%s)
                    """,
                    (track_id, file_path_list),
                )
                chunk_ids_by_file: Dict[str, list[str]] = {
                    file_path: list(chunks_list or [])
                    for file_path, chunks_list in cur.fetchall()
                }

                all_chunk_ids = [
                    chunk_id
                    for chunk_ids in chunk_ids_by_file.values()
                    for chunk_id in chunk_ids
                ]
                if not all_chunk_ids:
                    return

                cur.execute(
                    """
                    SELECT id, chunk_order_index, content, file_path
                    FROM lightrag_doc_chunks
                    WHERE id = ANY(%s)
                    ORDER BY file_path, chunk_order_index
                    """,
                    (all_chunk_ids,),
                )
                rows = cur.fetchall()
        except Exception as exc:
            logger.warning(
                "LightRAGAdapter: provenance backfill query failed – %s", exc
            )
            return
        finally:
            conn.close()

        hint_by_path: dict[str, int] = {}
        records = []
        for chunk_id, chunk_order_index, content, file_path in rows:
            record = build_provenance_record(
                retrieval_origin="lightrag",
                source_kind="lightrag_chunk",
                origin_id=chunk_id,
                source_unit_id=chunk_id,
                file_path=file_path,
                text=content,
                hint_start_line=hint_by_path.get(file_path, 1),
            )
            if record.end_line is not None:
                hint_by_path[file_path] = record.end_line + 1
            records.append(record)

        if records:
            self._provenance_repo.upsert_records(records)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve_async(
        self,
        query: str,
        *,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """Retrieve relevant nodes from LightRAG and return as ``NodeWithScore`` list.

        Parameters
        ----------
        query:
            The user question.
        mode:
            LightRAG query mode override.  If ``None``, uses the default
            set at construction time.
        top_k:
            Override for number of entities/relations retrieved.
        chunk_top_k:
            Override for number of text chunks retrieved.

        Returns
        -------
        List[NodeWithScore]
            One ``NodeWithScore`` per text chunk, plus additional nodes for
            knowledge-graph entities and relationships (at lower scores).
            Returns ``[]`` if LightRAG is unavailable or returns no context.
        """
        if not _LIGHTRAG_AVAILABLE or self._rag is None:
            return []
        if not self._initialized:
            await self.initialize()

        param = QueryParam(
            mode=mode or self._default_mode,
            only_need_context=True,
            top_k=top_k or self._top_k,
            chunk_top_k=chunk_top_k or self._chunk_top_k,
            max_total_tokens=self._max_total_tokens,
        )

        # ── Langfuse: wrap retrieval in a parent span ──────────────────────
        # All child LLM calls (already instrumented via @observe on _llm_func)
        # will be automatically nested under this span when Langfuse is enabled.
        if _LANGFUSE_ENABLED and _langfuse_client is not None:
            with _langfuse_client.start_as_current_observation(
                as_type="span",
                name="lightrag-retrieve",
                input={"query": query, "mode": mode or self._default_mode},
            ) as _span:
                try:
                    result = await self._rag.aquery_data(query, param=param)
                except Exception as exc:
                    _span.update(metadata={"error": str(exc)})
                    logger.warning(
                        "LightRAGAdapter: query failed – %s", exc, exc_info=True
                    )
                    return []
                nodes = self._convert_to_nodes(result, query)
                _span.update(
                    output={
                        "node_count": len(nodes),
                        "chunk_count": len(
                            [
                                n
                                for n in nodes
                                if n.node.metadata.get("type") == "lightrag_chunk"
                            ]
                        ),
                        "entity_count": len(
                            [
                                n
                                for n in nodes
                                if n.node.metadata.get("type") == "lightrag_entity"
                            ]
                        ),
                        "relation_count": len(
                            [
                                n
                                for n in nodes
                                if n.node.metadata.get("type") == "lightrag_relation"
                            ]
                        ),
                    }
                )
                return nodes
        else:
            # Langfuse not active – plain execution path (no overhead)
            try:
                result = await self._rag.aquery_data(query, param=param)
            except Exception as exc:
                logger.warning("LightRAGAdapter: query failed – %s", exc, exc_info=True)
                return []
            return self._convert_to_nodes(result, query)

    def retrieve(
        self,
        query: str,
        *,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """Synchronous wrapper around :meth:`retrieve_async`.

        Detects whether an event loop is already running (e.g. inside
        FastAPI / LangGraph async handlers) and avoids the ``run_until_complete``
        deadlock by using ``nest_asyncio`` as a fallback.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside an already-running loop (FastAPI, LangGraph, etc.).
            # nest_asyncio patches the loop to allow re-entrant run_until_complete.
            try:
                import nest_asyncio

                nest_asyncio.apply(loop)
            except ImportError:
                logger.warning(
                    "nest_asyncio not installed; sync retrieve() cannot run inside an "
                    "async context.  Install with: uv add nest-asyncio  — or call "
                    "retrieve_async() directly."
                )
                return []

        loop = _get_or_create_loop()
        return loop.run_until_complete(
            self.retrieve_async(query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k)
        )

    def _enrich_node_provenance(
        self,
        node: NodeWithScore,
        *,
        source_kind: str,
        origin_id: str,
    ) -> NodeWithScore:
        metadata = node.node.metadata
        metadata["retrieval_origin"] = "lightrag"
        metadata["source_kind"] = source_kind
        metadata["origin_id"] = origin_id

        record = self._provenance_repo.get_record("lightrag", source_kind, origin_id)
        if record:
            metadata.update({k: v for k, v in record.items() if v is not None})
            return node

        file_path = metadata.get("file_path") or metadata.get("source", "")
        content = node.node.text if hasattr(node.node, "text") else ""
        if source_kind == "lightrag_chunk" and file_path and content:
            record = build_provenance_record(
                retrieval_origin="lightrag",
                source_kind=source_kind,
                origin_id=origin_id,
                source_unit_id=origin_id,
                file_path=file_path,
                text=content,
            )
            metadata.update(record.to_metadata())
            self._provenance_repo.upsert_records([record])
            return node

        if file_path:
            metadata["canonical_path"] = canonicalize_path(file_path)
            metadata["source_doc_id"] = compute_source_doc_id(file_path)
        return node

    # ------------------------------------------------------------------
    # Conversion: LightRAG → NodeWithScore
    # ------------------------------------------------------------------

    def _convert_to_nodes(self, result: dict, query: str) -> List[NodeWithScore]:
        """Convert ``aquery_data()`` dict result to ``NodeWithScore`` list.

        ``aquery_data()`` returns::

            {
                "status": "success",
                "data": {
                    "chunks":        [{"reference_id", "content", "file_path", "chunk_id"}, ...],
                    "entities":      [{"entity_name", "entity_type", "description",
                                        "source_id", "file_path"}, ...],
                    "relationships": [{"src_id", "tgt_id", "description", "keywords",
                                        "weight", "file_path"}, ...],
                    "references":    [{"reference_id", "file_path"}, ...],
                },
                "metadata": {...},
            }
        """
        nodes: List[NodeWithScore] = []

        # aquery_data() returns a dict directly – no need for hasattr checks.
        if not isinstance(result, dict):
            logger.debug("LightRAGAdapter: unexpected result type %s", type(result))
            return []

        status = result.get("status", "")
        data = result.get("data", {})

        if status != "success" or not data:
            return nodes

        # ── 1. Text chunks (primary evidence) ─────────────────────────────────
        for chunk in data.get("chunks", []):
            content = chunk.get("content", "").strip()
            if not content:
                continue
            file_path = chunk.get("file_path", "")
            chunk_id = chunk.get("chunk_id", "")
            ref_id = chunk.get("reference_id", "")

            node = TextNode(
                id_=chunk_id or _stable_id(content),
                text=content,
                metadata={
                    "source": file_path,
                    "file_path": file_path,
                    "type": "lightrag_chunk",
                    "lightrag_ref_id": ref_id,
                    "chunk_id": chunk_id,
                },
            )
            nodes.append(
                self._enrich_node_provenance(
                    NodeWithScore(node=node, score=self._chunk_score),
                    source_kind="lightrag_chunk",
                    origin_id=chunk_id or node.node_id,
                )
            )

        # ── 2. Entity nodes (KG-derived factual summaries) ────────────────────
        for entity in data.get("entities", []):
            description = entity.get("description", "").strip()
            name = entity.get("entity_name", "").strip()
            if not description and not name:
                continue
            text = (
                f"[Entity: {name}]\n{description}"
                if description
                else f"[Entity: {name}]"
            )
            file_path = entity.get("file_path", "")
            node = TextNode(
                id_=_stable_id(f"entity:{name}"),
                text=text,
                metadata={
                    "source": file_path,
                    "file_path": file_path,
                    "type": "lightrag_entity",
                    "entity_name": name,
                    "entity_type": entity.get("entity_type", ""),
                    "source_id": entity.get("source_id", ""),
                    "reference_id": entity.get("reference_id", ""),
                },
            )
            entity_origin_id = entity.get("reference_id") or _stable_id(
                f"entity:{name}"
            )
            nodes.append(
                self._enrich_node_provenance(
                    NodeWithScore(node=node, score=self._entity_score),
                    source_kind="lightrag_entity",
                    origin_id=entity_origin_id,
                )
            )

        # ── 3. Relationship nodes (KG-derived relational context) ─────────────
        for rel in data.get("relationships", []):
            description = rel.get("description", "").strip()
            src = rel.get("src_id", "")
            tgt = rel.get("tgt_id", "")
            if not description:
                continue
            text = f"[Relation: {src} → {tgt}]\n{description}"
            file_path = rel.get("file_path", "")
            node = TextNode(
                id_=_stable_id(f"rel:{src}:{tgt}"),
                text=text,
                metadata={
                    "source": file_path,
                    "file_path": file_path,
                    "type": "lightrag_relation",
                    "src_id": src,
                    "tgt_id": tgt,
                    "keywords": rel.get("keywords", ""),
                    "weight": rel.get("weight", 1.0),
                    "source_id": rel.get("source_id", ""),
                    "reference_id": rel.get("reference_id", ""),
                },
            )
            relation_origin_id = rel.get("reference_id") or _stable_id(
                f"rel:{src}:{tgt}"
            )
            nodes.append(
                self._enrich_node_provenance(
                    NodeWithScore(node=node, score=self._relation_score),
                    source_kind="lightrag_relation",
                    origin_id=relation_origin_id,
                )
            )

        # ── 4. Fallback: no nodes extracted from structured data ────────────────
        if not nodes:
            logger.debug(
                "LightRAGAdapter: aquery_data returned no usable chunks/entities/relations"
            )
        logger.debug(
            "LightRAGAdapter: converted %d nodes (%d chunks, %d entities, %d relations)",
            len(nodes),
            len(data.get("chunks", [])),
            len(data.get("entities", [])),
            len(data.get("relationships", [])),
        )
        return nodes

    @staticmethod
    def _text_to_node(text: str, score: float = 0.7) -> NodeWithScore:
        """Wrap a plain text string as a single NodeWithScore."""
        return NodeWithScore(
            node=TextNode(
                id_=_stable_id(text[:128]),
                text=text,
                metadata={"type": "lightrag_context", "source": "lightrag"},
            ),
            score=score,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_id(text: str) -> str:
    """Generate a stable, deterministic node ID from content."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Return the running event loop or create a new one (for sync wrappers)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


# ---------------------------------------------------------------------------
# Public helpers for caller-injected Langfuse context
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def propagate_lightrag_attributes(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
):
    """Context manager that propagates user/session context to all Langfuse
    spans created inside the ``with`` block (including child ``retrieve_async``
    calls and their nested LLM-call spans).

    Usage::

        with propagate_lightrag_attributes(user_id="u123", session_id="s456"):
            nodes = await adapter.retrieve_async(query)

    When Langfuse is not enabled this is a no-op context manager with zero
    overhead.
    """
    if _LANGFUSE_ENABLED and _lf_propagate is not None:
        with _lf_propagate(
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            metadata=metadata or {},
        ):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialised)
# ---------------------------------------------------------------------------

_adapter_instance: Optional[LightRAGAdapter] = None


def get_lightrag_adapter() -> LightRAGAdapter:
    """Return the module-level singleton ``LightRAGAdapter``.

    The instance is created on first call.  ``initialize()`` is NOT
    automatically called – the caller must do that at startup (or the
    first ``retrieve()`` call will trigger it lazily).
    """
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = LightRAGAdapter()
    return _adapter_instance
