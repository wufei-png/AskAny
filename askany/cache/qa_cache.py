"""QA Semantic Cache Manager using GPTCache + PGVector."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gptcache import cache as gptcache
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

from askany.config import settings

if TYPE_CHECKING:
    from askany.main import SentenceTransformerEmbedding

logger = logging.getLogger(__name__)


class QACacheManager:
    """Thin wrapper around GPTCache for QA response caching.

    Uses BGE-m3 embeddings via SentenceTransformer, PGVector for similarity
    search, and PostgreSQL for response storage.

    Cache key = f"{query_type}:{query_text}"
    """

    def __init__(
        self,
        embed_model: SentenceTransformerEmbedding,
        postgres_url: str | None = None,
    ):
        self._embed_model = embed_model
        self._postgres_url = postgres_url or self._build_postgres_url()
        self._dimension = embed_model.dimension  # 1024 for BGE-m3
        self._initialized = False

    def _build_postgres_url(self) -> str:
        pg = settings.postgres_password.get_secret_value()
        return (
            f"postgresql://{settings.postgres_user}:{pg}"
            f"@{settings.postgres_host}:{settings.postgres_port}"
            f"/{settings.postgres_db}"
        )

    def build_cache_key(self, query: str, query_type: str) -> str:
        """Build composite cache key from query and type.

        Args:
            query: User query string
            query_type: QueryType value (AUTO/FAQ/DOCS/CODE)

        Returns:
            Composite key: f"{query_type}:{query}"
        """
        return f"{query_type}:{query.strip()}"

    def _build_embedding_func(self):
        """Build GPTCache-compatible embedding function from BGE-m3."""
        embed_model = self._embed_model

        def embedding_func(prompt: str) -> list[float]:
            # GPTCache passes full composite key (e.g., "AUTO:如何配置API")
            # Extract the actual query after the colon
            if ":" in prompt:
                actual_query = prompt.split(":", 1)[1]
            else:
                actual_query = prompt
            return embed_model._get_query_embedding(actual_query)

        return embedding_func

    def init(self) -> None:
        """Initialize GPTCache with PGVector + PostgreSQL."""
        if self._initialized:
            logger.warning("QACacheManager already initialized, skipping")
            return

        table_name = settings.qa_cache_postgres_table

        # PGVector for embedding storage (HNSW index)
        vector_base = VectorBase(
            "pgvector",
            dimension=self._dimension,
            url=self._postgres_url,
            collection_name=table_name,
        )

        # PostgreSQL for response storage
        cache_base = CacheBase(
            "postgresql",
            sql_url=self._postgres_url,
            table_name=table_name,
        )

        # Data manager combining both
        data_manager = get_data_manager(
            cache_base=cache_base,
            vector_base=vector_base,
        )

        # Initialize GPTCache
        gptcache.init(
            embedding_func=self._build_embedding_func(),
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
        )

        self._initialized = True
        logger.info(
            f"QACacheManager initialized: dimension={self._dimension}, "
            f"table={table_name}, threshold={settings.qa_cache_similarity_threshold}"
        )

    def get(self, query: str, query_type: str) -> str | None:
        """Get cached response for query.

        Args:
            query: User query string
            query_type: QueryType value (AUTO/FAQ/DOCS/CODE)

        Returns:
            Cached response string if hit (similarity >= threshold), None if miss.
        """
        if not self._initialized:
            logger.warning("QACacheManager not initialized, skipping cache lookup")
            return None

        cache_key = self.build_cache_key(query, query_type)

        try:
            if gptcache.embedding_func is None:
                logger.warning("GPTCache embedding_func not initialized")
                return None

            embedding = gptcache.embedding_func(cache_key)

            if gptcache.data_manager is None:
                logger.warning("GPTCache data_manager not initialized")
                return None

            search_results = gptcache.data_manager.search(embedding, top_k=1)

            if not search_results:
                logger.debug(f"Cache MISS: key={cache_key[:50]}... (no search results)")
                return None

            score = search_results[0][0]
            threshold = settings.qa_cache_similarity_threshold
            max_distance = 1.0 - threshold

            if score > max_distance:
                logger.debug(
                    f"Cache MISS: key={cache_key[:50]}... (score={score:.4f} > {max_distance:.4f})"
                )
                return None

            res_data = search_results[0]
            cache_data = gptcache.data_manager.get_scalar_data(res_data)

            if cache_data is None:
                logger.debug(
                    f"Cache MISS: key={cache_key[:50]}... (no cache data found)"
                )
                return None

            if hasattr(cache_data, "answers") and cache_data.answers:
                answer = cache_data.answers[0]
                if hasattr(answer, "answer"):
                    response: str | None = answer.answer
                else:
                    response = str(answer) if answer is not None else None
            else:
                response = str(cache_data) if cache_data is not None else None

            logger.debug(f"Cache HIT: key={cache_key[:50]}... (score={score:.4f})")
            return response
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, query: str, query_type: str, response: str) -> None:
        """Store query-response pair in cache.

        Args:
            query: User query string
            query_type: QueryType value (AUTO/FAQ/DOCS/CODE)
            response: LLM response string
        """
        if not self._initialized:
            logger.warning("QACacheManager not initialized, skipping cache set")
            return

        cache_key = self.build_cache_key(query, query_type)

        try:
            if gptcache.embedding_func is None or gptcache.data_manager is None:
                logger.warning("GPTCache not fully initialized")
                return

            embedding = gptcache.embedding_func(cache_key)

            gptcache.data_manager.save(
                question=cache_key,
                answer=response,
                embedding_data=embedding,
            )
            logger.debug(f"Cache SET: key={cache_key[:50]}...")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        if not self._initialized:
            return
        try:
            gptcache.flush()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def dimension(self) -> int:
        return self._dimension
