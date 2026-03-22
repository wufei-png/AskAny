"""QA Semantic Cache Manager using GPTCache + PGVector."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gptcache import Config
from gptcache import cache as gptcache
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

from askany.config import settings

if TYPE_CHECKING:
    from askany.main import SentenceTransformerEmbedding

logger = logging.getLogger(__name__)

# Cosine distance for normalized vectors ranges 0-2:
#   distance = 0 → identical (similarity = 1.0)
#   distance = 2 → opposite (similarity = -1)
# SearchDistanceEvaluation with positive=False: score = max_distance - distance
# For our threshold logic to work: max_distance=1.0, positive=False
#   score = 1.0 - distance
#   threshold = similarity_threshold = 0.9
#   HIT when: score >= 0.9 → distance <= 0.1 → similarity >= 0.9
COSINE_MAX_DISTANCE = 1.0  # matches 0-1 similarity scale after normalization


class QACacheManager:
    """Thin wrapper around GPTCache for QA response caching.

    Uses BGE-m3 embeddings via SentenceTransformer, PGVector for similarity
    search, and PostgreSQL for response storage.

    Cache key = f"{query_type}:{query_text}"

    Threshold behavior:
        - Cosine distance (normalized embeddings): 0 = identical, 2 = opposite
        - SearchDistanceEvaluation(positive=False): score = max_distance - distance
        - With max_distance=1.0: score = 1.0 - distance
        - Cache HIT when: score >= similarity_threshold (0.9)
        - This means: distance <= 0.1 → similarity >= 0.9
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
        # Uses vector_cosine_ops by default (distance = cosine distance)
        vector_base = VectorBase(
            "pgvector",
            dimension=self._dimension,
            url=self._postgres_url,
            collection_name=table_name,
            index_params={
                "index_type": "cosine",
                "params": {"lists": 100, "probes": 10},
            },
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

        # Initialize GPTCache with proper threshold configuration
        # similarity_threshold=0.9 means: score >= 0.9 → HIT
        # SearchDistanceEvaluation with positive=False: score = max_distance - distance
        # With max_distance=1.0: score = 1.0 - distance
        # HIT when: 1.0 - distance >= 0.9 → distance <= 0.1 → similarity >= 0.9
        gptcache.init(
            embedding_func=self._build_embedding_func(),
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(
                max_distance=COSINE_MAX_DISTANCE,
                positive=False,
            ),
            config=Config(similarity_threshold=settings.qa_cache_similarity_threshold),
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
            embedding_func = gptcache.embedding_func
            assert embedding_func is not None
            embedding = embedding_func(cache_key)
            search_results = gptcache.data_manager.search(embedding, top_k=1)  # type: ignore[union-attr]

            if not search_results:
                logger.debug(f"Cache MISS: key={cache_key[:50]}... (no search results)")
                return None

            # search_results[0] is (distance, id) tuple from vector store
            distance = search_results[0][0]

            # Apply threshold: score = max_distance - distance, HIT when score >= threshold
            # This gives: 1.0 - distance >= 0.9 → distance <= 0.1 → similarity >= 0.9
            score = COSINE_MAX_DISTANCE - distance
            threshold = settings.qa_cache_similarity_threshold

            if score < threshold:
                logger.debug(
                    f"Cache MISS: key={cache_key[:50]}... "
                    f"(score={score:.4f} < {threshold:.4f}, distance={distance:.4f})"
                )
                return None

            # Get the cached data
            cache_data = gptcache.data_manager.get_scalar_data(search_results[0])

            if cache_data is None:
                logger.debug(f"Cache MISS: key={cache_key[:50]}... (no cache data)")
                return None

            # Extract answer from cache data
            if hasattr(cache_data, "answers") and cache_data.answers:
                answer = cache_data.answers[0]
                response: str | None = (
                    answer.answer if hasattr(answer, "answer") else str(answer)
                )
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
            embedding_func = gptcache.embedding_func
            assert embedding_func is not None
            embedding = embedding_func(cache_key)
            gptcache.data_manager.save(  # type: ignore[union-attr]
                cache_key,  # question
                response,  # answer
                embedding,  # embedding_data
            )
            logger.debug(f"Cache SET: key={cache_key[:50]}...")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def clear(self) -> None:
        """Clear all cache entries from both PGVector and PostgreSQL."""
        if not self._initialized:
            return
        try:
            dm = gptcache.data_manager
            scalar_storage = getattr(dm, "s", None)
            vector_storage = getattr(dm, "v", None)
            if scalar_storage is None or vector_storage is None:
                return

            all_ids = scalar_storage.get_ids(deleted=False)
            if all_ids:
                vector_storage.delete(all_ids)
                for oid in all_ids:
                    scalar_storage.mark_deleted(oid)
                scalar_storage.clear_deleted_data()

            logger.info(f"Cache cleared: deleted {len(all_ids)} entries")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def dimension(self) -> int:
        return self._dimension
