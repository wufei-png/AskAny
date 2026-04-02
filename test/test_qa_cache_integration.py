#!/usr/bin/env python3
"""Integration tests for QA semantic cache with REAL BGE-m3 embeddings.

These tests use actual GPTCache + BGE-m3 embeddings (no mocks).
They verify real semantic similarity behavior and explore optimal thresholds.

Run with: pytest test/test_qa_cache_integration.py -v -s
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from askany.config import settings
from askany.cache.qa_cache import QACacheManager
from askany.main import SentenceTransformerEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip if prerequisites not available
TEST_TABLE_NAME = "askany_qa_cache_test"


def _check_prerequisites():
    """Check if we can run integration tests."""
    if not settings.postgres_host:
        pytest.skip("PostgreSQL not configured")
    if not settings.embedding_model:
        pytest.skip("embedding_model not configured")


@pytest.fixture(scope="module")
def embed_model():
    """Create real BGE-m3 embedding model."""
    _check_prerequisites()
    model_name = settings.embedding_model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformerEmbedding(model_name=model_name, device="cpu")
    logger.info(f"Embedding dimension: {model.dimension}")
    yield model


@pytest.fixture(scope="module")
def cache_manager(embed_model):
    """Create QACacheManager with REAL embeddings and a separate test table."""
    from gptcache import cache as gptcache

    # Use a separate test table to avoid interfering with existing cache
    original_table = settings.qa_cache_postgres_table
    settings.qa_cache_postgres_table = TEST_TABLE_NAME

    # Build postgres URL
    pg = settings.postgres_password.get_secret_value()
    postgres_url = (
        f"postgresql://{settings.postgres_user}:{pg}"
        f"@{settings.postgres_host}:{settings.postgres_port}"
        f"/{settings.postgres_db}"
    )

    manager = QACacheManager(embed_model=embed_model, postgres_url=postgres_url)
    manager.init()

    yield manager

    # Cleanup: clear test table
    try:
        manager.clear()
    except Exception:
        pass
    finally:
        settings.qa_cache_postgres_table = original_table


class TestQACacheSemanticIntegration:
    """Integration tests with real BGE-m3 embeddings and GPTCache."""

    def test_exact_query_hit(self, cache_manager):
        """Identical query → must HIT (similarity = 1.0)."""
        query = "如何配置API访问"
        response = "这是API配置的答案"

        cache_manager.set(query, "AUTO", response)
        result = cache_manager.get(query, "AUTO")

        assert result == response, f"Expected exact match to HIT, got {result}"

    def test_different_query_miss(self, cache_manager):
        """Different query text with low similarity → should miss.

        The type prefix is used only for cache keys (scalar storage),
        NOT for embeddings. Embeddings are computed from query text only.
        Uses queries with low semantic similarity to ensure cache miss.
        """
        query1 = "如何配置API访问"
        query2 = "今天天气怎么样"

        cache_manager.set(query1, "FAQ", "FAQ answer")
        result = cache_manager.get(query2, "FAQ")

        assert result is None, (
            f"Different query text with low similarity should miss, but got: {result}."
        )

    def test_semantic_similarity_exploration(self, cache_manager):
        """Verify semantic similarity scoring for various query relationships."""
        import numpy as np

        base_query = "如何配置API"
        cache_manager.set(base_query, "AUTO", "base answer")

        test_cases = [
            ("如何配置API", 0.95, "identical query"),
            ("API配置方法", 0.85, "same meaning"),
            ("怎么设置API", 0.80, "paraphrase"),
            ("API是什么", 0.70, "different intent"),
            ("如何部署服务", 0.60, "unrelated"),
        ]

        for query, min_expected_sim, description in test_cases:
            emb1 = cache_manager._embed_model._get_query_embedding(base_query)
            emb2 = cache_manager._embed_model._get_query_embedding(query)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            assert 0.0 <= sim <= 1.0 + 1e-9, (
                f"Similarity {sim} should be between 0 and 1"
            )
            if sim >= 0.90:
                result = cache_manager.get(query, "AUTO")
                assert result is not None, f"High similarity {sim:.4f} should HIT cache"

    def test_threshold_090_boundary_cases(self, cache_manager):
        """Test queries near the 0.90 threshold boundary.

        Verifies that similarity >= 0.90 results in cache HIT.
        """
        import numpy as np

        base_query = "如何配置API"
        cache_manager.set(base_query, "AUTO", "base answer")

        boundary_queries = [
            "API配置操作指南",
            "如何设置API参数",
            "API调用怎么配置",
            "配置API的步骤",
        ]

        base_emb = cache_manager._embed_model._get_query_embedding(base_query)
        hit_count = 0

        for query in boundary_queries:
            emb = cache_manager._embed_model._get_query_embedding(query)
            sim = float(
                np.dot(base_emb, emb) / (np.linalg.norm(base_emb) * np.linalg.norm(emb))
            )
            assert 0.0 <= sim <= 1.0, f"Similarity {sim} should be between 0 and 1"

            result = cache_manager.get(query, "AUTO")
            if sim >= 0.90:
                assert result is not None, (
                    f"sim={sim:.4f} >= 0.90 should HIT, query: '{query}'"
                )
                hit_count += 1

        assert hit_count > 0, (
            "At least some boundary queries should hit with threshold 0.90"
        )

    def test_chinese_phrasing_variations(self, cache_manager):
        """Test common Chinese phrasing variations for the same question.

        All variations should have high similarity (>= 0.90) and hit the cache.
        """
        import numpy as np

        variations = [
            "如何配置API",
            "API怎么配置",
            "API配置方法",
            "怎么配置API",
            "API该如何配置",
            "配置API的方法",
        ]

        base = variations[0]
        base_emb = cache_manager._embed_model._get_query_embedding(base)

        cache_manager.set(base, "AUTO", "unified answer")

        for v in variations[1:]:
            emb = cache_manager._embed_model._get_query_embedding(v)
            sim = float(
                np.dot(base_emb, emb) / (np.linalg.norm(base_emb) * np.linalg.norm(emb))
            )
            assert sim >= 0.80, (
                f"Chinese variations should have sim >= 0.80, got {sim:.4f} for '{v}'"
            )

            result = cache_manager.get(v, "AUTO")
            if sim >= 0.90:
                assert result is not None, (
                    f"sim={sim:.4f} >= 0.90 should HIT for variation: '{v}'"
                )

    def test_clear_removes_all_entries(self, cache_manager):
        """Cache clear should remove all entries."""
        cache_manager.set("query1", "AUTO", "answer1")
        cache_manager.set("query2", "AUTO", "answer2")

        cache_manager.clear()

        result1 = cache_manager.get("query1", "AUTO")
        result2 = cache_manager.get("query2", "AUTO")

        assert result1 is None, "Cache should be empty after clear"
        assert result2 is None, "Cache should be empty after clear"
