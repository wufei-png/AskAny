#!/usr/bin/env python3
"""Tests for QACacheManager."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


class TestQACacheManager:
    """Tests for QACacheManager."""

    @pytest.fixture
    def mock_embed_model(self):
        mock = MagicMock()
        mock.dimension = 1024
        mock._get_query_embedding.return_value = [0.1] * 1024
        return mock

    def test_build_cache_key_with_query_type(self, mock_embed_model):
        """Cache key includes query type prefix."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache"):
            manager = QACacheManager(embed_model=mock_embed_model)
            key = manager.build_cache_key("如何配置API", "AUTO")
            assert key == "AUTO:如何配置API"

    def test_build_cache_key_faq(self, mock_embed_model):
        """FAQ query type produces correct key."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache"):
            manager = QACacheManager(embed_model=mock_embed_model)
            key = manager.build_cache_key("如何配置API", "FAQ")
            assert key == "FAQ:如何配置API"

    def test_build_cache_key_docs(self, mock_embed_model):
        """DOCS query type produces correct key."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache"):
            manager = QACacheManager(embed_model=mock_embed_model)
            key = manager.build_cache_key("API配置", "DOCS")
            assert key == "DOCS:API配置"

    def test_build_cache_key_strips_whitespace(self, mock_embed_model):
        """Cache key strips leading/trailing whitespace from query."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache"):
            manager = QACacheManager(embed_model=mock_embed_model)
            key = manager.build_cache_key("  如何配置API  ", "AUTO")
            assert key == "AUTO:如何配置API"

    def test_init_creates_gptcache(self, mock_embed_model):
        """Cache initializes with PGVector and PostgreSQL backends."""
        from askany.cache.qa_cache import QACacheManager

        with (
            patch("askany.cache.qa_cache.gptcache") as mock_gptcache,
            patch("askany.cache.qa_cache.CacheBase") as mock_cache_base,
            patch("askany.cache.qa_cache.VectorBase") as mock_vector_base,
            patch("askany.cache.qa_cache.get_data_manager") as mock_data_manager,
        ):
            mock_vector_base.return_value = MagicMock()
            mock_cache_base.return_value = MagicMock()
            mock_data_manager.return_value = MagicMock()
            mock_gptcache.init = MagicMock()

            manager = QACacheManager(embed_model=mock_embed_model)
            manager.init()

            mock_gptcache.init.assert_called_once()
            call_kwargs = mock_gptcache.init.call_args.kwargs
            assert "embedding_func" in call_kwargs
            assert "data_manager" in call_kwargs
            assert "similarity_evaluation" in call_kwargs

    def test_get_returns_cached_response(self, mock_embed_model):
        """Cache get returns response on hit."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            mock_gptcache.get.return_value = "cached response"

            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = True

            result = manager.get("如何配置API", "AUTO")
            assert result == "cached response"
            mock_gptcache.get.assert_called_once_with("AUTO:如何配置API")

    def test_get_returns_none_on_miss(self, mock_embed_model):
        """Cache get returns None on miss."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            mock_gptcache.get.return_value = None

            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = True

            result = manager.get("如何配置API", "AUTO")
            assert result is None

    def test_set_stores_with_composite_key(self, mock_embed_model):
        """Cache set stores with composite key including query type."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = True
            manager.set("如何配置API", "FAQ", "FAQ answer")

            mock_gptcache.set.assert_called_once_with("FAQ:如何配置API", "FAQ answer")

    def test_clear_flushes_cache(self, mock_embed_model):
        """Cache clear calls gptcache.flush."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = True
            manager.clear()

            mock_gptcache.flush.assert_called_once()

    def test_get_skips_when_not_initialized(self, mock_embed_model):
        """Cache get returns None when not initialized."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = False

            result = manager.get("如何配置API", "AUTO")
            assert result is None
            mock_gptcache.get.assert_not_called()

    def test_set_skips_when_not_initialized(self, mock_embed_model):
        """Cache set does nothing when not initialized."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = False

            manager.set("如何配置API", "AUTO", "response")
            mock_gptcache.set.assert_not_called()

    def test_embedding_func_extracts_query(self, mock_embed_model):
        """Embedding function extracts actual query from composite key."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache"):
            manager = QACacheManager(embed_model=mock_embed_model)
            embed_fn = manager._build_embedding_func()

            # With composite key
            result = embed_fn("AUTO:如何配置API")
            mock_embed_model._get_query_embedding.assert_called_with("如何配置API")

            # Reset mock
            mock_embed_model._get_query_embedding.reset_mock()

            # Without composite key (edge case)
            result = embed_fn("如何配置API")
            mock_embed_model._get_query_embedding.assert_called_with("如何配置API")
