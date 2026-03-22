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
            with patch("askany.cache.qa_cache.settings") as mock_settings:
                mock_settings.qa_cache_similarity_threshold = 0.5
                mock_embedding = [0.1] * 1024
                mock_gptcache.embedding_func.return_value = mock_embedding
                mock_gptcache.data_manager.search.return_value = [
                    [0.1, 1]
                ]  # low distance = similar
                mock_cache_data = MagicMock()
                mock_cache_data.answers = [MagicMock(answer="cached response")]
                mock_gptcache.data_manager.get_scalar_data.return_value = (
                    mock_cache_data
                )

                manager = QACacheManager(embed_model=mock_embed_model)
                manager._initialized = True

                result = manager.get("如何配置API", "AUTO")
                assert result == "cached response"
                mock_gptcache.embedding_func.assert_called_once_with("AUTO:如何配置API")
                mock_gptcache.data_manager.search.assert_called_once()

    def test_get_returns_none_on_miss(self, mock_embed_model):
        """Cache get returns None when no similar entry found."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            mock_gptcache.embedding_func.return_value = [0.1] * 1024
            mock_gptcache.data_manager.search.return_value = []  # no results

            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = True

            result = manager.get("如何配置API", "AUTO")
            assert result is None

    def test_set_stores_with_composite_key(self, mock_embed_model):
        """Cache set stores with composite key including query type."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            mock_gptcache.embedding_func.return_value = [0.1] * 1024
            mock_gptcache.data_manager.save = MagicMock()

            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = True
            manager.set("如何配置API", "FAQ", "FAQ answer")

            mock_gptcache.data_manager.save.assert_called_once()
            call_args = mock_gptcache.data_manager.save.call_args
            assert call_args.args[0] == "FAQ:如何配置API"
            assert call_args.args[1] == "FAQ answer"

    def test_clear_deletes_all_entries(self, mock_embed_model):
        """Cache clear deletes all entries from underlying storage."""
        from askany.cache.qa_cache import QACacheManager

        mock_scalar = MagicMock()
        mock_scalar.get_ids.return_value = [1, 2, 3]
        mock_vector = MagicMock()

        mock_dm = MagicMock()
        mock_dm.s = mock_scalar
        mock_dm.v = mock_vector

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            mock_gptcache.data_manager = mock_dm
            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = True
            manager.clear()

            mock_scalar.get_ids.assert_called_once_with(deleted=False)
            mock_vector.delete.assert_called_once_with([1, 2, 3])
            self._assert_mark_deleted_calls(mock_scalar, [1, 2, 3])
            mock_scalar.clear_deleted_data.assert_called_once()

    def _assert_mark_deleted_calls(self, mock_scalar, expected_ids):
        """Helper to verify mark_deleted was called for each ID."""
        calls = mock_scalar.mark_deleted.call_args_list
        assert len(calls) == len(expected_ids)
        for i, oid in enumerate(expected_ids):
            assert calls[i].args[0] == oid

    def test_get_skips_when_not_initialized(self, mock_embed_model):
        """Cache get returns None when not initialized."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = False

            result = manager.get("如何配置API", "AUTO")
            assert result is None
            mock_gptcache.embedding_func.assert_not_called()

    def test_set_skips_when_not_initialized(self, mock_embed_model):
        """Cache set does nothing when not initialized."""
        from askany.cache.qa_cache import QACacheManager

        with patch("askany.cache.qa_cache.gptcache") as mock_gptcache:
            manager = QACacheManager(embed_model=mock_embed_model)
            manager._initialized = False

            manager.set("如何配置API", "AUTO", "response")
            mock_gptcache.embedding_func.assert_not_called()

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
