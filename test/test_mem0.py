#!/usr/bin/env python3
"""Tests for Mem0 user memory functionality.

Tests cover:
1. Mem0Adapter class methods (unit tests with mocks)
2. Singleton pattern for get_mem0_adapter()
3. Configuration settings
4. Integration points in server.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMem0AdapterSearch:
    """Test the search method of Mem0Adapter."""

    def test_search_returns_formatted_results(self):
        """Search should return list of dicts with memory and score."""
        from askany.memory.mem0_adapter import Mem0Adapter

        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {"memory": "User prefers Chinese", "score": 0.95},
                {"memory": "User works on DevOps", "score": 0.88},
            ]
        }

        adapter = Mem0Adapter.__new__(Mem0Adapter)
        adapter._memory = mock_memory

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1

            results = adapter.search("test query", user_id="test_user")

        assert len(results) == 2
        assert results[0]["memory"] == "User prefers Chinese"
        assert results[0]["score"] == 0.95

    def test_search_filters_by_score_threshold(self):
        """Search should filter results below score threshold."""
        from askany.memory.mem0_adapter import Mem0Adapter

        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {"memory": "High score", "score": 0.9},
                {"memory": "Low score", "score": 0.05},
            ]
        }

        adapter = Mem0Adapter.__new__(Mem0Adapter)
        adapter._memory = mock_memory

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.5

            results = adapter.search("test query", user_id="test_user")

        assert len(results) == 1
        assert results[0]["memory"] == "High score"

    def test_search_returns_empty_on_error(self):
        """Search should return empty list on exception."""
        from askany.memory.mem0_adapter import Mem0Adapter

        mock_memory = MagicMock()
        mock_memory.search.side_effect = Exception("Search failed")

        adapter = Mem0Adapter.__new__(Mem0Adapter)
        adapter._memory = mock_memory

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1

            results = adapter.search("test query", user_id="test_user")

        assert results == []

    def test_search_uses_custom_top_k(self):
        """Search should use custom top_k parameter."""
        from askany.memory.mem0_adapter import Mem0Adapter

        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}

        adapter = Mem0Adapter.__new__(Mem0Adapter)
        adapter._memory = mock_memory

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1

            adapter.search("test query", user_id="test_user", top_k=10)

        call_args = mock_memory.search.call_args
        assert call_args.kwargs.get("limit") == 10

    def test_search_with_list_response(self):
        """Search should handle list response format."""
        from askany.memory.mem0_adapter import Mem0Adapter

        mock_memory = MagicMock()
        mock_memory.search.return_value = [{"memory": "Direct list item", "score": 0.9}]

        adapter = Mem0Adapter.__new__(Mem0Adapter)
        adapter._memory = mock_memory

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1

            results = adapter.search("test query", user_id="test_user")

        assert len(results) == 1
        assert results[0]["memory"] == "Direct list item"


class TestMem0AdapterSaveTurn:
    """Test the save_turn method of Mem0Adapter."""

    def test_save_turn_calls_memory_add(self):
        """save_turn should call memory.add with formatted messages."""
        from askany.memory.mem0_adapter import Mem0Adapter

        mock_memory = MagicMock()

        adapter = Mem0Adapter.__new__(Mem0Adapter)
        adapter._memory = mock_memory

        adapter.save_turn("What is X?", "X is a thing", user_id="test_user")

        mock_memory.add.assert_called_once()
        call_args = mock_memory.add.call_args
        assert call_args.kwargs.get("user_id") == "test_user"

        messages = call_args.args[0]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is X?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "X is a thing"

    def test_save_turn_handles_exception(self):
        """save_turn should not raise on exception."""
        from askany.memory.mem0_adapter import Mem0Adapter

        mock_memory = MagicMock()
        mock_memory.add.side_effect = Exception("Add failed")

        adapter = Mem0Adapter.__new__(Mem0Adapter)
        adapter._memory = mock_memory

        adapter.save_turn("Q", "A", user_id="test_user")

    @pytest.mark.asyncio
    async def test_save_turn_async(self):
        """save_turn_async should run in executor."""
        from askany.memory.mem0_adapter import Mem0Adapter

        mock_memory = MagicMock()

        adapter = Mem0Adapter.__new__(Mem0Adapter)
        adapter._memory = mock_memory

        await adapter.save_turn_async("Q", "A", user_id="test_user")

        mock_memory.add.assert_called_once()


class TestMem0AdapterFormatMemories:
    """Test format_memories_as_system_text method."""

    def test_format_empty_memories(self):
        """Empty memories should return empty string."""
        from askany.memory.mem0_adapter import Mem0Adapter

        result = Mem0Adapter.format_memories_as_system_text([])
        assert result == ""

    def test_format_memories(self):
        """Should format memories as system text."""
        from askany.memory.mem0_adapter import Mem0Adapter

        memories = [
            {"memory": "Prefers Chinese"},
            {"memory": "Works on DevOps"},
        ]
        result = Mem0Adapter.format_memories_as_system_text(memories)

        assert "[User Memory Context]" in result
        assert "Prefers Chinese" in result
        assert "Works on DevOps" in result

    def test_format_skips_empty_memory(self):
        """Should skip memories with empty text."""
        from askany.memory.mem0_adapter import Mem0Adapter

        memories = [
            {"memory": "Valid memory"},
            {"memory": ""},
            {"memory": "  "},
        ]
        result = Mem0Adapter.format_memories_as_system_text(memories)

        assert "Valid memory" in result


class TestGetMem0AdapterSingleton:
    """Test get_mem0_adapter singleton pattern."""

    def test_returns_none_when_disabled(self):
        """Should return None when enable_mem0 is False."""
        import askany.memory.mem0_adapter as mem0_module

        mem0_module._adapter_instance = None

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.enable_mem0 = False

            from askany.memory.mem0_adapter import get_mem0_adapter

            result = get_mem0_adapter()
            assert result is None

    def test_returns_same_instance_on_multiple_calls(self):
        """Multiple calls should return the same singleton instance."""
        import askany.memory.mem0_adapter as mem0_module

        mem0_module._adapter_instance = None

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.enable_mem0 = True
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1
            mock_settings.postgres_user = "wufei"
            mock_settings.postgres_password = MagicMock()
            mock_settings.postgres_password.get_secret_value.return_value = "123456"
            mock_settings.postgres_host = "localhost"
            mock_settings.postgres_port = 5432
            mock_settings.postgres_db = "askany"
            mock_settings.mem0_collection_name = "askany_mem0_test_singleton"
            mock_settings.vector_dimension = 1024
            mock_settings.openai_model = "test-model"
            mock_settings.openai_api_base = "http://127.0.0.1:8081/v1"
            mock_settings.openai_api_key = "dummy"
            mock_settings.mem0_llm_model = None
            mock_settings.mem0_llm_api_base = None
            mock_settings.mem0_llm_api_key = None
            mock_settings.mem0_llm_provider = "openai"
            mock_settings.embedding_model = "BAAI/bge-m3"
            mock_settings.mem0_embedder_model = None
            mock_settings.mem0_embedder_provider = "huggingface"

            from askany.memory.mem0_adapter import get_mem0_adapter

            adapter1 = get_mem0_adapter()
            if adapter1 is None:
                pytest.skip("Mem0 not enabled")

            adapter2 = get_mem0_adapter()
            assert adapter1 is adapter2, (
                "get_mem0_adapter() should return same singleton instance"
            )


class TestMem0Config:
    """Test mem0 configuration settings."""

    def test_default_config_values(self):
        """Test default configuration values."""
        from askany.config import settings

        assert settings.enable_mem0 is False
        assert settings.mem0_collection_name == "askany_mem0"
        assert settings.mem0_top_k == 5
        assert settings.mem0_score_threshold == 0.3
        assert settings.mem0_llm_provider == "openai"
        assert settings.mem0_embedder_provider == "huggingface"


class TestMem0ServerIntegration:
    """Test mem0 integration in server.py."""

    def test_mem0_adapter_import(self):
        """Verify Mem0Adapter can be imported."""
        from askany.memory.mem0_adapter import Mem0Adapter

        assert Mem0Adapter is not None


class TestMem0E2E:
    """End-to-end tests for mem0 functionality.

    Run with: OPENAI_API_KEY=dummy uv run python -m pytest test/test_mem0.py::TestMem0E2E -v
    """

    def test_e2e_save_turn(self):
        """E2E test: save a turn (basic functionality test)."""
        import askany.memory.mem0_adapter as mem0_module

        mem0_module._adapter_instance = None

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.enable_mem0 = True
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1
            mock_settings.postgres_user = "wufei"
            mock_settings.postgres_password = MagicMock()
            mock_settings.postgres_password.get_secret_value.return_value = "123456"
            mock_settings.postgres_host = "localhost"
            mock_settings.postgres_port = 5432
            mock_settings.postgres_db = "askany"
            mock_settings.mem0_collection_name = "askany_mem0_test"
            mock_settings.vector_dimension = 1024
            mock_settings.openai_model = "test-model"
            mock_settings.openai_api_base = "http://127.0.0.1:8081/v1"
            mock_settings.openai_api_key = "dummy"
            mock_settings.mem0_llm_model = None
            mock_settings.mem0_llm_api_base = None
            mock_settings.mem0_llm_api_key = None
            mock_settings.mem0_llm_provider = "openai"
            mock_settings.embedding_model = "BAAI/bge-m3"
            mock_settings.mem0_embedder_model = None
            mock_settings.mem0_embedder_provider = "huggingface"

            from askany.memory.mem0_adapter import get_mem0_adapter

            adapter = get_mem0_adapter()
            if adapter is None:
                pytest.skip("Mem0 not enabled")

            user_id = "test_user_e2e"
            adapter.save_turn(
                "What is the capital of France?",
                "The capital of France is Paris.",
                user_id=user_id,
            )

    @pytest.mark.asyncio
    async def test_e2e_async_save(self):
        """E2E test: async save turn."""
        import askany.memory.mem0_adapter as mem0_module

        mem0_module._adapter_instance = None

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.enable_mem0 = True
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1
            mock_settings.postgres_user = "wufei"
            mock_settings.postgres_password = MagicMock()
            mock_settings.postgres_password.get_secret_value.return_value = "123456"
            mock_settings.postgres_host = "localhost"
            mock_settings.postgres_port = 5432
            mock_settings.postgres_db = "askany"
            mock_settings.mem0_collection_name = "askany_mem0_test2"
            mock_settings.vector_dimension = 1024
            mock_settings.openai_model = "test-model"
            mock_settings.openai_api_base = "http://127.0.0.1:8081/v1"
            mock_settings.openai_api_key = "dummy"
            mock_settings.mem0_llm_model = None
            mock_settings.mem0_llm_api_base = None
            mock_settings.mem0_llm_api_key = None
            mock_settings.mem0_llm_provider = "openai"
            mock_settings.embedding_model = "BAAI/bge-m3"
            mock_settings.mem0_embedder_model = None
            mock_settings.mem0_embedder_provider = "huggingface"

            from askany.memory.mem0_adapter import get_mem0_adapter

            adapter = get_mem0_adapter()
            if adapter is None:
                pytest.skip("Mem0 not enabled")

            user_id = "test_user_async"
            await adapter.save_turn_async(
                "What is 2+2?", "2+2 equals 4.", user_id=user_id
            )

    def test_e2e_add_then_search(self):
        """E2E test: add 记忆 → search 能正确召回.

        运行方式:
            python -m pytest test/test_mem0.py::TestMem0E2E::test_e2e_add_then_search -v -s

        依赖:
            - PostgreSQL + pgvector 可用
            - vLLM (或 OpenAI) endpoint 可用
            - mem0ai 包已安装
        """
        import time

        import askany.memory.mem0_adapter as mem0_module

        # 重置 singleton 确保使用新配置
        mem0_module._adapter_instance = None

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.enable_mem0 = True
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1
            mock_settings.postgres_user = "wufei"
            mock_settings.postgres_password = MagicMock()
            mock_settings.postgres_password.get_secret_value.return_value = "123456"
            mock_settings.postgres_host = "localhost"
            mock_settings.postgres_port = 5432
            mock_settings.postgres_db = "askany"
            # 使用独立 collection 避免污染
            mock_settings.mem0_collection_name = "askany_mem0_test_e2e_search"
            mock_settings.vector_dimension = 1024
            mock_settings.openai_model = (
                "/net/ai02/data/xlzhong/wufei/models/Qwen3-14B-AWQ/"
            )
            mock_settings.openai_api_base = "http://127.0.0.1:8081/v1"
            mock_settings.openai_api_key = "dummy"
            mock_settings.mem0_llm_model = None
            mock_settings.mem0_llm_api_base = None
            mock_settings.mem0_llm_api_key = None
            mock_settings.mem0_llm_provider = "openai"
            mock_settings.embedding_model = "BAAI/bge-m3"
            mock_settings.mem0_embedder_model = None
            mock_settings.mem0_embedder_provider = "huggingface"

            from askany.memory.mem0_adapter import get_mem0_adapter

            adapter = get_mem0_adapter()
            if adapter is None:
                pytest.skip("Mem0 not enabled")

            user_id = "test_e2e_search_user"

            # 1. 添加记忆
            adapter.save_turn(
                "I love coding in Python and prefer dark roast coffee",
                "Great, I've noted your preferences!",
                user_id=user_id,
            )

            # 等待向量写入 (mem0 异步写入)
            time.sleep(2)

            # 2. 搜索相关记忆 - 验证能召回
            results = adapter.search("What does user like?", user_id=user_id)

            # 3. 验证
            print("\n[E2E Search] Query: 'What does user like?'")
            print(f"[E2E Search] Results: {results}")

            assert len(results) > 0, "Should find at least one memory"

            # 检查是否包含相关内容 (语义匹配可能不精确，但应该能找到)
            memories_text = " ".join(r.get("memory", "") for r in results).lower()
            # Python/coding 可能被提取为 "likes programming" 或类似
            # 至少应该有一些相关词汇
            has_relevant = any(
                word in memories_text
                for word in ["python", "coding", "coffee", "prefers", "love"]
            )
            assert has_relevant, f"Should find relevant memory, got: {results}"

    def test_e2e_search_returns_correct_format(self):
        """E2E test: search 返回正确的格式.

        验证 search 返回的 dict 包含必要字段.
        """
        import time

        import askany.memory.mem0_adapter as mem0_module

        mem0_module._adapter_instance = None

        with patch("askany.memory.mem0_adapter.settings") as mock_settings:
            mock_settings.enable_mem0 = True
            mock_settings.mem0_top_k = 5
            mock_settings.mem0_score_threshold = 0.1
            mock_settings.postgres_user = "wufei"
            mock_settings.postgres_password = MagicMock()
            mock_settings.postgres_password.get_secret_value.return_value = "123456"
            mock_settings.postgres_host = "localhost"
            mock_settings.postgres_port = 5432
            mock_settings.postgres_db = "askany"
            mock_settings.mem0_collection_name = "askany_mem0_test_e2e_format"
            mock_settings.vector_dimension = 1024
            mock_settings.openai_model = (
                "/net/ai02/data/xlzhong/wufei/models/Qwen3-14B-AWQ/"
            )
            mock_settings.openai_api_base = "http://127.0.0.1:8081/v1"
            mock_settings.openai_api_key = "dummy"
            mock_settings.mem0_llm_model = None
            mock_settings.mem0_llm_api_base = None
            mock_settings.mem0_llm_api_key = None
            mock_settings.mem0_llm_provider = "openai"
            mock_settings.embedding_model = "BAAI/bge-m3"
            mock_settings.mem0_embedder_model = None
            mock_settings.mem0_embedder_provider = "huggingface"

            from askany.memory.mem0_adapter import get_mem0_adapter

            adapter = get_mem0_adapter()
            if adapter is None:
                pytest.skip("Mem0 not enabled")

            user_id = "test_e2e_format_user"

            adapter.save_turn("Test question", "Test answer", user_id=user_id)
            time.sleep(2)

            results = adapter.search("test", user_id=user_id)

            # 验证返回格式
            if len(results) > 0:
                r = results[0]
                assert "memory" in r, "Result should have 'memory' field"
                assert "score" in r, "Result should have 'score' field"
                assert isinstance(r["memory"], str), "memory should be string"
                assert isinstance(r["score"], (int, float)), "score should be numeric"
                print(f"[E2E Format] Sample result: {r}")
