#!/usr/bin/env python3
"""Tests for FAQQueryEngine - FAQ query engine with keyword and vector ensemble."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from llama_index.core import KeywordTableIndex, QueryBundle, VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, TextNode


class TestFAQQueryEngineInit:
    """Tests for FAQQueryEngine initialization."""

    def test_init_with_default_weights(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_reranker:
            mock_reranker_instance = MagicMock()
            mock_reranker.create.return_value = mock_reranker_instance

            engine = FAQQueryEngine(
                vector_index=mock_vector_index,
                keyword_index=mock_keyword_index,
                llm=mock_llm,
            )

        assert engine.vector_index == mock_vector_index
        assert engine.keyword_index == mock_keyword_index
        assert engine.llm == mock_llm
        assert engine.similarity_top_k == 5

    def test_init_with_custom_top_k(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_reranker:
            mock_reranker_instance = MagicMock()
            mock_reranker.create.return_value = mock_reranker_instance

            engine = FAQQueryEngine(
                vector_index=mock_vector_index,
                keyword_index=mock_keyword_index,
                llm=mock_llm,
                similarity_top_k=10,
            )

        assert engine.similarity_top_k == 10

    def test_init_with_custom_ensemble_weights(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_reranker:
            mock_reranker_instance = MagicMock()
            mock_reranker.create.return_value = mock_reranker_instance

            engine = FAQQueryEngine(
                vector_index=mock_vector_index,
                keyword_index=mock_keyword_index,
                llm=mock_llm,
                ensemble_weights=[0.3, 0.7],
            )

        assert engine.similarity_top_k == 5


class TestFAQQueryEngineRetrieve:
    """Tests for FAQQueryEngine.retrieve() method."""

    def test_retrieve_returns_nodes(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        mock_keyword_retriever = MagicMock()
        mock_keyword_index.as_retriever.return_value = mock_keyword_retriever

        mock_vector_retriever = MagicMock()
        mock_vector_index.as_retriever.return_value = mock_vector_retriever

        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes.side_effect = lambda nodes, **kw: nodes

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = mock_reranker
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1
                mock_settings.faq_second_score_threshold = 0.5

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                    similarity_top_k=5,
                )

        keyword_node = NodeWithScore(node=TextNode(text="keyword result"), score=0.9)
        vector_node = NodeWithScore(node=TextNode(text="vector result"), score=0.8)
        mock_keyword_retriever.retrieve.return_value = [keyword_node]
        mock_vector_retriever.retrieve.return_value = [vector_node]

        with patch("askany.rag.faq_query_engine.get_metrics") as mock_metrics:
            mock_metrics_instance = MagicMock()
            mock_metrics.return_value = mock_metrics_instance

            nodes = engine.retrieve("test query")

        assert len(nodes) == 2
        mock_keyword_retriever.retrieve.assert_called_once()
        mock_vector_retriever.retrieve.assert_called_once()

    def test_retrieve_with_metadata_filters(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        mock_keyword_retriever = MagicMock()
        mock_keyword_index.as_retriever.return_value = mock_keyword_retriever
        mock_vector_retriever = MagicMock()
        mock_vector_index.as_retriever.return_value = mock_vector_retriever

        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes.side_effect = lambda nodes, **kw: nodes

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = mock_reranker
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1
                mock_settings.faq_second_score_threshold = 0.0

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                    similarity_top_k=5,
                )

        faq_node = NodeWithScore(
            node=TextNode(
                text="问题: test\n答案: answer",
                metadata={"type": "faq", "id": "faq1", "category": "tech"},
            ),
            score=0.9,
        )
        mock_keyword_retriever.retrieve.return_value = [faq_node]
        mock_vector_retriever.retrieve.return_value = []

        with patch("askany.rag.faq_query_engine.get_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock()

            nodes = engine.retrieve("test query", metadata_filters={"category": "tech"})

        assert len(nodes) == 1
        assert nodes[0].node.metadata["category"] == "tech"


class TestFAQQueryEngineRetrieveWithScores:
    """Tests for FAQQueryEngine.retrieve_with_scores() method."""

    def test_retrieve_with_scores_returns_tuple(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        mock_keyword_retriever = MagicMock()
        mock_keyword_index.as_retriever.return_value = mock_keyword_retriever
        mock_vector_retriever = MagicMock()
        mock_vector_index.as_retriever.return_value = mock_vector_retriever

        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes.side_effect = lambda nodes, **kw: nodes

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = mock_reranker
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1
                mock_settings.faq_second_score_threshold = 0.0

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                    similarity_top_k=5,
                )

        node = NodeWithScore(node=TextNode(text="result"), score=0.85)
        mock_keyword_retriever.retrieve.return_value = [node]
        mock_vector_retriever.retrieve.return_value = []

        with patch("askany.rag.faq_query_engine.get_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock()

            nodes, top_score = engine.retrieve_with_scores("test query")

        assert isinstance(nodes, list)
        assert top_score == 0.85

    def test_retrieve_with_scores_empty_nodes_returns_zero_score(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        mock_keyword_retriever = MagicMock()
        mock_keyword_index.as_retriever.return_value = mock_keyword_retriever
        mock_vector_retriever = MagicMock()
        mock_vector_index.as_retriever.return_value = mock_vector_retriever

        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes.side_effect = lambda nodes, **kw: nodes

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = mock_reranker
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1
                mock_settings.faq_second_score_threshold = 0.0

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                    similarity_top_k=5,
                )

        mock_keyword_retriever.retrieve.return_value = []
        mock_vector_retriever.retrieve.return_value = []

        with patch("askany.rag.faq_query_engine.get_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock()

            nodes, top_score = engine.retrieve_with_scores("test query")

        assert nodes == []
        assert top_score == 0.0


class TestFAQQueryEngineSynthesizeFromNodes:
    """Tests for FAQQueryEngine.synthesize_from_nodes() method."""

    def test_synthesize_from_nodes_returns_response(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        mock_reranker = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = mock_reranker
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                    similarity_top_k=5,
                )

        mock_synthesize_response = MagicMock()
        mock_synthesize_response.__str__ = MagicMock(return_value="synthesized answer")
        engine.query_engine.synthesize = MagicMock(
            return_value=mock_synthesize_response
        )

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="问题: test\n答案: answer",
                    metadata={"type": "faq", "id": "faq1"},
                ),
                score=0.9,
            )
        ]

        result = engine.synthesize_from_nodes("test query", nodes)

        assert "synthesized answer" in result
        engine.query_engine.synthesize.assert_called_once()


class TestFAQQueryEngineHelperMethods:
    """Tests for FAQQueryEngine helper methods."""

    def test_filter_nodes_by_score(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = MagicMock()
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                )

        nodes = [
            NodeWithScore(node=TextNode(text="high score"), score=0.9),
            NodeWithScore(node=TextNode(text="medium score"), score=0.5),
            NodeWithScore(node=TextNode(text="low score"), score=0.1),
            NodeWithScore(node=TextNode(text="no score"), score=None),
        ]

        filtered = engine._filter_nodes_by_score(nodes, 0.3)

        assert len(filtered) == 3
        assert filtered[0].score == 0.9
        assert filtered[1].score == 0.5
        assert filtered[2].score is None

    def test_filter_nodes_by_metadata_exact_match(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = MagicMock()
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                )

        nodes = [
            NodeWithScore(
                node=TextNode(text="node1", metadata={"category": "tech"}), score=0.9
            ),
            NodeWithScore(
                node=TextNode(text="node2", metadata={"category": "biz"}), score=0.8
            ),
        ]

        filtered = engine._filter_nodes_by_metadata(nodes, {"category": "tech"})

        assert len(filtered) == 1
        assert filtered[0].node.metadata["category"] == "tech"

    def test_filter_nodes_by_metadata_tags(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = MagicMock()
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                )

        nodes = [
            NodeWithScore(
                node=TextNode(text="node1", metadata={"tags": "python,api,tech"}),
                score=0.9,
            ),
            NodeWithScore(
                node=TextNode(text="node2", metadata={"tags": "java,biz"}), score=0.8
            ),
        ]

        filtered = engine._filter_nodes_by_metadata(nodes, {"tags": "python"})

        assert len(filtered) == 1

    def test_extract_faq_references(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = MagicMock()
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                )

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="问题: What is API?\n答案: API is Application Programming Interface",
                    metadata={"type": "faq", "id": "faq1"},
                ),
                score=0.9,
            ),
            NodeWithScore(
                node=TextNode(text="Some code content", metadata={"type": "code"}),
                score=0.8,
            ),
        ]

        refs = engine._extract_faq_references(nodes)

        assert len(refs) == 1
        assert refs[0]["id"] == "faq1"
        assert "Application Programming Interface" in refs[0]["answer"]

    def test_extract_faq_references_deduplicates(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = MagicMock()
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                )

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="问题: Q1\n答案: A1",
                    metadata={"type": "faq", "id": "faq1"},
                ),
                score=0.9,
            ),
            NodeWithScore(
                node=TextNode(
                    text="问题: Q1\n答案: A1",
                    metadata={"type": "faq", "id": "faq1"},
                ),
                score=0.85,
            ),
        ]

        refs = engine._extract_faq_references(nodes)

        assert len(refs) == 1

    def test_format_faq_references(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = MagicMock()
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                )

        references = [{"id": "faq1", "answer": "Short answer"}]

        formatted = engine._format_faq_references(references)

        assert "faq1" in formatted
        assert "Short answer" in formatted
        assert "参考数据来源" in formatted

    def test_format_faq_references_truncates_long_answer(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = MagicMock()
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                )

        long_answer = "A" * 300
        references = [{"id": "faq1", "answer": long_answer}]

        formatted = engine._format_faq_references(references)

        assert "..." in formatted
        assert len(formatted) < len(long_answer) * 2

    def test_format_faq_references_empty(self):
        from askany.rag.faq_query_engine import FAQQueryEngine

        mock_vector_index = MagicMock(spec=VectorStoreIndex)
        mock_keyword_index = MagicMock(spec=KeywordTableIndex)
        mock_llm = MagicMock()

        with patch("askany.rag.faq_query_engine.SafeReranker") as mock_sr:
            mock_sr.create.return_value = MagicMock()
            with patch("askany.rag.faq_query_engine.settings") as mock_settings:
                mock_settings.device = "cpu"
                mock_settings.reranker_model = "test-model"
                mock_settings.faq_rerank_candidate_k = 20
                mock_settings.max_keywords_for_faq = 10
                mock_settings.query_fusion_num_queries = 1

                engine = FAQQueryEngine(
                    vector_index=mock_vector_index,
                    keyword_index=mock_keyword_index,
                    llm=mock_llm,
                )

        formatted = engine._format_faq_references([])

        assert formatted == ""
