"""Query router for different query types."""

from copy import deepcopy
from enum import Enum
from logging import getLogger
from typing import Dict, List, Optional, Union

from llama_index.core.schema import NodeWithScore, TextNode

from askany.config import settings
from askany.ingest.vector_store import VectorStoreManager
from askany.rag.faq_query_engine import FAQQueryEngine
from askany.rag.query_parser import parse_query_filters
from askany.rag.rag_query_engine import RAGQueryEngine

logger = getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def get_device() -> str:
    """Detect and return the appropriate device (cuda or cpu).

    Returns:
        Device string: "cuda" if CUDA is available, "cpu" otherwise
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available, using GPU")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


class QueryType(str, Enum):
    """Query type enumeration."""

    AUTO = "auto"
    FAQ = "faq"
    DOCS = "docs"
    CODE = "code"


class QueryRouter:
    """Router for different query types."""

    def __init__(
        self,
        docs_query_engine: RAGQueryEngine,
        faq_query_engine: Optional[Union[FAQQueryEngine, RAGQueryEngine]] = None,
    ):
        """Initialize query router.

        Args:
            docs_query_engine: Query engine for documentation
            faq_query_engine: Query engine for FAQ (optional, can be FAQQueryEngine or RAGQueryEngine)
        """
        self.docs_query_engine = docs_query_engine
        self.faq_query_engine = faq_query_engine

    def route(self, query: str, query_type: QueryType = QueryType.AUTO) -> str:
        """Route query to appropriate engine.

        Args:
            query: User query (may contain @tag filters)
            query_type: Type of query (auto/faq/docs/code)

        Returns:
            Generated response
        """
        # Parse filters from query (will be passed to engines)
        cleaned_query, metadata_filters = parse_query_filters(query)

        if query_type == QueryType.AUTO:
            return self._route_auto(cleaned_query, metadata_filters)

        if query_type == QueryType.FAQ and self.faq_query_engine:
            return self.faq_query_engine.query(cleaned_query, metadata_filters)
        elif query_type == QueryType.DOCS:
            return self.docs_query_engine.query(cleaned_query, metadata_filters)
        elif query_type == QueryType.CODE:
            # TODO: Implement code search
            return "Code search not yet implemented"
        else:
            # Fallback to docs
            return self.docs_query_engine.query(cleaned_query, metadata_filters)

    def _route_auto(
        self, query: str, metadata_filters: Optional[Dict[str, str]] = None
    ) -> str:
        """Smart auto routing with FAQ score checking and docs enhancement.

        Flow:
        1. Check if query is code-related (use code if yes)
        2. If not code, retrieve from FAQ
        3. If FAQ score is sufficient, return FAQ answer
        4. If FAQ score is insufficient, use FAQ answer as context and enhance with docs

        Args:
            query: User query (already cleaned, filters removed)
            metadata_filters: Optional metadata filters dict

        Returns:
            Generated response
        """
        # Step 1: Check if it's code-related
        if self._is_code_query(query):
            # TODO: Implement code search
            return "Code search not yet implemented"

        # Step 2: Try FAQ retrieval
        if not self.faq_query_engine:
            logger.warning("No FAQ engine found, fallback to docs for query: %s", query)
            # No FAQ engine, fallback to docs
            return self.docs_query_engine.query(query, metadata_filters)

        # Retrieve from FAQ and get top score (only retrieve once)
        faq_nodes, top_score = self.faq_query_engine.retrieve_with_scores(
            query, metadata_filters
        )

        # Step 3: Check if FAQ score is sufficient
        if top_score >= settings.faq_score_threshold and faq_nodes:
            # FAQ score is good enough, synthesize answer from already retrieved nodes
            return self.faq_query_engine.synthesize_from_nodes(
                query, faq_nodes
            )  # 压缩信息，调用llm总结

        # Step 4: FAQ score is insufficient or no results, enhance with docs

        # Retrieve docs using original query (no context added to retrieval)
        docs_nodes = self.docs_query_engine.retrieve(query, metadata_filters)

        # Merge FAQ nodes with docs nodes, but mark FAQ nodes with low reliability
        all_nodes = list(docs_nodes)
        faq_nodes_merged = False
        should_merge_low_confidence_faq = (
            faq_nodes
            and top_score < settings.faq_score_threshold
            and top_score >= settings.faq_second_score_threshold
        )
        if should_merge_low_confidence_faq:
            # Mark FAQ nodes as low reliability and merge them
            marked_faq_nodes = self._mark_faq_nodes_with_low_reliability(
                faq_nodes, top_score
            )
            # Use node_id as unique identifier (more reliable than hash)
            seen_ids = {self._get_node_id(node) for node in all_nodes}
            for faq_node in marked_faq_nodes:
                faq_node_id = self._get_node_id(faq_node)
                if faq_node_id not in seen_ids:
                    all_nodes.append(faq_node)  # Add at end (lower priority)
                    seen_ids.add(faq_node_id)
                    faq_nodes_merged = True

        # Create enhanced query with instructions about FAQ nodes
        if faq_nodes_merged:
            enhanced_query = f"""{query}

重要提示：以下内容中，标记为"[FAQ-低相关性]"的内容来自FAQ库，但这些信息的相关性分数({top_score:.2f})低于推荐阈值({settings.faq_score_threshold:.2f})。
请优先参考文档库中的信息（无标记的内容），FAQ内容仅作为补充参考，使用时请谨慎验证其准确性。"""
        else:
            enhanced_query = query

        # Synthesize final answer with merged nodes (FAQ + docs)
        return self.docs_query_engine.synthesize_from_nodes(
            query_str=enhanced_query, nodes=all_nodes, context=None
        )

    def _mark_faq_nodes_with_low_reliability(
        self, faq_nodes: List[NodeWithScore], score: float
    ) -> List[NodeWithScore]:
        """Mark FAQ nodes with low reliability prefix.

        Args:
            faq_nodes: List of FAQ nodes
            score: FAQ retrieval score

        Returns:
            List of marked FAQ nodes (with prefix added to content)
        """
        marked_nodes = []
        for faq_node in faq_nodes:
            # Create a deep copy to avoid modifying the original node
            marked_node = deepcopy(faq_node)

            # Add reliability warning prefix to the node content
            if isinstance(marked_node.node, TextNode):
                original_text = marked_node.node.text
                reliability_prefix = (
                    f"[FAQ-低相关性:分数={score:.2f},阈值={settings.faq_score_threshold:.2f}] "
                    f"以下内容来自FAQ库，相关性较低，请谨慎参考：\n\n"
                )
                marked_node.node.text = reliability_prefix + original_text
            else:
                # For non-TextNode, try to modify text_resource if available
                if (
                    hasattr(marked_node.node, "text_resource")
                    and marked_node.node.text_resource
                ):
                    original_text = marked_node.node.text_resource.text or ""
                    reliability_prefix = (
                        f"[FAQ-低相关性:分数={score:.2f},阈值={settings.faq_score_threshold:.2f}] "
                        f"以下内容来自FAQ库，相关性较低，请谨慎参考：\n\n"
                    )
                    marked_node.node.text_resource.text = (
                        reliability_prefix + original_text
                    )

            # Apply penalty to score
            if marked_node.score is not None:
                penalty_factor = 0.5  # Reduce score by 50%
                marked_node.score = marked_node.score * penalty_factor

            marked_nodes.append(marked_node)

        return marked_nodes

    def _get_node_id(self, node: NodeWithScore) -> str:
        """Get unique identifier for a node.

        Args:
            node: NodeWithScore object

        Returns:
            Unique identifier (node_id, id_, or hash)
        """
        # Try node_id first (most reliable)
        if hasattr(node.node, "node_id") and node.node.node_id:
            return node.node.node_id
        # Fallback to id_
        if hasattr(node.node, "id_") and node.node.id_:
            return node.node.id_
        # Fallback to hash
        if hasattr(node.node, "hash"):
            try:
                return node.node.hash
            except Exception:
                pass
        # Last resort: use node_id from NodeWithScore
        if hasattr(node, "node_id") and node.node_id:
            return node.node_id
        # If all else fails, use a combination of text and score
        node_text = (
            getattr(node.node, "text", "")
            or getattr(node.node, "get_content", lambda: "")()
        )
        return f"{node_text[:50]}_{node.score}"

    def _is_code_query(self, query: str) -> bool:
        """Check if query is code-related.

        Args:
            query: User query

        Returns:
            True if query is code-related
        """
        code_keywords = ["代码", "code", "函数", "function", "类", "class", "import"]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in code_keywords)

    # def _detect_query_type(self, query: str) -> QueryType:
    #     """Detect query type from query string.

    #     Args:
    #         query: User query

    #     Returns:
    #         Detected query type
    #     """
    #     # Simple keyword-based detection
    #     # TODO: Implement BERT-based classification as mentioned in README
    #     if self._is_code_query(query):
    #         return QueryType.CODE

    #     # TODO may be we need llm or classifier to detect the query type.
    #     query_lower = query.lower()
    #     # FAQ-related keywords
    #     faq_keywords = ["如何", "怎么", "为什么", "what", "how", "why", "error", "错误"]
    #     if any(keyword in query_lower for keyword in faq_keywords):
    #         return QueryType.FAQ

    #     # Default to docs
    #     return QueryType.DOCS


def create_query_router(
    vector_store_manager: VectorStoreManager,
    llm,
    _embed_model,
    device: Optional[str] = None,
):
    """Create query router with initialized engines.

    Args:
        vector_store_manager: Initialized vector store manager
        llm: Initialized LLM instance
        embed_model: Initialized embedding model instance
        device: Device to use for reranker ("cuda" or "cpu"). If None, auto-detect.
    """
    # Initialize global KeywordExtractorWrapper if using custom keyword index
    from askany.config import settings
    from askany.ingest.custom_keyword_index import (
        get_global_keyword_extractor,
        set_global_keyword_extractor,
    )
    from askany.ingest.keyword_extract_wrapper import KeywordExtractorWrapper

    if settings.using_custom_keyword_index:
        # Check if global extractor already exists
        global_extractor = get_global_keyword_extractor()
        if global_extractor is None:
            # Create and set global KeywordExtractorWrapper
            # Use priority from settings and pass embed_model for similarity filtering
            global_extractor = KeywordExtractorWrapper(
                priority=settings.keyword_extractor_priority,
                embed_model=_embed_model,
            )
            set_global_keyword_extractor(global_extractor)
            logger.info(
                f"Initialized global KeywordExtractorWrapper with priority={settings.keyword_extractor_priority}"
            )

    # Auto-detect device if not provided
    if device is None:
        device = get_device()
    # Get separate indexes for FAQ and docs
    faq_vector_index = vector_store_manager.get_faq_index()
    docs_vector_index = vector_store_manager.get_docs_index()
    faq_keyword_index = vector_store_manager.get_faq_keyword_index()
    docs_keyword_index = vector_store_manager.get_docs_keyword_index()

    # Fallback to legacy index if separate indexes not available
    if docs_vector_index is None:
        docs_vector_index = vector_store_manager.get_index()
        if docs_vector_index is None:
            raise ValueError(
                "No docs vector index available. Please run --ingest first."
            )

    # Create docs query engine (with optional keyword index for ensemble retrieval)
    if docs_vector_index and docs_keyword_index and settings.using_docs_keyword_index:
        # Use ensemble retriever (keyword + vector)
        docs_query_engine = RAGQueryEngine(
            index=docs_vector_index,
            keyword_index=docs_keyword_index,
            llm=llm,
            similarity_top_k=settings.docs_similarity_top_k,
            ensemble_weights=settings.docs_ensemble_weights,
        )
        print("Docs query engine created with ensemble retriever (keyword + vector)")
    elif docs_vector_index:
        # Docs vector index exists but no keyword index (vector-only, backward compatible)
        docs_query_engine = RAGQueryEngine(
            index=docs_vector_index,
            llm=llm,
            similarity_top_k=settings.docs_similarity_top_k,
        )
        print("Docs query engine created with vector-only retrieval")
    else:
        raise ValueError("No docs vector index available. Please run --ingest first.")

    # Create FAQ query engine (ensemble: keyword + vector + rerank)
    faq_query_engine = None
    if faq_vector_index and faq_keyword_index:
        # Use separate FAQ indexes
        logger.info("Creating FAQ query engine with ensemble retriever...")
        logger.info(f"Device: {device}, Reranker model: {settings.reranker_model}")
        try:
            faq_query_engine = FAQQueryEngine(
                vector_index=faq_vector_index,
                keyword_index=faq_keyword_index,
                llm=llm,
                similarity_top_k=settings.faq_similarity_top_k,
                ensemble_weights=settings.faq_ensemble_weights,
                device=device,
            )
            logger.info(
                "FAQ query engine created with ensemble retriever (keyword + vector + rerank)"
            )
            print(
                "FAQ query engine created with ensemble retriever (keyword + vector + rerank)"
            )
        except Exception as e:
            logger.error(f"Failed to create FAQ query engine: {e}", exc_info=True)
            raise
    elif faq_vector_index:
        # FAQ vector index exists but no keyword index
        print("Warning: No FAQ keyword index found, FAQ will use vector-only retrieval")
        faq_query_engine = RAGQueryEngine(
            index=faq_vector_index,
            llm=llm,
            similarity_top_k=settings.faq_vector_only_similarity_top_k,
        )
    else:
        # Fallback to legacy index if available
        legacy_index = vector_store_manager.get_index()
        legacy_keyword_index = vector_store_manager.get_keyword_index()
        if legacy_index and legacy_keyword_index:
            faq_query_engine = FAQQueryEngine(
                vector_index=legacy_index,
                keyword_index=legacy_keyword_index,
                llm=llm,
                similarity_top_k=settings.faq_similarity_top_k,
                ensemble_weights=settings.faq_ensemble_weights,
                device=device,
            )
            print(
                "FAQ query engine created with legacy indexes (keyword + vector + rerank)"
            )
        elif legacy_index:
            print("Warning: Using legacy index for FAQ (vector-only)")
            faq_query_engine = RAGQueryEngine(
                index=legacy_index,
                llm=llm,
                similarity_top_k=settings.faq_vector_only_similarity_top_k,
            )
        else:
            print("Warning: No FAQ index found, FAQ queries will be disabled")

    # Create router
    logger.info("Creating QueryRouter instance...")
    router = QueryRouter(
        docs_query_engine=docs_query_engine,
        faq_query_engine=faq_query_engine,
    )
    logger.info("QueryRouter created successfully.")

    return router
