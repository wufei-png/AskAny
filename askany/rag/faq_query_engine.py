"""FAQ query engine with KeywordTableIndex and VectorStoreIndex ensemble."""

from logging import getLogger
from typing import Dict, List, Optional, Tuple

from llama_index.core import KeywordTableIndex, QueryBundle, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever

# TODO maybe RouterRetriever?
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore

from askany.config import settings
from askany.rerank import SafeReranker

logger = getLogger(__name__)


class FAQQueryEngine:
    """FAQ query engine using ensemble retriever with keyword and vector indexes."""

    def __init__(
        self,
        vector_index: VectorStoreIndex,
        keyword_index: KeywordTableIndex,
        llm: LLM,
        similarity_top_k: int = 5,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        reranker_model: Optional[str] = None,
        ensemble_weights: List[float] = None,
        device: Optional[str] = None,
    ):
        """Initialize FAQ query engine.

        Args:
            vector_index: Vector store index
            keyword_index: Keyword table index
            llm: Language model for generation
            similarity_top_k: Number of top similar documents to retrieve
            response_mode: Response synthesis mode
            reranker_model: Reranker model name (defaults to settings)
            ensemble_weights: Weights for ensemble retriever [keyword_weight, vector_weight]
            device: Device to use for reranker ("cuda" or "cpu"). If None, auto-detect.
        """
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.llm = llm
        self.similarity_top_k = similarity_top_k

        # Use provided device, or fallback to settings.device
        self.device = device if device is not None else settings.device

        # Default weights: 0.5 for keyword, 0.5 for vector
        if ensemble_weights is None:
            ensemble_weights = [0.5, 0.5]
        # TODO   # 如果 keyword 检索更重要（精确匹配）
        #    ensemble_weights=[0.7, 0.3]  # keyword 70%, vector 30%

        #    # 如果语义理解更重要
        #    ensemble_weights=[0.3, 0.7]  # keyword 30%, vector 70%
        # Get rerank candidate count (number of nodes to retrieve before reranking)
        # This should be larger than similarity_top_k to allow reranker to truly filter
        # If similarity_top_k is -1, use a default large value for rerank_candidate_k
        if similarity_top_k == -1:
            rerank_candidate_k = getattr(settings, "faq_rerank_candidate_k", 50)
        else:
            rerank_candidate_k = getattr(
                settings, "faq_rerank_candidate_k", similarity_top_k * 4
            )
            if rerank_candidate_k < similarity_top_k:
                logger.warning(
                    f"faq_rerank_candidate_k ({rerank_candidate_k}) < similarity_top_k ({similarity_top_k}), "
                    f"setting to {similarity_top_k * 4} for proper reranking"
                )
                rerank_candidate_k = similarity_top_k * 4

        # Create keyword retriever (retrieve more candidates for reranking)
        # Pass max_keywords_per_query to control keyword extraction in query prompt
        keyword_retriever = keyword_index.as_retriever(
            similarity_top_k=rerank_candidate_k,
            max_keywords_per_query=settings.max_keywords_for_faq,
        )

        # Create vector retriever (retrieve more candidates for reranking)
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=rerank_candidate_k
        )

        # Create ensemble retriever using QueryFusionRetriever
        # Use RELATIVE_SCORE mode for weighted fusion, or SIMPLE for basic fusion
        # Retrieve more candidates so reranker can select the best ones
        ensemble_retriever = QueryFusionRetriever(
            retrievers=[keyword_retriever, vector_retriever],
            retriever_weights=ensemble_weights,
            mode=FUSION_MODES.RELATIVE_SCORE,  # Supports weighted fusion
            similarity_top_k=rerank_candidate_k,  # Retrieve more candidates for reranking
            use_async=False,  # Use sync mode for simplicity
            num_queries=settings.query_fusion_num_queries,
        )

        # Create reranker (optional)
        # Reranker will select top_n from the rerank_candidate_k candidates
        reranker_model = reranker_model or settings.reranker_model
        logger.info(
            f"reranker_model: {reranker_model}, if not exist, will download from huggingface"
        )
        # If similarity_top_k is -1, reranker should return all candidates
        reranker_top_n = (
            similarity_top_k if similarity_top_k > 0 else rerank_candidate_k
        )
        logger.info(
            f"Rerank strategy: retrieve {rerank_candidate_k} candidates, rerank to top {reranker_top_n if similarity_top_k > 0 else 'all'}"
        )
        logger.info(
            f"Creating reranker with top_n={reranker_top_n}, device={self.device}, model={reranker_model}"
        )
        logger.info("This may take a while if the reranker model needs to be loaded...")
        self.reranker = SafeReranker.create(
            top_n=reranker_top_n,  # Final number of nodes after reranking
            device=self.device,
            reranker_model=reranker_model,
        )
        logger.info(f"reranker successfully created: {self.reranker}")
        # Create query engine with reranker
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=ensemble_retriever,
            llm=llm,
            response_mode=response_mode,
            node_postprocessors=[self.reranker],
        )

    def query(
        self, query_str: str, metadata_filters: Optional[Dict[str, str]] = None
    ) -> str:
        """Query the FAQ system.

        Args:
            query_str: User query string (may contain @tag filters)
            metadata_filters: Optional metadata filters dict (if None, will parse from query)

        Returns:
            Generated response with reference information
        """

        # Synthesize answer from retrieved nodes
        query_bundle = QueryBundle(query_str)
        nodes = self.retrieve(query_str, metadata_filters)
        response = self.query_engine.synthesize(query_bundle, nodes)

        # Extract reference information from used nodes
        references = self._extract_faq_references(nodes)

        # Format response with references
        response_text = str(response)
        if references:
            response_text += self._format_faq_references(references)

        return response_text

    def retrieve(
        self, query_str: str, metadata_filters: Optional[Dict[str, str]] = None
    ) -> List:
        """Retrieve relevant documents without generation.

        Args:
            query_str: User query string (may contain @tag filters)
            metadata_filters: Optional metadata filters dict (if None, will parse from query)

        Returns:
            List of retrieved nodes (filtered by metadata if filters provided)
        """

        query_bundle = QueryBundle(query_str)
        nodes = self.query_engine.retrieve(query_bundle)

        # Apply metadata filtering if filters exist
        if metadata_filters:
            nodes = self._filter_nodes_by_metadata(nodes, metadata_filters)

        nodes = sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)

        # Filter by similarity threshold before truncation
        nodes = self._filter_nodes_by_score(nodes, settings.faq_second_score_threshold)
        # Limit to similarity_top_k nodes (reranker already filtered, but ensure after metadata filtering)
        # If similarity_top_k is -1, don't truncate
        if self.similarity_top_k > 0:
            nodes = nodes[: self.similarity_top_k]

        return nodes

    def retrieve_with_scores(
        self, query_str: str, metadata_filters: Optional[Dict[str, str]] = None
    ) -> Tuple[List, float]:
        """Retrieve relevant documents and return the top score.

        Args:
            query_str: User query string (may contain @tag filters)
            metadata_filters: Optional metadata filters dict (if None, will parse from query)

        Returns:
            Tuple of (list of retrieved nodes, top score)
        """

        nodes = self.retrieve(query_str, metadata_filters)

        # Get the top score (highest similarity score)
        top_score = nodes[0].score if nodes and nodes[0].score is not None else 0.0
        return nodes, top_score

    def synthesize_from_nodes(self, query_str: str, nodes: List) -> str:
        """Synthesize answer from retrieved nodes without re-retrieving.

        Args:
            query_str: User query string
            nodes: Retrieved nodes (from retrieve_with_scores)

        Returns:
            Generated response with reference information
        """
        query_bundle = QueryBundle(query_str)
        response = self.query_engine.synthesize(query_bundle, nodes)

        # Extract reference information from nodes
        references = self._extract_faq_references(nodes)

        # Format response with references
        response_text = str(response)
        if references:
            response_text += self._format_faq_references(references)

        return response_text

    def _extract_faq_references(
        self, nodes: List[NodeWithScore]
    ) -> List[Dict[str, str]]:
        """Extract FAQ reference information from nodes.

        Args:
            nodes: List of nodes used in the query

        Returns:
            List of dictionaries containing id and answer for each FAQ node
        """
        references = []
        seen_ids = set()

        for node in nodes:
            node_metadata = node.node.metadata if hasattr(node.node, "metadata") else {}

            # Check if this is a FAQ node
            if node_metadata.get("type") != "faq":
                continue

            # Get FAQ ID
            faq_id = node_metadata.get("id", "")
            if not faq_id or faq_id in seen_ids:
                continue

            seen_ids.add(faq_id)

            # Extract answer from node text
            # FAQ text format: "问题: {question}\n答案: {answer}"
            node_text = node.node.text if hasattr(node.node, "text") else ""
            answer = ""
            if "答案:" in node_text:
                answer = node_text.split("答案:")[-1].strip()

            references.append(
                {
                    "id": faq_id,
                    "answer": answer,
                }
            )

        return references

    def _format_faq_references(self, references: List[Dict[str, str]]) -> str:
        """Format FAQ references for display.

        Args:
            references: List of reference dictionaries

        Returns:
            Formatted reference string
        """
        if not references:
            return ""

        ref_text = "\n\n---\n**参考数据来源：**\n\n"
        for ref in references:
            ref_text += f"- **FAQ ID**: {ref['id']}\n"
            if ref["answer"]:
                # Truncate long answers for display
                answer_preview = (
                    ref["answer"][:200] + "..."
                    if len(ref["answer"]) > 200
                    else ref["answer"]
                )
                ref_text += f"  **答案**: {answer_preview}\n"
            ref_text += "\n"

        return ref_text

    def _filter_nodes_by_metadata(
        self, nodes: List[NodeWithScore], filters: Dict[str, str]
    ) -> List[NodeWithScore]:
        """Filter nodes by metadata criteria.

        Args:
            nodes: List of nodes to filter
            filters: Dictionary of metadata filters {key: value}

        Returns:
            Filtered list of nodes
        """
        filtered_nodes = []
        for node in nodes:
            node_metadata = node.node.metadata if hasattr(node.node, "metadata") else {}

            # Check if node matches all filter criteria
            matches = True
            for key, value in filters.items():
                # Check both direct key and meta_key format
                node_value = node_metadata.get(key) or node_metadata.get(f"meta_{key}")

                # Convert to string for comparison
                if node_value is None:
                    matches = False
                    break

                # Support exact match and substring match (for tags)
                node_value_str = str(node_value).lower()
                filter_value_str = str(value).lower()

                # For tags field, check if value is in comma-separated list
                if key == "tags" and "," in node_value_str:
                    tag_list = [tag.strip() for tag in node_value_str.split(",")]
                    if filter_value_str not in tag_list:
                        matches = False
                        break
                elif node_value_str != filter_value_str:
                    matches = False
                    break

            if matches:
                filtered_nodes.append(node)

        return filtered_nodes

    def _filter_nodes_by_score(
        self, nodes: List[NodeWithScore], threshold: float
    ) -> List[NodeWithScore]:
        """Filter nodes by similarity score threshold.

        Args:
            nodes: List of nodes to filter
            threshold: Minimum similarity score threshold

        Returns:
            Filtered list of nodes with score >= threshold
        """
        return [
            node
            for node in nodes
            if (node.score is None)
            or (node.score is not None and node.score >= threshold)
        ]
