"""RAG query engine for document retrieval and generation."""

from typing import Dict, List, Optional

from llama_index.core import KeywordTableIndex, QueryBundle, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore

from askany.config import settings
from askany.rerank import SafeReranker


class KeywordVectorAppendRetriever(BaseRetriever):
    """Custom retriever that appends keyword results to filtered vector results.
    
    This retriever:
    1. Accepts all keyword_retriever results (no threshold filtering)
    2. Applies reranker and threshold filtering only to vector_retriever results
    3. Appends keyword results to filtered vector results
    """

    def __init__(
        self,
        keyword_retriever: BaseRetriever,
        vector_retriever: BaseRetriever,
        reranker: Optional[SafeReranker] = None,
        similarity_threshold: float = 0.0,
        keyword_index: Optional[KeywordTableIndex] = None,
        similarity_top_k: int = 5,
    ) -> None:
        """Initialize the retriever.
        
        Args:
            keyword_retriever: Retriever for keyword-based search (all results accepted)
            vector_retriever: Retriever for vector-based search (filtered by reranker and threshold)
            reranker: Optional reranker to apply to vector results
            similarity_threshold: Minimum similarity score threshold for vector results
            keyword_index: Optional keyword table index for filtering by keyword limits
        """
        self._keyword_retriever = keyword_retriever
        self._vector_retriever = vector_retriever
        self._reranker = reranker
        self._similarity_threshold = similarity_threshold
        self._keyword_index = keyword_index
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _filter_keyword_nodes_by_limits(
        self, keyword_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """根据文件数量和匹配数量限制过滤关键字节点。

        从keyword_index中查找每个节点对应的关键词，按关键词分组。
        如果某个关键词命中了超过 one_keyword_max_file_num 的文件，
        或者命中的数量超过了 one_keyword_max_matches_num，
        则过滤掉该关键词对应的所有节点。

        Args:
            keyword_nodes: 关键字节点列表

        Returns:
            过滤后的关键字节点列表
        """
        if not keyword_nodes or not self._keyword_index:
            return keyword_nodes

        max_file_num = settings.one_keyword_max_file_num
        max_matches_num = settings.one_keyword_max_matches_num

        # 获取关键词表：keyword -> [node_ids]
        try:
            index_struct = self._keyword_index.index_struct
            if not hasattr(index_struct, "table") or not index_struct.table:
                # 如果没有关键词表，返回原始节点
                return keyword_nodes
            keyword_table = index_struct.table
        except Exception as e:
            print(f"Error accessing keyword table: {e}")
            return keyword_nodes

        # 只查找 keyword_nodes 中节点对应的关键词，避免遍历整个索引表
        # 收集所有需要查找的节点ID
        node_ids_to_find = {node.node.node_id for node in keyword_nodes}
        print(f"node_ids_to_find: ", node_ids_to_find)
        # with open("node_ids_to_find.txt", "w", encoding="utf-8") as f:  f.write(str(node_ids_to_find))
        # with open("keyword_table.txt", "w", encoding="utf-8") as f:  f.write(str(keyword_table))
        # 只遍历包含这些节点ID的关键词，构建反向映射：node_id -> [keywords]
        node_to_keywords: Dict[str, List[str]] = {}
        for keyword, node_ids in keyword_table.items():
            # print(f"keyword: {keyword}, node_ids: {node_ids}")
            # 只处理包含我们需要的节点ID的关键词
            for node_id in node_ids:
                if node_id in node_ids_to_find:
                    if node_id not in node_to_keywords:
                        node_to_keywords[node_id] = []
                    node_to_keywords[node_id].append(keyword)
        print(f"node_to_keywords: ", node_to_keywords)
        # 按关键词分组节点：keyword -> [nodes]
        keyword_to_nodes: Dict[str, List[NodeWithScore]] = {}
        nodes_without_keywords = []

        for node in keyword_nodes:
            node_id = node.node.node_id
            keywords = node_to_keywords.get(node_id, [])
            
            if not keywords:
                # 如果节点没有对应的关键词，保留它
                nodes_without_keywords.append(node)
            else:
                # 将节点添加到所有对应关键词的组中
                for keyword in keywords:
                    if keyword not in keyword_to_nodes:
                        keyword_to_nodes[keyword] = []
                    keyword_to_nodes[keyword].append(node)
        print(f"keyword_to_nodes: ", keyword_to_nodes)
        # 对每个关键词进行限制检查
        # 使用字典按node_id存储节点，避免重复（一个节点可能对应多个关键词）
        filtered_nodes_dict: Dict[str, NodeWithScore] = {}
        filtered_keywords = []

        for keyword, nodes in keyword_to_nodes.items():
            # 统计该关键词命中的文件数量（通过节点的metadata中的source字段去重）
            unique_files = set()
            for node in nodes:
                node_metadata = node.node.metadata if hasattr(node.node, "metadata") else {}
                source = node_metadata.get("source", "")
                if source:
                    unique_files.add(source)

            file_count = len(unique_files)
            matches_count = len(nodes)
            print(f"keyword: {keyword}, file_count: {file_count}, matches_count: {matches_count}")
            # 如果超过限制，过滤掉该关键词对应的所有节点
            if file_count > max_file_num or matches_count > max_matches_num:
                filtered_keywords.append(keyword)
                print(
                    f"Filtered keyword '{keyword}': file_count={file_count} (max={max_file_num}), "
                    f"matches_count={matches_count} (max={max_matches_num})"
                )
            else:
                # 保留该关键词的节点（使用node_id去重，因为一个节点可能对应多个关键词）
                for node in nodes:
                    node_id = node.node.node_id
                    if node_id not in filtered_nodes_dict:
                        filtered_nodes_dict[node_id] = node

        # 添加没有对应关键词的节点
        for node in nodes_without_keywords:
            node_id = node.node.node_id
            if node_id not in filtered_nodes_dict:
                filtered_nodes_dict[node_id] = node

        # 转换为列表
        filtered_nodes = list(filtered_nodes_dict.values())
        print(f"filtered_nodes: ", filtered_nodes)
        if filtered_keywords:
            print(
                f"Filtered {len(filtered_keywords)} keywords due to limits: {filtered_keywords}"
            )

        return filtered_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        # Get all keyword results (no filtering)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        
        # Get vector results
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        
        # Apply reranker to vector results if available
        if vector_nodes and self._reranker:
            vector_nodes = self._reranker.postprocess_nodes(
                vector_nodes, query_bundle=query_bundle
            )
        
        # Filter vector results by similarity threshold
        filtered_vector_nodes = [
            node
            for node in vector_nodes
            if node.score is not None and node.score >= self._similarity_threshold
        ]
        
        # Remove duplicates: track node IDs from vector results
        vector_node_ids = {node.node.node_id for node in filtered_vector_nodes}
        
        # Filter keyword nodes by limits (file count and match count)
        print("keyword_nodes: ",keyword_nodes[0].node.node_id)
        keyword_nodes = self._filter_keyword_nodes_by_limits(keyword_nodes)
        keyword_nodes = keyword_nodes[:self._similarity_top_k]
        # Add keyword nodes that are not already in vector results
        keyword_nodes_to_add = [
            node for node in keyword_nodes if node.node.node_id not in vector_node_ids
        ]
        for node in keyword_nodes_to_add:
            node.score = 1
        # Combine: vector results first (they are reranked and filtered), then keyword results
        combined_nodes = keyword_nodes_to_add + filtered_vector_nodes
        print(f"combined_nodes: ", len(combined_nodes))
        for node in combined_nodes:
            print(f"node: {node}")
        return combined_nodes


class RAGQueryEngine:
    """RAG query engine for retrieving and generating answers."""

    def __init__(
        self,
        index: VectorStoreIndex,
        llm: LLM,
        similarity_top_k: int = 5,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        keyword_index: Optional[KeywordTableIndex] = None,
        ensemble_weights: Optional[List[float]] = None,
    ):
        """Initialize RAG query engine.

        Args:
            index: Vector store index
            llm: Language model for generation
            similarity_top_k: Number of top similar documents to retrieve
            response_mode: Response synthesis mode
            keyword_index: Optional keyword table index for ensemble retrieval
            ensemble_weights: Weights for ensemble retriever [keyword_weight, vector_weight]
        """
        self.index = index
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.keyword_index = keyword_index
        self.device = settings.device

        # Get rerank candidate count (number of nodes to retrieve before reranking)
        # This should be larger than similarity_top_k to allow reranker to truly filter
        # If similarity_top_k is -1, use a default large value for rerank_candidate_k
        if similarity_top_k == -1:
            rerank_candidate_k = getattr(settings, "docs_rerank_candidate_k", 10)
        else:
            rerank_candidate_k = getattr(
                settings, "docs_rerank_candidate_k", similarity_top_k * 4
            )
            if rerank_candidate_k < similarity_top_k:
                rerank_candidate_k = similarity_top_k * 4

        # Create retriever
        if keyword_index:
            # Use ensemble retriever (keyword + vector)
            # Default weights: 0.5 for keyword, 0.5 for vector
            # if ensemble_weights is None:
            #     ensemble_weights = getattr(
            #         settings, "docs_ensemble_weights", [0.5, 0.5]
            #     )

            # Create keyword retriever (retrieve more candidates for reranking)
            # Pass max_keywords_per_query to control keyword extraction in query prompt
            keyword_retriever = keyword_index.as_retriever(
                # similarity_top_k=rerank_candidate_k, # 这里不限制关键词数量，keyword_retriever返回所有关键词,在KeywordVectorAppendRetriever中限制关键词数量
                max_keywords_per_query=settings.max_keywords_for_docs,
            )

            # Create vector retriever (retrieve more candidates for reranking)
            vector_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=rerank_candidate_k,
            )

            # Create reranker first (will be used in custom retriever)
            # If similarity_top_k is -1, reranker should return all candidates
            reranker_top_n = (
                similarity_top_k if similarity_top_k > 0 else rerank_candidate_k
            )
            reranker = SafeReranker.create(
                top_n=reranker_top_n,  # Final number of nodes after reranking
                device=self.device,
                reranker_model=settings.reranker_model,
            )
            self.reranker = reranker

            # Create custom retriever that appends keyword results to filtered vector results
            # Keyword results: all accepted (no threshold filtering)
            # Vector results: reranked and filtered by threshold
            retriever = KeywordVectorAppendRetriever(
                keyword_retriever=keyword_retriever,
                vector_retriever=vector_retriever,
                reranker=reranker,
                similarity_threshold=getattr(
                    settings, "docs_similarity_threshold", 0.0
                ),
                keyword_index=keyword_index,
                similarity_top_k=rerank_candidate_k,
            )
        else:
            # Use vector-only retriever (backward compatible)
            # If similarity_top_k is -1, use a large default value
            retriever_top_k = (
                similarity_top_k if similarity_top_k > 0 else rerank_candidate_k
            )
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=retriever_top_k,
            )
        # If similarity_top_k is -1, reranker should return all candidates
        if not keyword_index:
            # For vector-only case, create reranker and use it as postprocessor
            reranker_top_n = (
                similarity_top_k if similarity_top_k > 0 else rerank_candidate_k
            )
            self.reranker = SafeReranker.create(
                top_n=reranker_top_n,  # Final number of nodes after reranking
                device=self.device,
                reranker_model=settings.reranker_model,
            )
            # Create query engine with reranker as postprocessor
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=llm,
                response_mode=response_mode,
                node_postprocessors=[self.reranker],
            )
        else:
            # For keyword+vector case, reranker is already applied in custom retriever
            # No need to add it as postprocessor again
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=llm,
                response_mode=response_mode,
                node_postprocessors=[],  # Reranker is applied inside custom retriever
            )

    def query(
        self, query_str: str, metadata_filters: Optional[Dict[str, str]] = None
    ) -> str:
        """Query the RAG system.

        Args:
            query_str: User query string (may contain @tag filters)
            metadata_filters: Optional metadata filters dict (if None, will parse from query)

        Returns:
            Generated response with reference information
        """
        # Retrieve nodes (filtering logic is inside retrieve method)
        nodes = self.retrieve(query_str, metadata_filters)

        # Synthesize answer from retrieved nodes
        query_bundle = QueryBundle(query_str)
        response = self.query_engine.synthesize(query_bundle, nodes)

        # Extract reference information from used nodes
        references = self._extract_docs_references(nodes)

        # Format response with references
        response_text = str(response)
        if references:
            response_text += self._format_docs_references(references)

        return response_text

    def retrieve(
        self, query_str: str, metadata_filters: Optional[Dict[str, str]] = None
    ) -> List:
        """Retrieve relevant documents without generation.

        Args:
            query_str: User query string (may contain @tag filters)
            metadata_filters: Optional metadata filters dict (if None, will parse from query)

        Returns:
            List of retrieved nodes (filtered by metadata if filters provided, limited to similarity_top_k)
        """

        query_bundle = QueryBundle(query_str)
        nodes = self.query_engine.retriever.retrieve(query_bundle)

        # Apply metadata filtering if filters exist
        if metadata_filters:
            nodes = self._filter_nodes_by_metadata(nodes, metadata_filters)

        nodes = sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)

        # Filter by similarity threshold before truncation
        nodes = self._filter_nodes_by_score(nodes, settings.docs_similarity_threshold)
        # print(f"nodes after reranker: ", len(nodes))
        # for node in nodes:
        #     print(f"node: {node}")
        # Limit to similarity_top_k nodes (retriever may return more for reranking)
        # If similarity_top_k is -1, don't truncate
        if self.similarity_top_k > 0:
            nodes = nodes[: self.similarity_top_k]

        return nodes

    def synthesize_from_nodes(
        self, query_str: str, nodes: List, context: Optional[str] = None
    ) -> str:
        """Synthesize answer from retrieved nodes with optional context.

        Args:
            query_str: User query string
            nodes: Retrieved nodes
            context: Optional additional context (e.g., FAQ answer)

        Returns:
            Generated response with reference information
        """
        # If context is provided, modify the query to include it
        if context:
            enhanced_query = f"""用户问题: {query_str}

相关上下文信息:
{context}

请基于上述上下文信息和检索到的文档，提供完整和准确的答案。"""
            query_bundle = QueryBundle(enhanced_query)
        else:
            query_bundle = QueryBundle(query_str)

        response = self.query_engine.synthesize(query_bundle, nodes)

        # Extract reference information from nodes
        references = self._extract_docs_references(nodes)

        # Format response with references
        response_text = str(response)
        if references:
            response_text += self._format_docs_references(references)

        return response_text

    def _extract_docs_references(self, nodes: List[NodeWithScore]) -> Dict[str, List]:
        """Extract document reference information from nodes.

        Args:
            nodes: List of nodes used in the query

        Returns:
            Dictionary with 'markdown' (list of file paths) and 'faq' (list of FAQ references)
        """
        markdown_refs = []
        faq_refs = []
        seen_sources = set()
        seen_faq_ids = set()

        for node in nodes:
            node_metadata = node.node.metadata if hasattr(node.node, "metadata") else {}
            node_type = node_metadata.get("type", "")

            if node_type == "markdown":
                # Get source file path
                source = node_metadata.get("source", "")
                if source and source not in seen_sources:
                    seen_sources.add(source)
                    markdown_refs.append(source)
            elif node_type == "faq":
                # Extract FAQ reference (id and answer)
                faq_id = node_metadata.get("id", "")
                if faq_id and faq_id not in seen_faq_ids:
                    seen_faq_ids.add(faq_id)

                    # Extract answer from node text
                    node_text = node.node.text if hasattr(node.node, "text") else ""
                    answer = ""
                    if "答案:" in node_text:
                        answer = node_text.split("答案:")[-1].strip()

                    faq_refs.append(
                        {
                            "id": faq_id,
                            "answer": answer,
                        }
                    )

        return {
            "markdown": markdown_refs,
            "faq": faq_refs,
        }

    def _format_docs_references(self, references: Dict[str, List]) -> str:
        """Format document references for display.

        Args:
            references: Dictionary with 'markdown' and 'faq' lists

        Returns:
            Formatted reference string
        """
        markdown_refs = references.get("markdown", [])
        faq_refs = references.get("faq", [])

        if not markdown_refs and not faq_refs:
            return ""

        ref_text = "\n\n---\n**参考数据来源：**\n\n"

        # Format Markdown references
        if markdown_refs:
            ref_text += "**Markdown文档：**\n"
            for ref in markdown_refs:
                ref_text += f"- {ref}\n"
            ref_text += "\n"

        # Format FAQ references
        if faq_refs:
            ref_text += "**FAQ：**\n"
            for ref in faq_refs:
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
            node for node in nodes if (node.score is None) or (node.score is not None and node.score >= threshold)
        ]
