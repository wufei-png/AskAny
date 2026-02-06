"""Module for recording and formatting middle results from workflow nodes."""

from enum import Enum
from typing import Any, Dict, List, Optional
from askany.config import settings


class NodeType(str, Enum):
    """Node type enumeration for workflow nodes."""

    DIRECT_ANSWER_CHECK = "direct_answer_check"
    WEB_OR_RAG_CHECK = "web_or_rag_check"
    RAG_RETRIEVAL = "rag_retrieval"
    ANALYZE_RELEVANCE = "analyze_relevance"
    PROCESS_NO_RELEVANT = "process_no_relevant"
    PROCESS_SUB_QUERY = "process_sub_query"
    EXPAND_CONTEXT = "expand_context"


class MiddleResultRecorder:
    """Records and formats middle results from workflow nodes."""

    @staticmethod
    def record_direct_answer_check(
        middle_results: List[Dict[str, Any]], can_direct_answer: bool
    ) -> None:
        """Record direct answer check node result.

        Args:
            middle_results: List to append the result to
            can_direct_answer: Whether the question can be answered directly
        """
        if settings.return_middle_result == False:
            return
        result = {
            "node_type": NodeType.DIRECT_ANSWER_CHECK.value,
            "data": {
                "can_direct_answer": can_direct_answer,
            },
        }
        middle_results.append(result)

    @staticmethod
    def record_web_or_rag_check(
        middle_results: List[Dict[str, Any]],
        need_web_search: bool,
        need_rag_search: bool,
    ) -> None:
        """Record web or rag check node result.

        Args:
            middle_results: List to append the result to
            need_web_search: Whether web search is needed
            need_rag_search: Whether RAG search is needed
        """
        if settings.return_middle_result == False:
            return
        result = {
            "node_type": NodeType.WEB_OR_RAG_CHECK.value,
            "data": {
                "need_web_search": need_web_search,
                "need_rag_search": need_rag_search,
            },
        }
        middle_results.append(result)

    @staticmethod
    def record_rag_retrieval(
        middle_results: List[Dict[str, Any]],
        keywords: List[str],
        nodes_count: int,
    ) -> None:
        """Record RAG retrieval node result.

        Args:
            middle_results: List to append the result to
            keywords: List of extracted keywords
            nodes_count: Number of retrieved nodes
        """
        if settings.return_middle_result == False:
            return
        result = {
            "node_type": NodeType.RAG_RETRIEVAL.value,
            "data": {
                "keywords": keywords,
                "nodes_count": nodes_count,
            },
        }
        middle_results.append(result)

    @staticmethod
    def record_analyze_relevance(
        middle_results: List[Dict[str, Any]],
        relevant_file_paths: List[str],
        is_complete: bool,
    ) -> None:
        """Record analyze relevance node result.

        Args:
            middle_results: List to append the result to
            relevant_file_paths: List of relevant file paths
            is_complete: Whether the information is complete
        """
        if settings.return_middle_result == False:
            return
        result = {
            "node_type": NodeType.ANALYZE_RELEVANCE.value,
            "data": {
                "relevant_file_paths": relevant_file_paths,
                "is_complete": is_complete,
            },
        }
        middle_results.append(result)

    @staticmethod
    def record_process_no_relevant(
        middle_results: List[Dict[str, Any]],
        missing_info_keywords: List[str],
        sub_queries: List[str],
        hypothetical_answer: str,
    ) -> None:
        """Record process no relevant node result.

        Args:
            middle_results: List to append the result to
            missing_info_keywords: List of missing info keywords
            sub_queries: List of sub queries
            hypothetical_answer: Hypothetical answer for vector search
        """
        if settings.return_middle_result == False:
            return
        result = {
            "node_type": NodeType.PROCESS_NO_RELEVANT.value,
            "data": {
                "missing_info_keywords": missing_info_keywords,
                "sub_queries": sub_queries,
                "hypothetical_answer": hypothetical_answer,
            },
        }
        middle_results.append(result)

    @staticmethod
    def record_process_sub_query(
        middle_results: List[Dict[str, Any]],
        all_qa_context: List[Dict[str, str]],
    ) -> None:
        """Record process sub query node result.

        Args:
            middle_results: List to append the result to
            all_qa_context: List of Q&A context dicts with "query" and "answer" keys
        """
        if settings.return_middle_result == False:
            return
        result = {
            "node_type": NodeType.PROCESS_SUB_QUERY.value,
            "data": {
                "all_qa_context": all_qa_context,
            },
        }
        middle_results.append(result)

    @staticmethod
    def record_expand_context(
        middle_results: List[Dict[str, Any]],
        expanded_nodes_count: int,
    ) -> None:
        """Record expand context node result.

        Args:
            middle_results: List to append the result to
            expanded_nodes_count: Number of expanded nodes
        """
        if settings.return_middle_result == False:
            return
        result = {
            "node_type": NodeType.EXPAND_CONTEXT.value,
            "data": {
                "expanded_nodes_count": expanded_nodes_count,
            },
        }
        middle_results.append(result)

    @staticmethod
    def format_middle_results(
        middle_results: List[Dict[str, Any]], final_answer: str
    ) -> str:
        """Format middle results and final answer into a single string.

        Args:
            middle_results: List of middle result dictionaries
            final_answer: Final answer string

        Returns:
            Formatted string with all middle results and final answer
        """
        if settings.return_middle_result == False:
            return final_answer
        formatted_parts = []

        for result in middle_results:
            node_type = result["node_type"]
            data = result["data"]

            # Format data based on node type
            formatted_data = MiddleResultRecorder._format_node_data(node_type, data)
            formatted_parts.append(f"<_{node_type}>\n{formatted_data}\n</_{node_type}>")

        # Add final answer
        formatted_parts.append(final_answer)

        return "\n\n".join(formatted_parts)

    @staticmethod
    def _format_node_data(node_type: str, data: Dict[str, Any]) -> str:
        """Format node data based on node type.

        Args:
            node_type: Type of the node
            data: Data dictionary

        Returns:
            Formatted string representation of the data
        """
        if settings.return_middle_result == False:
            return ""
        if node_type == NodeType.DIRECT_ANSWER_CHECK.value:
            return f"can_direct_answer: {data.get('can_direct_answer', False)}"

        elif node_type == NodeType.WEB_OR_RAG_CHECK.value:
            return (
                f"need_web_search: {data.get('need_web_search', False)}\n"
                f"need_rag_search: {data.get('need_rag_search', False)}"
            )

        elif node_type == NodeType.RAG_RETRIEVAL.value:
            keywords = data.get("keywords", [])
            nodes_count = data.get("nodes_count", 0)
            keywords_str = ", ".join(keywords) if keywords else "[]"
            return f"keywords: [{keywords_str}]\nnodes_count: {nodes_count}"

        elif node_type == NodeType.ANALYZE_RELEVANCE.value:
            relevant_file_paths = data.get("relevant_file_paths", [])
            is_complete = data.get("is_complete", False)
            paths_str = "\n".join(f"  - {path}" for path in relevant_file_paths)
            return (
                f"relevant_file_paths:\n{paths_str if paths_str else '  []'}\n"
                f"is_complete: {is_complete}"
            )

        elif node_type == NodeType.PROCESS_NO_RELEVANT.value:
            missing_info_keywords = data.get("missing_info_keywords", [])
            sub_queries = data.get("sub_queries", [])
            hypothetical_answer = data.get("hypothetical_answer", "")
            keywords_str = (
                ", ".join(missing_info_keywords) if missing_info_keywords else "[]"
            )
            sub_queries_str = (
                "\n".join(f"  - {q}" for q in sub_queries) if sub_queries else "  []"
            )
            return (
                f"missing_info_keywords: [{keywords_str}]\n"
                f"sub_queries:\n{sub_queries_str}\n"
                f"hypothetical_answer: {hypothetical_answer}"
            )

        elif node_type == NodeType.PROCESS_SUB_QUERY.value:
            all_qa_context = data.get("all_qa_context", [])
            qa_parts = []
            for qa in all_qa_context:
                query = qa.get("query", "")
                answer = qa.get("answer", "")
                qa_parts.append(f"  问题: {query}\n  回答: {answer}")
            qa_str = "\n\n".join(qa_parts) if qa_parts else "  []"
            return f"all_qa_context:\n{qa_str}"

        elif node_type == NodeType.EXPAND_CONTEXT.value:
            expanded_nodes_count = data.get("expanded_nodes_count", 0)
            return f"expanded_nodes_count: {expanded_nodes_count}"

        else:
            # Fallback: format as JSON-like string
            import json

            return json.dumps(data, ensure_ascii=False, indent=2)
