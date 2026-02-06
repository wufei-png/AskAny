"""Agent workflow using LangGraph functionality."""

import json
import os
import re
from datetime import datetime
from logging import FileHandler, Formatter, getLogger
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from llama_index.core import QueryBundle
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from workflows.client import WorkflowClient

from askany.config import settings
from askany.ingest.custom_keyword_index import (
    get_global_keyword_extractor,
    set_global_keyword_extractor,
)
from askany.ingest.keyword_extract_wrapper import KeywordExtractorWrapper
from askany.rag.query_parser import parse_query_filters
from askany.rag.router import QueryRouter, QueryType
from askany.rerank import SafeReranker
from askany.workflow.AnalysisRelated_langchain import (
    NoRelevantResult,
    NoRelevantResultWithoutSubQueries,
    RelevanceAnalyzer,
    RelevantResult,
)
# from askany.workflow.FinalSummaryLlm import (
#     extract_docs_references,
#     format_docs_references,
#     generate_final_answer,

# )
from askany.workflow.FinalSummaryLlm_langchain import (
    FinalAnswerGenerator,
    extract_docs_references,
    format_docs_references,
)
from askany.workflow.firstStageRelevant_langchain import (
    DirectAnswerGenerator,
    WebOrRagAnswerGenerator,
)
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from askany.workflow.rewriteRetrive_langchain import QueryRewriteGenerator
from askany.workflow.SubProblemGenerator import SubProblemGenerator
from askany.workflow.WebSearchTool import WebSearchTool
from askany.workflow.token_control import truncate_nodes_by_tokens
from askany.workflow.middle_result_recorder import (
    MiddleResultRecorder,
    NodeType,
)

logger = getLogger(__name__)

debug = True  # Whether to enable debug mode / 是否开启调试模式
output_file = "workflow_langgraph.log"

# Setup debug logger / 设置调试日志记录器
debug_logger = None
if debug:
    debug_logger = getLogger("workflow_debug")
    debug_logger.setLevel(
        1
    )  # Set to lowest level to ensure all messages are logged / 设置为最低级别以确保所有消息都被记录
    # Remove all existing handlers / 移除所有现有的处理器
    debug_logger.handlers = []
    # Create file handler with real-time flush / 创建文件处理器，实时刷新
    file_handler = FileHandler(output_file, mode="a", encoding="utf-8")
    file_handler.setLevel(1)
    # Set format / 设置格式
    formatter = Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    debug_logger.addHandler(file_handler)
    debug_logger.propagate = (
        False  # Prevent propagation to root logger / 防止传播到根日志记录器
    )


async def process_parallel_group(
    agent_workflow: "AgentWorkflow",
    parallel_group: List[str],
    query_type: QueryType,
) -> str:
    """Process a parallel group (may contain multiple related questions that need serial processing).
    处理一个并行组（可能包含多个相关问题需要串行处理）。

    Args:
        agent_workflow: AgentWorkflow instance / AgentWorkflow 实例
        parallel_group: List of questions in the parallel group / 并行组中的问题列表
        query_type: Query type / 查询类型

    Returns:
        Processing result string / 处理结果字符串
    """
    if len(parallel_group) == 1:
        # Single question, process directly / 单个问题，直接处理
        logger.debug("并行组只有一个问题，直接处理: %s", parallel_group[0])
        initial_state: AgentState = {
            "query": parallel_group[0],
            "query_type": query_type,
            "can_direct_answer": False,
            "need_web_search": False,
            "need_rag_search": False,
            "web_or_rag_result_cached": False,
            "nodes": [],
            "keywords": [],
            "analysis": None,
            "iteration": 0,
            "no_relevant_result": None,
            "current_sub_query": None,
            "inner_previous_qa_context": [],
            "outer_previous_qa_context": [],
            "is_inner_sub_query_workflow": False,
            "is_outer_sub_query_workflow": False,
            "result": None,
            "middle_results": [],
        }
        result = await agent_workflow.graph.ainvoke(initial_state)
        answer = result.get("result", "抱歉，无法生成答案。")
        return f"问题: {parallel_group[0]}\n回答: {answer}"
    else:
        # Multiple related questions, process serially, put previous answer in outer_previous_qa_context
        # 多个相关问题，串行处理，将上一个问题的答案放到outer_previous_qa_context中
        logger.debug("并行组包含 %d 个相关问题，串行处理", len(parallel_group))
        outer_previous_qa_context: List[Dict[str, str]] = []

        for idx, sub_query in enumerate(parallel_group):
            logger.debug(
                "处理子问题 %d/%d - 子问题: %s",
                idx + 1,
                len(parallel_group),
                sub_query,
            )

            # Call workflow to process sub-question, don't modify query, use outer_previous_qa_context
            # 调用 workflow 处理子问题，不修改query，使用outer_previous_qa_context
            logger.debug(
                "准备调用workflow处理子问题, query: %s, query_type: %s, 已有 %d 个外部Q&A上下文",
                sub_query,
                query_type,
                len(outer_previous_qa_context),
            )
            try:
                initial_state: AgentState = {
                    "query": sub_query,  # 保持原始查询，不修改
                    "query_type": query_type,
                    "can_direct_answer": False,
                    "need_web_search": False,
                    "need_rag_search": False,
                    "web_or_rag_result_cached": False,
                    "nodes": [],
                    "keywords": [],
                    "analysis": None,
                    "iteration": 0,
                    "no_relevant_result": None,
                    "current_sub_query": None,
                    "inner_previous_qa_context": [],
                    "outer_previous_qa_context": outer_previous_qa_context.copy(),  # 传递之前的Q&A上下文
                    "is_inner_sub_query_workflow": False,
                    "is_outer_sub_query_workflow": True,  # 设置标志表示这是外部子问题workflow
                    "result": None,
                    "middle_results": [],
                }
                result = await agent_workflow.graph.ainvoke(initial_state)
                sub_answer = result.get("result", "抱歉，无法生成答案。")
                logger.debug(
                    "子问题 %d/%d workflow完成，结果长度: %d",
                    idx + 1,
                    len(parallel_group),
                    len(sub_answer),
                )
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e) if str(e) else repr(e)
                logger.error(
                    "子问题workflow调用失败 - 异常类型: %s, 异常消息: %s, 子问题: %s, query_type: %s",
                    error_type,
                    error_msg,
                    sub_query,
                    query_type,
                    exc_info=True,
                )
                sub_answer = f"抱歉，处理子问题时发生错误: {error_type}: {error_msg}"

            # Update context, accumulate all sub-questions' Q&A
            # 更新上下文，累积所有子问题的Q&A
            outer_previous_qa_context.append(
                {
                    "query": sub_query,
                    "answer": sub_answer,
                }
            )

        # Format return result / 格式化返回结果
        result_parts = []
        for qa_pair in outer_previous_qa_context:
            result_parts.append(f"问题: {qa_pair['query']}\n回答: {qa_pair['answer']}")
        return "\n\n".join(result_parts)


def _format_state_for_debug(state: "AgentState") -> str:
    """Format state information for debug output.
    格式化状态信息用于调试输出。

    Args:
        state: AgentState state dictionary / AgentState 状态字典

    Returns:
        Formatted state string / 格式化的状态字符串
    """
    formatted = {}
    for key, value in state.items():
        if key == "nodes":
            # Format node information / 格式化节点信息
            formatted[key] = {
                "count": len(value) if value else 0,
                "sample": (
                    [
                        {
                            "score": node.score,
                            "file_path": (
                                node.node.metadata.get("file_path")
                                or node.node.metadata.get("source")
                                or "unknown"
                            ),
                            "text_preview": (
                                node.node.get_content()[:100]
                                if hasattr(node.node, "get_content")
                                else (
                                    node.node.text[:100]
                                    if hasattr(node.node, "text")
                                    else ""
                                )
                            ),
                        }
                        for node in value[
                            :3
                        ]  # Only show first 3 nodes / 只显示前3个节点
                    ]
                    if value
                    else []
                ),
            }
        elif key == "analysis" and value is not None:
            # Format analysis result / 格式化分析结果
            formatted[key] = {
                "relevant_file_paths_count": len(value.relevant_file_paths)
                if hasattr(value, "relevant_file_paths")
                else 0,
                "is_complete": value.is_complete
                if hasattr(value, "is_complete")
                else None,
            }
        elif key == "no_relevant_result" and value is not None:
            # Format no relevant result / 格式化无相关结果
            formatted[key] = {
                "has_sub_queries": (
                    len(value.sub_queries) > 0
                    if hasattr(value, "sub_queries")
                    else False
                ),
                "sub_queries_count": (
                    len(value.sub_queries) if hasattr(value, "sub_queries") else 0
                ),
                "missing_info_keywords_count": (
                    len(value.missing_info_keywords)
                    if hasattr(value, "missing_info_keywords")
                    else 0
                ),
                "has_hypothetical_answer": (
                    bool(value.hypothetical_answer)
                    if hasattr(value, "hypothetical_answer")
                    else False
                ),
            }
        else:
            # For other fields, use value directly but limit string length
            # 对于其他字段，直接使用值，但限制字符串长度
            if isinstance(value, str) and len(value) > 200:
                formatted[key] = value[:200] + "..."
            else:
                formatted[key] = value
    return json.dumps(formatted, ensure_ascii=False, indent=2)


def _log_node_input(node_name: str, state: "AgentState"):
    """Log node input information to debug log.
    记录节点输入信息到调试日志。

    Args:
        node_name: Node name / 节点名称
        state: Input state / 输入状态
    """
    if debug and debug_logger:
        try:
            formatted_state = _format_state_for_debug(state)
            debug_logger.info(
                f"\n{'=' * 80}\n"
                f"节点: {node_name}\n"
                f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                f"输入状态:\n{formatted_state}\n"
                f"{'=' * 80}\n"
            )
            # Ensure immediate flush to file / 确保立即刷新到文件
            for handler in debug_logger.handlers:
                if isinstance(handler, FileHandler):
                    handler.flush()
        except Exception as e:
            debug_logger.warning(f"记录节点 {node_name} 输入时出错: {str(e)}")


# LangGraph State Definition
class AgentState(TypedDict):
    """Agent workflow state schema."""

    # Input
    query: str
    query_type: QueryType
    metadata_filters: Optional[Dict[str, str]]  # Metadata filters extracted from query

    # Direct answer check
    can_direct_answer: bool

    # Web/RAG routing
    need_web_search: bool
    need_rag_search: bool
    web_or_rag_result_cached: bool  # Flag indicating if web_or_rag_result is already cached from workflow_filter

    # RAG retrieval
    nodes: List[NodeWithScore]
    keywords: List[str]

    # Analysis
    analysis: Optional[RelevantResult]
    iteration: int

    # Sub-problem handling
    no_relevant_result: Optional[NoRelevantResult]
    current_sub_query: Optional[str]
    inner_previous_qa_context: List[
        Dict[str, str]
    ]  # List of {"query": str, "answer": str}
    outer_previous_qa_context: List[
        Dict[str, str]
    ]  # List of {"query": str, "answer": str}
    is_inner_sub_query_workflow: bool
    is_outer_sub_query_workflow: bool

    # Final output
    result: Optional[str]

    # Middle results for visualization
    middle_results: List[Dict[str, Any]]  # List of middle result dictionaries

    no_complete_answer: bool = False  # Whether to return no complete answer


class AgentWorkflow:
    """Agent workflow using LangGraph."""

    def __init__(
        self,
        router: QueryRouter,
        llm: LLM,
        base_path: Optional[str] = None,
        workflow_client: Optional["WorkflowClient"] = None,
        web_search_tool: Optional["WebSearchTool"] = None,
        local_file_search: Optional["LocalFileSearchTool"] = None,
    ):
        """Initialize AgentWorkflow.

        Args:
            router: Query router for RAG retrieval
            llm: Language model for analysis and generation
            base_path: Base path for local file search
            workflow_client: Optional workflow client for internal calls
            web_search_tool: Optional shared WebSearchTool instance. If None, will create new one.
            local_file_search: Optional shared LocalFileSearchTool instance. If None, will create new one.
        """
        self.router = router
        self.router.faq_query_engine = router.faq_query_engine
        self.llm = llm
        self.sub_problem_generator = SubProblemGenerator()
        self.workflow_client = workflow_client

        if base_path is None:
            base_path = settings.local_file_search_dir
            if base_path == "":
                base_path = None

        # Initialize keyword extractor wrapper
        # Use global instance if available (should be set in create_query_router if using_custom_keyword_index)
        # Otherwise create a new one for local file search only
        global_extractor = get_global_keyword_extractor()
        if global_extractor is not None:
            self.keyword_extractor = global_extractor
        else:
            # Create a local instance for LocalFileSearchTool only
            # Use priority from settings
            self.keyword_extractor = KeywordExtractorWrapper(
                priority=settings.keyword_extractor_priority
            )
            set_global_keyword_extractor(self.keyword_extractor)

        # Use shared local_file_search if provided, otherwise create new one
        if local_file_search is not None:
            self.local_file_search = local_file_search
        else:
            self.local_file_search = LocalFileSearchTool(
                base_path=base_path,
                keyword_extractor=self.keyword_extractor.GetKeywordExtractorFromTFIDF(),
            )

        # Initialize LLM for LangChain
        api_base = settings.openai_api_base
        api_key = settings.openai_api_key if settings.openai_api_key else ""
        self.chat_llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=api_key,
            base_url=api_base,
            temperature=settings.temperature,
            max_tokens=settings.output_tokens,
        )

        # Initialize reranker (shared instance) using SafeReranker
        self.reranker = SafeReranker.create(
            top_n=-1
        )  # Return all nodes, let caller decide

        # Use shared web_search_tool if provided, otherwise create new one
        if web_search_tool is not None:
            self.web_search_tool = web_search_tool
        else:
            try:
                self.web_search_tool = WebSearchTool()
            except ImportError:
                logger.warning(
                    "WebSearchTool not available, web search will be disabled"
                )
                self.web_search_tool = None

        # Initialize first stage relevance generators (using LangChain version)
        self.direct_answer_generator = DirectAnswerGenerator(llm=self.chat_llm)
        self.web_or_rag_generator = WebOrRagAnswerGenerator(
            llm=self.chat_llm,
            keyword_extractor=self.keyword_extractor.GetKeywordExtractorFromTFIDF(),
        )
        self.query_rewrite_generator = QueryRewriteGenerator(llm=self.chat_llm)
        self.relevance_analyzer = RelevanceAnalyzer(llm=self.chat_llm)
        self.final_answer_generator = FinalAnswerGenerator(llm=self.chat_llm)
        # Build the graph
        self._build_graph()

    def _build_graph(self):
        """Build and compile the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("direct_answer_check", self._direct_answer_check_node)
        workflow.add_node("web_or_rag_check", self._web_or_rag_check_node)
        workflow.add_node("rag_retrieval", self._rag_retrieval_node)
        workflow.add_node("analyze_relevance", self._analyze_relevance_node)
        workflow.add_node("process_no_relevant", self._process_no_relevant_node)
        workflow.add_node("process_sub_query", self._process_sub_query_node)
        workflow.add_node("expand_context", self._expand_context_node)
        workflow.add_node("generate_final_answer", self._generate_final_answer_node)

        # Add edges
        workflow.add_edge(START, "direct_answer_check")

        # Conditional edge from direct_answer_check
        workflow.add_conditional_edges(
            "direct_answer_check",
            self._route_after_direct_answer,
            {
                "direct_answer": END,
                "continue": "web_or_rag_check",
            },
        )

        # Conditional edge from web_or_rag_check
        workflow.add_conditional_edges(
            "web_or_rag_check",
            self._route_after_web_or_rag,
            {
                "web_only": "generate_final_answer",
                "rag": "rag_retrieval",
                "direct_answer": "generate_final_answer",
            },
        )

        # Edge from rag_retrieval to analyze_relevance
        workflow.add_edge("rag_retrieval", "analyze_relevance")

        # Conditional edge from analyze_relevance
        workflow.add_conditional_edges(
            "analyze_relevance",
            self._route_after_analyze,
            {
                "complete": "generate_final_answer",
                "incomplete": "expand_context",
                "no_relevant": "process_no_relevant",
                "max_iterations": "generate_final_answer",
            },
        )

        # Conditional edge from expand_context
        workflow.add_edge("expand_context", "analyze_relevance")

        # Conditional edge from process_no_relevant
        workflow.add_conditional_edges(
            "process_no_relevant",
            self._route_after_no_relevant,
            {
                "sub_query": "process_sub_query",
                "search": "analyze_relevance",
                "error": "generate_final_answer",
                "skip": "analyze_relevance",
            },
        )

        # Edge from process_sub_query to generate_final_answer
        workflow.add_edge("process_sub_query", "generate_final_answer")

        # Edge from generate_final_answer to END
        workflow.add_edge("generate_final_answer", END)

        self.graph = workflow.compile()

    # Node implementations

    def _direct_answer_check_node(self, state: AgentState) -> AgentState:
        """Step 0: 判断问题是否可以直接回答。"""
        _log_node_input("direct_answer_check", state)
        query = state.get("query", "")
        logger.debug("直接回答检查开始 - 查询: %s", query)

        # Check if question can be answered directly
        direct_answer_result = self.direct_answer_generator.generate(query)
        logger.debug(
            "直接回答检查完成 - 可以直接回答: %s",
            direct_answer_result.can_direct_answer,
        )

        # Record middle result
        middle_results = state.get("middle_results", [])
        MiddleResultRecorder.record_direct_answer_check(
            middle_results, direct_answer_result.can_direct_answer
        )

        if direct_answer_result.can_direct_answer:
            logger.debug("问题可以直接回答，生成直接答案")
            # Generate direct answer using LLM
            answer, reasoning = self.final_answer_generator.generate_final_answer(
                query, []
            )
            reasoning_str = reasoning if reasoning else ""
            return {
                **state,
                "can_direct_answer": True,
                "result": answer + "\n\n" + reasoning_str,
                "middle_results": middle_results,
            }

        # Cannot answer directly, proceed to check web/rag search
        logger.debug("问题不能直接回答，检查是否需要网络搜索或RAG检索")
        return {
            **state,
            "can_direct_answer": False,
            "middle_results": middle_results,
        }

    def _web_or_rag_check_node(self, state: AgentState) -> AgentState:
        """Step 0.5: 判断是否需要网络搜索或RAG检索。"""
        _log_node_input("web_or_rag_check", state)
        query = state.get("query", "")
        logger.debug("网络/RAG检查开始 - 查询: %s", query)

        # Check if web_or_rag_result is already cached from workflow_filter
        web_or_rag_result_cached = state.get("web_or_rag_result_cached", False)
        if web_or_rag_result_cached:
            # Use cached results from workflow_filter
            need_web = state.get("need_web_search", False)
            need_rag = state.get("need_rag_search", False)
            logger.debug(
                "使用缓存的网络/RAG检查结果 - 需要网络搜索: %s, 需要RAG检索: %s",
                need_web,
                need_rag,
            )
        else:
            # Check if web search or RAG search is needed
            web_or_rag_result = self.web_or_rag_generator.generate(query)
            logger.debug(
                "网络/RAG检查完成 - 需要网络搜索: %s, 需要RAG检索: %s",
                web_or_rag_result.need_web_search,
                web_or_rag_result.need_rag_search,
            )

            need_web = web_or_rag_result.need_web_search
            need_rag = web_or_rag_result.need_rag_search

        # Record middle result
        middle_results = state.get("middle_results", [])
        MiddleResultRecorder.record_web_or_rag_check(middle_results, need_web, need_rag)

        # Case 1: web=true, rag=false -> web search only
        if need_web and not need_rag:
            logger.debug("仅需要网络搜索")
            if self.web_search_tool is None:
                logger.warning("网络搜索工具不可用，直接生成答案")
                answer, reasoning = self.final_answer_generator.generate_final_answer(
                    query, []
                )
                reasoning_str = reasoning if reasoning else ""
                return {
                    **state,
                    "need_web_search": True,
                    "need_rag_search": False,
                    "result": answer + "\n\n" + reasoning_str,
                    "middle_results": middle_results,
                }
            web_nodes = self.web_search_tool.search(query)
            logger.debug("网络搜索完成 - 结果数: %d", len(web_nodes))
            # Check if we have nodes before generating answer
            if not web_nodes:
                logger.error("网络搜索未返回任何结果")
                return {
                    **state,
                    "need_web_search": True,
                    "need_rag_search": False,
                    "nodes": [],
                    "result": "抱歉，网络搜索未找到相关信息。",
                }

            # Generate answer from web search results
            answer = (
                web_nodes[0].node.get_content()
                if hasattr(web_nodes[0].node, "get_content")
                else web_nodes[0].node.text
            )
            if not answer:
                logger.error("网络搜索结果为空")
                return {
                    **state,
                    "need_web_search": True,
                    "need_rag_search": False,
                    "nodes": [],
                    "result": "抱歉，网络搜索未找到相关信息。",
                }
            logger.debug("网络搜索答案生成完成")
            return {
                **state,
                "need_web_search": True,
                "need_rag_search": False,
                "nodes": web_nodes,
                "result": answer + "\n",
                "middle_results": middle_results,
            }

        # Case 2: both false -> warning and direct answer
        if not need_web and not need_rag:
            logger.warning(
                "既不需要网络搜索也不需要RAG检索，但之前判断不能直接回答。"
                "这可能是判断错误，直接生成答案。"
            )
            answer, reasoning = self.final_answer_generator.generate_final_answer(
                query, []
            )
            reasoning_str = reasoning if reasoning else ""
            return {
                **state,
                "need_web_search": False,
                "need_rag_search": False,
                "result": answer + "\n\n" + reasoning_str,
                "middle_results": middle_results,
            }

        # Case 3: web=false, rag=true OR both true -> RAG retrieval (possibly with web)
        logger.debug(
            "需要RAG检索 (web=%s, rag=%s)，触发RAG检索",
            need_web,
            need_rag,
        )
        return {
            **state,
            "need_web_search": need_web,
            "need_rag_search": need_rag,
            "middle_results": middle_results,
        }

    def _rag_retrieval_node(self, state: AgentState) -> AgentState:
        """Step 1: RAG检索。"""
        _log_node_input("rag_retrieval", state)
        query = state.get("query", "")
        query_type = state.get("query_type", QueryType.AUTO)
        need_web = state.get("need_web_search", False)

        logger.debug("RAG检索开始 - 查询: %s, 查询类型: %s", query, query_type)

        # Validate query is not None
        if query is None:
            logger.error("查询不能为None，使用空字符串作为默认值")
            query = ""

        cleaned_query, metadata_filters = parse_query_filters(query)
        logger.debug(
            "查询解析完成 - 清理后查询: %s, 元数据过滤器: %s",
            cleaned_query,
            metadata_filters,
        )

        # Update state with cleaned_query and metadata_filters
        state = {
            **state,
            "query": cleaned_query,
            "metadata_filters": metadata_filters,
        }

        if settings.query_rewrite_bool:
            # Rewrite query for better RAG retrieval
            try:
                rewrite_result = self.query_rewrite_generator.generate(cleaned_query)
                rewritten_query = rewrite_result.rewritten_query.strip()
                if rewritten_query:
                    logger.debug(
                        "查询重写完成 - 原始查询: %s, 重写后查询: %s",
                        cleaned_query,
                        rewritten_query,
                    )
                    cleaned_query = rewritten_query
                    # Update state with rewritten query
                    state = {
                        **state,
                        "query": cleaned_query,
                    }
                else:
                    logger.debug("查询重写结果为空，使用原始查询")
            except Exception as e:
                logger.warning("查询重写失败，使用原始查询: %s", str(e))

        # Retrieve nodes directly using router's engines
        if query_type == QueryType.FAQ and self.router.faq_query_engine:
            logger.debug("使用FAQ查询引擎检索")
            nodes = self.router.faq_query_engine.retrieve(
                cleaned_query, metadata_filters
            )
            logger.debug("FAQ检索完成 - 检索到 %d 个节点", len(nodes))
        elif query_type == QueryType.DOCS:
            logger.debug("使用DOCS查询引擎检索")
            nodes = self.router.docs_query_engine.retrieve(
                cleaned_query, metadata_filters
            )
            logger.debug("DOCS检索完成 - 检索到 %d 个节点", len(nodes))
        else:
            # AUTO mode: try FAQ first, then docs
            logger.debug("AUTO模式 - 先尝试FAQ检索")
            if self.router.faq_query_engine:
                nodes, top_score = self.router.faq_query_engine.retrieve_with_scores(
                    cleaned_query, metadata_filters
                )
                logger.debug(
                    "FAQ检索完成 - 检索到 %d 个节点, 最高分数: %.4f, 阈值: %.4f",
                    len(nodes),
                    top_score,
                    settings.faq_score_threshold,
                )
                # If FAQ score is low, also retrieve from docs
                if top_score < settings.faq_score_threshold:
                    logger.debug("FAQ分数低于阈值，切换到DOCS检索")
                    docs_nodes = self.router.docs_query_engine.retrieve(
                        cleaned_query, metadata_filters
                    )
                    logger.debug("DOCS检索完成 - 检索到 %d 个节点", len(docs_nodes))
                    nodes = docs_nodes  # !只用docs的答案
            else:
                logger.debug("FAQ查询引擎不可用，使用DOCS查询引擎")
                nodes = self.router.docs_query_engine.retrieve(
                    cleaned_query, metadata_filters
                )
                logger.debug("DOCS检索完成 - 检索到 %d 个节点", len(nodes))

        # Now we have nodes from rag;

        # raise Exception("Stop here")
        # Extract keywords
        keywords, keywords_other = self._extract_keywords_from_query(cleaned_query)
        logger.debug(
            "关键词提取完成 - 提取到 %d 个关键词: %s, 过滤关键词数: %d",
            len(keywords),
            keywords,
            len(keywords_other),
        )

        # Tokenize query for sliding window search
        tokens = self.local_file_search.keyword_extractor.tokenize_text(
            cleaned_query, filter_stopwords=False
        )
        logger.debug("查询分词完成 - 分词数: %d, tokens: %s", len(tokens), tokens)

        # TODO first use tokens or keywords?
        # First attempt: search using extracted keywords
        print("keywords", keywords)
        keyword_nodes_tmp = (
            self.local_file_search.search_keyword_using_binary_algorithm(keywords)
        )
        keyword_nodes = self._search_results_to_nodes(keyword_nodes_tmp)

        # If no results found, try again with tokenized tokens
        # TODO now not use this, if use, add the min slide window size
        # if not keyword_nodes:
        #     logger.debug("使用keywords未找到结果，尝试使用直接分词进行搜索")
        #     keyword_nodes_tmp = self.local_file_search.search_keyword_using_binary_algorithm(tokens)

        #     for token in tokens:
        #         if token not in keywords:
        #             keywords.append(token)

        #     keyword_nodes = self._search_results_to_nodes(keyword_nodes_tmp)
        #     logger.debug("使用tokens搜索完成 - 节点数: %d", len(keyword_nodes))
        print("len(nodes): ", len(nodes))
        print("len(keyword_nodes): ", len(keyword_nodes))
        print(f"nodes: {nodes}")
        print(f"keyword_nodes: {keyword_nodes}")
        # raise Exception("Stop here")
        if keyword_nodes and len(keyword_nodes) > 0:
            nodes = self._merge_nodes(nodes, keyword_nodes)
            logger.debug("合并关键词搜索结果后节点数: %d", len(nodes))
        print("nodes: ", nodes)
        # Check if web search is also needed (when both web and rag are true)
        if need_web and self.web_search_tool:
            logger.debug("同时需要网络搜索，执行网络搜索")
            # Perform web search concurrently with RAG (if not already done)
            web_nodes = self.web_search_tool.search(query)
            logger.debug("网络搜索完成 - 结果数: %d", len(web_nodes))

            # Merge web search results with RAG results
            if web_nodes:
                nodes = self._merge_nodes(nodes, web_nodes)
                logger.debug("合并网络搜索结果后节点数: %d", len(nodes))

        # Rerank merged results (SafeReranker handles all edge cases)
        if self.reranker:
            query_bundle = QueryBundle(query)
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
            logger.debug("Reranker重排序完成 - 节点数: %d", len(nodes))

        # Record middle result
        middle_results = state.get("middle_results", [])
        MiddleResultRecorder.record_rag_retrieval(middle_results, keywords, len(nodes))

        return {
            **state,
            "nodes": nodes,
            "keywords": keywords,
            "keywords_other": keywords_other,  # Cache filtered out keywords for use in _process_no_relevant_node
            "iteration": 0,
            "query": cleaned_query,  # Ensure cleaned_query is in returned state
            "metadata_filters": metadata_filters,  # Ensure metadata_filters is in returned state
            "middle_results": middle_results,
        }

    def _analyze_relevance_node(self, state: AgentState) -> AgentState:
        """Step 2: LLM判断相关性和完整性。"""
        _log_node_input("analyze_relevance", state)
        query = state.get("query", "")
        iteration = state.get("iteration", 0)
        nodes = state.get("nodes", [])
        keywords = state.get("keywords", [])

        # Prepend outer and inner previous Q&A context nodes if exist
        outer_previous_qa_context = state.get("outer_previous_qa_context", [])
        inner_previous_qa_context = state.get("inner_previous_qa_context", [])
        if outer_previous_qa_context or inner_previous_qa_context:
            context_nodes = self._qa_context_to_nodes(
                outer_previous_qa_context, inner_previous_qa_context
            )
            nodes = context_nodes + nodes
            logger.debug(
                "添加了 %d 个外部Q&A上下文节点和 %d 个内部Q&A上下文节点到分析节点中",
                len(outer_previous_qa_context),
                len(inner_previous_qa_context),
            )

        logger.debug(
            "相关性分析开始 - 查询: %s, 迭代次数: %d, 节点数: %d",
            query,
            iteration,
            len(nodes),
        )

        # Limit total nodes to prevent infinite accumulation
        max_nodes = settings.max_nodes_to_llm
        logger.debug("当前节点总数: %d, 最大节点数限制: %d", len(nodes), max_nodes)
        if len(nodes) > max_nodes:
            logger.warning(
                "Too many nodes accumulated (%d > %d), truncating to prevent token overflow. "
                "This may indicate a loop or excessive context expansion.",
                len(nodes),
                max_nodes,
            )
            # Keep the highest scored nodes
            nodes = sorted(nodes, key=lambda n: n.score, reverse=True)[:max_nodes]
            logger.debug("节点截断后数量: %d", len(nodes))

        # Check max iterations
        logger.debug(
            "检查最大迭代次数 - 当前迭代: %d, 最大迭代: %d",
            iteration,
            settings.agent_max_iterations,
        )
        if iteration > settings.agent_max_iterations:
            logger.debug("达到最大迭代次数，生成最终答案")
            answer, reasoning = self.final_answer_generator.generate_final_answer(
                query, nodes
            )
            references = extract_docs_references(nodes)
            formatted_refs = format_docs_references(references)
            reasoning_str = reasoning if reasoning else ""
            return {
                **state,
                "nodes": nodes,
                "result": answer + formatted_refs + reasoning_str,
            }

        # Analyze
        logger.debug("开始LLM分析相关性和完整性")
        analysis = self.relevance_analyzer.analyze_relevance_and_completeness(
            query, nodes, keywords
        )
        logger.debug(
            "分析完成 - 相关文件路径数: %d, 是否完整: %s",
            len(analysis.relevant_file_paths),
            analysis.is_complete,
        )

        # Record middle result
        middle_results = state.get("middle_results", [])
        MiddleResultRecorder.record_analyze_relevance(
            middle_results,
            analysis.relevant_file_paths,
            analysis.is_complete,
        )

        # Check if complete
        is_relevant = len(analysis.relevant_file_paths) > 0
        is_complete = analysis.is_complete
        logger.debug(
            "检查完成条件 - 是否相关: %s, 是否完整: %s", is_relevant, is_complete
        )

        filtered_nodes = self.relevance_analyzer.filter_relevant_nodes(
            nodes, analysis.relevant_file_paths
        )
        logger.debug(
            "节点过滤完成 - 原始节点数: %d, 过滤后节点数: %d",
            len(nodes),
            len(filtered_nodes),
        )
        # 更新 state 的 nodes 为过滤后的节点
        nodes = filtered_nodes
        state["nodes"] = nodes

        # 处理逻辑：is_complete true 且有relevant_file_paths -> 针对relevant_file_paths回答
        if is_complete and is_relevant:
            logger.debug("查询已相关且完整，过滤相关节点并生成最终答案")
            # 过滤出只与 relevant_file_paths 相关的节点
            answer, reasoning = self.final_answer_generator.generate_final_answer(
                query, nodes
            )
            references = extract_docs_references(nodes)
            formatted_refs = format_docs_references(references)
            reasoning_str = reasoning if reasoning else ""
            return {
                **state,
                "nodes": nodes,
                "analysis": analysis,
                "result": answer + formatted_refs + reasoning_str,
                "middle_results": middle_results,
            }

        # 如果没有relevant_file_paths说明llm返回结果有误，返回没有结果llm判断失败
        if is_complete and not is_relevant:
            logger.error("LLM判断失败：is_complete=true但没有relevant_file_paths")
            return {
                **state,
                "analysis": analysis,
                "result": "抱歉，无法找到相关信息。LLM判断结果异常，请重试或调整问题。",
                "middle_results": middle_results,
            }

        # 否则 is_complete false
        logger.debug("迭代次数更新为: %d", iteration + 1)

        return {
            **state,
            "nodes": nodes,
            "keywords": keywords,  # 保存keywords以保持状态一致性
            "analysis": analysis,
            "iteration": iteration + 1,
            "middle_results": middle_results,
        }

    def _process_no_relevant_node(self, state: AgentState) -> AgentState:
        """处理没有任何相关文档的情况。"""
        _log_node_input("process_no_relevant", state)
        analysis = state.get("analysis")
        if analysis is None:
            return {**state, "result": "分析结果缺失"}

        is_relevant = len(analysis.relevant_file_paths) > 0
        if is_relevant:
            # 有相关文档，跳过此步骤
            logger.debug("有相关文档，跳过无相关文档处理步骤")
            return state

        # 检查是否是子问题workflow
        is_inner_sub_query_workflow = state.get("is_inner_sub_query_workflow", False)
        keywords = state.get("keywords", [])
        keywords_other = state.get("keywords_other", [])

        if is_inner_sub_query_workflow:
            # 子问题workflow中，直接使用query（在_process_sub_query_node中已经设置为sub_query）
            query_to_use = state.get("query", "")
            logger.debug(
                "子问题workflow中，使用不包含子问题的分析模型 - 查询: %s", query_to_use
            )
            no_relevant_result = (
                self.relevance_analyzer.analyze_no_relevant_without_sub_queries(
                    query_to_use, keywords
                )
            )
            logger.debug(
                "无相关文档分析完成（子问题workflow） - 关键词数: %d, 假想答案: %s",
                len(no_relevant_result.missing_info_keywords),
                "有" if no_relevant_result.hypothetical_answer else "无",
            )

            # Merge keywords from state, missing_info_keywords, and keywords_other, then update state keywords
            merged_keywords_set = (
                set(keywords)
                | set(no_relevant_result.missing_info_keywords)
                | set(keywords_other)
            )
            updated_keywords = list(merged_keywords_set)
            state["keywords"] = updated_keywords
            logger.debug(
                "合并关键词完成（子问题workflow） - 原始关键词数: %d, missing_info_keywords数: %d, keywords_other数: %d, 合并后: %d",
                len(keywords),
                len(no_relevant_result.missing_info_keywords),
                len(keywords_other),
                len(updated_keywords),
            )

            # Record middle result (for sub-query workflow, sub_queries is empty)
            middle_results = state.get("middle_results", [])
            MiddleResultRecorder.record_process_no_relevant(
                middle_results,
                no_relevant_result.missing_info_keywords,
                [],  # sub_queries is empty for NoRelevantResultWithoutSubQueries
                no_relevant_result.hypothetical_answer,
            )

            # 检查是否有missing_info_keywords或hypothetical_answer
            has_keywords = len(no_relevant_result.missing_info_keywords) > 0
            has_hypothetical = bool(no_relevant_result.hypothetical_answer)
            if has_keywords or has_hypothetical:
                logger.debug("检测到关键词或假想答案，触发并发检索")
                # Merge keywords_other into missing_info_keywords to enhance search
                original_missing_info_keywords_len = len(
                    no_relevant_result.missing_info_keywords
                )
                enhanced_missing_info_keywords = list(
                    set(no_relevant_result.missing_info_keywords) | set(keywords_other)
                )
                no_relevant_result.missing_info_keywords = (
                    enhanced_missing_info_keywords
                )
                logger.debug(
                    "增强missing_info_keywords完成（子问题workflow） - 原始数: %d, keywords_other数: %d, 增强后: %d",
                    original_missing_info_keywords_len,
                    len(keywords_other),
                    len(enhanced_missing_info_keywords),
                )
                # 触发并发检索
                new_nodes = self._concurrent_search(no_relevant_result, query_to_use)
                # Merge with existing nodes if needed
                existing_nodes = state.get("nodes", [])
                if settings.reserve_keywords_old_nodes and existing_nodes:
                    nodes = self._merge_nodes(existing_nodes, new_nodes)
                else:
                    nodes = new_nodes
                return {
                    **state,
                    "no_relevant_result": no_relevant_result,
                    "nodes": nodes,
                    "keywords_other": [],  # Clear keywords_other after enhancement
                    "middle_results": middle_results,
                }

            # 否则，不完整，且没有missing_info_keywords hypothetical_answer
            logger.error("LLM判断失败：无相关文档且无搜索策略（子问题workflow）")
            return {
                **state,
                "result": "抱歉，无法找到相关信息，且无法生成有效的搜索策略。请尝试调整问题或提供更多上下文。",
                "middle_results": middle_results,
            }

        # 非子问题workflow，使用完整的analyze_no_relevant
        query = state.get("query", "")
        logger.debug("开始分析无相关文档情况 - 查询: %s", query)
        no_relevant_result = self.relevance_analyzer.analyze_no_relevant(
            query, keywords
        )
        logger.debug(
            "无相关文档分析完成 - 子问题数: %d, 关键词数: %d, 假想答案: %s",
            len(no_relevant_result.sub_queries),
            len(no_relevant_result.missing_info_keywords),
            "有" if no_relevant_result.hypothetical_answer else "无",
        )

        # Merge keywords from state, missing_info_keywords, and keywords_other, then update state keywords
        merged_keywords_set = (
            set(keywords)
            | set(no_relevant_result.missing_info_keywords)
            | set(keywords_other)
        )
        updated_keywords = list(merged_keywords_set)
        state["keywords"] = updated_keywords
        logger.debug(
            "合并关键词完成 - 原始关键词数: %d, missing_info_keywords数: %d, keywords_other数: %d, 合并后: %d",
            len(keywords),
            len(no_relevant_result.missing_info_keywords),
            len(keywords_other),
            len(updated_keywords),
        )

        # Record middle result
        middle_results = state.get("middle_results", [])
        MiddleResultRecorder.record_process_no_relevant(
            middle_results,
            no_relevant_result.missing_info_keywords,
            no_relevant_result.sub_queries,
            no_relevant_result.hypothetical_answer,
        )

        # TODO 这里可能比较容易触发
        # 如果有sub_queries，触发子问题处理
        if no_relevant_result.sub_queries and settings.inner_sub_problems:
            logger.debug("检测到子问题，触发子问题处理流程")
            return {
                **state,
                "no_relevant_result": no_relevant_result,
                "middle_results": middle_results,
            }

        # 如果没有sub_queries，检查是否有missing_info_keywords或hypothetical_answer
        has_keywords = len(no_relevant_result.missing_info_keywords) > 0
        has_hypothetical = bool(no_relevant_result.hypothetical_answer)
        if has_keywords or has_hypothetical:
            logger.debug("检测到关键词或假想答案，触发并发检索")
            # Merge keywords_other into missing_info_keywords to enhance search
            original_missing_info_keywords_len = len(
                no_relevant_result.missing_info_keywords
            )
            enhanced_missing_info_keywords = list(
                set(no_relevant_result.missing_info_keywords) | set(keywords_other)
            )
            no_relevant_result.missing_info_keywords = enhanced_missing_info_keywords
            logger.debug(
                "增强missing_info_keywords完成 - 原始数: %d, keywords_other数: %d, 增强后: %d",
                original_missing_info_keywords_len,
                len(keywords_other),
                len(enhanced_missing_info_keywords),
            )
            new_nodes = self._concurrent_search(no_relevant_result, query)
            # Merge with existing nodes if needed
            existing_nodes = state.get("nodes", [])
            if settings.reserve_keywords_old_nodes and existing_nodes:
                nodes = self._merge_nodes(existing_nodes, new_nodes)
            else:
                nodes = new_nodes
            return {
                **state,
                "no_relevant_result": no_relevant_result,
                "nodes": nodes,
                "keywords_other": [],  # Clear keywords_other after enhancement
                "middle_results": middle_results,
            }

        # 否则，不完整，且没有sub_queries missing_info_keywords hypothetical_answer
        logger.error("LLM判断失败：无相关文档且无搜索策略")
        return {
            **state,
            "result": "抱歉，无法找到相关信息，且无法生成有效的搜索策略。请尝试调整问题或提供更多上下文。",
            "middle_results": middle_results,
        }

    async def _process_sub_query_node(self, state: AgentState) -> AgentState:
        """处理子问题：在一个循环中处理所有子问题，避免多次事件触发。"""
        _log_node_input("process_sub_query", state)
        no_relevant_result = state.get("no_relevant_result")
        if no_relevant_result is None:
            return {
                **state,
                "result": "子问题结果缺失",
            }

        sub_queries = no_relevant_result.sub_queries
        if not sub_queries or len(sub_queries) == 0:
            return {
                **state,
                "result": "子问题结果缺失",
            }
        if len(sub_queries) == 1:
            logger.warning("子问题数量为1 这不正常")
        inner_previous_qa_context = state.get("inner_previous_qa_context", [])

        logger.debug("开始处理 %d 个子问题", len(sub_queries))

        # 在一个循环中处理所有子问题
        # inner_previous_qa_context is now a list, so we'll append to it
        all_qa_context = (
            inner_previous_qa_context.copy() if inner_previous_qa_context else []
        )
        for idx, sub_query in enumerate(sub_queries):
            logger.debug(
                "处理子问题 %d/%d - 子问题: %s",
                idx + 1,
                len(sub_queries),
                sub_query,
            )

            # 对于LangGraph workflow，直接调用图的ainvoke方法（异步版本）
            # 不需要通过workflow_client，因为我们在同一个进程中
            # 使用原始query，不修改query，上下文通过inner_previous_qa_context传递
            logger.debug(
                "使用LangGraph workflow处理子问题: %s, 已有 %d 个内部Q&A上下文",
                sub_query,
                len(all_qa_context),
            )
            try:
                # 直接调用图的ainvoke方法，设置is_inner_sub_query_workflow=True以防止子问题再次分割
                # 构建初始状态，与invoke方法中的逻辑一致
                initial_state: AgentState = {
                    "query": sub_query,  # 使用原始query，不修改
                    "query_type": QueryType.AUTO,
                    "can_direct_answer": False,
                    "need_web_search": False,
                    "need_rag_search": False,
                    "web_or_rag_result_cached": False,
                    "nodes": [],
                    "keywords": [],
                    "analysis": None,
                    "iteration": 0,
                    "no_relevant_result": None,
                    "current_sub_query": None,
                    "inner_previous_qa_context": all_qa_context.copy(),  # 传递之前的内部Q&A上下文
                    "outer_previous_qa_context": state.get(
                        "outer_previous_qa_context", []
                    ),
                    "is_inner_sub_query_workflow": True,  # 设置标志防止子问题再次分割
                    "is_outer_sub_query_workflow": state.get(
                        "is_outer_sub_query_workflow", False
                    ),
                    "result": None,
                    "middle_results": [],
                }
                # 使用异步的ainvoke方法
                result = await self.graph.ainvoke(initial_state)
                sub_answer = result.get("result", "抱歉，无法生成答案。")
                logger.debug(
                    "子问题 %d/%d workflow完成，结果长度: %d",
                    idx + 1,
                    len(sub_queries),
                    len(sub_answer),
                )
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e) if str(e) else repr(e)
                logger.error(
                    "子问题workflow调用失败 - 异常类型: %s, 异常消息: %s, 子问题: %s",
                    error_type,
                    error_msg,
                    sub_query,
                    exc_info=True,
                )
                sub_answer = f"抱歉，处理子问题时发生错误: {error_type}: {error_msg}"

            # 更新上下文，累积所有子问题的Q&A
            all_qa_context.append(
                {
                    "query": sub_query,
                    "answer": sub_answer,
                }
            )

        # 所有子问题处理完成，更新上下文并返回最终答案
        logger.debug("所有子问题处理完成，生成最终答案")

        # Record middle result
        middle_results = state.get("middle_results", [])
        MiddleResultRecorder.record_process_sub_query(middle_results, all_qa_context)

        # Format the final answer from the Q&A context list
        result_parts = []
        for qa_pair in all_qa_context:
            result_parts.append(f"问题: {qa_pair['query']}\n回答: {qa_pair['answer']}")
        final_answer = "基于以下子问题的处理结果：\n\n" + "\n\n".join(result_parts)
        return {
            **state,
            "inner_previous_qa_context": all_qa_context,
            "result": final_answer,
            "middle_results": middle_results,
        }

    def _expand_context_node(self, state: AgentState) -> AgentState:
        """Step 4: 找到chunk所在位置 → 扩展上下文。"""
        _log_node_input("expand_context", state)
        analysis = state.get("analysis")
        nodes = state.get("nodes", [])

        if analysis is None:
            return state

        # Only expand context if relevant but incomplete
        is_relevant = len(analysis.relevant_file_paths) > 0
        is_complete = analysis.is_complete
        should_process = is_relevant and not is_complete
        logger.debug(
            "上下文扩展步骤开始 - 是否相关: %s, 是否完整: %s, 是否处理: %s",
            is_relevant,
            is_complete,
            should_process,
        )
        if not should_process:
            # Not relevant or already complete, skip expansion
            logger.debug("跳过上下文扩展步骤")
            return state

        logger.debug(
            "开始扩展上下文 - 节点数: %d, 扩展模式: %s, 扩展比例: %.2f",
            len(nodes),
            settings.expand_context_mode,
            settings.expand_context_ratio,
        )
        expanded_nodes = []
        for node in nodes:
            file_path = node.node.metadata.get("file_path") or node.node.metadata.get(
                "source"
            )
            if not file_path:
                logger.debug("节点缺少文件路径，跳过扩展")
                continue

            content = (
                node.node.get_content()
                if hasattr(node.node, "get_content")
                else node.node.text
            )
            if not content:
                logger.debug("节点内容为空，跳过扩展 - 文件路径: %s", file_path)
                continue

            logger.debug("扩展节点上下文 - 文件路径: %s", file_path)
            expanded = self.local_file_search.expand_context(
                content,
                file_path,
                expand_mode=settings.expand_context_mode,
                expand_ratio=settings.expand_context_ratio,
            )

            if expanded:
                logger.debug(
                    "节点扩展成功 - 文件路径: %s, 起始行: %s, 结束行: %s",
                    expanded.get("file_path"),
                    expanded.get("start_line"),
                    expanded.get("end_line"),
                )
                from llama_index.core.schema import TextNode

                expanded_node = TextNode(
                    text=expanded["content"],
                    metadata={
                        "file_path": expanded["file_path"],
                        "source": expanded["file_path"],
                        "type": "markdown",
                        "start_line": expanded["start_line"],
                        "end_line": expanded["end_line"],
                    },
                )
                # 使用临时分数，稍后通过reranker重新评分
                expanded_nodes.append(NodeWithScore(node=expanded_node, score=0.0))
            else:
                logger.debug("节点扩展失败 - 文件路径: %s", file_path)

        # 使用reranker对扩展节点重新评分 (SafeReranker handles all edge cases)
        if expanded_nodes and self.reranker:
            query = state.get("query", "")
            query_bundle = QueryBundle(query)
            expanded_nodes = self.reranker.postprocess_nodes(
                expanded_nodes, query_bundle=query_bundle
            )
            logger.debug("Reranker重排序完成 - 节点数: %d", len(expanded_nodes))

        logger.debug("上下文扩展完成 - 扩展节点数: %d", len(expanded_nodes))

        # 根据reserve_expanded_old_nodes决定是合并还是替换
        if expanded_nodes:
            if settings.reserve_expanded_old_nodes:
                logger.debug(
                    "合并扩展节点 - 原有节点数: %d, 扩展节点数: %d",
                    len(nodes),
                    len(expanded_nodes),
                )
                nodes = self._merge_nodes(nodes, expanded_nodes)
                logger.debug("合并后节点数: %d", len(nodes))
            else:
                logger.debug("替换节点 - 扩展节点数: %d", len(expanded_nodes))
                nodes = expanded_nodes
        # TODO here truncate nodes by tokens
        nodes, total_tokens, was_truncated = truncate_nodes_by_tokens(
            nodes, settings.llm_max_tokens
        )

        # Record middle result
        middle_results = state.get("middle_results", [])
        MiddleResultRecorder.record_expand_context(
            middle_results, len(expanded_nodes) if expanded_nodes else 0
        )

        return {
            **state,
            "nodes": nodes,
            "middle_results": middle_results,
        }

    def _generate_final_answer_node(self, state: AgentState) -> AgentState:
        """生成最终答案节点。"""
        _log_node_input("generate_final_answer", state)
        query = state.get("query", "")
        nodes = state.get("nodes", [])
        result = state.get("result")
        middle_results = state.get("middle_results", [])

        # Prepend outer and inner previous Q&A context nodes if exist
        outer_previous_qa_context = state.get("outer_previous_qa_context", [])
        inner_previous_qa_context = state.get("inner_previous_qa_context", [])

        if outer_previous_qa_context or inner_previous_qa_context:
            context_nodes = self._qa_context_to_nodes(
                outer_previous_qa_context, inner_previous_qa_context
            )
            nodes = context_nodes + nodes
            logger.debug(
                "添加了 %d 个外部Q&A上下文节点和 %d 个内部Q&A上下文节点到最终答案生成",
                len(outer_previous_qa_context),
                len(inner_previous_qa_context),
            )

        # If result already exists (from direct answer or web search), handle return_middle_result
        if result:
            if settings.return_middle_result:
                formatted_result = MiddleResultRecorder.format_middle_results(
                    middle_results, result
                )
                return {
                    **state,
                    "result": formatted_result,
                }
            return state

        # Requirement: Check if previous node is analyze_relevance and previous previous node is expand_context
        # and expand_node_force_end_line is True and model thinks it still cannot summarize/answer

        no_complete_answer = state.get("no_complete_answer", False)
        if no_complete_answer:
            logger.debug(
                "需求2条件满足：上一个节点是analyze_relevance，上上一个节点是expand_context，"
                "且expand_node_force_end_line为true，且模型认为依然不能总结回答，使用generate_not_complete_answer"
            )
            answer, reasoning = (
                self.final_answer_generator.generate_not_complete_answer(query, nodes)
            )
            references = extract_docs_references(nodes)
            formatted_refs = format_docs_references(references)
            reasoning_str = reasoning if reasoning else ""
            final_result = answer + formatted_refs + reasoning_str

            if settings.return_middle_result:
                formatted_result = MiddleResultRecorder.format_middle_results(
                    middle_results, final_result
                )
                return {
                    **state,
                    "result": formatted_result,
                }
            return {
                **state,
                "result": final_result,
            }

        # Otherwise, generate from nodes using normal method
        answer, reasoning = self.final_answer_generator.generate_final_answer(
            query, nodes
        )
        references = extract_docs_references(nodes)
        formatted_refs = format_docs_references(references)
        reasoning_str = reasoning if reasoning else ""
        final_result = answer + formatted_refs + reasoning_str

        if settings.return_middle_result:
            formatted_result = MiddleResultRecorder.format_middle_results(
                middle_results, final_result
            )
            return {
                **state,
                "result": formatted_result,
            }
        return {
            **state,
            "result": final_result,
        }

    # Routing functions

    def _route_after_direct_answer(
        self, state: AgentState
    ) -> Literal["direct_answer", "continue"]:
        """路由：直接回答检查后。"""
        can_direct_answer = state.get("can_direct_answer", False)
        route = "direct_answer" if can_direct_answer else "continue"
        if debug and debug_logger:
            debug_logger.info(
                f"路由决策 [direct_answer_check -> {route}] - can_direct_answer: {can_direct_answer}"
            )
            for handler in debug_logger.handlers:
                if isinstance(handler, FileHandler):
                    handler.flush()
        return route

    def _route_after_web_or_rag(
        self, state: AgentState
    ) -> Literal["web_only", "rag", "direct_answer"]:
        """路由：Web/RAG检查后。"""
        need_web = state.get("need_web_search", False)
        need_rag = state.get("need_rag_search", False)
        result = state.get("result")

        # If result already exists (from web search or direct answer), go to final answer
        if result:
            route = "direct_answer"
        # If only web search needed
        elif need_web and not need_rag:
            route = "web_only"
        # If RAG needed (with or without web)
        elif need_rag:
            route = "rag"
        # Fallback
        else:
            route = "direct_answer"

        if debug and debug_logger:
            debug_logger.info(
                f"路由决策 [web_or_rag_check -> {route}] - "
                f"need_web: {need_web}, need_rag: {need_rag}, has_result: {bool(result)}"
            )
            for handler in debug_logger.handlers:
                if isinstance(handler, FileHandler):
                    handler.flush()

        return route

    def _route_after_analyze(
        self, state: AgentState
    ) -> Literal["complete", "incomplete", "no_relevant", "max_iterations"]:
        """路由：相关性分析后。"""
        analysis = state.get("analysis")
        iteration = state.get("iteration", 0)
        result = state.get("result")
        middle_results = state.get("middle_results", [])

        # Requirement 2: Check if previous node is analyze_relevance and previous previous node is expand_context
        # and expand_node_force_end_line is True and model thinks it still cannot summarize/answer
        previous_node = None
        previous_previous_node = None
        if len(middle_results) >= 1:
            previous_node = middle_results[-1].get("node_type")
        if len(middle_results) >= 2:
            previous_previous_node = middle_results[-2].get("node_type")

        # Check requirement 2 condition
        requirement_met = (
            previous_node == NodeType.ANALYZE_RELEVANCE.value
            and previous_previous_node == NodeType.EXPAND_CONTEXT.value
            and settings.expand_node_force_end_line
            and analysis is not None
            and not analysis.is_complete
        )

        # If result already exists, go to final answer
        if result:
            route = "complete"
        # Requirement 2: If condition met, force to generate final answer (even if incomplete)
        elif requirement_met:
            logger.debug(
                "需求条件满足：上一个节点是analyze_relevance，上上一个节点是expand_context，"
                "且expand_node_force_end_line为true，且模型认为依然不能总结回答，强制生成最终答案"
            )
            route = "complete"
            state["no_complete_answer"] = True
        # Check max iterations
        elif iteration > settings.agent_max_iterations:
            route = "max_iterations"
        elif analysis is None:
            route = "no_relevant"
        else:
            is_relevant = len(analysis.relevant_file_paths) > 0
            is_complete = analysis.is_complete

            if is_complete and is_relevant:
                route = "complete"
            elif not is_relevant:
                route = "no_relevant"
            else:
                # is_complete false and is_relevant true
                route = "incomplete"

        if debug and debug_logger:
            is_relevant = len(analysis.relevant_file_paths) > 0 if analysis else False
            is_complete = analysis.is_complete if analysis else None
            debug_logger.info(
                f"路由决策 [analyze_relevance -> {route}] - "
                f"iteration: {iteration}, is_relevant: {is_relevant}, "
                f"is_complete: {is_complete}, has_result: {bool(result)}, no_complete_answer: {state.get('no_complete_answer', False)}"
            )
            for handler in debug_logger.handlers:
                if isinstance(handler, FileHandler):
                    handler.flush()

        return route

    def _route_after_no_relevant(
        self, state: AgentState
    ) -> Literal["sub_query", "search", "error", "skip"]:
        """路由：无相关文档处理后。"""
        analysis = state.get("analysis")
        if analysis is None:
            route = "error"
        else:
            is_relevant = len(analysis.relevant_file_paths) > 0
            if is_relevant:
                route = "skip"
            else:
                no_relevant_result = state.get("no_relevant_result")
                if no_relevant_result is None:
                    route = "error"
                # Check if has sub_queries (only for NoRelevantResult, not NoRelevantResultWithoutSubQueries)
                elif hasattr(no_relevant_result, "sub_queries"):
                    if len(no_relevant_result.sub_queries) > 0:
                        route = "sub_query"
                    else:
                        route = "error"
                else:
                    # Check if has keywords or hypothetical answer
                    has_keywords = len(no_relevant_result.missing_info_keywords) > 0
                    has_hypothetical = bool(no_relevant_result.hypothetical_answer)
                    if has_keywords or has_hypothetical:
                        route = "search"
                    else:
                        route = "error"

        if debug and debug_logger:
            analysis = state.get("analysis")
            no_relevant_result = state.get("no_relevant_result")
            is_relevant = len(analysis.relevant_file_paths) > 0 if analysis else False
            has_sub_queries = (
                len(no_relevant_result.sub_queries) > 0
                if no_relevant_result and hasattr(no_relevant_result, "sub_queries")
                else False
            )
            has_keywords = (
                len(no_relevant_result.missing_info_keywords) > 0
                if no_relevant_result
                and hasattr(no_relevant_result, "missing_info_keywords")
                else False
            )
            has_hypothetical = (
                bool(no_relevant_result.hypothetical_answer)
                if no_relevant_result
                and hasattr(no_relevant_result, "hypothetical_answer")
                else False
            )
            debug_logger.info(
                f"路由决策 [process_no_relevant -> {route}] - "
                f"is_relevant: {is_relevant}, has_sub_queries: {has_sub_queries}, "
                f"has_keywords: {has_keywords}, has_hypothetical: {has_hypothetical}"
            )
            for handler in debug_logger.handlers:
                if isinstance(handler, FileHandler):
                    handler.flush()

        return route

    # Helper methods

    def _concurrent_search(
        self,
        no_relevant_result: NoRelevantResult | NoRelevantResultWithoutSubQueries,
        query: str,
    ) -> List[NodeWithScore]:
        """并发检索：missing_info_keywords用原来的检索，hypothetical_answer用docs_query_engine.retrieve。"""

        def search_by_keywords():
            """使用关键词进行本地文件搜索。"""
            if not no_relevant_result.missing_info_keywords:
                return []
            logger.debug(
                "开始关键词搜索 - 关键词: %s",
                no_relevant_result.missing_info_keywords,
            )
            search_results = self.local_file_search.search_by_keywords(
                no_relevant_result.missing_info_keywords,
                expand_mode=settings.expand_context_mode,
                expand_ratio=settings.keyword_expand_ratio,
            )
            nodes = self._search_results_to_nodes(search_results)
            logger.debug("关键词搜索完成 - 节点数: %d", len(nodes))
            return nodes

        def search_keywords_from_vector_search():
            """使用关键词进行向量检索。"""
            if not no_relevant_result.missing_info_keywords:
                return []
            logger.debug("开始关键词向量检索")
            nodes = []
            for keyword in no_relevant_result.missing_info_keywords:
                new_nodes = self.router.docs_query_engine.retrieve(keyword)
                nodes = self._merge_nodes(nodes, new_nodes)
            logger.debug("关键词向量检索完成 - 节点数: %d", len(nodes))
            return nodes

        def search_by_hypothetical():
            """使用假想答案进行向量检索。"""
            if not no_relevant_result.hypothetical_answer:
                return []
            logger.debug("开始假想答案向量检索")
            nodes = self.router.docs_query_engine.retrieve(
                no_relevant_result.hypothetical_answer
            )
            logger.debug("假想答案向量检索完成 - 节点数: %d", len(nodes))
            return nodes

        # 并发执行三个检索（使用线程池而不是异步，因为函数不是async）
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            keyword_future = executor.submit(search_by_keywords)
            hypothetical_future = executor.submit(search_by_hypothetical)
            keywords_from_vector_search_future = executor.submit(
                search_keywords_from_vector_search
            )
            keyword_nodes = keyword_future.result()
            hypothetical_nodes = hypothetical_future.result()
            keywords_from_vector_search_nodes = (
                keywords_from_vector_search_future.result()
            )

        # 合并并去重
        all_nodes = self._merge_nodes(
            keyword_nodes, hypothetical_nodes, keywords_from_vector_search_nodes
        )
        logger.debug("并发检索完成 - 合并后节点数: %d", len(all_nodes))

        # 使用reranker重排序 (SafeReranker handles all edge cases)
        if all_nodes and self.reranker:
            query_bundle = QueryBundle(query)
            all_nodes = self.reranker.postprocess_nodes(all_nodes, query_bundle)
            logger.debug("Reranker重排序完成 - 节点数: %d", len(all_nodes))

        return all_nodes

    def _qa_context_to_nodes(
        self,
        outer_qa_context: List[Dict[str, str]],
        inner_qa_context: List[Dict[str, str]],
    ) -> List[NodeWithScore]:
        """Convert Q&A context lists to NodeWithScore objects.

        Generates nodes in the format:
        - node1: "问题1：\n回答1" (outer)
        - node2: "问题2：\n回答2" (outer)
        - node3: "问题3：\n    子问题1：\n     子问题回答1：" (inner with indentation)

        Args:
            outer_qa_context: List of dicts with "query" and "answer" keys for outer context
            inner_qa_context: List of dicts with "query" and "answer" keys for inner context

        Returns:
            List of NodeWithScore objects
        """
        nodes = []
        from llama_index.core.schema import TextNode

        # Add outer Q&A context nodes
        # Format: "问题1：\n回答1"
        for qa_pair in outer_qa_context:
            query = qa_pair.get("query", "")
            answer = qa_pair.get("answer", "")
            # Outer format: "问题：[query]\n回答：[answer]"
            text = f"问题：{query}\n回答：{answer}"

            node = TextNode(
                text=text,
                metadata={
                    "source": "outer_previous_qa_context",
                    "type": "qa_context",
                    "is_inner": False,
                },
            )
            nodes.append(NodeWithScore(node=node, score=1.0))

        # Add inner Q&A context nodes with indentation
        # Format: "问题3：\n    子问题1：\n     子问题回答1："
        for qa_pair in inner_qa_context:
            query = qa_pair.get("query", "")
            answer = qa_pair.get("answer", "")
            # Inner format: "问题：[query]\n    回答：[answer]" (answer is indented)
            text = f"问题：{query}\n    回答：{answer}"

            node = TextNode(
                text=text,
                metadata={
                    "source": "inner_previous_qa_context",
                    "type": "qa_context",
                    "is_inner": True,
                },
            )
            nodes.append(NodeWithScore(node=node, score=1.0))

        return nodes

    def _search_results_to_nodes(
        self, search_results: Dict[str, List[Dict[str, any]]]
    ) -> List[NodeWithScore]:
        """Convert search results to nodes."""
        nodes = []
        logger.debug("开始转换搜索结果为节点 - 关键词数: %d", len(search_results))

        for keyword, results in search_results.items():
            logger.debug("处理关键词: %s, 结果数: %d", keyword, len(results))
            for result in results:
                from llama_index.core.schema import TextNode

                node = TextNode(
                    text=result["content"],
                    metadata={
                        "file_path": result["file_path"],
                        "source": result["file_path"],
                        "type": "markdown",
                        "start_line": result["start_line"],
                        "end_line": result["end_line"],
                        "keyword": keyword,
                    },
                )
                # TODO score 0.8?
                nodes.append(NodeWithScore(node=node, score=0.8))

        logger.debug("搜索结果转换完成 - 总节点数: %d", len(nodes))
        return nodes

    def _merge_nodes(
        self,
        existing_nodes: List[NodeWithScore],
        new_nodes: List[NodeWithScore],
        third_nodes: List[NodeWithScore] = None,
    ) -> List[NodeWithScore]:
        """Merge nodes with strict overlap checking.

        Merge logic:
        1. Extract file_name from file_path
        2. Group nodes by file_name
        3. For nodes with same file_name, check if start_line and end_line overlap
        4. If overlap, verify that the overlapping lines have duplicate content
        5. If content is duplicate, merge the nodes (keep wider range)
        """
        if third_nodes is None:
            third_nodes = []

        logger.debug(
            "开始合并节点 - 已有节点数: %d, 新节点数: %d, 第三节点数: %d",
            len(existing_nodes),
            len(new_nodes),
            len(third_nodes),
        )

        # 合并所有节点以便统一处理
        all_nodes = list(existing_nodes) + list(new_nodes) + list(third_nodes)

        if not all_nodes:
            return []

        # 按文件名分组，同时收集没有文件路径的节点
        file_name_groups: Dict[str, List[NodeWithScore]] = {}
        nodes_without_path = []

        for node in all_nodes:
            path = node.node.metadata.get("file_path") or node.node.metadata.get(
                "source"
            )
            if not path:
                # 如果没有文件路径，单独收集（如QA上下文节点）
                nodes_without_path.append(node)
                continue

            file_name = os.path.basename(path)
            if file_name not in file_name_groups:
                file_name_groups[file_name] = []
            file_name_groups[file_name].append(node)

        merged_results = []
        skipped_count = 0

        # 对每个文件名组进行处理
        for file_name, nodes in file_name_groups.items():
            if len(nodes) == 1:
                # 只有一个节点，直接添加
                merged_results.append(nodes[0])
                continue

            # 按 start_line 排序
            nodes.sort(key=lambda n: n.node.metadata.get("start_line") or 0)

            # 合并重叠的结果
            merged = []
            current = nodes[0]

            for next_node in nodes[1:]:
                current_path = current.node.metadata.get(
                    "file_path"
                ) or current.node.metadata.get("source")
                current_start = current.node.metadata.get("start_line")
                current_end = current.node.metadata.get("end_line")
                next_path = next_node.node.metadata.get(
                    "file_path"
                ) or next_node.node.metadata.get("source")
                next_start = next_node.node.metadata.get("start_line")
                next_end = next_node.node.metadata.get("end_line")

                # 检查是否有有效的行号
                if (
                    current_start is None
                    or current_end is None
                    or next_start is None
                    or next_end is None
                ):
                    # 行号无效，都保留
                    merged.append(current)
                    current = next_node
                    continue

                # 检查是否完全相同（相同的路径、起始行和结束行）
                if (
                    current_path == next_path
                    and current_start == next_start
                    and current_end == next_end
                ):
                    # 完全相同，跳过下一个节点，保留当前节点（保留较高的分数）
                    if (
                        hasattr(next_node, "score")
                        and next_node.score
                        and hasattr(current, "score")
                        and current.score
                        and next_node.score > current.score
                    ):
                        current = next_node
                    skipped_count += 1
                    continue

                # 检查是否重叠：当前结果的结束行 >= 下一个结果的开始行
                if current_end >= next_start:
                    # 有重叠，检查重叠行的内容是否重复
                    overlap_start = max(current_start, next_start)
                    overlap_end = min(current_end, next_end)

                    # 获取重叠行的内容
                    current_overlap_content = self._get_overlap_content(
                        current_path, overlap_start, overlap_end, current
                    )
                    next_overlap_content = self._get_overlap_content(
                        next_path, overlap_start, overlap_end, next_node
                    )

                    # 检查重叠内容是否相同
                    if current_overlap_content and next_overlap_content:
                        # 去除首尾空白后比较
                        if (
                            current_overlap_content.strip()
                            == next_overlap_content.strip()
                        ):
                            # 内容重复，合并：取最小 start_line 和最大 end_line
                            merged_start = min(current_start, next_start)
                            merged_end = max(current_end, next_end)
                            print(
                                "success merge",
                                current_path,
                                current_start,
                                current_end,
                                next_start,
                                next_end,
                                merged_start,
                                merged_end,
                            )
                            # 重新获取合并后的内容
                            merged_content = (
                                self.local_file_search.get_file_content_by_lines(
                                    current_path, merged_start, merged_end
                                )
                            )
                            if merged_content:
                                from llama_index.core.schema import TextNode

                                # 创建合并后的节点，保留较高的分数
                                merged_score = max(
                                    current.score
                                    if hasattr(current, "score") and current.score
                                    else 0,
                                    next_node.score
                                    if hasattr(next_node, "score") and next_node.score
                                    else 0,
                                )

                                merged_node = TextNode(
                                    text=merged_content,
                                    metadata={
                                        "file_path": current_path,
                                        "source": current_path,
                                        "type": current.node.metadata.get(
                                            "type", "markdown"
                                        ),
                                        "start_line": merged_start,
                                        "end_line": merged_end,
                                    },
                                )
                                current = NodeWithScore(
                                    node=merged_node, score=merged_score
                                )
                                skipped_count += 1
                                continue

                    # 重叠但内容不同，都保留
                    merged.append(current)
                    current = next_node
                else:
                    # 没有重叠，保存当前结果，开始新的合并
                    merged.append(current)
                    current = next_node

            # 添加最后一个结果
            merged.append(current)
            merged_results.extend(merged)

        # 添加没有文件路径的节点（如QA上下文节点）
        merged_results.extend(nodes_without_path)

        logger.debug(
            "节点合并完成 - 合并后节点数: %d, 跳过重复节点数: %d",
            len(merged_results),
            skipped_count,
        )
        return merged_results

    def _get_overlap_content(
        self, file_path: str, overlap_start: int, overlap_end: int, node: NodeWithScore
    ) -> Optional[str]:
        """获取重叠行的内容。

        Args:
            file_path: 文件路径
            overlap_start: 重叠起始行
            overlap_end: 重叠结束行
            node: 节点对象

        Returns:
            重叠行的内容，如果获取失败返回None
        """
        try:
            # 首先尝试从文件读取
            content = self.local_file_search.get_file_content_by_lines(
                file_path, overlap_start, overlap_end
            )
            if content:
                return content

            # 如果从文件读取失败，尝试从节点内容中提取
            node_content = (
                node.node.get_content()
                if hasattr(node.node, "get_content")
                else node.node.text
            )
            if not node_content:
                return None

            node_start = node.node.metadata.get("start_line")
            node_end = node.node.metadata.get("end_line")

            if node_start is None or node_end is None:
                return None

            # 计算重叠部分在节点内容中的位置
            node_lines = node_content.split("\n")
            node_line_count = len(node_lines)

            # 计算重叠部分在节点中的相对行号
            overlap_in_node_start = max(0, overlap_start - node_start)
            overlap_in_node_end = min(node_line_count, overlap_end - node_start + 1)

            if overlap_in_node_start < overlap_in_node_end:
                overlap_lines = node_lines[overlap_in_node_start:overlap_in_node_end]
                return "\n".join(overlap_lines)

            return None
        except Exception as e:
            logger.debug("获取重叠内容失败: %s", e)
            return None

    def _extract_keywords_from_query(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract keywords from query using KeywordExtractorWrapper.

        Args:
            query: User query string

        Returns:
            Tuple of (filtered_keywords, keywords_other) where keywords_other are the filtered out keywords
        """
        logger.debug("开始提取关键词 - 查询: %s", query)
        try:
            # Use KeywordExtractorWrapper to extract keywords (TF-IDF priority)
            keywords, keywords_other = self.keyword_extractor.extract_keywords(query)
            logger.debug(
                "使用KeywordExtractorWrapper提取关键词完成 - 关键词数: %d, 过滤关键词数: %d",
                len(keywords),
                len(keywords_other),
            )
            return keywords, keywords_other
        except (ValueError, AttributeError) as e:
            # Fallback to simple extraction if model not trained or error occurs
            logger.debug(
                "KeywordExtractorWrapper提取失败，使用简单提取方法 - 错误: %s", str(e)
            )
            keywords = []
            words = re.split(r"[\s,，。、；;：:]+", query)
            for word in words:
                word = word.strip()
                if len(word) > 1:
                    keywords.append(word)
            keywords = keywords[:10]
            logger.debug("简单提取关键词完成 - 关键词数: %d", len(keywords))
            return keywords, []

    def invoke(
        self, query: str, query_type: QueryType = QueryType.AUTO, **kwargs
    ) -> str:
        """Invoke the workflow with a query.

        Args:
            query: User query string
            query_type: Query type (AUTO, FAQ, DOCS)
            **kwargs: Additional arguments, may include context dict from WorkflowServer

        Returns:
            Final answer string
        """
        if debug and debug_logger:
            debug_logger.info(
                f"\n{'#' * 80}\n"
                f"工作流开始执行\n"
                f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                f"查询: {query}\n"
                f"查询类型: {query_type}\n"
                f"{'#' * 80}\n"
            )
            for handler in debug_logger.handlers:
                if isinstance(handler, FileHandler):
                    handler.flush()

        # 从kwargs中读取is_inner_sub_query_workflow
        # WorkflowServer可能通过context参数传递，也可能直接通过kwargs传递
        # 如果kwargs中有context（dict），尝试从context中读取is_inner_sub_query_workflow
        is_inner_sub_query_workflow = kwargs.get("is_inner_sub_query_workflow", False)
        if not is_inner_sub_query_workflow and "context" in kwargs:
            context = kwargs.get("context")
            if isinstance(context, dict):
                is_inner_sub_query_workflow = context.get(
                    "is_inner_sub_query_workflow", False
                )
                # 如果context中有store（来自Context.to_dict()），尝试从store中读取
                if not is_inner_sub_query_workflow and "store" in context:
                    store = context.get("store", {})
                    if isinstance(store, dict):
                        is_inner_sub_query_workflow = store.get(
                            "is_inner_sub_query_workflow", False
                        )

        initial_state: AgentState = {
            "query": query,
            "query_type": query_type,
            "can_direct_answer": False,
            "need_web_search": False,
            "need_rag_search": False,
            "web_or_rag_result_cached": False,
            "nodes": [],
            "keywords": [],
            "analysis": None,
            "iteration": 0,
            "no_relevant_result": None,
            "current_sub_query": None,
            "inner_previous_qa_context": [],
            "outer_previous_qa_context": [],
            "is_inner_sub_query_workflow": is_inner_sub_query_workflow,
            "is_outer_sub_query_workflow": False,
            "result": None,
            "middle_results": [],
        }

        result = self.graph.invoke(initial_state)

        if debug and debug_logger:
            result_text = result.get("result", "抱歉，无法生成答案。")
            result_preview = (
                result_text[:200] + "..." if len(result_text) > 200 else result_text
            )
            debug_logger.info(
                f"\n{'#' * 80}\n"
                f"工作流执行完成\n"
                f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                f"结果预览: {result_preview}\n"
                f"结果长度: {len(result_text)} 字符\n"
                f"{'#' * 80}\n"
            )
            for handler in debug_logger.handlers:
                if isinstance(handler, FileHandler):
                    handler.flush()

        return result.get("result", "抱歉，无法生成答案。")
