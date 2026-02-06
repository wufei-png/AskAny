"""Minimal implementation of LangChain agent using create_agent.

This agent uses tools for RAG retrieval, web search, and local file search.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import textwrap as tw

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from askany.workflow.workflow_langgraph import AgentState
from askany.rag.router import QueryRouter, QueryType
from openai import APIStatusError
from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    FilesystemFileSearchMiddleware,
)
from langgraph.cache.memory import InMemoryCache
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from llama_index.core.schema import NodeWithScore

from askany.config import settings
from askany.ingest import VectorStoreManager
from askany.rag import create_query_router
from askany.prompts.prompt_manager import get_prompts
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from askany.workflow.WebSearchTool import WebSearchTool
from askany.workflow.FinalSummaryLlm_langchain import (
    extract_docs_references,
    format_docs_references,
)
from logging import getLogger
import logging
import sys
from tool.query_test import send_query

logger = logging.getLogger(__name__)
logger.setLevel(
    logging.DEBUG
)  # Set logger level, otherwise default is WARNING, INFO won't output / 设置 logger 的级别，否则默认是 WARNING，INFO 不会输出
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from pydantic import BaseModel, Field
from typing import Optional


class FinalSummaryResponse(BaseModel):
    """Final output structure / 最终输出的结构"""

    summary_answer: str = Field(
        description="Based on all provided reference content, if the content is relevant and complete, answer the user's question in natural, fluent paragraphs. If the context is insufficient to answer the question, return 'Reference content is insufficient to answer the question'. / 根据提供的所有参考内容，如果内容相关且完整，则以自然、流畅的段落形式，完整回答用户的问题。如果上下文不足以回答问题，请返回'参考内容不足以回答问题'。",
    )

    references: List[str] = Field(
        description="Reference content list. If reference is a web link, use the web link directly. If it's a local file, use the file path. If there's a line number, add it like: README.md (line: 177-181) / 参考内容列表, 参考内容如果为网页链接，直接用网页链接，如果是本地文件，用文件路径，如果有行号加上行号如：README.md (行号: 177-181)",
        default=[],
    )


# Initialize components (similar to main.py)
def initialize_components():
    """Initialize LLM, embedding model, vector store, and router."""
    from askany.main import (
        SentenceTransformerEmbedding,
        get_device,
        initialize_llm,
    )

    # Initialize LLM and embedding model
    llm, embed_model = initialize_llm()
    device = get_device()

    # Initialize vector store manager
    vector_store_manager = VectorStoreManager(embed_model, llm=llm)

    # Try to initialize separate indexes
    try:
        vector_store_manager.initialize_faq_index()
        vector_store_manager.initialize_docs_index()
        # logger.info("Using separate indexes for FAQ and docs")
    except Exception as e:
        # Fallback to legacy single index
        # logger.info(f"Separate indexes not available, using legacy index: {e}")
        vector_store_manager.initialize()

    # Create router
    # logger.info("Creating query router...")
    try:
        router = create_query_router(vector_store_manager, llm, embed_model, device)
        # logger.info("Query router created successfully.")
    except Exception as e:
        logger.error(f"Failed to create query router: {e}", exc_info=True)
        raise

    # Initialize web search tool
    # logger.info("Initializing web search tool...")
    try:
        web_search_tool = WebSearchTool()
        # logger.info("Web search tool initialized successfully.")
    except Exception as e:
        # logger.info(f"WebSearchTool not available: {e}")
        web_search_tool = None

    # Initialize local file search tool
    # logger.info("Initializing local file search tool...")
    base_path = (
        settings.local_file_search_dir if settings.local_file_search_dir else None
    )
    local_file_search = LocalFileSearchTool(base_path=base_path)
    # logger.info("Local file search tool initialized successfully.")

    # Initialize keyword extractor
    from askany.ingest.keyword_extract_wrapper import KeywordExtractorWrapper

    try:
        keyword_extractor = KeywordExtractorWrapper(
            priority=settings.keyword_extractor_priority
        )
        # logger.info("Keyword extractor initialized successfully.")
    except (ValueError, AttributeError) as e:
        logger.debug(
            "KeywordExtractorWrapper初始化失败，将在使用时使用简单提取方法 - 错误: %s",
            str(e),
        )
        keyword_extractor = None

    return llm, router, web_search_tool, local_file_search, keyword_extractor


# Create LangChain tools
def create_rag_tool(
    router: QueryRouter, local_file_search: LocalFileSearchTool, keyword_extractor=None
):
    """Create RAG retrieval tool.

    Args:
        router: QueryRouter instance for RAG retrieval
        local_file_search: LocalFileSearchTool instance for keyword-based search
        keyword_extractor: Optional shared KeywordExtractorWrapper instance. If None, will initialize new one.
    """
    # Use shared keyword_extractor if provided, otherwise initialize new one
    if keyword_extractor is None:
        from askany.ingest.keyword_extract_wrapper import KeywordExtractorWrapper

        try:
            keyword_extractor = KeywordExtractorWrapper(
                priority=settings.keyword_extractor_priority
            )
        except (ValueError, AttributeError) as e:
            logger.debug(
                "KeywordExtractorWrapper初始化失败，将在使用时使用简单提取方法 - 错误: %s",
                str(e),
            )
            keyword_extractor = None

    prompts = get_prompts()

    @tool(description=prompts.rag_search_description)
    def rag_search(query: str) -> str:
        """RAG search tool."""
        from askany.rag.query_parser import parse_query_filters
        from askany.rag.router import QueryType
        from llama_index.core.schema import TextNode
        import os
        import re

        query_type = "auto"  # 写死auto
        # Parse query filters
        cleaned_query, metadata_filters = parse_query_filters(query)

        # Convert query_type string to QueryType enum
        if query_type.lower() == "faq":
            query_type_enum = QueryType.FAQ
        elif query_type.lower() == "docs":
            query_type_enum = QueryType.DOCS
        else:
            query_type_enum = QueryType.AUTO

        # Retrieve nodes from RAG
        if query_type_enum == QueryType.FAQ and router.faq_query_engine:
            nodes = router.faq_query_engine.retrieve(cleaned_query, metadata_filters)
        elif query_type_enum == QueryType.DOCS:
            nodes = router.docs_query_engine.retrieve(cleaned_query, metadata_filters)
        else:
            # AUTO mode: try FAQ first, then docs
            if router.faq_query_engine:
                nodes, top_score = router.faq_query_engine.retrieve_with_scores(
                    cleaned_query, metadata_filters
                )
                if top_score < settings.faq_score_threshold:
                    nodes = router.docs_query_engine.retrieve(
                        cleaned_query, metadata_filters
                    )
            else:
                nodes = router.docs_query_engine.retrieve(
                    cleaned_query, metadata_filters
                )

        # Extract keywords and search local files (synchronized with workflow_langgraph.py:755-789)
        try:
            # Use the pre-initialized keyword_extractor
            if keyword_extractor is not None:
                keywords, keywords_other = keyword_extractor.extract_keywords(
                    cleaned_query
                )
                logger.debug(
                    "关键词提取完成 - 提取到 %d 个关键词: %s, 过滤关键词数: %d",
                    len(keywords),
                    keywords,
                    len(keywords_other),
                )
            else:
                raise ValueError("keyword_extractor not initialized")
        except (ValueError, AttributeError) as e:
            # Fallback to simple extraction if model not trained or error occurs
            logger.debug(
                "KeywordExtractorWrapper提取失败，使用简单提取方法 - 错误: %s", str(e)
            )
            keywords = []
            words = re.split(r"[\s,，。、；;：:]+", cleaned_query)
            for word in words:
                word = word.strip()
                if len(word) > 1:
                    keywords.append(word)
            keywords = keywords[:10]
            logger.debug("简单提取关键词完成 - 关键词数: %d", len(keywords))

        # Search using extracted keywords
        keyword_nodes = []
        if keywords:
            keyword_nodes_tmp = local_file_search.search_keyword_using_binary_algorithm(
                keywords
            )
            keyword_nodes = _search_results_to_nodes(
                keyword_nodes_tmp, local_file_search
            )
            logger.debug("关键词搜索完成 - 节点数: %d", len(keyword_nodes))
        print(f"keyword_nodes: {keyword_nodes}")
        # Merge keyword search results with RAG results
        if keyword_nodes and len(keyword_nodes) > 0:
            nodes = _merge_nodes(nodes, keyword_nodes, local_file_search)
            logger.debug("合并关键词搜索结果后节点数: %d", len(nodes))

        # Format nodes as string
        if not nodes:
            return (
                "未找到相关信息。\n"
                "提示：如果RAG搜索未找到结果，建议尝试使用search_local_files_by_keywords工具进行本地文件搜索，"
                "因为本地文件搜索使用精确关键词匹配，可能找到RAG搜索遗漏的内容。"
            )

        result_parts = []
        for i, node in enumerate(nodes, 1):
            content = (
                node.node.get_content()
                if hasattr(node.node, "get_content")
                else node.node.text
            )
            file_path = (
                node.node.metadata.get("file_path")
                or node.node.metadata.get("source")
                or "unknown"
            )
            result_parts.append(
                f"[Result {i}]\nSource: {file_path}\nContent: {content}\n"
            )

        return "\n".join(result_parts)

    return rag_search


def _search_results_to_nodes(
    search_results: Dict[str, List[Dict[str, Any]]],
    local_file_search: LocalFileSearchTool,
) -> List[NodeWithScore]:
    """Convert search results to nodes (synchronized with workflow_langgraph.py:1773-1800)."""
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
    existing_nodes: List[NodeWithScore],
    new_nodes: List[NodeWithScore],
    local_file_search: LocalFileSearchTool,
) -> List[NodeWithScore]:
    """Merge nodes with strict overlap checking (simplified version of workflow_langgraph.py:1802-1951).

    Merge logic:
    1. Extract file_name from file_path
    2. Group nodes by file_name
    3. For nodes with same file_name, check if start_line and end_line overlap
    4. If overlap, verify that the overlapping lines have duplicate content
    5. If content is duplicate, merge the nodes (keep wider range)
    """
    logger.debug(
        "开始合并节点 - 已有节点数: %d, 新节点数: %d",
        len(existing_nodes),
        len(new_nodes),
    )

    # 合并所有节点以便统一处理
    all_nodes = list(existing_nodes) + list(new_nodes)

    if not all_nodes:
        return []

    # 按文件名分组，同时收集没有文件路径的节点
    file_name_groups: Dict[str, List[NodeWithScore]] = {}
    nodes_without_path = []

    for node in all_nodes:
        path = node.node.metadata.get("file_path") or node.node.metadata.get("source")
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
                current_overlap_content = _get_overlap_content(
                    current_path, overlap_start, overlap_end, current, local_file_search
                )
                next_overlap_content = _get_overlap_content(
                    next_path, overlap_start, overlap_end, next_node, local_file_search
                )

                # 检查重叠内容是否相同
                if current_overlap_content and next_overlap_content:
                    # 去除首尾空白后比较
                    if current_overlap_content.strip() == next_overlap_content.strip():
                        # 内容重复，合并：取最小 start_line 和最大 end_line
                        merged_start = min(current_start, next_start)
                        merged_end = max(current_end, next_end)
                        logger.debug(
                            "成功合并节点: %s, %d-%d 和 %d-%d -> %d-%d",
                            current_path,
                            current_start,
                            current_end,
                            next_start,
                            next_end,
                            merged_start,
                            merged_end,
                        )
                        # 重新获取合并后的内容
                        merged_content = local_file_search.get_file_content_by_lines(
                            current_path, merged_start, merged_end
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

    # 添加没有文件路径的节点
    merged_results.extend(nodes_without_path)

    logger.debug(
        "节点合并完成 - 合并后节点数: %d, 跳过重复节点数: %d",
        len(merged_results),
        skipped_count,
    )

    return merged_results


def _get_overlap_content(
    file_path: str,
    start_line: int,
    end_line: int,
    node: NodeWithScore,
    local_file_search: LocalFileSearchTool,
) -> Optional[str]:
    """Get content for overlap range from file or node (simplified version)."""
    try:
        # Try to get from file first
        content = local_file_search.get_file_content_by_lines(
            file_path, start_line, end_line
        )
        if content:
            return content
    except Exception:
        pass

    # Fallback: extract from node content if file read fails
    node_content = (
        node.node.get_content() if hasattr(node.node, "get_content") else node.node.text
    )
    if node_content:
        # Simple extraction: split by lines and get overlap range
        lines = node_content.split("\n")
        node_start = node.node.metadata.get("start_line", 1)
        node_end = node.node.metadata.get("end_line", len(lines))

        # Calculate relative positions
        if start_line >= node_start and end_line <= node_end:
            rel_start = start_line - node_start
            rel_end = end_line - node_start
            if 0 <= rel_start < len(lines) and 0 <= rel_end < len(lines):
                return "\n".join(lines[rel_start : rel_end + 1])

    return None


def create_web_search_tool(web_search_tool_instance: WebSearchTool):
    """Create web search tool."""
    prompts = get_prompts()

    @tool(description=prompts.web_search_description)
    def web_search(query: str) -> str:
        """Web search tool."""
        if web_search_tool_instance is None:
            return "Web search is not available."

        nodes = web_search_tool_instance.search(query)
        if not nodes:
            return "No web search results found."

        # Format nodes as string
        result_parts = []
        for i, node in enumerate(nodes, 1):
            content = (
                node.node.get_content()
                if hasattr(node.node, "get_content")
                else node.node.text
            )
            source = (
                node.node.metadata.get("source")
                or node.node.metadata.get("url")
                or "unknown"
            )
            result_parts.append(
                f"[Web Result {i}]\nSource: {source}\nContent: {content}\n"
            )

        return "\n".join(result_parts)

    return web_search


def create_local_file_search_tools(local_file_search_instance: LocalFileSearchTool):
    """Create local file search tools."""
    prompts = get_prompts()

    @tool(description=prompts.local_file_search_description)
    def search_local_files_by_keywords(keywords: List[str]) -> str:
        """Local file search tool."""
        # Use keywords directly as a list
        keyword_list = [k.strip() for k in keywords if k and k.strip()]

        if not keyword_list:
            return "No keywords provided."

        # Search using LocalFileSearchTool
        results = local_file_search_instance.search_by_keywords(keyword_list)

        if not results:
            return "No results found for the given keywords."

        # Format results as string
        result_parts = []
        for keyword, matches in results.items():
            result_parts.append(f"\n[Keyword: {keyword}]")
            for i, match in enumerate(matches, 1):
                file_path = match.get("file_path", "unknown")
                start_line = match.get("start_line", "?")
                end_line = match.get("end_line", "?")
                content = match.get("content", "")[:500]  # Limit content length
                result_parts.append(
                    f"  Match {i}:\n"
                    f"    File: {file_path}\n"
                    f"    Lines: {start_line}-{end_line}\n"
                    f"    Content: {content}...\n"
                )

        return "\n".join(result_parts)

    @tool(description=prompts.get_file_content_description)
    def get_file_content(file_path: str, start_line: int, end_line: int) -> str:
        """Get file content tool."""
        content = local_file_search_instance.get_file_content_by_lines(
            file_path, start_line, end_line
        )

        if content is None:
            return f"Could not retrieve content from {file_path} (lines {start_line}-{end_line})."

        return f"[File: {file_path}, Lines: {start_line}-{end_line}]\n{content}"

    return search_local_files_by_keywords, get_file_content


# Create system prompt middleware to guide agent behavior
@dynamic_prompt
def agent_system_prompt(request: ModelRequest) -> str:
    # 生成系统提示词，指导agent的工具使用策略。
    """Generate system prompt to guide agent's tool usage strategy."""
    prompts = get_prompts()
    #     return """You are an intelligent assistant that helps users search and find information.
    return prompts.agent_system_prompt


# 1. **Determine Web vs RAG (web_or_rag_check)**:
#    - Analyze the question type to determine what information source is needed
#    - If the question involves **real-time information, latest news, current events**, use web_search
#    - If the question involves **local documents, FAQ, technical documentation, configuration guides**, use RAG tools
#    - Key decision criteria:
#      * Timeliness: Need latest information → web_search
#      * Domain-specific: Company internal docs, technical specs → RAG (rag_search + search_local_files_by_keywords)
#      * General: Common knowledge, public information → web_search

# 2. **RAG Retrieval Strategy (rag_retrieval)**:
#    When RAG is needed, try in the following order:
#    a. **rag_search**: Use vector similarity search in documents and FAQ
#    b. **search_local_files_by_keywords**: If rag_search finds no results, use exact keyword matching
#       - Extract keywords from the question (nouns, technical terms, technical names)
#       - Keywords should be meaningful terms, avoid stop words
#    c. **glob_search**: If you need to find specific file names or path patterns
#    d. **grep_search**: If you need to use regex for exact text matching
#    e. **get_file_content**: After finding relevant files, retrieve content

# Core Principles:
# ✓ Intelligently select tools based on question nature: use what you need
# ✓ Web search: Real-time info, public knowledge, latest updates
# ✓ RAG search: Local docs, FAQ, technical specs, internal knowledge
# ✓ When one tool fails, try other related tools
# ✓ After finding information, use get_file_content to get complete content before answering

# Examples:
# - "What is the latest Kubernetes version?" → web_search (real-time info)
# - "How to configure company's Docker environment?" → rag_search + search_local_files_by_keywords (local docs)
# - "What's the difference between Python list and tuple?" → Direct answer or web_search (general knowledge)
# - "What is the viper system deployment process?" → rag_search + search_local_files_by_keywords (internal docs)"""


def should_retry_model(exc: Exception) -> bool:
    # Retry on 5xx errors (server/transient)
    if isinstance(exc, APIStatusError):
        if 500 <= exc.status_code < 600:
            return True
        # Retry on 400 when output is truncated/invalid JSON (model hit max_tokens or malformed tool_calls)
        if exc.status_code == 400:
            msg = (getattr(exc, "message", None) or str(exc) or "").lower()
            if "json_invalid" in msg or "eof while parsing" in msg:
                return True
    # Some layers wrap the error; check message for truncated JSON
    msg = str(exc).lower()
    if "json_invalid" in msg or "eof while parsing" in msg:
        return True
    return False


def create_agent_with_tools(
    router=None,
    web_search_tool=None,
    local_file_search=None,
    llm_instance=None,
    keyword_extractor=None,
):
    """Create LangChain agent with all tools.

    Args:
        router: Optional shared QueryRouter instance. If None, will initialize new one.
        web_search_tool: Optional shared WebSearchTool instance. If None, will initialize new one.
        local_file_search: Optional shared LocalFileSearchTool instance. If None, will initialize new one.
        llm_instance: Optional shared LLM instance. If None, will initialize new one.
        keyword_extractor: Optional shared KeywordExtractorWrapper instance. If None, will initialize new one.

    Returns:
        Created agent instance
    """
    # Initialize components if not provided (for backward compatibility)
    if (
        router is None
        or web_search_tool is None
        or local_file_search is None
        or llm_instance is None
        or keyword_extractor is None
    ):
        # logger.info("Initializing components...")
        _llm, _router, _web_search_tool, _local_file_search, _keyword_extractor = (
            initialize_components()
        )
        if router is None:
            router = _router
        if web_search_tool is None:
            web_search_tool = _web_search_tool
        if local_file_search is None:
            local_file_search = _local_file_search
        if llm_instance is None:
            llm_instance = _llm
        if keyword_extractor is None:
            keyword_extractor = _keyword_extractor
        # logger.info("Components initialized.")

    # Create LangChain LLM
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else ""
    # Use at least 8192 max_tokens to reduce "Invalid JSON: EOF while parsing" when
    # model outputs long tool_calls or structured response that gets truncated
    chat_llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=api_key,
        base_url=api_base,
        temperature=settings.temperature,
        max_tokens=max(settings.output_tokens, 8192),
        timeout=settings.llm_timeout,
    )

    # Create tools
    tools = []

    # RAG tool (pass local_file_search and keyword_extractor for keyword-based search enhancement)
    rag_tool = create_rag_tool(router, local_file_search, keyword_extractor)
    tools.append(rag_tool)

    # Web search tool
    if web_search_tool is not None:
        web_tool = create_web_search_tool(web_search_tool)
        tools.append(web_tool)

    # Local file search tools
    search_tool, get_content_tool = create_local_file_search_tools(local_file_search)
    tools.append(search_tool)
    tools.append(get_content_tool)

    # Create agent with system prompt middleware and other middleware
    # logger.info(f"Creating agent with {len(tools)} tools...")

    # Build middleware list
    # Note: Middleware order matters - they execute in the order listed
    # Recommended order: System prompt -> Limits -> Retries -> Summarization -> Human-in-the-loop -> Utilities
    middleware_list = [
        # 1. System prompt middleware (custom) - Must be first to set behavior
        agent_system_prompt,
        # 2. Model call limit: Limit the number of model calls to prevent excessive costs
        # Should be early to catch runaway agents quickly
        ModelCallLimitMiddleware(
            thread_limit=50,  # Maximum model calls across all runs in a thread
            run_limit=20,  # Maximum model calls per single invocation
            exit_behavior="end",  # Graceful termination when limit reached
        ),
        # 3. Tool call limit: Control tool execution by limiting call counts
        # Global limit - applies to all tools
        ToolCallLimitMiddleware(
            thread_limit=100,  # Maximum tool calls across all runs in a thread
            run_limit=30,  # Maximum tool calls per single invocation
        ),
        # Tool-specific limits for web_search (more restrictive)
        ToolCallLimitMiddleware(
            tool_name="web_search",
            thread_limit=10,  # Limit web searches per thread
            run_limit=5,  # Limit web searches per invocation
        ),
        # 4. Model retry: Automatically retry failed model calls with exponential backoff
        # Handles API rate limits, temporary failures, network issues
        ModelRetryMiddleware(
            max_retries=1,  # Maximum retry attempts after initial call (total: 4 attempts)
            backoff_factor=2.0,  # Exponential backoff multiplier (1s, 2s, 4s, ...)
            initial_delay=1.0,  # Initial delay in seconds
            max_delay=60.0,  # Maximum delay cap (won't exceed 60s)
            jitter=True,  # Add random jitter to avoid thundering herd problem
            retry_on=should_retry_model,
            on_failure="continue",  # Return error message instead of raising exception
        ),
        # 5. Tool retry: Automatically retry failed tool calls with exponential backoff
        # Handles transient tool failures (network errors, timeouts, etc.)
        ToolRetryMiddleware(
            max_retries=1,  # Maximum retry attempts after initial call (total: 4 attempts)
            backoff_factor=2.0,  # Exponential backoff multiplier (1s, 2s, 4s, ...)
            initial_delay=1.0,  # Initial delay in seconds
            max_delay=60.0,  # Maximum delay cap (won't exceed 60s)
            jitter=True,  # Add random jitter to avoid thundering herd problem
            on_failure="return_message",  # Return error message instead of raising exception
        ),
        # 6. Summarization: Automatically summarize conversation history when approaching token limits
        # Should be after retries to avoid summarizing during retry attempts
        SummarizationMiddleware(
            model=chat_llm,  # Use same model for summarization
            trigger=(
                "tokens",
                int(settings.llm_max_tokens * 0.9),
            ),  # Trigger when approaching 4w tokens (adjust based on model context window)
            # Alternative: trigger=[("tokens", 8000), ("messages", 50)] for multiple conditions
            keep=("messages", 20),  # Keep last 20 messages before summarization
        ),
        # 7. Human-in-the-loop: Pause execution for human approval of tool calls
        # Configure to require approval for sensitive tools (e.g., web_search, file writes)
        # Note: Requires checkpointer to be configured in create_agent for state persistence
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Example: require approval for web_search tool
                # Uncomment and configure as needed:
                # "web_search": {
                #     "allowed_decisions": ["approve", "edit", "reject"],
                # },
                # Currently disabled - set to empty dict to disable human-in-the-loop
                # To enable: add tool names and their allowed decision types
            }
        ),
        # 8. Filesystem file search: Provide Glob and Grep search tools over filesystem files
        # This automatically adds glob_search and grep_search tools to the agent
        # Should be last as it adds tools rather than modifying behavior
        # Note: This is a complement to LocalFileSearchTool - use for exact filename matching and regex search
        FilesystemFileSearchMiddleware(
            root_path=settings.local_file_search_dir
            or "data/markdown",  # Use same path as LocalFileSearchTool
            use_ripgrep=True,  # Use ripgrep for faster search (falls back to Python regex if unavailable)
            max_file_size_mb=10,  # Skip files larger than 10MB to avoid memory issues
        ),
    ]
    from langchain.agents.structured_output import ProviderStrategy

    # Create cache instance for graph execution caching
    # Note: InMemoryCache 是 LangGraph 的图执行缓存，用于缓存节点执行结果
    # 对于"相同问题直接返回答案"的优化，请使用 QuestionAnswerCache（在 invoke_with_retry 中自动使用）
    cache = InMemoryCache()
    agent = create_agent(
        model=chat_llm,
        system_prompt=agent_system_prompt,
        tools=tools,
        middleware=middleware_list,
        # debug=True,
        cache=cache,
        # context_schema=AgentState,
        response_format=FinalSummaryResponse,
    )

    return agent


def format_agent_response_with_references(response_content: str) -> str:
    """Format agent response with document references.

    Args:
        response_content: The agent's response content

    Returns:
        Formatted response with references appended
    """

    global _last_retrieved_nodes

    # If we have retrieved nodes, add references
    if _last_retrieved_nodes:
        references = extract_docs_references(_last_retrieved_nodes)
        formatted_refs = format_docs_references(references)

        if formatted_refs:
            return response_content + formatted_refs

    return response_content


def extract_all_tool_calls(result: dict) -> List[dict]:
    """Extract all tool_calls from all AIMessages in the result.

    Args:
        result: Agent invoke result dictionary

    Returns:
        List of tool call dictionaries with 'name', 'args', and 'type' keys
    """
    tool_calls_list = []

    if "messages" in result:
        messages = result["messages"]
        for message in messages:
            # Check if message has tool_calls attribute
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    # tool_call can be a dict or an object with attributes
                    if isinstance(tool_call, dict):
                        # Extract name, args, and type from dict
                        tool_call_dict = {
                            "name": tool_call.get("name", ""),
                            "args": tool_call.get("args", {}),
                            "type": tool_call.get("type", "tool_call"),
                        }
                        if tool_call_dict["name"] == "FinalSummaryResponse":
                            continue
                    else:
                        # Extract from object attributes
                        tool_call_dict = {
                            "name": getattr(tool_call, "name", ""),
                            "args": getattr(tool_call, "args", {}),
                            "type": getattr(tool_call, "type", "tool_call"),
                        }
                        if tool_call_dict["name"] == "FinalSummaryResponse":
                            continue
                    tool_calls_list.append(tool_call_dict)

    return tool_calls_list


def extract_and_format_response(result: dict) -> str:
    """Extract and format response from agent result.

    Handles both structured_response (FinalSummaryResponse) and regular messages.
    Formats according to query5 copy.log format.

    Args:
        result: Agent invoke result dictionary

    Returns:
        Formatted response string
    """
    # Handle string input (for test responses)
    if isinstance(result, str):
        return result

    # Extract structured response if available
    agent_response = ""

    if "structured_response" in result:
        structured_response = result["structured_response"]
        if isinstance(structured_response, FinalSummaryResponse):
            # Format response according to query5 copy.log format
            formatted_output = structured_response.summary_answer

            # Add references section if references exist
            if structured_response.references:
                formatted_output += "\n\n---\n**参考数据来源：**\n"
                for ref in structured_response.references:
                    formatted_output += f"- {ref}\n"

            agent_response = formatted_output
        else:
            agent_response = str(structured_response)

    # Fallback to original message extraction logic
    elif "messages" in result:
        messages = result["messages"]
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                agent_response = last_message.content
            else:
                agent_response = str(last_message)

    tool_calls_list = extract_all_tool_calls(result)
    if tool_calls_list:
        tool_calls_str = (
            f"\n\n工具调用：{json.dumps(tool_calls_list, ensure_ascii=False)}"
        )
        agent_response += tool_calls_str
    return agent_response


def invoke_with_retry(agent, messages_input: dict, max_retries: int = 2):
    """Invoke agent with retry on truncated/invalid JSON (json_invalid, EOF while parsing)."""
    for attempt in range(max_retries):
        try:
            return agent.invoke(messages_input)
        except Exception as e:
            err = str(e).lower()
            if ("json_invalid" in err or "eof while parsing" in err) and (
                attempt < max_retries - 1
            ):
                logger.warning(
                    "Model output truncated/invalid JSON, retrying (%s/%s): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                continue
            raise
    raise RuntimeError("invoke_with_retry: unreachable")


def write_result_to_file(
    question: str,
    answer: str,
    duration: float | None = None,
    result_file: str = "result.json",
):
    """Write question, answer and duration to result file as JSON array.

    Args:
        question: User question (问题)
        answer: Agent answer (回答)
        result_file: Path to result file (default: result.json)
        duration: Elapsed time in seconds (optional). Stored as 耗时; null if not provided.
    """
    result_path = Path(project_root) / result_file

    # Load existing array or start with empty list
    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
        if not isinstance(data, list):
            data = []
    else:
        data = []

    # Append new entry: 问题, 回答, 耗时
    entry = {
        "question": question,
        "answer": answer,
        "time": round(duration, 2) if duration is not None else None,
    }
    data.append(entry)

    # Write back
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def answer_test(query_list=None, multi_turn_conversations=None):
    """Main function to test the agent.

    Args:
        query_list: List of single questions to test
        multi_turn_conversations: List of multi-turn conversation flows
    """
    import time

    # Create agent
    agent = create_agent_with_tools()

    # Test query
    # logger.info("\n" + "=" * 80)
    # logger.info("Agent created successfully!")
    # logger.info("=" * 80)

    # Test single questions
    if query_list:
        # logger.info("\n" + "=" * 80)
        # logger.info("Starting Single Question Tests")
        # logger.info("=" * 80)

        for i, question in enumerate(query_list, 1):
            # logger.info(f"\n[Question {i}/{len(query_list)}]")
            # logger.info("=" * 80)
            # logger.info(f"Query: {question}")
            # logger.info("=" * 80)
            print("question: ", question)
            try:
                time_start = time.time()

                # create_agent
                result = invoke_with_retry(
                    agent, {"messages": [{"role": "user", "content": question}]}
                )
                time_end = time.time()

                # Extract and format response
                agent_response = extract_and_format_response(result)
                # logger.info(f"\nAgent Response:\n{agent_response}\n")
                tool_calls_list = extract_all_tool_calls(result)
                if tool_calls_list:
                    tool_calls_str = f"\n\n工具调用：{json.dumps(tool_calls_list, ensure_ascii=False)}"
                    agent_response += tool_calls_str

                # limit_agnet
                # base_url=f"http://127.0.0.1:{settings.api_port}"
                # agent_response=send_query(
                #     base_url,
                #     question,
                #     QueryType.AUTO,
                #     description="AUTO mode with Docs query (should route to Docs)",
                #     model=settings.openai_model,
                # )
                # time_end = time.time()

                # # Write to result file
                write_result_to_file(
                    question, agent_response, duration=time_end - time_start
                )

                logger.info(f"Time taken: {time_end - time_start:.2f} seconds")
                logger.info("=" * 80)

            except Exception as e:
                # logger.info(f"\n❌ Error processing question: {e}\n")
                import traceback

                traceback.print_exc()
                # logger.info("=" * 80)

    # Test multi-turn conversations
    if multi_turn_conversations:
        # logger.info("\n" + "=" * 80)
        # logger.info("Starting Multi-Turn Conversation Tests")
        # logger.info("=" * 80)

        for conv_idx, conversation in enumerate(multi_turn_conversations, 1):
            # logger.info(f"\n[Conversation {conv_idx}/{len(multi_turn_conversations)}]")
            # logger.info("=" * 80)
            # logger.info(f"Conversation Type: {conversation.get('type', 'Unknown')}")
            # logger.info("=" * 80)

            # Initialize conversation state
            messages = []

            for turn_idx, turn in enumerate(conversation.get("turns", []), 1):
                user_message = turn.get("user", "")
                # logger.info(f"\n--- Turn {turn_idx} ---")
                # logger.info(f"User: {user_message}")

                try:
                    # Add user message to conversation
                    messages.append({"role": "user", "content": user_message})

                    time_start = time.time()
                    result = invoke_with_retry(agent, {"messages": messages})
                    time_end = time.time()

                    # Extract and format response
                    agent_response = extract_and_format_response(result)
                    # logger.info(f"Agent: {agent_response}")
                    # logger.info(f"Time taken: {time_end - time_start:.2f} seconds")

                    # Extract all tool calls and append to response
                    tool_calls_list = extract_all_tool_calls(result)
                    if tool_calls_list:
                        tool_calls_str = f"\n\n工具调用：{json.dumps(tool_calls_list, ensure_ascii=False)}"
                        agent_response += tool_calls_str

                    # Write to result file
                    write_result_to_file(
                        user_message, agent_response, duration=time_end - time_start
                    )

                    # Update messages for next turn
                    # if "messages" in result:
                    #     messages = result["messages"]
                    messages.append({"role": "assistant", "content": agent_response})
                except Exception as e:
                    # logger.info(f"\n❌ Error in turn {turn_idx}: {e}\n")
                    import traceback

                    traceback.print_exc()
                    break

            # logger.info("=" * 80)

    # logger.info("\n" + "=" * 80)
    # logger.info("All Tests Completed!")
    # logger.info("=" * 80)


def test_agent_ui():
    """Main function to test the agent."""
    # Create agent
    agent = create_agent_with_tools()

    # Test query
    # logger.info("\n" + "=" * 80)
    # logger.info("Agent created successfully!")
    # logger.info("=" * 80)
    # logger.info("\nYou can now use the agent like this:")
    # logger.info('  result = agent.invoke({"messages": [{"role": "user", "content": "your question"}]})')
    # logger.info("\nExample:")
    # logger.info('  result = agent.invoke({"messages": [{"role": "user", "content": "What is the FAQ about?"}]})')
    # logger.info("=" * 80)

    # Interactive mode
    import time

    # logger.info("\nEntering interactive mode. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                # logger.info("Goodbye!")
                break

            if not user_input:
                continue

            time_start = time.time()
            result = invoke_with_retry(
                agent, {"messages": [{"role": "user", "content": user_input}]}
            )
            time_end = time.time()

            # Extract and format response
            print("result: ", result)
            agent_response = extract_and_format_response(result)
            # logger.info(f"\nAgent: {agent_response}\n")
            tool_calls_list = extract_all_tool_calls(result)
            if tool_calls_list:
                tool_calls_str = (
                    f"\n\n工具调用：{json.dumps(tool_calls_list, ensure_ascii=False)}"
                )
                agent_response += tool_calls_str
            # Write to result file
            write_result_to_file(
                user_input, agent_response, duration=time_end - time_start
            )

        except KeyboardInterrupt:
            # logger.info("\n\nGoodbye!")
            break
        except Exception as e:
            # logger.info(f"\nError: {e}\n")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    # Single question test list
    # 15 questions that can be answered via RAG (conceptual, process-oriented)
    # test_agent_ui()
    # exit()
    from question import all_questions
    from question import all_conversations

    # logger.info("=" * 80)
    print("=" * 80)
    # Run tests
    answer_test(
        query_list=all_questions,
        # query_list=[],
        multi_turn_conversations=all_conversations,
    )
