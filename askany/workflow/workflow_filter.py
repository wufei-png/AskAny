"""Workflow filter for pre-processing queries before sub-problem extraction.

This module provides a filter that attempts to answer queries directly or via web search
before falling back to sub-problem extraction.
"""

import logging
import re
from typing import Optional
from llama_index.core.schema import QueryBundle
from pathlib import Path
import sys
# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.workflow.firstStageRelevant_langchain import (
    DirectAnswerGenerator,
    WebOrRagAnswerGenerator,
)
from askany.workflow.FinalSummaryLlm_langchain import FinalAnswerGenerator
from askany.workflow.WebSearchTool import WebSearchTool
from askany.rerank import SafeReranker

logger = logging.getLogger(__name__)


class WorkflowFilterResult:
    """Result from workflow filter processing."""

    def __init__(
        self,
        have_result: bool,
        need_web_search: bool,
        need_rag_search: bool,
        result: Optional[str] = None,
    ):
        """Initialize WorkflowFilterResult.

        Args:
            have_result: Whether a result was generated
            result: The generated answer if have_result is True
        """
        self.have_result = have_result
        self.need_web_search = need_web_search
        self.need_rag_search = need_rag_search
        self.result = result


class WorkflowFilter:
    """Filter workflow that attempts to answer queries before sub-problem extraction.

    This filter implements a simplified workflow:
    1. Check if query can be answered directly
    2. If not, check if web search is needed (without RAG)
    3. If web search is needed and available, perform search and generate answer
    4. Otherwise, return have_result=False to trigger sub-problem extraction
    """

    def __init__(
        self,
        direct_answer_generator: DirectAnswerGenerator,
        web_or_rag_generator: WebOrRagAnswerGenerator,
        final_answer_generator: FinalAnswerGenerator,
        web_search_tool: Optional[WebSearchTool] = None,
        reranker: Optional[SafeReranker] = None,
    ):
        """Initialize WorkflowFilter.

        Args:
            direct_answer_generator: Generator for checking if query can be answered directly
            web_or_rag_generator: Generator for checking if web or RAG search is needed
            final_answer_generator: Generator for final answer generation
            web_search_tool: Optional web search tool
            reranker: Optional reranker for web search results
        """
        self.direct_answer_generator = direct_answer_generator
        self.web_or_rag_generator = web_or_rag_generator
        self.final_answer_generator = final_answer_generator
        self.web_search_tool = web_search_tool
        self.reranker = reranker

    def process(self, query: str) -> WorkflowFilterResult:
        """Process a query through the filter workflow.

        Args:
            query: User query string

        Returns:
            WorkflowFilterResult with have_result flag and optional result
        """
        logger.debug("工作流过滤器开始处理 - 查询: %s", query)

        # Step 0: Check if query contains valid URL (http:// or https://)
        # TODO 只要有url就只执行网络搜索? 这样可能不合理
        url_pattern = r'https?://[^\s]+'
        if re.search(url_pattern, query):
            logger.debug("检测到查询中包含URL链接，直接执行网络搜索")
            if self.web_search_tool is None:
                logger.warning("网络搜索工具不可用，无法搜索")
                return WorkflowFilterResult(have_result=False, need_web_search=True, need_rag_search=False)

            web_nodes = self.web_search_tool.search(query)
            logger.debug("网络搜索完成 - 结果数: %d", len(web_nodes))

            # Use reranker if available
            if web_nodes and self.reranker:
                query_bundle = QueryBundle(query)
                web_nodes = self.reranker.postprocess_nodes(web_nodes, query_bundle)
                logger.debug("Reranker重排序完成 - 节点数: %d", len(web_nodes))

            # Check if we have nodes before generating answer
            if not web_nodes:
                logger.warning("网络搜索未返回任何结果")
                return WorkflowFilterResult(have_result=False, need_web_search=True, need_rag_search=False)

            # Generate answer from web search results
            answer, reasoning = self.final_answer_generator.generate_final_answer(
                query, web_nodes
            )
            logger.debug("网络搜索答案生成完成")
            reasoning_str = reasoning if reasoning else ""
            result_text = answer + "\n\n" + reasoning_str
            return WorkflowFilterResult(have_result=True, need_web_search=True, need_rag_search=False, result=result_text)

        # Step 1: Check if query can be answered directly
        logger.debug("步骤1: 检查是否可以直接回答")
        direct_answer_result = self.direct_answer_generator.generate(query)
        logger.debug(
            "直接回答检查完成 - 可以直接回答: %s",
            direct_answer_result.can_direct_answer,
        )

        if direct_answer_result.can_direct_answer:
            logger.debug("问题可以直接回答，生成直接答案")
            answer, reasoning = self.final_answer_generator.generate_final_answer(
                query, []
            )
            reasoning_str = reasoning if reasoning else ""
            result_text = answer + "\n\n" + reasoning_str
            logger.debug("直接答案生成完成")
            return WorkflowFilterResult(have_result=True, need_web_search=False, need_rag_search=False, result=result_text)

        # Step 2: Check if web search or RAG is needed
        logger.debug("步骤2: 检查是否需要网络搜索或RAG检索")
        web_or_rag_result = self.web_or_rag_generator.generate(query)
        logger.debug(
            "网络/RAG检查完成 - 需要网络搜索: %s, 需要RAG检索: %s",
            web_or_rag_result.need_web_search,
            web_or_rag_result.need_rag_search,
        )

        need_web = web_or_rag_result.need_web_search
        need_rag = web_or_rag_result.need_rag_search

        # Step 3: If web search is needed and RAG is not needed, perform web search
        if need_web and not need_rag:
            logger.debug("仅需要网络搜索，执行网络搜索")
            if self.web_search_tool is None:
                logger.warning("网络搜索工具不可用，无法搜索")
                return WorkflowFilterResult(have_result=False, need_web_search=True, need_rag_search=False)

            web_nodes = self.web_search_tool.search(query)
            logger.debug("网络搜索完成 - 结果数: %d", len(web_nodes))

            # Use reranker if available
            if web_nodes and self.reranker:
                query_bundle = QueryBundle(query)
                web_nodes = self.reranker.postprocess_nodes(web_nodes, query_bundle)
                logger.debug("Reranker重排序完成 - 节点数: %d", len(web_nodes))

            # Check if we have nodes before generating answer
            if not web_nodes:
                logger.warning("网络搜索未返回任何结果")
                return WorkflowFilterResult(have_result=False, need_web_search=True, need_rag_search=False)

            # Generate answer from web search results
            answer, reasoning = self.final_answer_generator.generate_final_answer(
                query, web_nodes
            )
            logger.debug("网络搜索答案生成完成")
            reasoning_str = reasoning if reasoning else ""
            result_text = answer + "\n\n" + reasoning_str
            return WorkflowFilterResult(have_result=True, need_web_search=True, need_rag_search=False, result=result_text)

        # Step 4: If RAG is needed or both are false, return have_result=False
        # This will trigger sub-problem extraction in server.py
        logger.debug(
            "需要RAG检索或无法确定搜索方式，返回have_result=False以触发子问题提取"
        )
        return WorkflowFilterResult(have_result=False, need_web_search=need_web, need_rag_search=need_rag)

if __name__ == "__main__":
    from askany.workflow.firstStageRelevant_langchain import DirectAnswerGenerator, WebOrRagAnswerGenerator
    from askany.workflow.FinalSummaryLlm_langchain import FinalAnswerGenerator
    from askany.workflow.WebSearchTool import WebSearchTool
    from askany.rerank import SafeReranker

    direct_answer_generator = DirectAnswerGenerator()
    web_or_rag_generator = WebOrRagAnswerGenerator()
    final_answer_generator = FinalAnswerGenerator()
    web_search_tool = WebSearchTool()
    reranker = SafeReranker.create(
        top_n=-1
    )  # Return all nodes, let caller decide
    workflow_filter = WorkflowFilter(
        direct_answer_generator=direct_answer_generator,
        web_or_rag_generator=web_or_rag_generator,
        final_answer_generator=final_answer_generator,
        web_search_tool=web_search_tool,
        reranker=reranker,
    )
    result =  workflow_filter.process("https://xueqiu.com/8244815919/327993547 在这一文中的机器人技术中，与美国合作的厂家中有什么特别的吗")
    print(result.result)