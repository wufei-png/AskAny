"""Agent workflow using llama_index Workflow functionality."""

import re
from logging import getLogger
from typing import TYPE_CHECKING, Dict, List, Optional

from llama_index.core import QueryBundle
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from openai import OpenAI

if TYPE_CHECKING:
    from workflows.client import WorkflowClient

from askany.config import settings
from askany.ingest.keyword_extract_wrapper import KeywordExtractorWrapper
from askany.rag.query_parser import parse_query_filters
from askany.rag.router import QueryRouter, QueryType
from askany.workflow.AnalysisRelated import (
    NoRelevantResult,
    NoRelevantResultWithoutSubQueries,
    RelevantResult,
    analyze_no_relevant,
    analyze_no_relevant_without_sub_queries,
    analyze_relevance_and_completeness,
)
from askany.workflow.FinalSummaryLlm import (
    extract_docs_references,
    format_docs_references,
    generate_final_answer,
)
from askany.workflow.firstStageRelevant import (
    DirectAnswerGenerator,
    WebOrRagAnswerGenerator,
)
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from askany.workflow.SubProblemGenerator import SubProblemGenerator
from askany.workflow.WebSearchTool import WebSearchTool

logger = getLogger(__name__)

# WorkflowServer and WorkflowClient are imported from workflows package


# Workflow Events
class RAGRetrievalEvent(Event):
    """RAG检索事件。"""

    nodes: List[NodeWithScore]
    keywords: List[str]


class AnalysisEvent(Event):
    """分析事件。"""

    analysis: RelevantResult
    nodes: List[NodeWithScore]
    keywords: List[str]


class KeywordSearchEvent(Event):
    """关键词搜索事件。"""

    search_results: Dict[str, List[Dict[str, any]]]
    new_nodes: List[NodeWithScore]


class ContextExpansionEvent(Event):
    """上下文扩展事件。"""

    expanded_nodes: List[NodeWithScore]


class SubQueryWorkflowEvent(Event):
    """子问题工作流事件。"""

    sub_query: str
    previous_qa_context: str  # 之前的问题和回答上下文


class NoRelevantSearchEvent(Event):
    """无相关文档时的搜索事件。"""

    nodes: List[NodeWithScore]


class DirectAnswerEvent(Event):
    """直接回答事件。"""

    query: str


class RAGRetrievalTriggerEvent(Event):
    """触发RAG检索的事件。"""

    query: str


# class WebSearchEvent(Event):
#     """网络搜索事件。"""

#     nodes: List[NodeWithScore]
#     query: str


class AgentWorkflowLlama(Workflow):
    """Agent workflow using llama_index Workflow."""

    def __init__(
        self,
        router: QueryRouter,
        llm: LLM,
        base_path: Optional[str] = None,
        workflow_client: Optional["WorkflowClient"] = None,
        num_concurrent_runs: int = settings.num_concurrent_runs,
    ):
        """Initialize AgentWorkflow.

        Args:
            router: Query router for RAG retrieval
            llm: Language model for analysis and generation
            base_path: Base path for local file search
            workflow_client: Optional workflow client for internal calls
        """
        super().__init__(num_concurrent_runs=num_concurrent_runs)
        self.router = router
        self.llm = llm
        self.sub_problem_generator = SubProblemGenerator()
        self.workflow_client = workflow_client
        if base_path is None:
            base_path = settings.local_file_search_dir
            if base_path == "":
                base_path = None
        self.local_file_search = LocalFileSearchTool(base_path=base_path)

        # Initialize keyword extractor wrapper with TF-IDF priority
        self.keyword_extractor = KeywordExtractorWrapper(priority="tfidf")

        api_base = settings.openai_api_base
        api_key = settings.openai_api_key if settings.openai_api_key else ""
        self.client = OpenAI(api_key=api_key, base_url=api_base)

        # Initialize reranker (shared instance)
        self.reranker = self._create_reranker()

        # Initialize web search tool
        try:
            self.web_search_tool = WebSearchTool()
        except ImportError:
            logger.warning("WebSearchTool not available, web search will be disabled")
            self.web_search_tool = None

        # Initialize first stage relevance generators
        self.direct_answer_generator = DirectAnswerGenerator(client=self.client)
        self.web_or_rag_generator = WebOrRagAnswerGenerator(client=self.client)

    @step
    async def direct_answer_check(
        self, ctx: Context, ev: StartEvent
    ) -> DirectAnswerEvent | StopEvent:
        """Step 0: 判断问题是否可以直接回答。"""
        query = ev.get("query")
        logger.debug("直接回答检查开始 - 查询: %s", query)

        # Check if question can be answered directly
        direct_answer_result = self.direct_answer_generator.generate(query)
        logger.debug(
            "直接回答检查完成 - 可以直接回答: %s, 理由: %s",
            direct_answer_result.can_direct_answer,
            direct_answer_result.reasoning,
        )

        if direct_answer_result.can_direct_answer:
            logger.debug("问题可以直接回答，生成直接答案")
            # Generate direct answer using LLM
            answer, reasoning = generate_final_answer(query, [], self.client)
            return StopEvent(result=answer + "\n\n" + reasoning)

        # Cannot answer directly, proceed to check web/rag search
        logger.debug("问题不能直接回答，检查是否需要网络搜索或RAG检索")
        return DirectAnswerEvent(query=query)

    @step
    async def web_or_rag_check(
        self, ctx: Context, ev: DirectAnswerEvent
    ) -> RAGRetrievalTriggerEvent | StopEvent:
        """Step 0.5: 判断是否需要网络搜索或RAG检索。"""
        query = ev.query
        logger.debug("网络/RAG检查开始 - 查询: %s", query)

        # Check if web search or RAG search is needed
        web_or_rag_result = self.web_or_rag_generator.generate(query)
        logger.debug(
            "网络/RAG检查完成 - 需要网络搜索: %s, 需要RAG检索: %s, 理由: %s",
            web_or_rag_result.need_web_search,
            web_or_rag_result.need_rag_search,
            web_or_rag_result.reasoning,
        )

        need_web = web_or_rag_result.need_web_search
        need_rag = web_or_rag_result.need_rag_search

        # Case 1: web=true, rag=false -> web search only
        if need_web and not need_rag:
            logger.debug("仅需要网络搜索")
            if self.web_search_tool is None:
                logger.warning("网络搜索工具不可用，直接生成答案")
                answer, reasoning = generate_final_answer(query, [], self.client)
                return StopEvent(result=answer + "\n\n" + reasoning)
            web_nodes = self.web_search_tool.search(query)
            logger.debug("网络搜索完成 - 结果数: %d", len(web_nodes))

            # Use reranker if available
            if web_nodes and self.reranker:
                logger.debug("使用reranker对网络搜索结果重新排序")
                query_bundle = QueryBundle(query)
                reranked_nodes = self.reranker.postprocess_nodes(
                    web_nodes, query_bundle=query_bundle
                )
                logger.debug("Reranker重排序完成 - 节点数: %d", len(reranked_nodes))
                web_nodes = reranked_nodes

            # Generate answer from web search results and return StopEvent
            answer, reasoning = generate_final_answer(query, web_nodes, self.client)
            logger.debug("网络搜索答案生成完成")
            return StopEvent(result=answer + "\n\n" + reasoning)

        # Case 2: both false -> warning and direct answer
        if not need_web and not need_rag:
            logger.warning(
                "既不需要网络搜索也不需要RAG检索，但之前判断不能直接回答。"
                "这可能是判断错误，直接生成答案。"
            )
            answer, reasoning = generate_final_answer(query, [], self.client)
            return StopEvent(result=answer + "\n\n" + reasoning)

        # Case 3: web=false, rag=true OR both true -> RAG retrieval (possibly with web)
        # Store the web search flag for later use in rag_retrieval
        await ctx.store.set("need_web_search", need_web)
        await ctx.store.set("need_rag_search", need_rag)
        logger.debug(
            "需要RAG检索 (web=%s, rag=%s)，触发RAG检索事件",
            need_web,
            need_rag,
        )
        # Return RAGRetrievalTriggerEvent to trigger rag_retrieval step
        # Use a separate event type to avoid re-triggering web_or_rag_check
        # The rag_retrieval step will check need_web_search flag and perform web search if needed
        return RAGRetrievalTriggerEvent(query=query)

    # @step
    # async def web_search_answer(self, ctx: Context, ev: WebSearchEvent) -> StopEvent:
    #     """处理仅网络搜索的情况，生成答案。"""
    #     query = ev.query
    #     nodes = ev.nodes
    #     logger.debug("网络搜索答案生成开始 - 查询: %s, 节点数: %d", query, len(nodes))

    #     # Use reranker if available
    #     if nodes and self.reranker:
    #         logger.debug("使用reranker对网络搜索结果重新排序")
    #         query_bundle = QueryBundle(query)
    #         reranked_nodes = self.reranker.postprocess_nodes(
    #             nodes, query_bundle=query_bundle
    #         )
    #         logger.debug("Reranker重排序完成 - 节点数: %d", len(reranked_nodes))
    #         nodes = reranked_nodes

    #     # Generate answer from web search results
    #     answer, reasoning = generate_final_answer(query, nodes, self.client)
    #     logger.debug("网络搜索答案生成完成")
    #     return StopEvent(result=answer + "\n\n" + reasoning)

    @step
    async def rag_retrieval(
        self, ctx: Context, ev: RAGRetrievalTriggerEvent
    ) -> RAGRetrievalEvent:
        """Step 1: RAG检索。"""

        query = ev.query
        query_type = await ctx.store.get("query_type", QueryType.AUTO)

        logger.debug("RAG检索开始 - 查询: %s, 查询类型: %s", query, query_type)

        # Validate query is not None
        if query is None:
            logger.error("查询不能为None，使用空字符串作为默认值")
            query = ""

        # 从context中获取is_sub_query_workflow标志（如果存在）
        # 这个标志在调用子问题时通过context传递
        # Context 从外部传入的 context dict 初始化，可以通过 store.get 获取
        is_sub_query_workflow = await ctx.store.get("is_sub_query_workflow", False)
        if is_sub_query_workflow:
            logger.debug(
                "检测到子问题workflow上下文（通过context），设置标志防止再次分割"
            )

        await ctx.store.set("query", query)
        await ctx.store.set("query_type", query_type)
        await ctx.store.set("iteration", 0)

        cleaned_query, metadata_filters = parse_query_filters(query)
        logger.debug(
            "查询解析完成 - 清理后查询: %s, 元数据过滤器: %s",
            cleaned_query,
            metadata_filters,
        )

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
                    ## TODO no use of settings.faq_second_score_threshold
                    docs_nodes = self.router.docs_query_engine.retrieve(
                        cleaned_query, metadata_filters
                    )
                    logger.debug("DOCS检索完成 - 检索到 %d 个节点", len(docs_nodes))
                    # nodes = list(nodes) + list(docs_nodes)
                    nodes = docs_nodes  # !只用docs的答案
            else:
                logger.debug("FAQ查询引擎不可用，使用DOCS查询引擎")
                nodes = self.router.docs_query_engine.retrieve(
                    cleaned_query, metadata_filters
                )
                logger.debug("DOCS检索完成 - 检索到 %d 个节点", len(nodes))

        # Extract keywords
        keywords = self._extract_keywords_from_query(cleaned_query)
        logger.debug("关键词提取完成 - 提取到 %d 个关键词: %s", len(keywords), keywords)

        # Check if web search is also needed (when both web and rag are true)
        need_web = await ctx.store.get("need_web_search", False)
        if need_web and self.web_search_tool:
            logger.debug("同时需要网络搜索，执行网络搜索")
            import asyncio

            # Perform web search concurrently with RAG (if not already done)
            web_nodes = await asyncio.to_thread(self.web_search_tool.search, query)
            logger.debug("网络搜索完成 - 结果数: %d", len(web_nodes))

            # Merge web search results with RAG results
            if web_nodes:
                nodes = self._merge_nodes(nodes, web_nodes)
                logger.debug("合并网络搜索结果后节点数: %d", len(nodes))

                # Rerank merged results
                if self.reranker:
                    logger.debug("使用reranker对合并结果重新排序")
                    query_bundle = QueryBundle(query)
                    reranked_nodes = self.reranker.postprocess_nodes(
                        nodes, query_bundle=query_bundle
                    )
                    logger.debug("Reranker重排序完成 - 节点数: %d", len(reranked_nodes))
                    nodes = reranked_nodes

        return RAGRetrievalEvent(nodes=nodes, keywords=keywords)

    @step
    async def analyze_relevance(
        self,
        ctx: Context,
        ev: RAGRetrievalEvent | ContextExpansionEvent | NoRelevantSearchEvent,
    ) -> AnalysisEvent | StopEvent:
        """Step 2: LLM判断相关性和完整性。"""
        # Get current state
        query = await ctx.store.get("query")
        iteration = await ctx.store.get("iteration", 0)
        logger.debug(
            "相关性分析开始 - 查询: %s, 迭代次数: %d, 事件类型: %s",
            query,
            iteration,
            type(ev).__name__,
        )

        # Collect nodes and keywords
        if isinstance(ev, RAGRetrievalEvent):
            all_nodes = ev.nodes
            all_keywords = ev.keywords
            logger.debug(
                "RAGRetrievalEvent - 节点数: %d, 关键词数: %d",
                len(all_nodes),
                len(all_keywords),
            )
        elif isinstance(ev, NoRelevantSearchEvent):
            logger.debug(
                "NoRelevantSearchEvent - 新节点数: %d, 保留旧节点: %s",
                len(ev.nodes),
                settings.reserve_keywords_old_nodes,
            )
            if settings.reserve_keywords_old_nodes:
                existing_nodes = await ctx.store.get("nodes", [])
                logger.debug(
                    "合并节点 - 已有节点数: %d, 新节点数: %d",
                    len(existing_nodes),
                    len(ev.nodes),
                )
                all_nodes = self._merge_nodes(existing_nodes, ev.nodes)
                logger.debug("合并后节点数: %d", len(all_nodes))
            else:
                all_nodes = ev.nodes
            all_keywords = await ctx.store.get("keywords", [])
        elif isinstance(ev, ContextExpansionEvent):
            logger.debug(
                "ContextExpansionEvent - 扩展节点数: %d, 保留旧节点: %s",
                len(ev.expanded_nodes),
                settings.reserve_expanded_old_nodes,
            )
            if settings.reserve_expanded_old_nodes:
                existing_nodes = await ctx.store.get("nodes", [])
                logger.debug(
                    "合并节点 - 已有节点数: %d, 扩展节点数: %d",
                    len(existing_nodes),
                    len(ev.expanded_nodes),
                )
                all_nodes = self._merge_nodes(existing_nodes, ev.expanded_nodes)
                logger.debug("合并后节点数: %d", len(all_nodes))
            else:
                all_nodes = ev.expanded_nodes
            all_keywords = await ctx.store.get("keywords", [])

        else:
            logger.warning("Unknown event type: %s", type(ev))
            all_nodes = await ctx.store.get("nodes", [])
            all_keywords = await ctx.store.get("keywords", [])

        # Limit total nodes to prevent infinite accumulation
        # Estimate: each node ~1000-5000 tokens, limit to ~20 nodes to stay within token budget
        max_nodes = settings.max_nodes_to_llm
        logger.debug("当前节点总数: %d, 最大节点数限制: %d", len(all_nodes), max_nodes)
        if len(all_nodes) > max_nodes:
            logger.warning(
                "Too many nodes accumulated (%d > %d), truncating to prevent token overflow. "
                "This may indicate a loop or excessive context expansion.",
                len(all_nodes),
                max_nodes,
            )
            # Keep the highest scored nodes
            all_nodes = sorted(all_nodes, key=lambda n: n.score, reverse=True)[
                :max_nodes
            ]
            logger.debug("节点截断后数量: %d", len(all_nodes))

        await ctx.store.set("nodes", all_nodes)

        # Check max iterations
        # TODO here return I don't know
        logger.debug(
            "检查最大迭代次数 - 当前迭代: %d, 最大迭代: %d",
            iteration,
            settings.agent_max_iterations,
        )
        if iteration > settings.agent_max_iterations:
            logger.debug("达到最大迭代次数，生成最终答案")
            answer, reasoning = generate_final_answer(query, all_nodes, self.client)
            references = extract_docs_references(all_nodes)
            formatted_refs = format_docs_references(references)
            return StopEvent(result=answer + formatted_refs + reasoning)

        # Analyze
        logger.debug("开始LLM分析相关性和完整性")
        analysis = analyze_relevance_and_completeness(
            query, all_nodes, all_keywords, self.client
        )
        logger.debug(
            "分析完成 - 相关文件路径数: %d, 是否完整: %s",
            len(analysis.relevant_file_paths),
            analysis.is_complete,
        )
        await ctx.store.set("keywords", all_keywords)
        # Check if complete
        is_relevant = len(analysis.relevant_file_paths) > 0
        is_complete = analysis.is_complete
        logger.debug(
            "检查完成条件 - 是否相关: %s, 是否完整: %s", is_relevant, is_complete
        )

        # 处理逻辑：is_complete true 且有relevant_file_paths -> 针对relevant_file_paths回答
        if is_complete and is_relevant:
            logger.debug("查询已相关且完整，生成最终答案")
            answer, reasoning = generate_final_answer(query, all_nodes, self.client)
            references = extract_docs_references(all_nodes)
            formatted_refs = format_docs_references(references)
            return StopEvent(result=answer + formatted_refs + reasoning)

        # 如果没有relevant_file_paths说明llm返回结果有误，返回没有结果llm判断失败
        if is_complete and not is_relevant:
            logger.error("LLM判断失败：is_complete=true但没有relevant_file_paths")
            return StopEvent(
                result="抱歉，无法找到相关信息。LLM判断结果异常，请重试或调整问题。"
            )

        # 否则 is_complete false
        await ctx.store.set("iteration", iteration + 1)
        logger.debug("迭代次数更新为: %d", iteration + 1)

        return AnalysisEvent(analysis=analysis, nodes=all_nodes, keywords=all_keywords)

    @step
    async def process_no_relevant(
        self, ctx: Context, ev: AnalysisEvent
    ) -> SubQueryWorkflowEvent | NoRelevantSearchEvent | StopEvent | None:
        """处理没有任何相关文档的情况。"""
        is_relevant = len(ev.analysis.relevant_file_paths) > 0
        if is_relevant:
            # 有相关文档，跳过此步骤
            logger.debug("有相关文档，跳过无相关文档处理步骤")
            return None

        # 检查是否是子问题workflow
        is_sub_query_workflow = await ctx.store.get("is_sub_query_workflow", False)
        keywords = await ctx.store.get("keywords", [])

        if is_sub_query_workflow:
            # 子问题workflow中，使用当前子问题的查询（如果有的话，否则使用原始查询）
            current_sub_query = await ctx.store.get("current_sub_query")
            query_to_use = (
                current_sub_query if current_sub_query else await ctx.store.get("query")
            )
            logger.debug(
                "子问题workflow中，使用不包含子问题的分析模型 - 查询: %s", query_to_use
            )
            no_relevant_result = analyze_no_relevant_without_sub_queries(
                query_to_use, keywords, self.client
            )
            logger.debug(
                "无相关文档分析完成（子问题workflow） - 关键词数: %d, 假想答案: %s",
                len(no_relevant_result.missing_info_keywords),
                "有" if no_relevant_result.hypothetical_answer else "无",
            )

            # 检查是否有missing_info_keywords或hypothetical_answer
            has_keywords = len(no_relevant_result.missing_info_keywords) > 0
            has_hypothetical = bool(no_relevant_result.hypothetical_answer)
            if has_keywords or has_hypothetical:
                logger.debug("检测到关键词或假想答案，触发并发检索")
                # 触发并发检索（使用NoRelevantResultWithoutSubQueries）
                # 现在 _concurrent_search 支持两种类型，可以直接使用
                return await self._concurrent_search(
                    ctx, no_relevant_result, query_to_use
                )

            # 否则，不完整，且没有missing_info_keywords hypothetical_answer
            logger.error("LLM判断失败：无相关文档且无搜索策略（子问题workflow）")
            return StopEvent(
                result="抱歉，无法找到相关信息，且无法生成有效的搜索策略。请尝试调整问题或提供更多上下文。"
            )

        # 非子问题workflow，使用完整的analyze_no_relevant
        query = await ctx.store.get("query")
        logger.debug("开始分析无相关文档情况 - 查询: %s", query)
        no_relevant_result = analyze_no_relevant(query, keywords, self.client)
        logger.debug(
            "无相关文档分析完成 - 子问题数: %d, 关键词数: %d, 假想答案: %s",
            len(no_relevant_result.sub_queries),
            len(no_relevant_result.missing_info_keywords),
            "有" if no_relevant_result.hypothetical_answer else "无",
        )

        # 如果有sub_queries，触发子问题处理
        if no_relevant_result.sub_queries:
            logger.debug("检测到子问题，触发子问题处理流程")
            # 保存no_relevant_result到context
            await ctx.store.set("no_relevant_result", no_relevant_result)
            # 返回第一个子问题
            return SubQueryWorkflowEvent(
                sub_query=no_relevant_result.sub_queries[0],
                previous_qa_context="",
            )

        # 如果没有sub_queries，检查是否有missing_info_keywords或hypothetical_answer
        has_keywords = len(no_relevant_result.missing_info_keywords) > 0
        has_hypothetical = bool(no_relevant_result.hypothetical_answer)
        if has_keywords or has_hypothetical:
            logger.debug("检测到关键词或假想答案，触发并发检索")
            # 保存no_relevant_result到context
            await ctx.store.set("no_relevant_result", no_relevant_result)
            # 触发并发检索
            return await self._concurrent_search(ctx, no_relevant_result, query)

        # 否则，不完整，且没有sub_queries missing_info_keywords hypothetical_answer
        logger.error("LLM判断失败：无相关文档且无搜索策略")
        return StopEvent(
            result="抱歉，无法找到相关信息，且无法生成有效的搜索策略。请尝试调整问题或提供更多上下文。"
        )

    async def _concurrent_search(
        self,
        ctx: Context,
        no_relevant_result: NoRelevantResult | NoRelevantResultWithoutSubQueries,
        query: str,
    ) -> NoRelevantSearchEvent:
        """并发检索：missing_info_keywords用原来的检索，hypothetical_answer用docs_query_engine.retrieve。

        支持 NoRelevantResult 和 NoRelevantResultWithoutSubQueries 两种类型，
        因为它们都包含相同的字段：missing_info_keywords 和 hypothetical_answer。
        """
        import asyncio

        async def search_by_keywords():
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

        async def search_by_hypothetical():
            """使用假想答案进行向量检索。"""
            if not no_relevant_result.hypothetical_answer:
                return []
            logger.debug("开始假想答案向量检索")
            nodes = self.router.docs_query_engine.retrieve(
                no_relevant_result.hypothetical_answer
            )
            logger.debug("假想答案向量检索完成 - 节点数: %d", len(nodes))
            return nodes

        # 并发执行两个检索
        keyword_nodes, hypothetical_nodes = await asyncio.gather(
            search_by_keywords(), search_by_hypothetical()
        )

        # 合并并去重
        all_nodes = self._merge_nodes(keyword_nodes, hypothetical_nodes)
        logger.debug("并发检索完成 - 合并后节点数: %d", len(all_nodes))

        # 使用reranker重排序
        if all_nodes and self.reranker:
            logger.debug("使用reranker重排序节点")
            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(
                all_nodes, query_bundle=query_bundle
            )
            logger.debug("Reranker重排序完成 - 节点数: %d", len(reranked_nodes))
            all_nodes = reranked_nodes

        return NoRelevantSearchEvent(nodes=all_nodes)

    @step
    async def process_sub_query(
        self, ctx: Context, ev: SubQueryWorkflowEvent
    ) -> StopEvent:
        """处理子问题：在一个循环中处理所有子问题，避免多次事件触发。"""
        no_relevant_result: NoRelevantResult = await ctx.store.get("no_relevant_result")
        sub_queries = no_relevant_result.sub_queries
        previous_qa_context = await ctx.store.get("previous_qa_context", "")

        logger.debug("开始处理 %d 个子问题", len(sub_queries))

        # 在一个循环中处理所有子问题
        all_qa_context = previous_qa_context
        for idx, sub_query in enumerate(sub_queries):
            logger.debug(
                "处理子问题 %d/%d - 子问题: %s",
                idx + 1,
                len(sub_queries),
                sub_query,
            )

            # 构建包含之前上下文的查询
            enhanced_query = sub_query
            if all_qa_context:
                enhanced_query = f"{all_qa_context}\n\n当前问题: {sub_query}"

            # 使用workflow_client来调用workflow（如果可用）
            # 这样可以复用同一个workflow实例，并且可以通过server管理
            if self.workflow_client is not None:
                logger.debug("使用workflow_client处理子问题: %s", sub_query)
                logger.debug(
                    "enhanced_query长度: %d, 内容预览: %s",
                    len(enhanced_query),
                    enhanced_query[:200],
                )
                try:
                    # Import here to avoid circular import
                    # 传递context，设置is_sub_query_workflow=True以防止子问题再次分割
                    # 必须使用Context.to_dict()正确序列化context，否则ctx.store无法读取
                    from workflows.context import Context, JsonSerializer

                    from askany.workflow.workflow_server import run_workflow_via_client

                    temp_ctx = Context(self)
                    await temp_ctx.store.set("is_sub_query_workflow", True)
                    sub_context = temp_ctx.to_dict(serializer=JsonSerializer())
                    logger.debug(
                        "准备调用run_workflow_via_client, workflow_client类型: %s",
                        type(self.workflow_client).__name__,
                    )
                    sub_answer = await run_workflow_via_client(
                        self.workflow_client,
                        enhanced_query,
                        QueryType.AUTO,
                        context=sub_context,
                    )
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
                        "子问题workflow调用失败 - 异常类型: %s, 异常消息: %s, 子问题: %s, enhanced_query长度: %d",
                        error_type,
                        error_msg,
                        sub_query,
                        len(enhanced_query),
                        exc_info=True,
                    )
                    sub_answer = (
                        f"抱歉，处理子问题时发生错误: {error_type}: {error_msg}"
                    )
            else:
                sub_answer = "抱歉，服务器内部错误，无法处理此子问题。"

            # 更新上下文，累积所有子问题的Q&A
            all_qa_context = (
                f"{all_qa_context}\n\n问题: {sub_query}\n回答: {sub_answer}"
                if all_qa_context
                else f"问题: {sub_query}\n回答: {sub_answer}"
            )

        # 所有子问题处理完成，更新上下文并返回最终答案
        await ctx.store.set("previous_qa_context", all_qa_context)
        logger.debug("所有子问题处理完成，生成最终答案")
        final_answer = f"基于以下子问题的处理结果：\n\n{all_qa_context}"
        return StopEvent(result=final_answer)

    @step
    async def expand_context(
        self, ctx: Context, ev: AnalysisEvent
    ) -> ContextExpansionEvent | None:
        """Step 4: 找到chunk所在位置 → 扩展上下文。"""
        # Only expand context if relevant but incomplete
        is_relevant = len(ev.analysis.relevant_file_paths) > 0
        is_complete = ev.analysis.is_complete
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
            return None

        logger.debug(
            "开始扩展上下文 - 节点数: %d, 扩展模式: %s, 扩展比例: %.2f",
            len(ev.nodes),
            settings.expand_context_mode,
            settings.expand_context_ratio,
        )
        expanded_nodes = []
        for node in ev.nodes:
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

        # 使用reranker对扩展节点重新评分
        if expanded_nodes and self.reranker:
            logger.debug("使用reranker对扩展节点重新评分")
            query = await ctx.store.get("query")
            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(
                expanded_nodes, query_bundle=query_bundle
            )
            logger.debug("Reranker重新评分完成 - 节点数: %d", len(reranked_nodes))
            expanded_nodes = reranked_nodes

        logger.debug("上下文扩展完成 - 扩展节点数: %d", len(expanded_nodes))
        return ContextExpansionEvent(expanded_nodes=expanded_nodes)

    # Helper methods

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
                nodes.append(NodeWithScore(node=node, score=0.8))

        logger.debug("搜索结果转换完成 - 总节点数: %d", len(nodes))
        return nodes

    def _merge_nodes(
        self, existing_nodes: List[NodeWithScore], new_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Merge nodes, avoiding duplicates."""
        merged = list(existing_nodes)
        logger.debug(
            "开始合并节点 - 已有节点数: %d, 新节点数: %d",
            len(existing_nodes),
            len(new_nodes),
        )

        existing_paths = {
            (
                n.node.metadata.get("file_path") or n.node.metadata.get("source"),
                n.node.metadata.get("start_line"),
                n.node.metadata.get("end_line"),
            )
            for n in existing_nodes
        }

        added_count = 0
        for new_node in new_nodes:
            path = new_node.node.metadata.get(
                "file_path"
            ) or new_node.node.metadata.get("source")
            start = new_node.node.metadata.get("start_line")
            end = new_node.node.metadata.get("end_line")
            if (path, start, end) not in existing_paths:
                merged.append(new_node)
                existing_paths.add((path, start, end))
                added_count += 1

        logger.debug(
            "节点合并完成 - 合并后节点数: %d, 新增节点数: %d, 跳过重复节点数: %d",
            len(merged),
            added_count,
            len(new_nodes) - added_count,
        )
        return merged

    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract keywords from query using KeywordExtractorWrapper.

        Args:
            query: User query string

        Returns:
            List of extracted keywords
        """
        logger.debug("开始提取关键词 - 查询: %s", query)
        try:
            # Use KeywordExtractorWrapper to extract keywords (TF-IDF priority)
            keywords = self.keyword_extractor.extract_keywords(query)
            logger.debug(
                "使用KeywordExtractorWrapper提取关键词完成 - 关键词数: %d",
                len(keywords),
            )
            return keywords
        except (ValueError, AttributeError) as e:
            # Fallback to simple extraction if model not trained or error occurs
            # This can happen if the TF-IDF model hasn't been trained yet
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
            return keywords

    def _create_reranker(self) -> Optional[BaseNodePostprocessor]:
        """创建reranker实例（与FAQQueryEngine使用同一个实例）。

        Returns:
            Reranker实例，如果创建失败则返回None
        """
        try:
            reranker_model = settings.reranker_model
            reranker_type = getattr(settings, "reranker_type", "sentence_transformer")

            # Auto-detect device
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info("CUDA available, using GPU for reranker")
                else:
                    device = "cpu"
                    logger.info("Using CPU for reranker")
            except ImportError:
                device = "cpu"
                logger.info("torch not available, using CPU for reranker")

            if reranker_type == "flag_embedding":
                # Try to use FlagEmbeddingReranker if available
                try:
                    try:
                        from llama_index.postprocessor.flag_embedding_reranker import (
                            FlagEmbeddingReranker,
                        )
                    except ImportError:
                        from llama_index.core.postprocessor.flag_embedding_reranker import (
                            FlagEmbeddingReranker,
                        )

                    reranker = FlagEmbeddingReranker(
                        model=reranker_model,
                        top_n=-1,  # Return all nodes, let caller decide
                        device=device,
                    )
                    logger.info("FlagEmbeddingReranker created successfully")
                    return reranker
                except ImportError:
                    logger.warning(
                        "FlagEmbeddingReranker not available, "
                        "falling back to SentenceTransformerRerank"
                    )

            # Default: Use SentenceTransformerRerank
            try:
                reranker = SentenceTransformerRerank(
                    model=reranker_model,
                    top_n=-1,  # Return all nodes, let caller decide
                    trust_remote_code=True,
                    device=device,
                )
                logger.info("SentenceTransformerRerank created successfully")
                return reranker
            except TypeError:
                # If device parameter is not supported
                logger.warning(
                    "SentenceTransformerRerank does not support device parameter, "
                    "using auto-detection"
                )
                reranker = SentenceTransformerRerank(
                    model=reranker_model,
                    top_n=-1,
                    trust_remote_code=True,
                )
                logger.info("SentenceTransformerRerank created successfully")
                return reranker
        except Exception as e:
            logger.error("Failed to create reranker: %s", str(e))
            return None
