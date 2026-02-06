"""Workflow server wrapper for AgentWorkflow (LangGraph)."""

from logging import getLogger
from typing import Optional

from askany.rag.router import QueryType
from askany.workflow.workflow_langgraph import AgentWorkflow

logger = getLogger(__name__)


async def run_workflow_via_client(
    agent_workflow: AgentWorkflow,
    query: str,
    query_type: Optional[QueryType] = None,
    context: Optional[dict] = None,
) -> str:
    """Run workflow using AgentWorkflow directly.

    Args:
        agent_workflow: AgentWorkflow instance
        query: Query string
        query_type: Query type (optional, defaults to AUTO)
        context: Optional context dict (currently unused, kept for compatibility)

    Returns:
        Workflow result as string
    """
    if query_type is None:
        query_type = QueryType.AUTO

    logger.debug("AgentWorkflow: Running workflow with query: %s", query)
    logger.debug(
        "AgentWorkflow: query_type=%s, context=%s",
        query_type,
        context,
    )

    try:
        # Extract is_sub_query_workflow from context if provided
        is_sub_query_workflow = False
        if context:
            is_sub_query_workflow = context.get("is_sub_query_workflow", False)

        # Build initial state
        initial_state = {
            "query": query,
            "query_type": query_type,
            "can_direct_answer": False,
            "need_web_search": False,
            "need_rag_search": False,
            "nodes": [],
            "keywords": [],
            "analysis": None,
            "iteration": 0,
            "no_relevant_result": None,
            "current_sub_query": None,
            "previous_qa_context": "",
            "is_sub_query_workflow": is_sub_query_workflow,
            "result": None,
        }

        # Use async invoke method
        logger.debug("AgentWorkflow: 调用graph.ainvoke开始")
        result = await agent_workflow.graph.ainvoke(initial_state)
        logger.debug(
            "AgentWorkflow: graph.ainvoke完成, result类型=%s",
            type(result).__name__,
        )

        # Extract result
        result_str = result.get("result", "抱歉，无法处理此查询。")
        if result_str is None:
            logger.warning("AgentWorkflow: Workflow returned None result")
            return "抱歉，无法处理此查询。"

        logger.debug(
            "AgentWorkflow: Workflow completed, result length: %d", len(result_str)
        )
        return result_str
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else repr(e)
        logger.error(
            "AgentWorkflow.ainvoke失败 - 异常类型: %s, 异常消息: %s, query长度: %d, query_type: %s",
            error_type,
            error_msg,
            len(query),
            query_type,
            exc_info=True,
        )
        raise
