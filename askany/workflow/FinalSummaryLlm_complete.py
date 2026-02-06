"""Final summary LLM calls for generating final answers."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core.schema import NodeWithScore
from openai import OpenAI
from pydantic import BaseModel, Field
from openai import OpenAI

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings
from askany.workflow.token_control import (
    check_and_truncate_messages,
    truncate_nodes_by_tokens,
)


class FinalSummaryResponse(BaseModel):
    """RAG Workflow 最终输出的结构。"""

    summary_answer: str = Field(
        description="根据提供的所有参考内容，如果内容相关且完整，则以自然、流畅的段落形式，完整回答用户的问题。如果上下文不足以回答问题，请返回'参考内容不足以回答问题'。",
    )


def format_final_answer_input(query: str, nodes: List[NodeWithScore]) -> str:
    """格式化输入用于生成最终答案。

    Args:
        query: 用户问题
        nodes: 最终使用的节点列表

    Returns:
        格式化的提示文本
    """
    context_parts = []
    for node in nodes:
        file_path = node.node.metadata.get("file_path") or node.node.metadata.get(
            "source", "Unknown File"
        )
        content = (
            node.node.get_content()
            if hasattr(node.node, "get_content")
            else node.node.text
        )
        context_parts.append(
            f"### 参考文件路径: {file_path}\n内容:\n```\n{content[: settings.max_content_length]}...\n```\n---"
        )

    full_prompt_content = (
        f"--- 任务要求 ---\n"
        f"请根据以下参考文件内容，以自然、流畅的段落形式，完整回答用户的问题。\n"
        f"--- 用户问题 ---\n"
        f"**问题:** {query}\n"
        f"--- 参考文件内容 ---\n"
        f"{chr(10).join(context_parts)}\n"
        f"--- 结束 ---"
    )
    return full_prompt_content


def generate_final_answer_complete(
    query: str, nodes: List[NodeWithScore], client: OpenAI
) -> Tuple[str, Optional[str]]:
    """生成最终答案（使用 completion 接口，不使用 structured output）。

    使用普通的 chat.completions.create 接口，直接返回模型生成的文本内容，
    不要求 JSON 格式输出。这样可以避免 structured output 可能带来的问题。

    Args:
        query: 用户问题
        nodes: 最终使用的节点列表
        client: OpenAI客户端

    Returns:
        Tuple[str, Optional[str]]: (最终答案字符串, reasoning内容（如果存在）)
    """
    # Truncate nodes to fit within token limit
    # Reserve tokens for prompt template (query, system message, etc.)
    truncated_nodes, node_tokens, nodes_truncated = truncate_nodes_by_tokens(
        nodes,
        max_tokens=settings.llm_max_tokens,
        reserve_for_prompt=1500,  # Reserve for query, system message
    )

    if nodes_truncated:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "Nodes truncated in generate_final_answer: "
            "original=%d nodes, kept=%d nodes",
            len(nodes),
            len(truncated_nodes),
        )

    formatted_input = format_final_answer_input(query, truncated_nodes)

    # Check and truncate messages if needed
    messages = [
        {
            "role": "system",
            "content": "你是一个运维助手，基于上下文内容回答用户的问题。",
        },
        {
            "role": "user",
            "content": formatted_input,
        },
    ]

    truncated_messages, total_tokens, messages_truncated = check_and_truncate_messages(
        messages,
        max_tokens=settings.llm_max_tokens,
    )

    if messages_truncated:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "Messages truncated in generate_final_answer: total_tokens=%d/%d",
            total_tokens,
            settings.llm_max_tokens,
        )

    try:
        # 使用普通的 completion 接口，不使用 structured output
        logger.debug(f"Truncated messages: {truncated_messages}")
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=truncated_messages,
            max_tokens=settings.output_tokens,
        )
        logger.debug(f"Completion: {completion}")
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        error_msg = str(e)
        if "length limit" in error_msg.lower() or "LengthFinishReasonError" in str(
            type(e)
        ):
            logger.error(
                "LLM response hit token limit. This may indicate the model is generating "
                "excessively long responses. Error: %s",
                error_msg,
            )
            # Return a fallback response
            return (
                "抱歉，由于响应过长，无法生成完整答案。请尝试将问题分解为更小的子问题，或提供更具体的查询。",
                "响应长度超过限制，可能是由于模型生成了过长的内容。",
            )
        elif "timeout" in error_msg.lower() or "Timeout" in str(type(e)):
            logger.error(
                "LLM request timed out. Error: %s",
                error_msg,
            )
            return (
                "抱歉，请求超时。请稍后重试，或尝试将问题分解为更小的子问题。",
                "请求超时，可能是由于模型响应时间过长。",
            )
        else:
            # Re-raise other exceptions
            raise

    # 直接从 message.content 获取响应内容
    response_message = completion.choices[0].message
    summary_answer = response_message.content

    # 尝试获取 reasoning 字段（如果存在）
    long_reasoning = getattr(response_message, "reasoning", None)

    # 打印调试信息
    import logging

    logger = logging.getLogger(__name__)
    logger.debug(
        f"Completion response - answer length: {len(summary_answer) if summary_answer else 0}"
    )
    if long_reasoning:
        logger.debug(f"Reasoning length: {len(long_reasoning)}")

    return (
        summary_answer,
        long_reasoning,
    )


def extract_docs_references(nodes: List[NodeWithScore]) -> Dict[str, List]:
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


def format_docs_references(references: Dict[str, List]) -> str:
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


if __name__ == "__main__":
    from llama_index.core.schema import Node, NodeWithScore

    # Get configuration from settings
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None

    # Create OpenAI client directly from configuration
    client_api_key = api_key if api_key else ""
    client = OpenAI(
        api_key=client_api_key,
        base_url=api_base,
    )

    print(f"Using LLM: {type(client)}")
    print(f"API Base: {api_base}")
    print(f"Model: {settings.openai_model}")
    print("-" * 80)

    # Create test nodes
    query = "API响应没有收到，怎么办？"
    nodes = []
    node1 = Node(
        metadata={
            "file_path": "data/markdown/system-update-guide.md",
            "source": "data/markdown/system-update-guide.md",
            "type": "markdown",
            "start_line": 1,
            "end_line": 100,
        },
    )
    node1.set_content(
        """
⭐️⭐️ 系统组件更新指南 ⭐️⭐️

# 版本更新说明 <a name="system-update-info"></a>
描述：
此更新会添加数据库表的一个新字段，升级脚本会自动更新数据库表结构。

后向兼容：更新后可以后向兼容旧版本的数据，可以在系统运行期间进行组件更新。

需要注意的是在更新后**如果想要再回退到旧版本**，数据库表的新增字段信息将无法使用，推荐将更新后的新数据或所有数据备份，然后再回退。

相关链接：
更新文档：
https://docs.example.com/system-update-guide

"""
    )
    node2 = Node(
        metadata={
            "file_path": "data/markdown/api-troubleshooting-guide.md",
            "source": "data/markdown/api-troubleshooting-guide.md",
            "type": "markdown",
            "start_line": 1,
            "end_line": 100,
        },
    )
    node2.set_content(
        """
             # 为什么API没有返回响应?

API没有返回响应是一个可能性很多的问题, 下面按照数据流的先后顺序来检查

- 检查API服务状态, 通过监控系统(参考监控文档), 或者管理面板(参考管理文档), 或者[API健康检查接口](./api-health-check.md) 来检查服务的状态是否正常, 具体可以参考[API状态文档](api-status.md)
- 检查请求参数是否符合预期, 比如请求类型是否错误(比如本来应该发送POST请求却用了GET), 请求参数(比如参数值超出了允许范围), 超时时间设置是否合理(比如超时时间设置为120秒)等等.
- 检查API配置相关的参数是否符合预期, 参考[API配置调优指南](../../config/api-tuning.md)
- 检查存储是否满了, 检查消息队列是否写入失败, 通过关键词"queue", "storage" 过滤应用日志, 有那个报错, 就去解决这些组件的问题.

"""
    )
    nodes.append(NodeWithScore(node=node1, score=0.6))
    nodes.append(NodeWithScore(node=node2, score=0.9))

    # Test the function using completion interface
    result, reasoning = generate_final_answer_complete(query, nodes, client)
    print("Final Answer:")
    print(result)
    print("Reasoning:")
    print(reasoning)
    print("-" * 80)

    # # Test extract_docs_references
    # references = extract_docs_references(nodes)
    # print("Document References:")
    # print(references)
    # print("-" * 80)

    # # Test format_docs_references
    # formatted_refs = format_docs_references(references)
    # print("Formatted References:")
    # print(formatted_refs)
    # print("-" * 80)
