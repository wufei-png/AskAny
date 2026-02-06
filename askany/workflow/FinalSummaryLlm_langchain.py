"""Final summary LLM calls for generating final answers (LangChain version)."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings
from askany.prompts.prompt_manager import get_prompts
from askany.workflow.token_control import (
    check_and_truncate_messages,
    truncate_nodes_by_tokens,
)


class FinalSummaryResponse(BaseModel):
    """RAG Workflow 最终输出的结构。"""

    summary_answer: str = Field(
        description="根据提供的所有参考内容，如果内容相关且完整，则以自然、流畅的段落形式，完整回答用户的问题。如果上下文不足以回答问题，请返回'参考内容不足以回答问题'。",
    )
    reasoning: Optional[str] = Field(
        description="简要解释回答的依据和推理过程",
        default=None,
    )


class FinalAnswerGenerator:
    """Generator for final answers using LangChain."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize FinalAnswerGenerator.

        Args:
            llm: LangChain ChatOpenAI instance. If None, creates one from settings.
        """
        if llm is None:
            api_base = settings.openai_api_base
            api_key = settings.openai_api_key if settings.openai_api_key else None
            model = settings.openai_model

            # Create ChatOpenAI client from configuration
            client_api_key = api_key if api_key else ""
            self.llm = ChatOpenAI(
                model=model,
                api_key=client_api_key,
                base_url=api_base,
                temperature=settings.temperature,
                max_tokens=settings.output_tokens,
            )

            print(f"Using LLM: {type(self.llm)}")
            print(f"API Base: {api_base}")
            print(f"Model: {model}")
            print("-" * 80)
        else:
            self.llm = llm

        # 不再使用 structured output，直接使用 completion 接口
        # 这样可以避免 structured output 可能带来的问题（如换行符问题）

    def _format_final_answer_input(self, query: str, nodes: List[NodeWithScore]) -> str:
        """格式化输入用于生成最终答案。

        Args:
            query: 用户问题
            nodes: 最终使用的节点列表

        Returns:
            格式化的提示文本
        """
        prompts = get_prompts()
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
                f"### 参考文件路径: {file_path}\n"
                f"内容:\n"
                f"```\n{content[:settings.max_content_length]}...\n```\n"
                f"---"
            )

        # Handle empty nodes case
        if not context_parts:
            return prompts.final_answer_no_context_task.format(query=query)
        else:
            context = chr(10).join(context_parts)
            return prompts.final_answer_task.format(query=query, context=context)

    def generate_final_answer(
        self, query: str, nodes: List[NodeWithScore]
    ) -> Tuple[str, Optional[str]]:
        """生成最终答案。

        Args:
            query: 用户问题
            nodes: 最终使用的节点列表

        Returns:
            Tuple of (answer, reasoning)
        """
        import logging

        logger = logging.getLogger(__name__)

        # Truncate nodes to fit within token limit
        # Reserve tokens for prompt template (query, system message, etc.)
        truncated_nodes, node_tokens, nodes_truncated = truncate_nodes_by_tokens(
            nodes,
            max_tokens=settings.llm_max_tokens,
            reserve_for_prompt=1500,  # Reserve for query, system message
        )

        if nodes_truncated:
            logger.warning(
                "Nodes truncated in generate_final_answer: "
                "original=%d nodes, kept=%d nodes",
                len(nodes),
                len(truncated_nodes),
            )

        formatted_input = self._format_final_answer_input(query, truncated_nodes)

        # Check and truncate messages if needed
        prompts = get_prompts()
        messages = [
            SystemMessage(content=prompts.final_answer_system),
            HumanMessage(content=formatted_input),
        ]

        # Convert to dict format for truncation check
        messages_dict = [
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": messages[1].content},
        ]

        truncated_messages_dict, total_tokens, messages_truncated = (
            check_and_truncate_messages(
                messages_dict,
                max_tokens=settings.llm_max_tokens,
            )
        )

        if messages_truncated:
            logger.warning(
                "Messages truncated in generate_final_answer: total_tokens=%d/%d",
                total_tokens,
                settings.llm_max_tokens,
            )
            # Rebuild messages from truncated dict
            messages = [
                SystemMessage(content=truncated_messages_dict[0]["content"]),
                HumanMessage(content=truncated_messages_dict[1]["content"]),
            ]

        # 使用普通的 completion 接口，不使用 structured output
        try:
            response = self.llm.invoke(messages)
            print(f"Response: {response}")
        except Exception as e:
            # Handle LengthFinishReasonError and other parsing errors
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

        # 直接从 response.content 获取响应内容
        summary_answer = response.content if hasattr(response, "content") else str(response)
        
        # 尝试获取 reasoning 字段（如果存在）
        reasoning = getattr(response, "reasoning", None)
        
        # 打印调试信息
        print(f"Completion response - answer length: {len(summary_answer) if summary_answer else 0}")
        if reasoning:
            print(f"Reasoning length: {len(reasoning)}")
        
        # 尝试获取 token usage（如果可用）
        if hasattr(response, "response_metadata"):
            token_usage = response.response_metadata.get("token_usage", {})
            if token_usage:
                print(f"Token Usage: {token_usage}")

        return (summary_answer, reasoning)

    def _format_not_complete_answer_input(self, query: str, nodes: List[NodeWithScore]) -> str:
        """格式化输入用于生成不完整答案。

        Args:
            query: 用户问题
            nodes: 最终使用的节点列表

        Returns:
            格式化的提示文本
        """
        prompts = get_prompts()
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
                f"### 参考文件路径: {file_path}\n"
                f"内容:\n"
                f"```\n{content[:settings.max_content_length]}...\n```\n"
                f"---"
            )

        # Handle empty nodes case
        if not context_parts:
            return prompts.final_answer_no_context_task.format(query=query)
        else:
            context = chr(10).join(context_parts)
            return prompts.not_complete_answer_task.format(query=query, context=context)

    def generate_not_complete_answer(
        self, query: str, nodes: List[NodeWithScore]
    ) -> Tuple[str, Optional[str]]:
        """生成不完整答案（基于不完整的资料）。

        Args:
            query: 用户问题
            nodes: 最终使用的节点列表

        Returns:
            Tuple of (answer, reasoning)
        """
        import logging

        logger = logging.getLogger(__name__)

        # Truncate nodes to fit within token limit
        # Reserve tokens for prompt template (query, system message, etc.)
        truncated_nodes, node_tokens, nodes_truncated = truncate_nodes_by_tokens(
            nodes,
            max_tokens=settings.llm_max_tokens,
            reserve_for_prompt=1500,  # Reserve for query, system message
        )

        if nodes_truncated:
            logger.warning(
                "Nodes truncated in generate_not_complete_answer: "
                "original=%d nodes, kept=%d nodes",
                len(nodes),
                len(truncated_nodes),
            )

        formatted_input = self._format_not_complete_answer_input(query, truncated_nodes)
        print(f"Formatted input: {formatted_input}")
        # Check and truncate messages if needed
        prompts = get_prompts()
        messages = [
            SystemMessage(content=prompts.not_complete_answer_system),
            HumanMessage(content=formatted_input),
        ]

        # Convert to dict format for truncation check
        messages_dict = [
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": messages[1].content},
        ]

        truncated_messages_dict, total_tokens, messages_truncated = (
            check_and_truncate_messages(
                messages_dict,
                max_tokens=settings.llm_max_tokens,
            )
        )

        if messages_truncated:
            logger.warning(
                "Messages truncated in generate_not_complete_answer: total_tokens=%d/%d",
                total_tokens,
                settings.llm_max_tokens,
            )
            # Rebuild messages from truncated dict
            messages = [
                SystemMessage(content=truncated_messages_dict[0]["content"]),
                HumanMessage(content=truncated_messages_dict[1]["content"]),
            ]

        # 使用普通的 completion 接口，不使用 structured output
        try:
            response = self.llm.invoke(messages)
            print(f"Response: {response}")
        except Exception as e:
            # Handle LengthFinishReasonError and other parsing errors
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

        # 直接从 response.content 获取响应内容
        summary_answer = response.content if hasattr(response, "content") else str(response)
        
        # 尝试获取 reasoning 字段（如果存在）
        reasoning = getattr(response, "reasoning", None)
        
        # 打印调试信息
        print(f"Completion response - answer length: {len(summary_answer) if summary_answer else 0}")
        if reasoning:
            print(f"Reasoning length: {len(reasoning)}")
        
        # 尝试获取 token usage（如果可用）
        if hasattr(response, "response_metadata"):
            token_usage = response.response_metadata.get("token_usage", {})
            if token_usage:
                print(f"Token Usage: {token_usage}")

        return (summary_answer, reasoning)


def extract_docs_references(nodes: List[NodeWithScore]) -> Dict[str, List]:
    """Extract document reference information from nodes.

    Args:
        nodes: List of nodes used in the query

    Returns:
        Dictionary with 'markdown' (list of dicts with 'source', 'start_line', 'end_line') and 'faq' (list of FAQ references)
    """
    markdown_refs = []
    faq_refs = []
    seen_sources = set()
    seen_faq_ids = set()

    for node in nodes:
        node_metadata = node.node.metadata if hasattr(node.node, "metadata") else {}
        node_type = node_metadata.get("type", "")

        if node_type == "markdown":
            # Get source file path and line numbers
            source = node_metadata.get("source", "")
            if source and source not in seen_sources:
                seen_sources.add(source)
                start_line = node_metadata.get("start_line")
                end_line = node_metadata.get("end_line")
                markdown_refs.append({
                    "source": source,
                    "start_line": start_line,
                    "end_line": end_line,
                })
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
            # Handle both dict format (with line numbers) and string format (backward compatibility)
            if isinstance(ref, dict):
                source = ref.get("source", "")
                start_line = ref.get("start_line")
                end_line = ref.get("end_line")
                if start_line is not None and end_line is not None:
                    ref_text += f"- {source} (行号: {start_line}-{end_line})\n"
                else:
                    ref_text += f"- {source}\n"
            else:
                # Backward compatibility: if ref is a string
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


# Export function for backward compatibility
def generate_final_answer(
    query: str,
    nodes: List[NodeWithScore],
    client=None,  # Deprecated, kept for backward compatibility
) -> Tuple[str, Optional[str]]:
    """Backward compatibility wrapper."""
    generator = FinalAnswerGenerator()
    return generator.generate_final_answer(query, nodes)


if __name__ == "__main__":
    from llama_index.core.schema import Node, NodeWithScore

    # Get configuration from settings
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None

    print(f"API Base: {api_base}")
    print(f"Model: {settings.openai_model}")
    print("-" * 80)

    # Create generator
    generator = FinalAnswerGenerator()

    # Create test nodes
    query = "解析结果没有收到，怎么办？"
    nodes = []
    node1 = Node(
        metadata={
            "file_path": "data/markdown/xxx-FAQ-1-break-changes.md",
            "source": "data/markdown/xxx-FAQ-1-break-changes.md",
            "type": "markdown",
            "start_line": 1,
            "end_line": 100,
        },
    )
    node1.set_content(
        """
⭐️⭐️ VPS 的重大改动 ⭐️⭐️

# 前后版本不兼容 <a name="vps-system-info"></a>
描述：
此更新会添加jobs表的一个字段，升级chart会自动更新数据库表结构。

后向兼容：更新后可以后向兼容老的vps job，可以在job运行期间进行vps的chart更新。

需要注意的是在更新后**如果想要再回退到老版本**，jobs表的新增字段信息将无法用上，推荐将更新后的新增job或所有job删除，然后再回退。

相关链接：
ones需求：
https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/9CLVdLmf/page/Bu1LuN3E

"""
    )
    node2 = Node(
        metadata={
            "file_path": "data/markdown/xxx-FAQ-2-no-object-info.md",
            "source": "data/markdown/xxx-FAQ-2-no-object-info.md",
            "type": "markdown",
            "start_line": 1,
            "end_line": 100,
        },
    )
    node2.set_content(
        """
             # 为什么下游没有收到解析结果?

没有收到解析结果是一个可能性很多的问题, 下面按照数据流的先后顺序来检查

- 任务状态, 通过百夫长(参考百夫长文档), 或者CMS(参考CMS文档), 或者[VPS 任务接口](./VPS-FAQ-1-task-scripts copy.md) 来检查任务的状态是否正常, 具体可以参考[VPS 任务状态文档](VPS-FAQ-3-task-status.md)
- 检查任务参数是否符合预期, 比如任务类型是否错误(比如本来应该下人脸的任务下成了人群), 任务ROI(比如ROI 画在了无人的区域), 激活时间过长(比如打架事件激活时间设置为120s)等等.
- 检查解析精度相关的参数是否符合预期, 参考[解析精度调优指南](../../process_tune/index.md)
- 检查OSG存储是否满了, 检查kafka 是否写入失败, 通过关键词"kafka", "osg" 过滤vps worker日志, 有那个报错, 就去解决这些组件的问题.

"""
    )
    nodes.append(NodeWithScore(node=node1, score=0.6))
    nodes.append(NodeWithScore(node=node2, score=0.9))

    # Test the function
    result, reasoning = generator.generate_final_answer(query, nodes)
    print("Final Answer:")
    print(result)
    print("Reasoning:")
    print(reasoning)
    print("-" * 80)

    # Test extract_docs_references
    references = extract_docs_references(nodes)
    print("Document References:")
    print(references)
    print("-" * 80)

    # Test format_docs_references
    formatted_refs = format_docs_references(references)
    print("Formatted References:")
    print(formatted_refs)
    print("-" * 80)
