"""Analysis related LLM calls for relevance and completeness checking (LangChain version)."""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field, field_validator

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings
from askany.prompts.prompt_manager import get_prompts
from askany.workflow.token_control import (
    check_and_truncate_messages,
)

# Linux 文件系统中最基础的禁止字符
FORBIDDEN_CHARS_PATTERN = re.compile(r"[\x00]")


class RelevantResult(BaseModel):
    """分析结果的数据结构。"""

    relevant_file_paths: List[str] = Field(  # TODO maybe return score？
        description="与用户问题相关的文件路径列表。如果没有任何内容相关，请返回空列表。",
        default_factory=list,
    )

    is_complete: bool = Field(
        description="现有内容是否已经足以完整回答用户的问题。如果需要更多信息，设为 False。"
    )

    reasoning: str = Field(
        description="简要解释为什么认为这些文件相关或不相关，以及为什么认为信息完整或不完整（思维链）。"
    )

    @field_validator("relevant_file_paths", mode="after")
    @classmethod
    def validate_and_check_existence(cls, paths: List[str]) -> List[str]:
        """校验路径是否符合语法，并检查文件在机器上是否真实存在。"""
        existing_paths = []
        non_existing_paths = []
        special_identifiers = []

        for path in paths:
            # 1. 语法校验（防止注入或截断）
            if FORBIDDEN_CHARS_PATTERN.search(path):
                raise ValueError(f"路径包含非法字符 (Null Byte): {path}")
            if not path.strip():
                raise ValueError("路径不能为空或仅包含空白字符。")

            # 2. 检查是否为特殊标识符（无需文件系统验证）
            # 网络搜索标识符（web_search_*）
            if path.startswith("web_search_"):
                special_identifiers.append(path)
                continue
            # 工作流上下文标识符
            if path in ("outer_previous_qa_context", "inner_previous_qa_context"):
                special_identifiers.append(path)
                continue

            # 3. 真实性校验 (I/O 操作)
            if os.path.exists(path):
                existing_paths.append(path)
            else:
                non_existing_paths.append(path)

        # 4. 报错逻辑：如果存在任何一个不存在的路径（非特殊标识符），则报错
        if non_existing_paths:
            error_message = (
                f"以下 {len(non_existing_paths)} 个文件路径不存在于文件系统中，验证失败: {non_existing_paths}. "
                f"【注意】已检查到有效的路径有: {existing_paths}。"
                f"【注意】特殊标识符（无需验证）: {special_identifiers}。"
            )
            # raise ValueError(error_message)//不raise 只是打印，避免整个问题失败
            print(f"AnalysisRelated_langchain.py: {error_message}")
        return paths


class NoRelevantResult(BaseModel):
    """处理没有任何相关文档的情况时的分析结果。"""

    missing_info_keywords: List[str] = Field(
        description="如果关键字有缺失，参考已经有的关键词，生成不同于已有的，用于搜索的缺失或不同角度的关键字列表。",
        default_factory=list,
    )

    # 关键优化：问题分解（Sub-Queries）而不是简单的关键词
    sub_queries: List[str] = Field(
        description="如果问题可以拆解，检查原问题是否可以拆解为若干子问题，这些子问题必须按逻辑上的依赖性排序，先解决前面的问题，再依据前问题答案解决后面的问题。",
        default_factory=list,
    )

    hypothetical_answer: str = Field(
        description="生成一个假想答案，用于向量相似度检索。",
        default="",
    )

    # reasoning: str = Field(
    #     description="简要解释为什么没有任何文档相关，以及为什么生成这些关键词、子问题或假想答案（思维链）。"
    # )


class NoRelevantResultWithoutSubQueries(BaseModel):
    """处理没有任何相关文档的情况时的分析结果（不包含子问题，用于子问题workflow中）。"""

    missing_info_keywords: List[str] = Field(
        description="如果关键字有缺失，参考已经有的关键词，生成不同于已有的，用于搜索的缺失或不同角度的关键字列表。",
        default_factory=list,
    )

    hypothetical_answer: str = Field(
        description="生成一个假想答案，用于向量相似度检索。",
        default="",
    )

    # reasoning: str = Field(
    #     description="简要解释为什么没有任何文档相关，以及为什么生成这些关键词或假想答案（思维链）。"
    # )


class GenerateKeywords(BaseModel):
    """生成关键词。"""

    missing_info_keywords: List[str] = Field(
        description="参考已经有的关键词，生成不同于已有的，用于搜索的缺失或不同角度的关键字列表。",
        default_factory=list,
    )


class RelevanceAnalyzer:
    """Analyzer for relevance and completeness checking using LangChain."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize RelevanceAnalyzer.

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

        # Create structured LLM for relevance analysis
        self.structured_llm_relevance = self.llm.with_structured_output(
            schema=RelevantResult,
            method="json_schema",
            include_raw=False,
            strict=True,
            tools=None,
        )

        # Create structured LLM for no relevant analysis
        self.structured_llm_no_relevant = self.llm.with_structured_output(
            schema=NoRelevantResult,
            method="json_schema",
            include_raw=False,
            strict=True,
            tools=None,
        )

        # Create structured LLM for no relevant analysis without sub queries
        self.structured_llm_no_relevant_without_sub = self.llm.with_structured_output(
            schema=NoRelevantResultWithoutSubQueries,
            method="json_schema",
            include_raw=False,
            strict=True,
            tools=None,
        )

        # Create structured LLM for simple keyword generation
        self.structured_llm_generate_keywords = self.llm.with_structured_output(
            schema=GenerateKeywords,
            method="json_schema",
            include_raw=False,
            strict=True,
            tools=None,
        )

    def _format_structured_input(
        self, query: str, nodes: List[NodeWithScore], keywords: List[str]
    ) -> str:
        """格式化输入用于分析相关性和完整性。

        Args:
            query: 用户问题
            nodes: 检索到的节点列表
            keywords: 关键词列表

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
                f"```\n{content[: settings.max_content_length]}...\n```\n"
                f"---"
            )
        # 这里不需要关键词 直接根据文件内容判断相关性
        context = chr(10).join(context_parts)
        return prompts.relevance_analysis_task.format(query=query, context=context)

    def analyze_relevance_and_completeness(
        self,
        query: str,
        nodes: List[NodeWithScore],
        keywords: List[str],
    ) -> RelevantResult:
        """使用LLM分析相关性和完整性。

        Args:
            query: 用户问题
            nodes: 检索到的节点列表
            keywords: 关键词列表

        Returns:
            RelevantResult 分析结果
        """
        import logging

        logger = logging.getLogger(__name__)

        # Truncate nodes to fit within token limit
        # Reserve tokens for prompt template (query, keywords, system message, etc.)
        # truncated_nodes, node_tokens, nodes_truncated = truncate_nodes_by_tokens(
        #     nodes,
        #     max_tokens=settings.llm_max_tokens,
        #     reserve_for_prompt=2000,  # Reserve for query, keywords, system message
        # )
        # TODO 更精准的预估，这个可能估不准
        # Start with all nodes, will be truncated if token limit exceeded
        current_nodes = nodes
        # if nodes_truncated:
        #     logger.warning(
        #         "Nodes truncated in analyze_relevance_and_completeness: "
        #         "original=%d nodes, kept=%d nodes",
        #         len(nodes),
        #         len(truncated_nodes),
        #     )

        # Call LLM with structured output using LangChain
        # Retry with half nodes if token limit exceeded
        max_retries = 2
        for attempt in range(max_retries):
            # Format input with current nodes
            formatted_input = self._format_structured_input(
                query, current_nodes, keywords
            )

            # Check and truncate messages if needed
            prompts = get_prompts()
            messages = [
                SystemMessage(content=prompts.relevance_analysis_system),
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
                    "Messages truncated in analyze_relevance_and_completeness: "
                    "total_tokens=%d/%d",
                    total_tokens,
                    settings.llm_max_tokens,
                )
                # Rebuild messages from truncated dict
                messages = [
                    SystemMessage(content=truncated_messages_dict[0]["content"]),
                    HumanMessage(content=truncated_messages_dict[1]["content"]),
                ]

            try:
                result = self.structured_llm_relevance.invoke(messages)
                print(f"RelevanceAnalyzer Response content: {result}")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                # Check if error is due to token limit exceeded
                is_token_limit_error = (
                    "maximum context length" in error_msg
                    or "context length" in error_msg
                    or ("tokens" in error_msg and "request" in error_msg)
                    or ("400" in error_msg and "tokens" in error_msg)
                )

                if is_token_limit_error and attempt < max_retries - 1:
                    # Truncate nodes to half
                    original_count = len(current_nodes)
                    current_nodes = current_nodes[: len(current_nodes) // 2]
                    logger.warning(
                        "Token limit exceeded in analyze_relevance_and_completeness, "
                        "truncating nodes from %d to %d and retrying (attempt %d/%d). Error: %s",
                        original_count,
                        len(current_nodes),
                        attempt + 1,
                        max_retries,
                        str(e),
                    )
                    # Continue to retry with truncated nodes
                    continue
                else:
                    # Re-raise if not token limit error or max retries reached
                    logger.error(
                        "Error in analyze_relevance_and_completeness: %s",
                        str(e),
                        exc_info=True,
                    )
                    raise

    def filter_relevant_nodes(
        self, nodes: List[NodeWithScore], relevant_file_paths: List[str]
    ) -> List[NodeWithScore]:
        """过滤出与相关文件路径匹配的节点。

        Args:
            nodes: 节点列表
            relevant_file_paths: 相关文件路径列表

        Returns:
            过滤后的节点列表，只包含文件路径在 relevant_file_paths 中的节点
        """
        import logging

        logger = logging.getLogger(__name__)

        if not relevant_file_paths:
            logger.debug("relevant_file_paths 为空，返回空列表")
            return []

        # 将 relevant_file_paths 转换为集合以提高查找效率
        relevant_paths_set = set(relevant_file_paths)
        filtered_nodes = []

        for node in nodes:
            # 获取节点的文件路径
            file_path = node.node.metadata.get("file_path") or node.node.metadata.get(
                "source"
            )
            if file_path and file_path in relevant_paths_set:
                filtered_nodes.append(node)
            elif not file_path:
                logger.debug("节点缺少文件路径，跳过过滤")

        logger.debug(
            "节点过滤完成 - 原始节点数: %d, 过滤后节点数: %d, 相关文件路径数: %d",
            len(nodes),
            len(filtered_nodes),
            len(relevant_file_paths),
        )

        return filtered_nodes

    def _format_no_relevant_input(self, query: str, keywords: List[str]) -> str:
        """格式化输入用于分析没有任何相关文档的情况。

        Args:
            query: 用户问题
            keywords: 关键词列表

        Returns:
            格式化的提示文本
        """
        prompts = get_prompts()
        return prompts.no_relevant_task.format(query=query, keywords=keywords)

    def analyze_no_relevant(
        self,
        query: str,
        keywords: List[str],
    ) -> NoRelevantResult:
        """使用LLM分析没有任何相关文档的情况，生成搜索策略。

        Args:
            query: 用户问题
            keywords: 关键词列表

        Returns:
            NoRelevantResult 分析结果
        """
        import logging

        logger = logging.getLogger(__name__)

        formatted_input = self._format_no_relevant_input(query, keywords)

        # Check and truncate messages if needed
        prompts = get_prompts()
        messages = [
            SystemMessage(content=prompts.no_relevant_system),
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
                "Messages truncated in analyze_no_relevant: total_tokens=%d/%d",
                total_tokens,
                settings.llm_max_tokens,
            )
            # Rebuild messages from truncated dict
            messages = [
                SystemMessage(content=truncated_messages_dict[0]["content"]),
                HumanMessage(content=truncated_messages_dict[1]["content"]),
            ]

        # Call LLM with structured output using LangChain
        result = self.structured_llm_no_relevant.invoke(messages)

        assert isinstance(result, NoRelevantResult)
        # Check and filter duplicate keywords
        original_keywords = result.missing_info_keywords.copy()
        filtered_keywords = [
            kw for kw in result.missing_info_keywords if kw not in keywords
        ]
        duplicate_keywords = [kw for kw in original_keywords if kw in keywords]

        if duplicate_keywords:
            logger.warning(
                f"发现重复的关键词（已过滤）: {duplicate_keywords}。"
                f"原始关键词: {original_keywords}，过滤后: {filtered_keywords}"
            )

        # If all keywords were filtered out, use simple keyword generation
        if not filtered_keywords and original_keywords:
            logger.warning(
                "所有生成的关键词都与已搜索关键字重复，使用简化提示词重新生成关键词"
            )
            filtered_keywords = self._generate_keywords_simple(query, keywords)
            # Filter again to ensure no duplicates
            filtered_keywords = [kw for kw in filtered_keywords if kw not in keywords]
            if not filtered_keywords:
                logger.warning("简化提示词生成的关键词仍然全部重复，返回空列表")

        # Update result with filtered keywords
        result.missing_info_keywords = filtered_keywords
        return result

    def _format_no_relevant_without_sub_queries_input(
        self, query: str, keywords: List[str]
    ) -> str:
        """格式化输入用于分析没有任何相关文档的情况（不包含子问题）。

        Args:
            query: 用户问题
            keywords: 关键词列表

        Returns:
            格式化的提示文本
        """
        prompts = get_prompts()
        return prompts.no_relevant_without_sub_task.format(
            query=query, keywords=keywords
        )

    def _format_simple_keywords_input(self, query: str, keywords: List[str]) -> str:
        """格式化输入用于简化关键词生成（避免任务过重）。

        Args:
            query: 用户问题
            keywords: 关键词列表

        Returns:
            格式化的提示文本
        """
        prompts = get_prompts()
        return prompts.simple_keywords_task.format(query=query, keywords=keywords)

    def _generate_keywords_simple(self, query: str, keywords: List[str]) -> List[str]:
        """使用简化的提示词生成关键词（避免任务过重）。

        Args:
            query: 用户问题
            keywords: 关键词列表

        Returns:
            生成的关键词列表
        """
        import logging

        logger = logging.getLogger(__name__)

        formatted_input = self._format_simple_keywords_input(query, keywords)

        # Check and truncate messages if needed
        prompts = get_prompts()
        messages = [
            SystemMessage(content=prompts.simple_keywords_system),
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
                "Messages truncated in _generate_keywords_simple: total_tokens=%d/%d",
                total_tokens,
                settings.llm_max_tokens,
            )
            # Rebuild messages from truncated dict
            messages = [
                SystemMessage(content=truncated_messages_dict[0]["content"]),
                HumanMessage(content=truncated_messages_dict[1]["content"]),
            ]

        # Call LLM with structured output using LangChain
        result = self.structured_llm_generate_keywords.invoke(messages)
        assert isinstance(result, GenerateKeywords)
        return result.missing_info_keywords

    def analyze_no_relevant_without_sub_queries(
        self,
        query: str,
        keywords: List[str],
    ) -> NoRelevantResultWithoutSubQueries:
        """使用LLM分析没有任何相关文档的情况，生成搜索策略（不包含子问题，用于子问题workflow中）。

        Args:
            query: 用户问题
            keywords: 关键词列表

        Returns:
            NoRelevantResultWithoutSubQueries 分析结果
        """
        import logging

        logger = logging.getLogger(__name__)

        formatted_input = self._format_no_relevant_without_sub_queries_input(
            query, keywords
        )

        # Check and truncate messages if needed
        prompts = get_prompts()
        messages = [
            SystemMessage(content=prompts.no_relevant_without_sub_system),
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
                "Messages truncated in analyze_no_relevant_without_sub_queries: "
                "total_tokens=%d/%d",
                total_tokens,
                settings.llm_max_tokens,
            )
            # Rebuild messages from truncated dict
            messages = [
                SystemMessage(content=truncated_messages_dict[0]["content"]),
                HumanMessage(content=truncated_messages_dict[1]["content"]),
            ]

        # Call LLM with structured output using LangChain
        result = self.structured_llm_no_relevant_without_sub.invoke(messages)
        assert isinstance(result, NoRelevantResultWithoutSubQueries)
        # Check and filter duplicate keywords
        original_keywords = result.missing_info_keywords.copy()
        filtered_keywords = [
            kw for kw in result.missing_info_keywords if kw not in keywords
        ]
        duplicate_keywords = [kw for kw in original_keywords if kw in keywords]

        if duplicate_keywords:
            logger.warning(
                f"发现重复的关键词（已过滤）: {duplicate_keywords}。"
                f"原始关键词: {original_keywords}，过滤后: {filtered_keywords}"
            )

        # If all keywords were filtered out, use simple keyword generation
        if not filtered_keywords and original_keywords:
            logger.warning(
                "所有生成的关键词都与已搜索关键字重复，使用简化提示词重新生成关键词"
            )
            filtered_keywords = self._generate_keywords_simple(query, keywords)
            # Filter again to ensure no duplicates
            filtered_keywords = [kw for kw in filtered_keywords if kw not in keywords]
            if not filtered_keywords:
                logger.warning("简化提示词生成的关键词仍然全部重复，返回空列表")

        # Update result with filtered keywords
        result.missing_info_keywords = filtered_keywords

        return result


# Export functions for backward compatibility
def analyze_relevance_and_completeness(
    query: str,
    nodes: List[NodeWithScore],
    keywords: List[str],
    client=None,  # Deprecated, kept for backward compatibility
) -> RelevantResult:
    """Backward compatibility wrapper."""
    analyzer = RelevanceAnalyzer()
    return analyzer.analyze_relevance_and_completeness(query, nodes, keywords)


def analyze_no_relevant(
    query: str,
    keywords: List[str],
    client=None,  # Deprecated, kept for backward compatibility
) -> NoRelevantResult:
    """Backward compatibility wrapper."""
    analyzer = RelevanceAnalyzer()
    return analyzer.analyze_no_relevant(query, keywords)


def analyze_no_relevant_without_sub_queries(
    query: str,
    keywords: List[str],
    client=None,  # Deprecated, kept for backward compatibility
) -> NoRelevantResultWithoutSubQueries:
    """Backward compatibility wrapper."""
    analyzer = RelevanceAnalyzer()
    return analyzer.analyze_no_relevant_without_sub_queries(query, keywords)


if __name__ == "__main__":
    from llama_index.core.schema import Node, NodeWithScore

    # Get configuration from settings
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None

    print(f"API Base: {api_base}")
    print(f"Model: {settings.openai_model}")
    print("-" * 80)

    # Create analyzer
    analyzer = RelevanceAnalyzer()

    # Create test nodes
    query = "如何更新系统组件？需要哪些步骤？"
    nodes = []
    node1 = Node(
        metadata={"file_path": "data/markdown/system-update-guide.md"},
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
        metadata={"file_path": "data/markdown/api-troubleshooting-guide.md"},
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
    nodes.append(NodeWithScore(node=node1, score=0.8))
    nodes.append(NodeWithScore(node=node2, score=0.7))
    keywords = ["系统", "更新", "组件"]

    # Test the function
    result = analyzer.analyze_relevance_and_completeness(query, nodes, keywords)
    print("Analysis Result1:")
    print("Relevant file paths:", result.relevant_file_paths)
    print("Is complete:", result.is_complete)
    print("-" * 80)

    query = "API响应没有收到，怎么办？"
    keywords = ["API", "响应"]

    # Test the function
    result = analyzer.analyze_relevance_and_completeness(query, nodes, keywords)
    print("Analysis Result2:")
    print("Relevant file paths:", result.relevant_file_paths)
    print("Is complete:", result.is_complete)
    print("-" * 80)
    query = "数据没有在数据库中找到，怎么办？"
    keywords = ["数据", "数据库"]

    # Test the function
    result = analyzer.analyze_no_relevant(query, keywords)
    print("Analysis Result3:")
    print("Missing info keywords:", result.missing_info_keywords)
    print("Sub queries:", result.sub_queries)
    print("Hypothetical answer:", result.hypothetical_answer)
    print("-" * 80)

    query = "数据成功写入消息队列了，但没有在数据库中找到，怎么办？"
    keywords = ["数据", "数据库"]

    # Test the function
    result = analyzer.analyze_no_relevant_without_sub_queries(query, keywords)
    print("Analysis Result4:")
    print("Missing info keywords:", result.missing_info_keywords)
    print("Hypothetical answer:", result.hypothetical_answer)
    print("-" * 80)
