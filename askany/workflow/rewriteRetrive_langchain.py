"""Query rewrite generator for optimizing queries for RAG retrieval (LangChain version)."""

import sys
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings  # noqa: E402


class QueryRewriteResult(BaseModel):
    """
    查询重写结果
    """

    rewritten_query: str = Field(
        description="重写后的查询语句，优化为更适合RAG知识库检索的形式。保持原意、明确化概念等，使其更容易匹配知识库中的相关内容。如果原查询已经很清晰，可以保持基本不变",
        default="",
    )

SYS_PROMPT="""
将用户的查询重写为更适合RAG知识库检索的形式: 简化表达，明确化概念，更正错别字。如果原查询已经很清晰，可以保持基本不变。不要增加问题。
将问句改为解决方案陈述句，如：

原始查询：VPS中的人脸外扩参数是什么意思？
返回：VPS人脸外扩参数

原始查询：多模态app没有告警, 我应该怎么排查？
返回：多模态app没有告警

原始查询：vis对接gb28181国标需要配置哪些信息？
返回：vis对接gb28181国标配置

原始查询：属性库查询慢如何优化
返回：属性库查询性能优化
"""
class QueryRewriteGenerator:
    """Generator for rewriting user queries to optimize RAG retrieval."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize QueryRewriteGenerator.

        Args:
            llm: LangChain ChatOpenAI instance. If None, creates one from settings.
        """
        if llm is None:
            api_base = settings.openai_api_base
            api_key = settings.openai_api_key if settings.openai_api_key else None
            model = settings.openai_model

            # Create ChatOpenAI client from configuration
            # For vLLM, api_key can be None or empty string, but ChatOpenAI requires it
            # Use empty string as fallback for vLLM (vLLM typically doesn't require auth)
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

        # Create structured LLM with system message
        self.structured_llm = self.llm.with_structured_output(
            schema=QueryRewriteResult,
            method="json_schema",
            include_raw=False,
            strict=True,
            tools=None,
        )

    def generate(self, query: str) -> QueryRewriteResult:
        """Generate rewritten query optimized for RAG retrieval.

        Args:
            query: Original user query string

        Returns:
            QueryRewriteResult containing rewritten_query
        """
        # Format prompt
        prompt = self._format_prompt(query)

        # Call LLM with structured output using LangChain messages
        result = self.structured_llm.invoke(
            [
                SystemMessage(
                    content=SYS_PROMPT
                ),
                HumanMessage(content=prompt),
            ]
        )

        return result

    def _format_prompt(self, query: str) -> str:
        """Format prompt for query rewrite generation.

        Args:
            query: User query string

        Returns:
            Formatted prompt string
        """
        return f"""
用户原始查询：
{query}"""


if __name__ == "__main__":
    query_rewrite_generator = QueryRewriteGenerator()
    queries=["介绍算法编排2.0","app打包流程变更了吗？有什么变化"]
    for query in queries:
        result = query_rewrite_generator.generate(query)
        print(result.rewritten_query)

# NVIDIA T4 GPU硬件部署与深度学习环境配置方法，包括CUDA驱动安装、AI模型特征数据库存储优化、大数据聚类归档系统兼容性设置
# VPS部署的多模态AI应用（图像识别/视频分析）告警机制失效，如何排查特征数据库异常或大数据聚类归档导致的监控异常？
