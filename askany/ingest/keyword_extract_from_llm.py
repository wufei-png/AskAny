"""Keyword extraction using LLM with Chinese prompts."""

import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
import textwrap as tw

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings  # noqa: E402
import logging

logger = logging.getLogger(__name__)
from cachetools import LRUCache, cachedmethod


class KeywordExtractionResult(BaseModel):
    """关键词提取结果结构"""

    keywords: List[str] = Field(
        description="提取的关键词列表, 数量不超过指定的最大关键词数量。"
    )


class KeywordFilterResult(BaseModel):
    """关键词过滤结果结构"""

    filtered_keywords: List[str] = Field(
        description="过滤后的关键词列表，只保留比较长的关键词或者特定领域的词，去掉太常见的词语。"
    )


class KeywordExtractorFromLLM:
    """使用 LLM 进行关键词提取的生成器"""

    def __init__(self, client: Optional[OpenAI] = None, max_keywords: int = 3):
        """初始化 KeywordExtractorFromLLM

        Args:
            client: OpenAI 客户端，如果为 None 则从配置创建
            max_keywords: 最大关键词数量，默认为 3
        """
        if client is None:
            api_base = settings.openai_api_base
            api_key = settings.openai_api_key if settings.openai_api_key else None
            model = settings.openai_model

            # Create OpenAI client directly from configuration
            # For vLLM, api_key can be None or empty string, but OpenAI client requires it
            # Use empty string as fallback for vLLM (vLLM typically doesn't require auth)
            client_api_key = api_key if api_key else ""
            self.client = OpenAI(
                api_key=client_api_key,
                base_url=api_base,
                timeout=settings.llm_timeout,
            )

            print(f"Using LLM: {type(self.client)}")
            print(f"API Base: {api_base}")
            print(f"Model: {model}")
            print("-" * 80)
        else:
            self.client = client

        self.max_keywords = max_keywords
        self.min_question_length = 3
        self.cache = LRUCache(maxsize=1024)

    @cachedmethod(lambda self: self.cache)
    def extract_keywords(self, question: str) -> List[str]:
        """从问题中提取关键词

        Args:
            question: 用户问题字符串

        Returns:
            关键词列表
        """
        # 如果问题长度小于最小长度阈值，直接返回空列表
        # Python 中 len() 对中英文的处理：
        # - 英文：每个字符（字母、数字、标点）占 1 个长度单位
        # - 中文：每个中文字符（汉字、中文标点）占 1 个长度单位
        # 例如：len("hi") = 2, len("你好") = 2, len("hello") = 5, len("你好吗") = 3
        # 注意：虽然中文字符在显示宽度上通常占 2 个字符宽度（全角），但在 Python 字符串长度计算中仍为 1
        if len(question) < self.min_question_length:
            return [question]

        # Format system prompt with template instructions (without question)
        system_prompt = self._format_system_prompt()
        # Format user message with question prefix
        user_content = self._format_user_message(question)

        # Call LLM with structured output
        completion = self.client.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            response_format=KeywordExtractionResult,
            temperature=settings.temperature,
            top_p=settings.top_p,
        )

        # Parse response
        response_content = completion.choices[0].message
        if not response_content.parsed:
            raise ValueError(
                "Failed to parse keyword extraction result from LLM response"
            )

        # Validate and limit keywords length
        keywords = response_content.parsed.keywords

        # 去重，重复次数多的排在前面
        keyword_counter = Counter(keywords)
        # 按照出现次数降序排序，然后按照原始顺序排序（保持稳定性）
        keywords = [
            keyword
            for keyword, _ in sorted(
                keyword_counter.items(), key=lambda x: (-x[1], keywords.index(x[0]))
            )
        ]

        if len(keywords) > self.max_keywords:
            keywords = keywords[: self.max_keywords]

        return keywords

    def _format_system_prompt(self) -> str:
        """格式化系统提示词

        Returns:
            格式化后的系统提示词字符串
        """
        return tw.dedent(f"""\
你是关键词提取助手，负责从用户问题中提取关键词。

任务要求：
1. 请根据用户问题，从文本中提取最多 {self.max_keywords} 个关键词。
2. 优先选择有助于检索答案的关键信息，避免常见停用词。
3. 对明显的拼写错误进行纠正。

拼写纠正示例：
- deplyment → deployment
- moudle → module
- defualt → default
- paramter → parameter
""")

    def _format_user_message(self, question: str) -> str:
        """格式化用户消息，添加问题前缀

        Args:
            question: 用户问题字符串

        Returns:
            格式化后的用户消息字符串
        """
        return f"问题：{question}"

    def filter_keywords(self, keywords: List[str], query: str) -> List[str]:
        """过滤关键词，去掉太常见的词语，只保留比较长的关键词或者特定领域的词

        Args:
            keywords: 待过滤的关键词列表
            query: 用户问题字符串
        Returns:
            过滤后的关键词列表
        """
        if not keywords:
            return []

        # Format system prompt for filtering
        system_prompt = self._format_filter_system_prompt()
        # print(f"System prompt: {system_prompt}")
        # Format user message with keywords
        user_content = self._format_filter_user_message(keywords, query)
        # print(f"User content: {user_content}")
        # Call LLM with structured output
        completion = self.client.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            response_format=KeywordFilterResult,
            temperature=settings.temperature,
            top_p=settings.top_p,
            timeout=settings.llm_timeout,
        )

        # Parse response
        response_content = completion.choices[0].message
        if not response_content.parsed:
            logger.error("Failed to parse keyword filter result from LLM response")
            return keywords

        return response_content.parsed.filtered_keywords

    def _format_filter_system_prompt(self) -> str:
        """格式化过滤系统提示词

        Returns:
            格式化后的系统提示词字符串
        """
        # TODO 关键词小库优化
        #         return tw.dedent(f"""\
        # 你是关键词过滤助手。

        # 你的任务是：
        # 从用户问题提取出的关键词列表里，过滤掉对知识检索定位价值低的词，仅保留具备明确检索区分度的关键词。

        # 【过滤原则】
        # - 保留词：
        #     - 产品名、系统名、组件名、架构名、专有技术名词
        #     - 如果关键词是明确的技术 / 产品 / 组件 / 工具名词（如数据库、中间件、基础设施、平台），
        #     且与同一问题中的其他关键词存在直接技术或业务关系（如配置、部署、依赖、使用），
        #     一律保留，不得因为“常见”或“不在知识库”而过滤。
        # - 过滤词：
        #     - 语义泛化、缺乏检索区分度的抽象词（如“操作”“方法”“方案”“问题”“使用”等）
        #     - “常见词”仅指抽象泛指词，不包括具体的技术或基础设施名词。

        # 【上下文关联规则】
        # - 如果某个词本身较常见，但与其他关键词存在明确上下文或业务关联，
        #   且可能在同一技术文档中共同出现，则不要过滤。

        # 示例说明

        # 示例 1：
        # 问题：console 不显示验证码
        # 原始关键词：["console", "验证码"]
        # 说明：
        # - console 是系统关键名词，必须保留；
        # - 验证码是常见词，但与 console 存在直接功能关联，可能在同一文档中出现，因此不应过滤。

        # 示例 2：
        # 问题：如何选择时空库架构？如一主一从，实际怎么操作？
        # 原始关键词：["时空库", "一主一从", "架构", "操作"]
        # 说明：
        # - 时空库、一主一从 是核心业务与架构关键词，必须保留；
        # - 架构 虽然较常见，但与 时空库 强相关，不过滤；
        # - 操作 语义过于宽泛，与大量场景相关，应过滤掉。
        # """)

        return tw.dedent(f"""\
你是关键词过滤助手。

你的任务是：  
从用户问题提取出的关键词列表里，过滤掉对知识检索定位价值低的词，仅保留具备明确检索区分度的关键词 
**注意: 不要对关键词进行拆分或者修改，只进行过滤！**

【过滤原则】  
- 保留词：  
    - 产品名、系统名、组件名、架构名、专有技术名词， 以及这些名词附属词，如描述它们的形容词，动词，
例如: 当关键词包含 fail, error, timeout, not show 等异常描述时，它们与主体名词结合构成了唯一的搜索意图，禁止过滤。
    - 如果关键词是明确的技术 / 产品 / 组件 / 工具名词（如数据库、中间件、基础设施、平台），
    且与同一问题中的其他关键词存在直接技术或业务关系（如配置、部署、依赖、使用），
    一律保留，不得因为“常见”或“可能不在知识库”而过滤。

- 过滤词：  
    - 语义泛化、缺乏检索区分度的抽象词（如“操作”“方法”“方案”“问题”“使用”等）  
    - “常见词”仅指抽象泛指词，不包括具体的技术或基础设施名词。

【上下文关联规则】  
- 如果某个词本身较常见，但与其他关键词存在明确上下文或业务关联，
  且可能在同一技术文档中共同出现，则不要过滤。

示例说明  

示例 1：  
问题：console 不显示验证码  
原始关键词：["console", "验证码"]  
说明：  
- console 是系统关键名词，必须保留；  
- 验证码是常见词，但与 console 存在直接功能关联，可能在同一文档中出现，因此不应过滤。  

示例 2：  
问题：如何选择时空库架构？如一主一从，实际怎么操作？  
原始关键词：["时空库", "一主一从", "架构", "操作"]  
说明：  
- 时空库、一主一从 是核心业务与架构关键词，必须保留；  
- 架构 虽然较常见，但与 时空库 强相关，不过滤；  
- 操作 语义过于宽泛，与大量场景相关，应过滤掉。 
""")

    def _format_filter_user_message(self, keywords: List[str], query: str) -> str:
        """格式化过滤用户消息

        Args:
            keywords: 待过滤的关键词列表
            query: 用户问题字符串
        Returns:
            格式化后的用户消息字符串
        """
        keywords_str = "、".join(keywords)
        return f"用户问题：{query}\n关键词列表：{keywords_str}"


if __name__ == "__main__":
    # Get configuration from settings
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None
    model = settings.openai_model

    print("Using LLM: KeywordExtractorFromLLM")
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print("-" * 80)

    # Test KeywordExtractorFromLLM
    extractor = KeywordExtractorFromLLM(max_keywords=3)

    # Test 3: Chinese question
    # print("Test 3: Chinese question")
    # question3 = "激活率是什么意思？"
    # keywords3 = extractor.extract_keywords(question3)
    # print(f"Question: {question3}")
    # print(f"Keywords: {keywords3}")
    # print("-" * 80)
