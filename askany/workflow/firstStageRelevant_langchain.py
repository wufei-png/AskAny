"""SubProblemGenerator for decomposing user queries into sub-problems (LangChain version)."""

import re
import sys
from pathlib import Path
from typing import Optional, cast
from logging import getLogger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from numpy import False_
from pydantic import BaseModel, Field

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings  # noqa: E402
from askany.ingest.keyword_extract_from_tfidf import KeywordExtractorFromTFIDF
from askany.prompts.prompt_manager import get_prompts


import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Linux 文件系统中最基础的禁止字符
FORBIDDEN_CHARS_PATTERN = re.compile(r"[\x00]")


class DirectAnswerResult(BaseModel):
    """
    相关性判断
    """

    can_direct_answer: bool = Field(
        description="判断问题是否可以直接根据已有知识回答,不依赖外部知识库或者网络搜索.",
        default=False,
    )
    reasoning: str = Field(description="简要解释你的依据。")


class WebOrRagAnswer(BaseModel):
    """
    相关性判断
    """

    need_web_search: bool = Field(
        description="判断问题是否可以直接通过网络搜索找到答案.",
        default=False,
    )
    need_rag_search: bool = Field(
        description="判断问题是否需要通过rag知识库检索找到答案,知识库的内容为: 对图片视频用ai模型分析, 特征数据库存储检索, 大数据聚类归档的业务数据知识库.",
        default=False,
    )
    # reasoning: str = Field(description="简要解释你的依据。")


class DirectAnswerGenerator:
    """Generator for decomposing user queries into sub-problems."""

    def __init__(self, llm: Optional[ChatOpenAI] = None, keyword_extractor: Optional[KeywordExtractorFromTFIDF] = None):
        """Initialize DirectAnswerGenerator.

        Args:
            llm: LangChain ChatOpenAI instance. If None, creates one from settings.
        """
        if keyword_extractor is None:
            self.keyword_extractor = KeywordExtractorFromTFIDF()
        else:
            self.keyword_extractor = keyword_extractor

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
            schema=DirectAnswerResult,
            method="json_schema",
            include_raw=False,
            strict=True,
            tools=None,
        )

    def generate(self, query: str) -> DirectAnswerResult:
        """Generate first stage relevant result from user query.

        Args:
            query: User query string

        Returns:
            DirectAnswerResult containing can_direct_answer
        """
        # Format prompt
        prompt = self._format_prompt(query)

        # Call LLM with structured output using LangChain messages
        prompts = get_prompts()
        result = self.structured_llm.invoke(
            [
                SystemMessage(
                    content=prompts.direct_answer_system
                ),
                HumanMessage(content=prompt),
            ]
        )
        #assert result is DirectAnswerResult
        assert isinstance(result, DirectAnswerResult), f"Expected DirectAnswerResult, got {type(result)}"
        result = cast(DirectAnswerResult, result)
        
        # recheck can_direct_answer
        if result.can_direct_answer:
            keywords = self.keyword_extractor.extract_keywords_set(query)
            #TODO 需要优化word_freq.txt 再使用
            if keywords and len(keywords) > 0:
                for keyword in keywords:
                    frequency = self.keyword_extractor.get_frequency_in_freqfile(keyword)
                    print(f"关键词 {keyword} 在word_freq.txt中出现，频率为 {frequency}")
                    if frequency <= settings.freq_in_rag_threshold:
                        print(f"关键词 {keyword} 在word_freq.txt中出现，且频率小于10，我们推翻了llm的判断，需要进行网络搜索")
                        result.can_direct_answer = False
                        break
                result.can_direct_answer = False

        return result

    def _format_prompt(self, query: str) -> str:
        """Format prompt for sub-problem generation.

        Args:
            query: User query string

        Returns:
            Formatted prompt string
        """
        prompts = get_prompts()
        return prompts.direct_answer_task.format(query=query)


class WebOrRagAnswerGenerator:
    """Generator for decomposing user queries into sub-problems."""

    def __init__(self, llm: Optional[ChatOpenAI] = None, keyword_extractor: Optional[KeywordExtractorFromTFIDF] = None):
        """Initialize WebOrRagAnswerGenerator.

        Args:
            llm: LangChain ChatOpenAI instance. If None, creates one from settings.
            keyword_extractor: KeywordExtractorFromTFIDF instance. If None, creates one from settings.
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

        # Initialize keyword extractor
        if keyword_extractor is None:
            self.keyword_extractor = KeywordExtractorFromTFIDF()
        else:
            self.keyword_extractor = keyword_extractor

        # Create structured LLM with system message
        self.structured_llm = self.llm.with_structured_output(
            schema=WebOrRagAnswer,
            method="json_schema",
            include_raw=False,
            strict=True,
            tools=None,
        )

    def generate(self, query: str) -> WebOrRagAnswer:
        """Generate web or rag answer result from user query.

        Args:
            query: User query string

        Returns:
            WebOrRagAnswer containing need_web_search and need_rag_search
        """
        # Format prompt
        prompt = self._format_prompt(query)

        # Call LLM with structured output using LangChain messages
        prompts = get_prompts()
        result = self.structured_llm.invoke(
            [
                SystemMessage(
                    content=prompts.web_or_rag_system
                ),
                HumanMessage(content=prompt),
            ]
        )

        # Type assertion to ensure result is WebOrRagAnswer for IDE type checking
        assert isinstance(result, WebOrRagAnswer), f"Expected WebOrRagAnswer, got {type(result)}"
        result = cast(WebOrRagAnswer, result)

        # recheck need_rag_search
        # If need_rag_search is False, try keyword extraction to check if query matches domain
        keywords = None
        have_extracted_keywords = False
        if not result.need_rag_search:
            have_extracted_keywords = True
            keywords = self.keyword_extractor.extract_keywords_set(query)
            if keywords and len(keywords) > 0:  # If keywords are extracted, set need_rag_search to True
                logger.info(f"关键词提取有结果，我们推翻了llm的判断，需要进行RAG检索")
                result.need_rag_search = True
        
        # recheck need_web_search
        if result.need_rag_search and result.need_web_search:
            if have_extracted_keywords:
                if keywords and len(keywords) > 0:
                    print(f"关键词提取有结果，通常来讲这种情况是用户想要知道一些业务知识，而不是通用知识，所以不需要进行网络搜索")
                    result.need_web_search = False
            else:
                keywords = self.keyword_extractor.extract_keywords_set(query)
                if keywords and len(keywords) > 0:
                    print(f"关键词提取有结果，通常来讲这种情况是用户想要知道一些业务知识，而不是通用知识，所以不需要进行网络搜索")
                    result.need_web_search = False
                else:
                    print(f"关键词提取没有结果，我们不推翻llm的判断，需要进行网络搜索")
                    result.need_web_search = True

        return result

    def _format_prompt(self, query: str) -> str:
        """Format prompt for web or rag answer generation.

        Args:
            query: User query string

        Returns:
            Formatted prompt string
        """
        prompts = get_prompts()
        return prompts.web_or_rag_task.format(query=query)


if __name__ == "__main__":
    # Get configuration from settings
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None
    model = settings.openai_model
    direct_answer = DirectAnswerGenerator()
    web_rag_answer_generator = WebOrRagAnswerGenerator()
    query="什么是激活率？"
    result = direct_answer.generate(query)
    print(f"Result: {result}")
    print(f"Can direct answer: {result.can_direct_answer}")
    # print(f"Reasoning: {result.reasoning}")
    print("-" * 80)
    if not result.can_direct_answer:
        result = web_rag_answer_generator.generate(query)
        print(f"{query} Result: {result}")
        print(f"Need web search: {result.need_web_search}")
        print(f"Need rag search: {result.need_rag_search}")
        # print(f"Reasoning: {result.reasoning}")
        print("-" * 80)
    raise Exception("Stop here")
    result = web_rag_answer_generator.generate(
            """在中美当前的关系下，普通人如何投资美股？"""
        )
    print(f"在中美当前的关系下，普通人如何投资美股？Result: {result}")
    print(f"Need web search: {result.need_web_search}")
    print(f"Need rag search: {result.need_rag_search}")
    # print(f"Reasoning: {result.reasoning}")
    print("-" * 80)
    
    
    result = web_rag_answer_generator.generate(
            """k8s如何重启deployment？"""
        )
    print(f"k8s如何重启deployment？Result: {result}")
    print(f"Need web search: {result.need_web_search}")
    print(f"Need rag search: {result.need_rag_search}")
    # print(f"Reasoning: {result.reasoning}")
    print("-" * 80)