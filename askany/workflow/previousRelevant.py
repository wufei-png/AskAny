"""Final summary LLM calls for generating final answers (LangChain version)."""

try:
    from askany.observability.langfuse_setup import get_langfuse_callback_handler
except ImportError:
    get_langfuse_callback_handler = lambda: None  # noqa: E731

import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings


class PreviousRelevantResponse(BaseModel):
    """RAG Workflow 最终输出的结构。"""

    relevant_qa_indexes: list[int] = Field(
        description="历史qa中相关的qa对话的下标列表,如果没有相关的,返回空列表"
    )


class PreviousRelevantor:
    """Generator for final answers using LangChain."""

    def __init__(self, llm: ChatOpenAI | None = None):
        """Initialize PreviousRelevantor.

        Args:
            llm: LangChain ChatOpenAI instance. If None, creates one from settings.
        """
        if llm is None:
            api_base = settings.openai_api_base
            api_key = settings.openai_api_key if settings.openai_api_key else None
            model = settings.openai_model

            # Create ChatOpenAI client from configuration
            client_api_key = api_key if api_key else ""
            _lf_handler = get_langfuse_callback_handler()
            self.llm = ChatOpenAI(
                model=model,
                api_key=client_api_key,
                base_url=api_base,
                temperature=settings.temperature,
                max_tokens=settings.output_tokens,
                callbacks=[_lf_handler] if _lf_handler else None,
            )

            print(f"Using LLM: {type(self.llm)}")
            print(f"API Base: {api_base}")
            print(f"Model: {model}")
            print("-" * 80)
        else:
            self.llm = llm

        # 不再使用 structured output，直接使用 completion 接口
        # 这样可以避免 structured output 可能带来的问题（如换行符问题）


if __name__ == "__main__":
    # Get configuration from settings
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None

    print(f"API Base: {api_base}")
    print(f"Model: {settings.openai_model}")
    print("-" * 80)

    # Create generator
    generator = PreviousRelevantor()
