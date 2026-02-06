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
from askany.workflow.token_control import (
    check_and_truncate_messages,
    truncate_nodes_by_tokens,
)


class PreviousRelevantResponse(BaseModel):
    """RAG Workflow 最终输出的结构。"""

    relevant_qa_indexes: List[int] = Field(
        description="历史qa中相关的qa对话的下标列表,如果没有相关的,返回空列表"
    )


class PreviousRelevantor:
    """Generator for final answers using LangChain."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
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


if __name__ == "__main__":
    from llama_index.core.schema import Node, NodeWithScore

    # Get configuration from settings
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None

    print(f"API Base: {api_base}")
    print(f"Model: {settings.openai_model}")
    print("-" * 80)

    # Create generator
    generator = PreviousRelevantor()
