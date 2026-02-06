"""SubProblemGenerator for decomposing user queries into sub-problems."""

import re
import sys
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings
from askany.prompts.prompt_manager import get_prompts

# Linux 文件系统中最基础的禁止字符
FORBIDDEN_CHARS_PATTERN = re.compile(r"[\x00]")


class SubProblemStructure(BaseModel):
    """
    子问题结构：二级列表，第一级为并行执行的问题组，第二级为串行执行的相关问题。
    """

    parallel_groups: List[List[str]] = Field(
        description="子问题列表",
        default_factory=list,
    )
    # reasoning: str = Field(
    #     description="简要解释为什么这样分解问题，以及问题之间的关系。"
    # )


class SubProblemGenerator:
    """Generator for decomposing user queries into sub-problems."""

    def __init__(self, client: Optional[OpenAI] = None):
        """Initialize SubProblemGenerator.

        Args:
            llm: Language model for generating sub-problems
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
            )

            print(f"Using LLM: {type(client)}")
            print(f"API Base: {api_base}")
            print(f"Model: {model}")
            print("-" * 80)
        else:
            self.client = client

    def generate(self, query: str) -> SubProblemStructure:
        """Generate sub-problems from user query.

        Args:
            query: User query string

        Returns:
            SubProblemStructure containing parallel groups of sub-problems
        """
        # Get prompts from PromptManager
        prompts = get_prompts()
        user_prompt = prompts.sub_problem_task.format(query=query)

        # Call LLM with structured output
        completion = self.client.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": prompts.sub_problem_system,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format=SubProblemStructure,
            temperature=settings.temperature,
            top_p=settings.top_p,
        )

        # Parse response
        response_content = completion.choices[0].message
        if not response_content.parsed:
            raise ValueError("Failed to parse sub-problem structure from LLM response")
        print(f"SubProblemGenerator Response content: {response_content}")
        return response_content.parsed


## 这里的不要删除问题细节需要严格测试一下，确保不会删除问题中的细节。

if __name__ == "__main__":
    # Get configuration from settings
    api_base = settings.openai_api_base
    api_key = settings.openai_api_key if settings.openai_api_key else None
    model = settings.openai_model

    print("Using LLM: SubProblemGenerator")
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print("-" * 80)

    # Test SubProblemGenerator
    generator = SubProblemGenerator()

    # Test 1: Simple single question
    print("Test 1: Simple single question")
    query1 = "cassadnra组件的concurent_reads 有什么用"
    result1 = generator.generate(query1)
    print(f"Query: {query1}")
    print(f"Parallel groups: {result1.parallel_groups}")
    # print(f"Reasoning: {result1.reasoning}")
    print("-" * 80)
    raise Exception("Stop here")
    # Test 2: Complex question with multiple sub-problems

    # Test 3
    print("Test 3")
    query2 = "中美的关系现在如何？主要在针对什么问题进行交锋？"
    result2 = generator.generate(query2)
    print(f"Query: {query2}")
    print(f"Parallel groups: {result2.parallel_groups}")
    # print(f"Reasoning: {result2.reasoning}")
    print("-" * 80)

    print("Test 5")
    query2 = "https://xueqiu.com/8244815919/327993547 在这一文中的机器人技术中，与美国合作的厂家中有什么特别的吗"
    result2 = generator.generate(query2)
    print(f"Query: {query2}")
    print(f"Parallel groups: {result2.parallel_groups}")
    # print(f"Reasoning: {result2.reasoning}")
    print("-" * 80)
