

# 五、我给你的最终架构建议（明确）


### 推荐架构（现实可落地）


```pgsql
LangGraph
  ├── Decision / Loop / Retry
  ├── State (query / nodes / keywords / iteration)
  ↓
LlamaIndex
  ├── Retriever
  ├── Reranker
  ├── Chunk expansion
  └── Index / Storage

```

**LangGraph 负责“怎么想”，
LlamaIndex 负责“去哪找”。**


# Q:
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

# Linux 文件系统中最基础的禁止字符
FORBIDDEN_CHARS_PATTERN = re.compile(r"[\x00]")


class SubProblemStructure(BaseModel):
    """
    子问题结构：二级列表，第一级为并行执行的问题组，第二级为串行执行的相关问题。
    """

    parallel_groups: List[List[str]] = Field(
        description="子问题列表，第一级列表表示可以并行执行的问题组，第二级列表表示需要串行执行的相关问题。"
        "例如：[[问题1], [问题2, 问题3]] 表示问题1可以独立执行，问题2和问题3需要串行执行且问题3依赖问题2的答案。"
        "如果只有一个问题，返回 [[问题]]。"
        "如果多个问题不相关，返回 [[问题1], [问题2], ...]。"
        "如果多个问题相关，返回 [[问题1, 问题2, ...]]，按逻辑顺序排列。",
        default_factory=list,
    )
    reasoning: str = Field(
        description="简要解释为什么这样分解问题，以及问题之间的关系。"
    )


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

    def generate(self, query: str) -&gt; SubProblemStructure:
        """Generate sub-problems from user query.

        Args:
            query: User query string

        Returns:
            SubProblemStructure containing parallel groups of sub-problems
        """
        # Create OpenAI client from LLM if it's OpenAI-compatible
        # Otherwise, use the LLM directly

        # Format prompt
        prompt = self._format_prompt(query)

        # Call LLM with structured output
        completion = self.client.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个问题分析助手，负责将用户的问题分解为子问题。"
                    "你需要判断问题数量、问题之间的关系（相关或不相关），并按照逻辑顺序组织问题。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=SubProblemStructure,
        )

        # Parse response
        response_content = completion.choices[0].message
        if not response_content.parsed:
            raise ValueError("Failed to parse sub-problem structure from LLM response")

        return response_content.parsed

    def _format_prompt(self, query: str) -&gt; str:
        """Format prompt for sub-problem generation.

        Args:
            query: User query string

        Returns:
            Formatted prompt string
        """
        return f"""请分析以下用户问题，并将其分解为子问题。

用户问题：
{query}

要求：
1. 如果只有一个问题，返回 [[问题]]
2. 如果有多个问题且问题之间不相关，返回 [[问题1], [问题2], ...]，每个问题独立执行
3. 如果有多个问题且问题之间相关，返回 [[问题1, 问题2, ...]]，按逻辑顺序排列，后续问题要依赖前面问题的答案
4. 不要删除问题中的细节

请以JSON格式输出结果，包含parallel_groups和reasoning字段。"""


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
    query1 = "如何安装viper"
    result1 = generator.generate(query1)
    print(f"Query: {query1}")
    print(f"Parallel groups: {result1.parallel_groups}")
    print(f"Reasoning: {result1.reasoning}")
    print("-" * 80)


这是我目前自定义的某一个tool，另外还有分析文档相关性，判断是否应该网络搜索，rag搜索，或者直接回答，以及本地文件搜索工具，支持grep关键字，扩展上下文的tool，这些tool如何迁移到langgraph中？
# A:
这个问题问得**非常到位**，而且你贴的这段 `SubProblemGenerator` 代码本身已经是**工程级 Tool 实现**，不是玩具。
我会按**“怎么迁移，而不是要不要迁移”**来回答。



# 一、先给你一个明确结论（不模糊）



**你现在这些“自定义 tool”，在 LangGraph 中不需要重写，也不需要改成 LangChain Tool。**


**它们最合适的迁移方式是：**
👉 **作为 LangGraph 的 Node（节点），而不是 Tool。**



这是一个非常重要的观念转变。



# 二、关键概念澄清：Tool \neq LangGraph Node


### 你现在的“tool”实际是什么？


以 `SubProblemGenerator` 为例，它是：


- 有明确输入（query）
- 有确定输出（结构化 Pydantic）
- 不做自由工具选择
- 不依赖 ReAct / tool calling

**本质是一个“确定性计算步骤”，而不是 LLM Tool。**


### 在 LangGraph 里最自然的归属


| 现在 | LangGraph |
| ---- | ---- |
| tool / helper class | **Node（状态变换函数）** |
| ctx.store | **State 字段** |
| Event | **State 更新 + Conditional Edge** |



# 三、迁移的总体原则（非常重要）


### ✅ 原则一：**业务逻辑 0 重写**


- `SubProblemGenerator.generate()`
- 文档相关性分析
- web / rag 判断
- grep / context expand

👉 **全部原样保留**



### ✅ 原则二：**只改“谁来调、怎么串”**


你只迁移：


- 控制流
- 状态流
- loop / branch


# 四、统一 State 设计（迁移的核心）


在 LangGraph 里，**你要做的第一件事不是写 node，而是定义 State。**


```python
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str

    # sub-problem
    sub_problems: Optional[list[list[str]]]
    sub_reasoning: Optional[str]

    # retrieval
    rag_nodes: Optional[list]
    web_nodes: Optional[list]
    local_nodes: Optional[list]

    # analysis
    relevance: Optional[str]   # relevant / partial / irrelevant
    completeness: Optional[str]

    # context
    expanded_nodes: Optional[list]
    keywords: Optional[list[str]]

    # control
    iteration: int
    need_web: bool
    need_rag: bool

    # final
    answer: Optional[str]

```


**这是迁移成败的 80%。**




# 五、SubProblemGenerator → LangGraph Node（示例）


### 原代码：完全不动


```python
generator = SubProblemGenerator(client=openai_client)

```

### LangGraph Node 包装


```python
def sub_problem_node(state: AgentState) -&gt; AgentState:
    result = generator.generate(state["query"])

    return {
        **state,
        "sub_problems": result.parallel_groups,
        "sub_reasoning": result.reasoning,
    }

```

**没有 Tool 定义，没有 decorator，没有 adapter。**



# 六、你提到的其它 Tool，如何一一映射


我直接给你一个**对照表**（非常实用）。



## 1️⃣ 文档相关性 / 完整性分析


### 你现在


- `analyze_relevance_and_completeness`
- 返回结构化结果

### LangGraph


```python
def relevance_judge_node(state: AgentState) -&gt; AgentState:
    result = analyze_relevance_and_completeness(
        state["query"], state["rag_nodes"]
    )
    return {
        **state,
        "relevance": result.relevance,
        "completeness": result.completeness,
    }

```

### 控制流


```python
graph.add_conditional_edges(
    "relevance_judge",
    lambda s: s["relevance"],
    {
        "complete": "final_answer",
        "irrelevant": "keyword_search",
        "partial": "expand_context",
    }
)

```


## 2️⃣ 是否 Web / RAG / Direct Answer 判断


### 你现在


- `direct_answer_check`
- `web_or_rag_check`

### LangGraph 做法


- **拆成两个 node**
- 用 conditional edge 连接

```python
def route_node(state):
    result = web_or_rag_generator.generate(state["query"])
    return {
        **state,
        "need_web": result.need_web_search,
        "need_rag": result.need_rag_search,
    }

```


## 3️⃣ 本地文件搜索（grep）


### 你现在


- `LocalFileSearchTool`

### LangGraph Node


```python
def local_search_node(state):
    nodes = local_file_search.search(
        state["keywords"] or state["query"]
    )
    return {**state, "local_nodes": nodes}

```


## 4️⃣ 扩展上下文（chunk 前后扩展）


### LangGraph Node


```python
def expand_context_node(state):
    expanded = expand_chunks(state["rag_nodes"])
    return {**state, "expanded_nodes": expanded}

```


## 5️⃣ 关键词生成 + 搜索 Loop


```python
def keyword_gen_node(state):
    keywords = keyword_extractor.extract(state["query"])
    return {**state, "keywords": keywords}

```

然后回到 `rag_retrieval`。



# 七、Loop 控制（你现在最痛的点）


LangGraph 原生支持：


```python
def should_continue(state):
    if state["iteration"] &gt;= 3:
        return "stop"
    return "continue"

graph.add_conditional_edges(
    "relevance_judge",
    should_continue,
    {
        "continue": "keyword_gen",
        "stop": "final_answer",
    }
)

```

**你不需要再在工具里关心“第几轮”。**



# 八、你完全不需要做的事情（重要）


❌ 不需要把这些改成 LangChain Tool
❌ 不需要写 `@tool` decorator
❌ 不需要让 LLM 自动选工具
❌ 不需要 Prompt 里暴露工具描述


你现在是**确定性编排，不是自由 Agent**。



# 九、迁移路径建议（非常具体）


### 第一阶段（1–2 天）


- 定义 `AgentState`
- 把 **SubProblemGenerator + relevance judge** 迁成 LangGraph

### 第二阶段


- 把 keyword / expand / rag loop 迁移
- 保留 LlamaIndex RAG

### 第三阶段


- 接 LangServe
- 对接 OpenWebUI


# 十、一句话总结（给你定心）



**你现在这些 tool，都是“高质量 Node 候选者”。**


**LangGraph 不要求你重写能力，只要求你把“谁在什么时候跑”说清楚。**




如果你愿意，下一步我可以：


1. **直接用你现有这些模块，给你写一个完整 LangGraph skeleton**
2. **帮你设计最小 State（防止过度设计）**
3. **指出哪些 tool 不该迁、哪些必须迁**
4. **把 SubProblem 并行执行映射到 LangGraph 并行节点**

你想先做哪一步？