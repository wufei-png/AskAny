# LangChain/LangGraph 迁移与组件解析
_Exported on 2025/12/21 at GMT+8 0:50:28 from Cursor (2.2.35)_

---

**User**

@dev_readme/from_llamaindex_to_langchain/from_llamaindex_to_langgraph.md 我想要做迁移，对于目前的能力，有哪些langchain，langgraph可以提供的能力做优化？
目前我观察到SummarizationMiddleware 可以在消息过长截断，请解释Middleware的工作原理
checkpointer功能的原理和使用介绍，在该项目中可以做什么
TodoListMiddleware 可以替代@askany/workflow/SubProblemGenerator.py 吗，还是他们的本质都是提示词工程，区别不大？
@dev_readme/langchain_future/roadmap.md:145 LocalFileSearchTool 可以在使用这里的@askany/workflow/LocalFileSearchTool.py 自定义逻辑一轮无效后 再用吗，如何实现.
一个个解释，你可以调用mcp工具查询，我接入了 name为docs-langchain

---

**Cursor**


查看相关文件，了解项目现状，然后逐一解答。



查询 LangChain 文档以获取准确信息：



查看项目中的 workflow 实现，以便给出更具体的建议：



基于文档和项目代码，逐一解答：

## 1. SummarizationMiddleware 的工作原理

`SummarizationMiddleware` 用于在消息过长时自动截断并压缩历史消息。

### 工作原理：

1. 触发条件：在每次模型调用前检查，支持：
   - 按 token 数：`trigger=("tokens", 4000)`
   - 按消息数：`trigger=("messages", 10)`
   - 组合条件：`trigger=[("tokens", 4000), ("messages", 10)]`

2. 压缩过程：
   - 使用指定的模型（通常是小模型如 `gpt-4o-mini`）对旧消息进行摘要
   - 用摘要替换原始旧消息
   - 保留最近的 N 条消息（`keep=("messages", 20)`）

3. 执行时机：在 `wrap_model_call` 阶段，每次调用模型前检查并处理

### 在你的项目中的应用：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",  # 使用小模型做摘要
            trigger=("tokens", 4000),  # 超过4000 tokens时触发
            keep=("messages", 20)  # 保留最近20条消息
        )
    ],
    checkpointer=checkpointer,
)
```

优势：
- 自动管理长对话上下文
- 降低 token 消耗
- 保留关键信息（最近消息 + 历史摘要）

---

## 2. Checkpointer 功能的原理和使用介绍

### 工作原理：

1. 状态快照：在每个 super-step（节点执行后）保存完整状态快照，包括：
   - 所有 state 字段的值
   - 配置信息（`thread_id` 等）
   - 元数据
   - 下一步要执行的节点

2. Thread 关联：每个 checkpoint 关联一个 `thread_id`，支持多会话隔离

3. 状态恢复：可以从任意 checkpoint 恢复执行

### 实现类型：

- `InMemorySaver`：内存存储，适合开发测试
- `SqliteSaver`：SQLite 持久化，适合本地部署
- `PostgresSaver`：PostgreSQL 持久化，适合生产环境

### 在你的项目中的应用：

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

# 使用 PostgreSQL 持久化（适合生产环境）
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# 或者使用 SQLite（适合本地）
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")

# 编译图时传入 checkpointer
graph = workflow.compile(checkpointer=checkpointer)

# 使用 thread_id 管理会话
config: RunnableConfig = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke({"messages": "..."}, config)
```

在你的项目中的价值：
1. 会话持久化：用户对话状态可跨请求保持
2. 错误恢复：失败后可从 checkpoint 恢复
3. 调试：可查看任意步骤的状态
4. 人机交互：支持暂停等待人工输入后继续

示例：结合 SummarizationMiddleware 使用：

```python
# 状态压缩 + 持久化
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=PostgresSaver.from_conn_string("postgresql://..."),
)
```

---

## 3. TodoListMiddleware vs SubProblemGenerator

### TodoListMiddleware 的特点：

- 提供 `write_todos` 工具，由模型在运行时决定是否使用
- 动态规划：可在执行过程中更新任务列表
- 面向工具调用：模型自主决定何时创建/更新 todo list
- 适合：需要动态调整计划的复杂任务

### SubProblemGenerator 的特点：

- 显式调用：在特定节点主动调用，返回结构化结果
- 静态分解：一次性将问题分解为子问题结构
- 结构化输出：返回 `parallel_groups`（并行/串行组）
- 适合：需要明确控制问题分解流程的场景

### 对比分析：

| 特性 | TodoListMiddleware | SubProblemGenerator |
|------|-------------------|---------------------|
| 调用方式 | 模型自主决定（工具调用） | 显式节点调用 |
| 输出格式 | 动态 todo list | 结构化 `parallel_groups` |
| 控制粒度 | 模型控制 | 代码控制 |
| 适用场景 | 开放式任务规划 | 需要并行/串行执行的问题分解 |

### 建议：

不建议完全替代，原因：
1. 你的 `SubProblemGenerator` 返回 `parallel_groups`，明确区分并行/串行，更适合你的工作流
2. 你的工作流需要精确控制问题分解的时机和方式
3. `TodoListMiddleware` 更适合开放式任务规划，而非结构化问题分解

可以结合使用：
- 在需要动态调整计划时，让模型使用 `write_todos`
- 在需要结构化分解时，继续使用 `SubProblemGenerator`

---

## 4. LocalFileSearchTool 与 FileSearchMiddleware 的结合使用

### 实现方案：

可以在自定义工具一轮无效后，再使用 LangChain 的 `FileSearchMiddleware`。实现方式：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import FileSearchMiddleware
from langchain.tools import tool
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool

# 1. 将你的 LocalFileSearchTool 包装为 LangChain Tool
@tool
def local_file_search(keywords: str) -> str:
    """使用自定义逻辑搜索本地文件。
    
    Args:
        keywords: 逗号分隔的关键词列表
        
    Returns:
        搜索结果
    """
    tool_instance = LocalFileSearchTool()
    keyword_list = [k.strip() for k in keywords.split(",")]
    results = tool_instance.search_by_keywords(
        keyword_list,
        expand_mode="markdown",
        expand_ratio=5
    )
    
    # 格式化结果
    formatted_results = []
    for keyword, matches in results.items():
        for match in matches:
            formatted_results.append(
                f"文件: {match['file_path']}\n"
                f"行号: {match['start_line']}-{match['end_line']}\n"
                f"内容: {match['content']}\n"
            )
    
    return "\n---\n".join(formatted_results) if formatted_results else "未找到相关文件"

# 2. 创建带 fallback 的 agent
agent = create_agent(
    model="gpt-4o",
    tools=[local_file_search],  # 优先使用自定义工具
    middleware=[
        FileSearchMiddleware(
            # 当自定义工具无效时，使用内置的文件搜索
            # 可以配置搜索路径等
        )
    ],
)
```

### 更优雅的实现：自定义 Middleware

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool

class LocalFileSearchFallbackMiddleware(AgentMiddleware):
    """自定义文件搜索中间件：优先使用 LocalFileSearchTool，失败后使用 FileSearchMiddleware"""
    
    def __init__(self, base_path: str = None):
        self.local_tool = LocalFileSearchTool(base_path=base_path)
        self.fallback_middleware = FileSearchMiddleware()
    
    def wrap_tool_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 检查是否是文件搜索相关的工具调用
        if self._is_file_search_request(request):
            # 先尝试使用自定义逻辑
            try:
                result = self._try_local_search(request)
                if result and result.get("found"):
                    # 自定义工具成功，直接返回
                    return ModelResponse(
                        content=result["content"],
                        tool_calls=[]  # 标记为已处理
                    )
            except Exception as e:
                print(f"LocalFileSearchTool failed: {e}")
            
            # 自定义工具失败，fallback 到 FileSearchMiddleware
            return self.fallback_middleware.wrap_tool_call(request, handler)
        
        # 其他工具调用正常处理
        return handler(request)
    
    def _is_file_search_request(self, request: ModelRequest) -> bool:
        # 判断是否是文件搜索请求
        # 可以根据 tool_calls 或其他特征判断
        return False  # 实现你的判断逻辑
    
    def _try_local_search(self, request: ModelRequest) -> dict:
        # 尝试使用 LocalFileSearchTool
        # 解析请求中的关键词等
        # 返回 {"found": True/False, "content": "..."}
        return {"found": False, "content": ""}

# 使用
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        LocalFileSearchFallbackMiddleware(base_path="/path/to/docs")
    ],
)
```

### 推荐方案：在节点中实现 fallback

在你的 LangGraph 节点中实现：

```python
def file_search_node(state: AgentState) -> AgentState:
    """文件搜索节点：优先使用 LocalFileSearchTool，失败后使用 FileSearchMiddleware"""
    query = state.get("query", "")
    keywords = extract_keywords(query)  # 你的关键词提取逻辑
    
    # 第一轮：使用自定义 LocalFileSearchTool
    local_tool = LocalFileSearchTool()
    results = local_tool.search_by_keywords(keywords)
    
    if not results or all(not matches for matches in results.values()):
        # 第一轮无效，使用 FileSearchMiddleware（通过工具调用）
        # 这里可以调用 agent 的 file_search 工具
        fallback_results = call_file_search_middleware(query)
        return {
            **state,
            "local_nodes": fallback_results,
            "search_method": "fallback"
        }
    
    return {
        **state,
        "local_nodes": results,
        "search_method": "local"
    }
```

---

## 总结与建议

### 迁移优化建议：

1. 使用 `SummarizationMiddleware` 管理长对话上下文
2. 使用 `PostgresSaver` 或 `SqliteSaver` 实现状态持久化
3. 保留 `SubProblemGenerator`，不替换为 `TodoListMiddleware`
4. 在节点中实现 LocalFileSearchTool → FileSearchMiddleware 的 fallback 逻辑

### 架构示例：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)
```

需要我帮你实现具体的代码吗？

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",  # 使用小模型做摘要
            trigger=("tokens", 4000),  # 超过4000 tokens时触发
            keep=("messages", 20)  # 保留最近20条消息
        )
    ],
    checkpointer=checkpointer,
)
```

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

# 使用 PostgreSQL 持久化（适合生产环境）
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# 或者使用 SQLite（适合本地）
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")

# 编译图时传入 checkpointer
graph = workflow.compile(checkpointer=checkpointer)

# 使用 thread_id 管理会话
config: RunnableConfig = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke({"messages": "..."}, config)
```

```python
# 状态压缩 + 持久化
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=PostgresSaver.from_conn_string("postgresql://..."),
)
```

```python
from langchain.agents import create_agent
from langchain.agents.middleware import FileSearchMiddleware
from langchain.tools import tool
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool

# 1. 将你的 LocalFileSearchTool 包装为 LangChain Tool
@tool
def local_file_search(keywords: str) -> str:
    """使用自定义逻辑搜索本地文件。
    
    Args:
        keywords: 逗号分隔的关键词列表
        
    Returns:
        搜索结果
    """
    tool_instance = LocalFileSearchTool()
    keyword_list = [k.strip() for k in keywords.split(",")]
    results = tool_instance.search_by_keywords(
        keyword_list,
        expand_mode="markdown",
        expand_ratio=5
    )
    
    # 格式化结果
    formatted_results = []
    for keyword, matches in results.items():
        for match in matches:
            formatted_results.append(
                f"文件: {match['file_path']}\n"
                f"行号: {match['start_line']}-{match['end_line']}\n"
                f"内容: {match['content']}\n"
            )
    
    return "\n---\n".join(formatted_results) if formatted_results else "未找到相关文件"

# 2. 创建带 fallback 的 agent
agent = create_agent(
    model="gpt-4o",
    tools=[local_file_search],  # 优先使用自定义工具
    middleware=[
        FileSearchMiddleware(
            # 当自定义工具无效时，使用内置的文件搜索
            # 可以配置搜索路径等
        )
    ],
)
```

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool

class LocalFileSearchFallbackMiddleware(AgentMiddleware):
    """自定义文件搜索中间件：优先使用 LocalFileSearchTool，失败后使用 FileSearchMiddleware"""
    
    def __init__(self, base_path: str = None):
        self.local_tool = LocalFileSearchTool(base_path=base_path)
        self.fallback_middleware = FileSearchMiddleware()
    
    def wrap_tool_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 检查是否是文件搜索相关的工具调用
        if self._is_file_search_request(request):
            # 先尝试使用自定义逻辑
            try:
                result = self._try_local_search(request)
                if result and result.get("found"):
                    # 自定义工具成功，直接返回
                    return ModelResponse(
                        content=result["content"],
                        tool_calls=[]  # 标记为已处理
                    )
            except Exception as e:
                print(f"LocalFileSearchTool failed: {e}")
            
            # 自定义工具失败，fallback 到 FileSearchMiddleware
            return self.fallback_middleware.wrap_tool_call(request, handler)
        
        # 其他工具调用正常处理
        return handler(request)
    
    def _is_file_search_request(self, request: ModelRequest) -> bool:
        # 判断是否是文件搜索请求
        # 可以根据 tool_calls 或其他特征判断
        return False  # 实现你的判断逻辑
    
    def _try_local_search(self, request: ModelRequest) -> dict:
        # 尝试使用 LocalFileSearchTool
        # 解析请求中的关键词等
        # 返回 {"found": True/False, "content": "..."}
        return {"found": False, "content": ""}

# 使用
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        LocalFileSearchFallbackMiddleware(base_path="/path/to/docs")
    ],
)
```

```python
def file_search_node(state: AgentState) -> AgentState:
    """文件搜索节点：优先使用 LocalFileSearchTool，失败后使用 FileSearchMiddleware"""
    query = state.get("query", "")
    keywords = extract_keywords(query)  # 你的关键词提取逻辑
    
    # 第一轮：使用自定义 LocalFileSearchTool
    local_tool = LocalFileSearchTool()
    results = local_tool.search_by_keywords(keywords)
    
    if not results or all(not matches for matches in results.values()):
        # 第一轮无效，使用 FileSearchMiddleware（通过工具调用）
        # 这里可以调用 agent 的 file_search 工具
        fallback_results = call_file_search_middleware(query)
        return {
            **state,
            "local_nodes": fallback_results,
            "search_method": "fallback"
        }
    
    return {
        **state,
        "local_nodes": results,
        "search_method": "local"
    }
```

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)
```

---

**User**

def file_search_node(state: AgentState) -> AgentState:
    """文件搜索节点：优先使用 LocalFileSearchTool，失败后使用 FileSearchMiddleware"""
    query = state.get("query", "")
    keywords = extract_keywords(query)  # 你的关键词提取逻辑
    
    # 第一轮：使用自定义 LocalFileSearchTool
    local_tool = LocalFileSearchTool()
    results = local_tool.search_by_keywords(keywords)
    
    if not results or all(not matches for matches in results.values()):
        # 第一轮无效，使用 FileSearchMiddleware（通过工具调用）
        # 这里可以调用 agent 的 file_search 工具
        fallback_results = call_file_search_middleware(query)
        return {
            **state,
            "local_nodes": fallback_results,
            "search_method": "fallback"
        }
    
    return {
        **state,
        "local_nodes": results,
        "search_method": "local"
    }
节点每次运行都要初始化LocalFileSearchTool吗

---

**Cursor**


检查 `LocalFileSearchTool` 的初始化开销，并给出优化建议：



分析 `LocalFileSearchTool` 的初始化开销，并提供优化方案：



问题分析：`LocalFileSearchTool` 的初始化开销较大，因为 `_find_markdown_files()` 需要遍历整个目录树查找所有 markdown 文件。每次节点运行时都初始化会带来性能问题。

优化方案：在 LangGraph 中共享工具实例，避免重复初始化。推荐方案如下：

## 方案 1：使用闭包/工厂函数（推荐）

在编译图时创建共享实例：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from askany.config import settings

class AgentState(TypedDict):
    query: str
    local_nodes: Optional[dict]
    search_method: Optional[str]
    # ... 其他字段

def create_file_search_node(local_file_search_tool: LocalFileSearchTool):
    """创建文件搜索节点，使用共享的工具实例"""
    
    def file_search_node(state: AgentState) -> AgentState:
        """文件搜索节点：优先使用 LocalFileSearchTool，失败后使用 FileSearchMiddleware"""
        query = state.get("query", "")
        keywords = extract_keywords(query)  # 你的关键词提取逻辑
        
        # 使用共享的工具实例（不需要每次初始化）
        results = local_file_search_tool.search_by_keywords(keywords)
        
        if not results or all(not matches for matches in results.values()):
            # 第一轮无效，使用 FileSearchMiddleware
            fallback_results = call_file_search_middleware(query)
            return {
                **state,
                "local_nodes": fallback_results,
                "search_method": "fallback"
            }
        
        return {
            **state,
            "local_nodes": results,
            "search_method": "local"
        }
    
    return file_search_node

# 在创建图时初始化一次
def create_workflow():
    # 只初始化一次
    local_file_search_tool = LocalFileSearchTool(
        base_path=settings.local_file_search_dir
    )
    
    # 创建节点时传入共享实例
    file_search_node = create_file_search_node(local_file_search_tool)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("file_search", file_search_node)
    # ... 添加其他节点和边
    
    return workflow.compile()
```

## 方案 2：使用模块级单例（简单但不够灵活）

```python
# 在模块级别创建单例
_local_file_search_tool = None

def get_local_file_search_tool(base_path: Optional[str] = None) -> LocalFileSearchTool:
    """获取 LocalFileSearchTool 单例"""
    global _local_file_search_tool
    if _local_file_search_tool is None:
        _local_file_search_tool = LocalFileSearchTool(base_path=base_path)
    return _local_file_search_tool

def file_search_node(state: AgentState) -> AgentState:
    """文件搜索节点：使用单例工具"""
    query = state.get("query", "")
    keywords = extract_keywords(query)
    
    # 使用单例，不会重复初始化
    local_tool = get_local_file_search_tool()
    results = local_tool.search_by_keywords(keywords)
    
    # ... 后续逻辑
```

## 方案 3：使用 LangGraph 的 Context（最优雅）

通过 `RunnableConfig` 传递共享资源：

```python
from langchain_core.runnables import RunnableConfig
from typing import Any

class WorkflowContext:
    """工作流上下文，存储共享资源"""
    def __init__(self):
        self.local_file_search_tool = LocalFileSearchTool(
            base_path=settings.local_file_search_dir
        )
        # 可以存储其他共享资源
        self.sub_problem_generator = SubProblemGenerator()
        # ...

# 全局上下文（或从配置中获取）
_workflow_context = None

def get_workflow_context() -> WorkflowContext:
    """获取工作流上下文"""
    global _workflow_context
    if _workflow_context is None:
        _workflow_context = WorkflowContext()
    return _workflow_context

def file_search_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """文件搜索节点：从上下文获取工具"""
    query = state.get("query", "")
    keywords = extract_keywords(query)
    
    # 从上下文获取共享工具实例
    context = get_workflow_context()
    results = context.local_file_search_tool.search_by_keywords(keywords)
    
    # ... 后续逻辑
```

## 方案 4：使用类封装（最符合 OOP 风格）

```python
class WorkflowNodes:
    """封装所有节点，共享资源作为实例变量"""
    
    def __init__(self):
        # 初始化一次，所有节点共享
        self.local_file_search_tool = LocalFileSearchTool(
            base_path=settings.local_file_search_dir
        )
        self.sub_problem_generator = SubProblemGenerator()
        # ... 其他共享资源
    
    def file_search_node(self, state: AgentState) -> AgentState:
        """文件搜索节点"""
        query = state.get("query", "")
        keywords = extract_keywords(query)
        
        # 使用实例变量，不需要每次初始化
        results = self.local_file_search_tool.search_by_keywords(keywords)
        
        if not results or all(not matches for matches in results.values()):
            fallback_results = call_file_search_middleware(query)
            return {
                **state,
                "local_nodes": fallback_results,
                "search_method": "fallback"
            }
        
        return {
            **state,
            "local_nodes": results,
            "search_method": "local"
        }
    
    def sub_problem_node(self, state: AgentState) -> AgentState:
        """子问题生成节点"""
        result = self.sub_problem_generator.generate(state["query"])
        return {
            **state,
            "sub_problems": result.parallel_groups,
        }
    # ... 其他节点

# 使用
nodes = WorkflowNodes()

workflow = StateGraph(AgentState)
workflow.add_node("file_search", nodes.file_search_node)
workflow.add_node("sub_problem", nodes.sub_problem_node)
# ...
```

## 推荐方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 方案1：闭包 | 灵活，资源明确 | 需要显式传递 | 资源明确且固定的场景 |
| 方案2：单例 | 简单直接 | 不够灵活，难以测试 | 简单项目 |
| 方案3：Context | 符合 LangGraph 设计 | 需要额外抽象 | 复杂工作流 |
| 方案4：类封装 | 符合 OOP，易维护 | 需要类实例 | 大型项目（推荐） |

## 最终推荐实现

结合你的项目结构，推荐使用方案 4（类封装），与现有 `workflow_llamaindex.py` 的风格一致：

```python
from langgraph.graph import StateGraph
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from askany.workflow.SubProblemGenerator import SubProblemGenerator
from askany.config import settings

class LangGraphWorkflow:
    """LangGraph 工作流，封装所有节点和共享资源"""
    
    def __init__(self, router, llm, base_path: Optional[str] = None):
        """初始化工作流，所有资源只初始化一次"""
        self.router = router
        self.llm = llm
        
        # 共享资源：只初始化一次
        if base_path is None:
            base_path = settings.local_file_search_dir
        
        self.local_file_search_tool = LocalFileSearchTool(base_path=base_path)
        self.sub_problem_generator = SubProblemGenerator()
        self.direct_answer_generator = DirectAnswerGenerator()
        self.web_or_rag_generator = WebOrRagAnswerGenerator()
        # ... 其他共享资源
    
    def file_search_node(self, state: AgentState) -> AgentState:
        """文件搜索节点：使用共享的工具实例"""
        query = state.get("query", "")
        keywords = extract_keywords(query)
        
        # 使用共享实例，不会重复初始化
        results = self.local_file_search_tool.search_by_keywords(keywords)
        
        if not results or all(not matches for matches in results.values()):
            fallback_results = call_file_search_middleware(query)
            return {
                **state,
                "local_nodes": fallback_results,
                "search_method": "fallback"
            }
        
        return {
            **state,
            "local_nodes": results,
            "search_method": "local"
        }
    
    def build_graph(self) -> StateGraph:
        """构建并编译图"""
        workflow = StateGraph(AgentState)
        
        # 添加节点（使用绑定方法，自动访问 self）
        workflow.add_node("file_search", self.file_search_node)
        workflow.add_node("sub_problem", self.sub_problem_node)
        # ... 其他节点
        
        # 添加边
        workflow.add_edge(START, "file_search")
        # ...
        
        return workflow.compile()

# 使用
workflow_instance = LangGraphWorkflow(router, llm)
graph = workflow_instance.build_graph()
```

这样，`LocalFileSearchTool` 只会在 `LangGraphWorkflow.__init__()` 时初始化一次，所有节点共享同一个实例，避免重复的文件系统遍历。

需要我帮你实现完整的迁移代码吗？

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from askany.config import settings

class AgentState(TypedDict):
    query: str
    local_nodes: Optional[dict]
    search_method: Optional[str]
    # ... 其他字段

def create_file_search_node(local_file_search_tool: LocalFileSearchTool):
    """创建文件搜索节点，使用共享的工具实例"""
    
    def file_search_node(state: AgentState) -> AgentState:
        """文件搜索节点：优先使用 LocalFileSearchTool，失败后使用 FileSearchMiddleware"""
        query = state.get("query", "")
        keywords = extract_keywords(query)  # 你的关键词提取逻辑
        
        # 使用共享的工具实例（不需要每次初始化）
        results = local_file_search_tool.search_by_keywords(keywords)
        
        if not results or all(not matches for matches in results.values()):
            # 第一轮无效，使用 FileSearchMiddleware
            fallback_results = call_file_search_middleware(query)
            return {
                **state,
                "local_nodes": fallback_results,
                "search_method": "fallback"
            }
        
        return {
            **state,
            "local_nodes": results,
            "search_method": "local"
        }
    
    return file_search_node

# 在创建图时初始化一次
def create_workflow():
    # 只初始化一次
    local_file_search_tool = LocalFileSearchTool(
        base_path=settings.local_file_search_dir
    )
    
    # 创建节点时传入共享实例
    file_search_node = create_file_search_node(local_file_search_tool)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("file_search", file_search_node)
    # ... 添加其他节点和边
    
    return workflow.compile()
```

```python
# 在模块级别创建单例
_local_file_search_tool = None

def get_local_file_search_tool(base_path: Optional[str] = None) -> LocalFileSearchTool:
    """获取 LocalFileSearchTool 单例"""
    global _local_file_search_tool
    if _local_file_search_tool is None:
        _local_file_search_tool = LocalFileSearchTool(base_path=base_path)
    return _local_file_search_tool

def file_search_node(state: AgentState) -> AgentState:
    """文件搜索节点：使用单例工具"""
    query = state.get("query", "")
    keywords = extract_keywords(query)
    
    # 使用单例，不会重复初始化
    local_tool = get_local_file_search_tool()
    results = local_tool.search_by_keywords(keywords)
    
    # ... 后续逻辑
```

```python
from langchain_core.runnables import RunnableConfig
from typing import Any

class WorkflowContext:
    """工作流上下文，存储共享资源"""
    def __init__(self):
        self.local_file_search_tool = LocalFileSearchTool(
            base_path=settings.local_file_search_dir
        )
        # 可以存储其他共享资源
        self.sub_problem_generator = SubProblemGenerator()
        # ...

# 全局上下文（或从配置中获取）
_workflow_context = None

def get_workflow_context() -> WorkflowContext:
    """获取工作流上下文"""
    global _workflow_context
    if _workflow_context is None:
        _workflow_context = WorkflowContext()
    return _workflow_context

def file_search_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """文件搜索节点：从上下文获取工具"""
    query = state.get("query", "")
    keywords = extract_keywords(query)
    
    # 从上下文获取共享工具实例
    context = get_workflow_context()
    results = context.local_file_search_tool.search_by_keywords(keywords)
    
    # ... 后续逻辑
```

```python
class WorkflowNodes:
    """封装所有节点，共享资源作为实例变量"""
    
    def __init__(self):
        # 初始化一次，所有节点共享
        self.local_file_search_tool = LocalFileSearchTool(
            base_path=settings.local_file_search_dir
        )
        self.sub_problem_generator = SubProblemGenerator()
        # ... 其他共享资源
    
    def file_search_node(self, state: AgentState) -> AgentState:
        """文件搜索节点"""
        query = state.get("query", "")
        keywords = extract_keywords(query)
        
        # 使用实例变量，不需要每次初始化
        results = self.local_file_search_tool.search_by_keywords(keywords)
        
        if not results or all(not matches for matches in results.values()):
            fallback_results = call_file_search_middleware(query)
            return {
                **state,
                "local_nodes": fallback_results,
                "search_method": "fallback"
            }
        
        return {
            **state,
            "local_nodes": results,
            "search_method": "local"
        }
    
    def sub_problem_node(self, state: AgentState) -> AgentState:
        """子问题生成节点"""
        result = self.sub_problem_generator.generate(state["query"])
        return {
            **state,
            "sub_problems": result.parallel_groups,
        }
    # ... 其他节点

# 使用
nodes = WorkflowNodes()

workflow = StateGraph(AgentState)
workflow.add_node("file_search", nodes.file_search_node)
workflow.add_node("sub_problem", nodes.sub_problem_node)
# ...
```

```python
from langgraph.graph import StateGraph
from askany.workflow.LocalFileSearchTool import LocalFileSearchTool
from askany.workflow.SubProblemGenerator import SubProblemGenerator
from askany.config import settings

class LangGraphWorkflow:
    """LangGraph 工作流，封装所有节点和共享资源"""
    
    def __init__(self, router, llm, base_path: Optional[str] = None):
        """初始化工作流，所有资源只初始化一次"""
        self.router = router
        self.llm = llm
        
        # 共享资源：只初始化一次
        if base_path is None:
            base_path = settings.local_file_search_dir
        
        self.local_file_search_tool = LocalFileSearchTool(base_path=base_path)
        self.sub_problem_generator = SubProblemGenerator()
        self.direct_answer_generator = DirectAnswerGenerator()
        self.web_or_rag_generator = WebOrRagAnswerGenerator()
        # ... 其他共享资源
    
    def file_search_node(self, state: AgentState) -> AgentState:
        """文件搜索节点：使用共享的工具实例"""
        query = state.get("query", "")
        keywords = extract_keywords(query)
        
        # 使用共享实例，不会重复初始化
        results = self.local_file_search_tool.search_by_keywords(keywords)
        
        if not results or all(not matches for matches in results.values()):
            fallback_results = call_file_search_middleware(query)
            return {
                **state,
                "local_nodes": fallback_results,
                "search_method": "fallback"
            }
        
        return {
            **state,
            "local_nodes": results,
            "search_method": "local"
        }
    
    def build_graph(self) -> StateGraph:
        """构建并编译图"""
        workflow = StateGraph(AgentState)
        
        # 添加节点（使用绑定方法，自动访问 self）
        workflow.add_node("file_search", self.file_search_node)
        workflow.add_node("sub_problem", self.sub_problem_node)
        # ... 其他节点
        
        # 添加边
        workflow.add_edge(START, "file_search")
        # ...
        
        return workflow.compile()

# 使用
workflow_instance = LangGraphWorkflow(router, llm)
graph = workflow_instance.build_graph()
```

