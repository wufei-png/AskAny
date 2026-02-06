# LangChain/LangGraph 优化特性清单

基于文档调研和项目分析，以下是除了 `LocalFileSearchTool` 和 `SummarizationMiddleware` 之外，可以提升效果或功能性的 LangChain/LangGraph 特性。

## 目录
1. [错误处理和重试](#错误处理和重试)
2. [性能优化](#性能优化)
3. [安全和隐私](#安全和隐私)
4. [流式处理](#流式处理)
5. [并行处理](#并行处理)
6. [缓存机制](#缓存机制)
7. [监控和可观测性](#监控和可观测性)
8. [动态路由和决策](#动态路由和决策)
9. [人机交互](#人机交互)
10. [状态管理优化](#状态管理优化)

---

## 1. 错误处理和重试

### 1.1 ToolRetryMiddleware ⚠️ 不适用于你的架构

**功能**：自动重试失败的工具调用，处理网络错误、超时等临时故障。

**⚠️ 重要说明**：
`ToolRetryMiddleware` 是为 **Agent 模式**设计的（LLM 自主选择工具），**不适用于你的架构**！

**为什么不适用**：
- `create_agent` 会 `bind_tools`，所有工具定义会污染提示词
- 你的节点是**确定性的**，显式调用工具（如 `web_search_tool.search(query)`），不是让 LLM 选择工具
- 在分析相关性阶段，如果绑定了工具，会污染提示词，LLM 可能误调用工具

**正确的实现方式**：
请参考 📄 [tool_retry_correct_implementation.md](./tool_retry_correct_implementation.md)

**推荐方案：在工具类内部实现重试**：
```python
# askany/workflow/WebSearchTool.py
class WebSearchTool:
    def search(self, query: str) -> List[NodeWithScore]:
        """搜索网络（带自动重试）"""
        last_exception = None
        delay = 1.0
        
        for attempt in range(3):  # max_retries
            try:
                return self._search_impl(query)
            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < 2:
                    logger.warning(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2.0  # backoff_factor
                else:
                    logger.error("Failed after 3 attempts")
                    return []  # 或 raise
        return []
```

**在你的项目中的应用**：
- ✅ 在 `WebSearchTool.search()` 内部实现重试（你已经在做）
- ✅ 在 `LocalFileSearchTool` 内部实现重试
- ✅ 在 RAG 检索内部实现重试
- ❌ **不要**使用 `create_agent` + `ToolRetryMiddleware`

### 1.2 ModelRetryMiddleware ⚠️ 需要适配你的架构

**功能**：自动重试失败的模型调用，处理 API 限流、临时错误等。

**适用场景**：
- LLM API 调用失败（如 OpenAI API 限流）
- vLLM 服务临时不可用
- 网络抖动导致的模型调用失败

**⚠️ 注意**：
`ModelRetryMiddleware` 是为 `create_agent` 设计的。如果你的节点直接调用 LLM（不使用 `create_agent`），需要适配。

**方案 A：使用 wrap_model_call（推荐）**
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI
import time

@wrap_model_call
def model_retry_middleware(request: ModelRequest, handler) -> ModelResponse:
    """自定义模型重试中间件"""
    max_retries = 3
    backoff_factor = 2.0
    initial_delay = 1.0
    retry_on = (ConnectionError, TimeoutError, APIError)
    
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return handler(request)
        except retry_on as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(f"Model call failed, retrying in {delay}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error("Model call failed after all retries")
                raise
    
    if last_exception:
        raise last_exception

# 在节点中使用
def analyze_relevance_node(state: AgentState) -> AgentState:
    """分析相关性节点（带重试）"""
    model = ChatOpenAI(model="gpt-4o")
    model_with_retry = model.with_config({"middleware": [model_retry_middleware]})
    # 使用 model_with_retry 调用
```

**方案 B：在节点函数内部实现重试（更简单）**
```python
def analyze_relevance_and_completeness(query, nodes, client):
    """分析相关性（带重试）"""
    max_retries = 3
    delay = 1.0
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.parse(...)
            return completion.choices[0].message.parsed
        except (ConnectionError, TimeoutError, APIError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"LLM call failed, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error("LLM call failed after all retries")
                raise
```

**在你的项目中的应用**：
- ✅ 在 `analyze_relevance_and_completeness` 中添加重试
- ✅ 在 `DirectAnswerGenerator.generate` 中添加重试
- ✅ 在 `SubProblemGenerator.generate` 中添加重试
- ✅ 可以替换 `AutoRetryVLLM` 中的手动重试逻辑

### 1.3 LangGraph 节点级重试策略
**功能**：在 LangGraph 节点级别配置重试策略。

**实现示例**：
```python
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node(
    "rag_retrieval",
    rag_retrieval_node,
    retry_policy={
        "max_attempts": 3,
        "initial_delay": 1.0,
        "backoff_factor": 2.0,
    }
)
```

---

## 2. 性能优化

### 2.1 Batch 批处理
**功能**：批量处理多个独立请求，提高吞吐量和降低成本。

**适用场景**：
- 并行处理多个子问题
- 批量检索多个文档
- 批量分析多个查询的相关性

**实现示例**：
```python
# 批量处理多个查询
queries = [
    "如何更新组件？",
    "如何查看日志？"
]

# 批量调用模型
responses = model.batch(queries)
for response in responses:
    print(response.content)
```

**在你的项目中的应用**：
- 在 `process_parallel_group` 中使用 batch 处理并行子问题
- 批量分析多个文档的相关性

### 2.2 Prompt Caching（提示词缓存）
**功能**：缓存重复的提示词部分，降低延迟和成本。

**适用场景**：
- 系统提示词重复使用
- 文档模板重复使用
- 固定格式的提示词

**实现示例**：
```python
from langchain_openai import ChatOpenAI

# OpenAI 自动缓存（隐式）
model = ChatOpenAI(model="gpt-4o")

# Anthropic 显式缓存
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    middleware=[AnthropicPromptCachingMiddleware()]
)

# 使用 prompt_cache_key 标记可缓存部分
messages = [
    {"role": "system", "content": "你是一个运维助手"},  # 可缓存
    {"role": "user", "content": query}  # 不缓存
]
```

**在你的项目中的应用**：
- 缓存 `DirectAnswerGenerator` 和 `WebOrRagAnswerGenerator` 的系统提示词
- 缓存 `SubProblemGenerator` 的提示词模板

### 2.3 动态模型选择
**功能**：根据任务复杂度动态选择模型（小模型处理简单任务，大模型处理复杂任务）。

**实现示例**：
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_openai import ChatOpenAI

basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂度选择模型"""
    message_count = len(request.state["messages"])
    query_complexity = estimate_complexity(request.state["messages"][-1].content)
    
    if message_count > 10 or query_complexity > 0.7:
        model = advanced_model
    else:
        model = basic_model
    
    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,
    tools=[...],
    middleware=[dynamic_model_selection]
)
```

**在你的项目中的应用**：
- 简单问题使用 `gpt-4o-mini`，复杂问题使用 `gpt-4o`
- 根据查询类型选择不同模型

---

## 3. 安全和隐私

### 3.1 PIIMiddleware（个人身份信息中间件）
**功能**：自动检测和脱敏敏感信息（邮箱、信用卡号、IP地址等）。

**适用场景**：
- 处理用户输入中的敏感信息
- 日志记录前脱敏
- 符合隐私法规要求

**实现示例**：
```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        PIIMiddleware(
            strategy="redact",  # 或 "block"
            pii_types=["email", "credit_card", "ip_address"],
        ),
    ],
)
```

**在你的项目中的应用**：
- 在 API 入口处添加 PII 检测
- 日志记录前脱敏敏感信息

---

## 4. 流式处理

### 4.1 Streaming 流式响应
**功能**：实时流式返回结果，提升用户体验。

**实现示例**：
```python
from langgraph.graph import StateGraph

# 流式调用
for chunk in graph.stream({"messages": "..."}, stream_mode="updates"):
    print(chunk)

# 流式事件
for event in graph.stream_events({"messages": "..."}, version="v2"):
    print(event)
```

**在你的项目中的应用**：
- LangServe 接口支持流式返回
- 实时显示检索进度
- 实时显示生成过程

### 4.2 Stream Mode 配置
**功能**：控制流式输出的粒度。

**选项**：
- `"values"`: 每次状态更新
- `"messages"`: 每次消息更新
- `"updates"`: 每个节点更新

---

## 5. 并行处理

### 5.1 LangGraph 并行节点
**功能**：在 LangGraph 中实现真正的并行执行。

**实现示例**：
```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgentState)

# 添加并行节点
workflow.add_node("keyword_search", keyword_search_node)
workflow.add_node("hypothetical_search", hypothetical_search_node)
workflow.add_node("web_search", web_search_node)

# 从 START 并行执行
workflow.add_edge(START, "keyword_search")
workflow.add_edge(START, "hypothetical_search")
workflow.add_edge(START, "web_search")

# 聚合节点等待所有并行节点完成
workflow.add_node("aggregate", aggregate_node)
workflow.add_edge("keyword_search", "aggregate")
workflow.add_edge("hypothetical_search", "aggregate")
workflow.add_edge("web_search", "aggregate")
```

**在你的项目中的应用**：
- 替换 `_concurrent_search` 中的 `asyncio.gather`
- 实现真正的并行检索（关键词检索 + 假设答案检索）

---

## 6. 缓存机制

### 6.1 Semantic Cache（语义缓存）
**功能**：基于语义相似度缓存查询结果，避免重复计算。

**适用场景**：
- 相似查询的缓存
- 降低重复检索成本
- 提高响应速度

**实现示例**：
```python
from langchain.cache import SemanticCache
from langchain_openai import OpenAIEmbeddings

cache = SemanticCache(
    embedding=OpenAIEmbeddings(),
    similarity_threshold=0.8,  # 相似度阈值
)

# 使用缓存
from langchain.globals import set_llm_cache
set_llm_cache(cache)
```

**在你的项目中的应用**：
- 缓存相似查询的 RAG 检索结果
- 缓存相关性分析结果

### 6.2 In-Memory Cache
**功能**：内存缓存，适合开发测试。

**实现示例**：
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

---

## 7. 监控和可观测性

### 7.1 LangSmith 集成
**功能**：完整的追踪、监控和调试能力。

**功能特性**：
- 请求追踪
- 性能监控
- 成本分析
- 错误追踪
- 提示词版本管理

**实现示例**：
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "askany-workflow"
```

**在你的项目中的应用**：
- 追踪所有 LLM 调用
- 分析性能瓶颈
- 监控成本
- 调试问题

### 7.2 自定义追踪
**功能**：添加自定义追踪点。

**实现示例**：
```python
from langchain_core.tracers import LangChainTracer

tracer = LangChainTracer()
# 自动追踪所有调用
```

---

## 8. 动态路由和决策

### 8.1 Conditional Edges（条件边）
**功能**：根据状态动态决定下一步执行哪个节点。

**实现示例**：
```python
from langgraph.graph import StateGraph, START, END

def should_use_rag(state: AgentState) -> str:
    """根据状态决定是否使用 RAG"""
    if state.get("need_rag_search"):
        return "rag_retrieval"
    elif state.get("need_web_search"):
        return "web_search"
    else:
        return "direct_answer"

workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_node)
workflow.add_conditional_edges(
    "classify",
    should_use_rag,
    {
        "rag_retrieval": "rag_node",
        "web_search": "web_node",
        "direct_answer": "answer_node",
    }
)
```

**在你的项目中的应用**：
- 替换 `direct_answer_check` 和 `web_or_rag_check` 的逻辑
- 实现更清晰的条件路由

### 8.2 Command 对象
**功能**：在工具中动态控制图执行流程。

**实现示例**：
```python
from langgraph.types import Command
from langchain.tools import tool

@tool
def dynamic_reroute(query: str) -> Command:
    """根据查询动态决定路由"""
    if "紧急" in query:
        return Command(
            update={"priority": "high"},
            goto="urgent_handler"
        )
    return Command(update={"priority": "normal"})
```

---

## 9. 人机交互

### 9.1 HumanInTheLoopMiddleware
**功能**：在敏感操作前暂停，等待人工审核。

**适用场景**：
- 数据库写操作
- 文件删除操作
- 敏感信息查询

**实现示例**：
```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[database_write_tool, file_delete_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "database_write_tool": True,
                "file_delete_tool": True,
            },
            description_prefix="操作等待审核",
        ),
    ],
    checkpointer=InMemorySaver(),
)

# 执行时会暂停等待审核
result = agent.invoke({"messages": "删除文件 X"}, config)
# 审核通过后继续
result = agent.invoke(None, config)  # 继续执行
```

**在你的项目中的应用**：
- 敏感查询的人工审核
- 危险操作前的确认

---

## 10. 状态管理优化

### 10.1 Checkpointer 持久化
**功能**：持久化状态，支持恢复和调试。

**实现示例**：
```python
from langgraph.checkpoint.postgres import PostgresSaver

# PostgreSQL 持久化（生产环境）
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/dbname"
)

# SQLite 持久化（本地开发）
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")

graph = workflow.compile(checkpointer=checkpointer)
```

**在你的项目中的应用**：
- 会话状态持久化
- 错误恢复
- 调试和审计

### 10.2 State 更新优化
**功能**：使用 Command 精确控制状态更新。

**实现示例**：
```python
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@tool
def clear_history() -> Command:
    """清空对话历史"""
    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )
```

---

## 11. Rate Limiting（速率限制）

### 11.1 InMemoryRateLimiter
**功能**：限制 API 调用速率，避免触发限流。

**实现示例**：
```python
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.tools import TavilySearchResults

# 限制为每秒 0.1 次请求
rate_limiter = InMemoryRateLimiter(requests_per_second=0.1)

tool = TavilySearchResults(rate_limiter=rate_limiter)
```

**在你的项目中的应用**：
- WebSearchTool 的速率限制
- LLM API 调用的速率限制

---

## 12. 消息元数据

### 12.1 Message Metadata
**功能**：为消息添加元数据，支持追踪和路由。

**实现示例**：
```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content="查询用户信息",
    name="alice",  # 用户标识
    id="msg_123",  # 消息ID
    metadata={
        "user_id": "user_123",
        "session_id": "session_456",
        "priority": "high",
    }
)
```

**在你的项目中的应用**：
- 追踪用户会话
- 路由不同用户的消息
- 优先级处理

---

## 实施优先级建议

### 高优先级（立即实施）
1. **ToolRetryMiddleware** - 提升系统稳定性
2. **ModelRetryMiddleware** - 统一错误处理
3. **Checkpointer 持久化** - 支持会话和恢复
4. **LangSmith 集成** - 监控和调试

### 中优先级（近期实施）
5. **SummarizationMiddleware** - 管理长对话
6. **Batch 批处理** - 提升并行处理性能
7. **Prompt Caching** - 降低成本
8. **Streaming** - 提升用户体验

### 低优先级（长期优化）
9. **PIIMiddleware** - 隐私保护
10. **HumanInTheLoopMiddleware** - 敏感操作审核
11. **Semantic Cache** - 缓存优化
12. **动态模型选择** - 成本优化

---

## 参考资源

- [LangChain Middleware 文档](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph)
- [LangSmith 文档](https://docs.langchain.com/langsmith)
- [LangChain v1.0 发布说明](https://blog.langchain.com/langchain-langgraph-1dot0/)
