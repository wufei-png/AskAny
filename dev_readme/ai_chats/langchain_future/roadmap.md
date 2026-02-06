- 支持多种模式
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)


- 

Batch
https://docs.langchain.com/oss/python/langchain/models#batch
Batching a collection of independent requests to a model can significantly improve performance and reduce costs, as the processing can be done in parallel:
Batch
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
batch 支持


Prompt caching
Many providers offer prompt caching features to reduce latency and cost on repeat processing of the same tokens. These features can be implicit or explicit:
Implicit prompt caching: providers will automatically pass on cost savings if a request hits a cache. Examples: OpenAI and Gemini.
Explicit caching: providers allow you to manually indicate cache points for greater control or to guarantee cost savings. Examples:
ChatOpenAI (via prompt_cache_key)
Anthropic’s AnthropicPromptCachingMiddleware
Gemini.
AWS Bedrock





Message metadata
Add metadata
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)

聊天室接入robot？


Updating state:
Use Command to update the agent’s state or control the graph’s execution flow:
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime

# Update the conversation history by removing all messages
@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""

    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
    runtime: ToolRuntime
) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})
压缩上下文 同时相关的： short-memory checkpoint
To summarize message history in an agent, use the built-in SummarizationMiddleware:

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""


ClearToolUsesEdit


================================== Ai Message ==================================

Your name is Bob!
"""


TodoListMiddleware

ToolRetryMiddleware
ModelRetryMiddleware


https://docs.langchain.com/oss/python/langchain/middleware/built-in#file-search 可以在某个节点使用吗 LocalFileSearchTool一轮找不到就用这个？


human in the loop

---

## 完整优化特性清单

详细的 LangChain/LangGraph 优化特性说明，请参考：
📄 [optimization_features.md](./optimization_features.md)

### 快速参考

**高优先级（立即实施）**：
- ✅ **ToolRetryMiddleware** - 自动重试失败的工具调用
- ✅ **ModelRetryMiddleware** - 自动重试失败的模型调用
- ✅ **Checkpointer 持久化** - PostgreSQL/SQLite 状态持久化
- ✅ **LangSmith 集成** - 完整的追踪和监控

**中优先级（近期实施）**：
- ✅ **SummarizationMiddleware** - 管理长对话上下文
- ✅ **Batch 批处理** - 并行处理多个请求
- ✅ **Prompt Caching** - 降低成本和延迟
- ✅ **Streaming** - 实时流式响应

**低优先级（长期优化）**：
- ✅ **PIIMiddleware** - 敏感信息脱敏
- ✅ **HumanInTheLoopMiddleware** - 敏感操作审核
- ✅ **Semantic Cache** - 语义相似度缓存
- ✅ **动态模型选择** - 根据复杂度选择模型
- ✅ **Rate Limiting** - API 调用速率限制
- ✅ **并行节点** - LangGraph 真正的并行执行
