# 多轮对话检索功能实现方案

## 需求概述

每轮次后,checkpointer只需要加载chat_history List[struct qa_pair{"query": str, "answer": str, nodes: List[NodeWithScore]}]:
    # 构建初始状态
    initial_state: AgentState = {
        "conversation_id": thread_id,
        "turn_number": len(current_state.values.get("chat_history", [])) // 2 + 1 if current_state.values else 1,
        "chat_history": current_state.values.get("chat_history", []) if current_state.values else [],
        "needed_chat_history": [],
    }


在direct_answer_check_node中先判断如果是第一次问,现有流程

否则: 根据历史对话和新问题判断相关性 编写@previousRelevant.py 实现
- 直接调用一次llm判断 判断历史qa中相关的qa对话, 随后更新needed_chat_history 将相关的历史qa对[struct qa_pair{"query": str, "answer": str, nodes: List[NodeWithScore]}]设置到state的  needed_chat_history 中, 如果needed_chat_history数组长度为0, 说明无关, 这一轮不使用历史问答状态, 依然走到后续的web_or_rag_check节点

- 如果相关: 遍历needed_chat_history里的所有nodes 做_merge_nodes函数合并,如果token超过了设置的llm_max_tokens一半, SummaryFromLlm进行总结压缩

@previousRelevant.py实现另外一个llm调用输入为needed_chat_history和 此次query,


并用合并后的nodes以及历史会话判断是否可以解决问题 如果可以 直接回答 否则判断有哪些相关的nodes和query保留 并重写rag和websearch的Query



- 简化掉turn_number,直接用len(chat_history)
- Checkpointer**: 已有PostgreSQL数据库，使用`PostgresSaver


## 技术调研

### 1. OpenWebUI前端传递的信息

根据代码分析，OpenWebUI前端在请求中会传递以下信息：
- `chat_id`: 对话的唯一标识符（UUID）
- `session_id`: WebSocket会话ID
- `messages`: 完整的对话历史（包含role和content）
- `parent_id`: 父消息ID
- `parent_message`: 父消息对象

这些信息在 `backend/open_webui/main.py` 的 `chat_completion` 函数中被提取到 `metadata` 中。

### 2. LangGraph状态持久化机制

LangGraph支持通过checkpoint机制进行状态持久化：
- **Thread ID**: 每个对话会话使用唯一的`thread_id`来区分
- **Checkpointer**: 支持多种存储后端（MemorySaver、Redis、PostgreSQL、MongoDB等）
- **State History**: 可以通过`graph.get_state_history(config)`获取完整的历史状态
- **Metadata**: 每个checkpoint包含metadata，其中`step`字段可以表示轮次

### 3. 当前AskAny实现情况

- **FastAPI Server** (`askany/api/server.py`): 
  - 当前只接收`messages`数组，未处理`chat_id`
  - 需要扩展以支持`chat_id`和对话历史
  
- **LangGraph Workflow** (`askany/workflow/workflow_langgraph.py`):
  - 使用`AgentState`定义状态，包含query、nodes、analysis等字段
  - 当前没有配置checkpointer，状态不持久化
  - 需要添加conversation_id、turn_number等字段

## 实现方案

### 1. 对话标识与轮次管理

#### 1.1 对话ID (Conversation ID / Thread ID)

- **来源**: 使用OpenWebUI传递的`chat_id`作为`thread_id`
- **传递方式**: 
  - 方案A: 通过请求体参数传递（推荐）
  - 方案B: 通过HTTP Header传递（如`X-Chat-Id`）
- **生成规则**: 
  - 如果请求中没有`chat_id`，后端生成新的UUID
  - 如果已有`chat_id`，使用该ID作为`thread_id`

#### 1.2 轮次判断

- **轮次定义**: 每次用户提问 → 系统完整回复 = 1轮
- **轮次获取方式**:
  - 方式1: 从checkpoint的`metadata.step`获取（LangGraph自动维护）
  - 方式2: 自定义`turn_number`字段在state中维护
  - 方式3: 通过`get_state_history(config)`获取历史checkpoint数量
- **推荐**: 使用方式1（metadata.step）+ 方式2（自定义turn_number）双重保障

### 2. FastAPI接口扩展

#### 2.1 请求模型扩展

在`ChatCompletionRequest`中添加可选字段：

```python
class ChatCompletionRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[ChatMessage]
    temperature: float = settings.temperature
    max_tokens: Optional[int] = None
    stream: bool = False
    chat_id: Optional[str] = None  # 新增：对话ID
    thread_id: Optional[str] = None  # 新增：LangGraph thread_id（与chat_id相同）
```

#### 2.2 接口处理逻辑

在`chat_completions`函数中：
1. 提取`chat_id`（从请求体或header）
2. 如果没有`chat_id`，生成新的UUID
3. 将`chat_id`作为`thread_id`传递给LangGraph workflow
4. 从messages中提取历史对话上下文

### 3. LangGraph状态扩展

#### 3.1 AgentState扩展

在`workflow_langgraph.py`中扩展`AgentState`：

```python
class AgentState(TypedDict):
    # 现有字段...
    
    # 多轮对话相关字段
    conversation_id: str  # 对话唯一ID（thread_id）
    turn_number: int  # 当前轮次（从1开始）
    chat_history: List[Dict[str, str]]  # 完整对话历史 [{"role": "user/assistant", "content": "..."}]
    previous_nodes: List[NodeWithScore]  # 上一轮查询的nodes
    previous_state_summary: Optional[Dict[str, Any]]  # 上一轮状态摘要
    is_first_turn: bool  # 是否为第一轮对话
```

#### 3.2 Checkpointer配置

在`AgentWorkflow.__init__`中配置checkpointer：

```python
from langgraph.checkpoint.memory import MemorySaver
# 或使用其他存储后端
# from langgraph.checkpoint.postgres import PostgresSaver
# from langgraph.checkpoint.redis import RedisSaver

def __init__(self, ...):
    # ... 现有初始化代码 ...
    
    # 配置checkpointer
    checkpointer = MemorySaver()  # 开发环境
    # checkpointer = PostgresSaver(...)  # 生产环境使用PostgreSQL
    # checkpointer = RedisSaver(...)  # 生产环境使用Redis
    
    # 在_build_graph中编译时传入checkpointer
    self.graph = workflow.compile(checkpointer=checkpointer)
```

#### 3.3 状态恢复与更新

在workflow执行前：
1. 使用`thread_id`从checkpointer恢复历史状态
2. 判断是否为第一轮（通过`get_state_history`或检查state）
3. 如果是第一轮，初始化新状态
4. 如果不是第一轮，从历史状态中恢复`previous_nodes`、`chat_history`等

### 4. 多轮对话检索逻辑

#### 4.1 第一轮判断

在workflow开始节点（如`direct_answer_check`）中：
```python
def _direct_answer_check_node(self, state: AgentState) -> AgentState:
    # 检查是否为第一轮
    conversation_id = state.get("conversation_id")
    config = {"configurable": {"thread_id": conversation_id}}
    
    # 获取历史状态
    history = self.graph.get_state_history(config)
    is_first_turn = len(history) == 0
    
    if is_first_turn:
        # 第一轮：走现有流程
        state["is_first_turn"] = True
        state["turn_number"] = 1
        # ... 现有逻辑 ...
    else:
        # 多轮对话处理
        state["is_first_turn"] = False
        # ... 多轮对话逻辑 ...
```

#### 4.2 相关性判断

在非第一轮时，添加相关性判断节点：

```python
def _check_relevance_node(self, state: AgentState) -> AgentState:
    """判断新问题与历史对话的相关性"""
    current_query = state.get("query")
    chat_history = state.get("chat_history", [])
    previous_nodes = state.get("previous_nodes", [])
    
    # 使用LLM判断相关性
    relevance_result = self.relevance_analyzer.check_query_relevance(
        current_query, chat_history, previous_nodes
    )
    
    if not relevance_result.is_relevant:
        # 无关：丢弃历史状态，走原workflow
        state["previous_nodes"] = []
        state["previous_state_summary"] = None
        # 但保留chat_history用于上下文
    else:
        # 相关：合并历史nodes
        # ... 合并逻辑 ...
    
    return state
```

#### 4.3 历史Nodes合并与截断

```python
def _merge_previous_nodes(self, state: AgentState) -> AgentState:
    """合并历史nodes并进行token截断"""
    current_nodes = state.get("nodes", [])
    previous_nodes = state.get("previous_nodes", [])
    
    # 合并nodes（去重）
    merged_nodes = self._merge_nodes(previous_nodes, current_nodes)
    
    # Token截断（从前面开始截断）
    truncated_nodes, total_tokens, was_truncated = truncate_nodes_by_tokens(
        merged_nodes, 
        settings.llm_max_tokens,
        truncate_from_start=True  # 从前面开始截断
    )
    
    state["nodes"] = truncated_nodes
    return state
```

#### 4.4 判断是否可直接回答

```python
def _check_can_answer_with_history(self, state: AgentState) -> AgentState:
    """使用合并后的nodes和历史会话判断是否可以解决问题"""
    query = state.get("query")
    nodes = state.get("nodes", [])
    chat_history = state.get("chat_history", [])
    
    # 使用LLM判断是否可以基于现有信息回答
    can_answer = self.relevance_analyzer.can_answer_with_context(
        query, nodes, chat_history
    )
    
    if can_answer:
        # 可以直接回答
        answer, reasoning = self.final_answer_generator.generate_final_answer(
            query, nodes, chat_history
        )
        state["result"] = answer
    else:
        # 需要进一步检索
        # 判断哪些相关的nodes和query保留
        relevant_info = self.relevance_analyzer.extract_relevant_info(
            query, nodes, chat_history
        )
        # 重写RAG和WebSearch的Query
        state["query"] = relevant_info.rewritten_query
        state["nodes"] = relevant_info.relevant_nodes
    
    return state
```

#### 4.5 Query重写

```python
def _rewrite_query_with_context(self, state: AgentState) -> AgentState:
    """根据历史对话和保留的nodes重写查询"""
    original_query = state.get("query")
    chat_history = state.get("chat_history", [])
    relevant_nodes = state.get("nodes", [])
    
    # 使用LLM重写查询
    rewritten_query = self.query_rewrite_generator.rewrite_with_context(
        original_query, chat_history, relevant_nodes
    )
    
    state["query"] = rewritten_query
    return state
```

### 5. 状态保存

#### 5.1 执行时传入thread_id

在`invoke`或`ainvoke`方法中：

```python
def invoke(self, query: str, query_type: QueryType = QueryType.AUTO, **kwargs):
    # 从kwargs中获取thread_id
    thread_id = kwargs.get("thread_id") or kwargs.get("conversation_id")
    if not thread_id:
        # 生成新的thread_id
        import uuid
        thread_id = str(uuid.uuid4())
    
    # 构建config
    config = {"configurable": {"thread_id": thread_id}}
    
    # 获取历史状态
    current_state = self.graph.get_state(config)
    
    # 构建初始状态
    initial_state: AgentState = {
        "conversation_id": thread_id,
        "turn_number": len(current_state.values.get("chat_history", [])) // 2 + 1 if current_state.values else 1,
        "chat_history": current_state.values.get("chat_history", []) if current_state.values else [],
        "previous_nodes": current_state.values.get("nodes", []) if current_state.values else [],
        # ... 其他字段 ...
    }
    
    # 执行workflow
    result = self.graph.invoke(initial_state, config=config)
    
    return result.get("result", "抱歉，无法生成答案。")
```

#### 5.2 保存状态到checkpoint

LangGraph会自动保存状态到checkpointer，但需要在关键节点更新状态：

```python
def _generate_final_answer_node(self, state: AgentState) -> AgentState:
    # ... 生成答案逻辑 ...
    
    # 更新chat_history
    chat_history = state.get("chat_history", [])
    chat_history.append({
        "role": "user",
        "content": state.get("query", "")
    })
    chat_history.append({
        "role": "assistant", 
        "content": state.get("result", "")
    })
    
    # 保存当前nodes作为previous_nodes供下一轮使用
    state["previous_nodes"] = state.get("nodes", [])
    state["chat_history"] = chat_history
    state["turn_number"] = state.get("turn_number", 0) + 1
    
    return state
```

### 6. API接口设计

#### 6.1 主对话接口

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 提取chat_id
    chat_id = request.chat_id or request.thread_id
    if not chat_id:
        # 从messages中尝试提取（如果前端通过其他方式传递）
        # 或生成新的chat_id
        import uuid
        chat_id = str(uuid.uuid4())
    
    # 提取用户消息
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    user_query = user_messages[-1].content if user_messages else ""
    
    # 构建chat_history
    chat_history = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    
    # 调用workflow
    response_text = await process_query_with_subproblems(
        agent_workflow_global,
        workflow_filter_global,
        user_query,
        query_type,
        thread_id=chat_id,  # 传递thread_id
        chat_history=chat_history  # 传递历史
    )
    
    # 返回响应（可包含turn_number等信息）
    return ChatCompletionResponse(
        # ... 现有字段 ...
        # 可选：添加metadata包含turn_number等
    )
```

#### 6.2 历史查询接口（可选）

```python
@app.get("/v1/conversations/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """获取对话历史"""
    config = {"configurable": {"thread_id": conversation_id}}
    history = agent_workflow_global.graph.get_state_history(config)
    
    return {
        "conversation_id": conversation_id,
        "turns": len(history),
        "history": [
            {
                "turn": i + 1,
                "state": snapshot.values,
                "metadata": snapshot.metadata
            }
            for i, snapshot in enumerate(history)
        ]
    }
```

### 7. 存储后端选择

#### 7.1 开发环境
- **MemorySaver**: 内存存储，重启后丢失，适合开发测试

#### 7.2 生产环境推荐
- **PostgreSQL**: 如果已有PostgreSQL数据库，使用`PostgresSaver`
- **Redis**: 高性能，适合高并发场景
- **MongoDB**: 灵活的文档存储

配置示例（PostgreSQL）：
```python
from langgraph.checkpoint.postgres import PostgresSaver, AsyncPostgresSaver

# 同步版本
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/dbname"
)

# 异步版本（推荐）
checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/dbname"
)
```

### 8. 注意事项

#### 8.1 状态大小控制
- 限制`chat_history`的最大长度（如最近20轮）
- 限制`previous_nodes`的数量和token数
- 定期清理过旧的历史状态

#### 8.2 并发安全
- 使用数据库锁或Redis锁确保同一`thread_id`的并发请求安全
- 考虑使用乐观锁或版本号机制

#### 8.3 Schema兼容性
- State schema变更时考虑向后兼容
- 使用版本号标识state schema版本

#### 8.4 性能优化
- 历史状态查询使用缓存
- 批量保存checkpoint而非每次节点执行都保存
- 考虑异步保存checkpoint

## 实现步骤

1. **Phase 1: 基础框架**
   - 扩展`ChatCompletionRequest`添加`chat_id`字段
   - 在FastAPI中提取并传递`chat_id`
   - 配置MemorySaver checkpointer（开发环境）

2. **Phase 2: 状态扩展**
   - 扩展`AgentState`添加多轮对话字段
   - 实现状态恢复逻辑
   - 实现第一轮判断逻辑

3. **Phase 3: 多轮对话逻辑**
   - 实现相关性判断节点
   - 实现历史nodes合并与截断
   - 实现Query重写逻辑

4. **Phase 4: 生产环境优化**
   - 切换到生产级checkpointer（PostgreSQL/Redis）
   - 实现状态大小控制
   - 实现并发安全机制
   - 性能优化

5. **Phase 5: 测试与验证**
   - 单元测试
   - 集成测试
   - 多轮对话场景测试
