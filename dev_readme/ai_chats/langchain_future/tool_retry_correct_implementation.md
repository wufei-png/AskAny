# ToolRetryMiddleware 的正确实现方式

## 问题分析

你的担心是完全正确的！`create_agent` + `ToolRetryMiddleware` 不适用于你的架构，因为：

1. **你的节点是确定性的**：每个节点是函数，显式调用工具，不是让 LLM 选择工具
2. **工具调用是直接的**：`web_search_tool.search(query)`，不是通过 `bind_tools`
3. **分析节点不需要工具**：`analyze_relevance_and_completeness` 是纯 LLM 调用

如果使用 `create_agent`，会导致：
- ❌ 所有工具被 `bind_tools`，工具定义污染提示词
- ❌ 在分析相关性阶段，LLM 看到不需要的工具定义
- ❌ LLM 可能错误地尝试调用工具

---

## 正确的实现方案

### 方案 1：在工具类内部实现重试（推荐，你已经在做）

**优点**：
- 不污染提示词
- 每个工具独立控制重试逻辑
- 简单直接

**实现**：

```python
# askany/workflow/WebSearchTool.py
import time
from typing import List
from functools import wraps

def retry_on_failure(max_retries=3, backoff_factor=2.0, initial_delay=1.0, 
                     retry_on=(ConnectionError, TimeoutError)):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
                        raise
                except Exception as e:
                    # 非重试异常直接抛出
                    raise
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

class WebSearchTool:
    @retry_on_failure(
        max_retries=3,
        backoff_factor=2.0,
        initial_delay=1.0,
        retry_on=(ConnectionError, TimeoutError, requests.RequestException)
    )
    def search(self, query: str) -> List[NodeWithScore]:
        """搜索网络（带自动重试）"""
        # 你的现有实现
        response = requests.post(self.api_url, json=data, timeout=self.timeout)
        # ...
```

**在你的项目中的应用**：
- ✅ `WebSearchTool.search()` - 已实现（可以优化）
- ✅ `LocalFileSearchTool` - 可以添加重试
- ✅ RAG 检索 - 可以添加重试

---

### 方案 2：使用 LangChain 的 Retry Utilities（不通过 Middleware）

**优点**：
- 使用 LangChain 的标准工具
- 可以统一配置
- 不污染提示词

**实现**：

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.utils import Input, Output
from typing import Callable, TypeVar, ParamSpec
import time
import logging

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

def create_retry_wrapper(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    retry_on: tuple = (Exception,),
):
    """创建重试包装器"""
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        def retry_func(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
                        raise
                except Exception as e:
                    # 非重试异常直接抛出
                    raise
            
            if last_exception:
                raise last_exception
        
        return retry_func
    return wrapper

# 使用
class WebSearchTool:
    def __init__(self, ...):
        # ...
        # 包装 search 方法
        self.search = create_retry_wrapper(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            retry_on=(ConnectionError, TimeoutError, requests.RequestException)
        )(self._search_impl)
    
    def _search_impl(self, query: str) -> List[NodeWithScore]:
        """实际的搜索实现（不带重试）"""
        # 你的现有实现
        response = requests.post(self.api_url, json=data, timeout=self.timeout)
        # ...
```

---

### 方案 3：在 LangGraph 节点级别添加重试（针对特定节点）

**优点**：
- 在节点级别控制重试
- 可以针对不同节点配置不同策略
- 不影响工具本身

**实现**：

```python
from langgraph.graph import StateGraph
from functools import wraps
import asyncio

def retry_node(max_retries=3, backoff_factor=2.0, initial_delay=1.0):
    """节点重试装饰器"""
    def decorator(node_func):
        @wraps(node_func)
        async def wrapper(state: AgentState):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return await node_func(state)
                except (ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Node {node_func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"Node {node_func.__name__} failed after {max_retries} attempts")
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator

# 使用
class LangGraphWorkflow:
    @retry_node(max_retries=3, backoff_factor=2.0, initial_delay=1.0)
    async def web_search_node(self, state: AgentState) -> AgentState:
        """网络搜索节点（带重试）"""
        query = state.get("query", "")
        nodes = self.web_search_tool.search(query)  # 工具调用
        return {**state, "web_nodes": nodes}
    
    @retry_node(max_retries=2)  # 不同节点可以不同配置
    async def rag_retrieval_node(self, state: AgentState) -> AgentState:
        """RAG 检索节点（带重试）"""
        query = state.get("query", "")
        nodes = self.router.retrieve(query)  # 工具调用
        return {**state, "rag_nodes": nodes}
    
    # 分析节点不需要重试（纯 LLM 调用）
    def analyze_relevance_node(self, state: AgentState) -> AgentState:
        """分析相关性节点（不需要重试，纯 LLM）"""
        query = state.get("query", "")
        nodes = state.get("rag_nodes", [])
        result = analyze_relevance_and_completeness(query, nodes, self.client)
        return {**state, "relevance": result}
```

---

### 方案 4：使用 LangChain 的 Runnable.retry（如果工具是 Runnable）

如果你的工具实现了 `Runnable` 接口，可以使用 LangChain 的内置重试：

```python
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input, Output

class WebSearchRunnable(Runnable[str, List[NodeWithScore]]):
    """将 WebSearchTool 包装为 Runnable"""
    
    def __init__(self, tool: WebSearchTool):
        self.tool = tool
    
    def invoke(self, input: str, config=None) -> List[NodeWithScore]:
        return self.tool.search(input)

# 使用 retry
web_search_runnable = WebSearchRunnable(web_search_tool).retry(
    stop_after_attempt=3,
    wait_exponential=multiplier=2.0,
    initial_delay=1.0,
    retry_if=lambda e: isinstance(e, (ConnectionError, TimeoutError))
)
```

---

## 推荐方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **方案1：工具内部重试** | 简单、直接、不污染提示词 | 每个工具需要单独实现 | ✅ **推荐** - 你的当前架构 |
| **方案2：Retry Wrapper** | 统一配置、可复用 | 需要包装每个工具方法 | 工具类较多时 |
| **方案3：节点级重试** | 节点级别控制 | 重试粒度较粗 | 需要节点级别重试时 |
| **方案4：Runnable.retry** | LangChain 标准方式 | 需要实现 Runnable | 工具是 Runnable 时 |

---

## 在你的项目中的具体实现

### 1. 优化 WebSearchTool（你已经在做，可以改进）

```python
# askany/workflow/WebSearchTool.py
class WebSearchTool:
    def __init__(self, ..., max_retries=3, retry_backoff=2.0, retry_delay=1.0):
        # ...
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_delay = retry_delay
    
    def search(self, query: str) -> List[NodeWithScore]:
        """搜索网络（带自动重试）"""
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                return self._search_impl(query)
            except (ConnectionError, TimeoutError, requests.RequestException) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Web search failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
                else:
                    logger.error(f"Web search failed after {self.max_retries} attempts")
                    return []  # 或 raise，根据你的需求
        
        return []
    
    def _search_impl(self, query: str) -> List[NodeWithScore]:
        """实际的搜索实现"""
        # 你的现有实现
        response = requests.post(self.api_url, json=data, timeout=self.timeout)
        # ...
```

### 2. 为 LocalFileSearchTool 添加重试

```python
# askany/workflow/LocalFileSearchTool.py
class LocalFileSearchTool:
    def __init__(self, ..., max_retries=3):
        # ...
        self.max_retries = max_retries
    
    def search_by_keywords(self, keywords: List[str], ...) -> Dict:
        """根据关键字搜索（带重试）"""
        for attempt in range(self.max_retries):
            try:
                return self._search_impl(keywords, ...)
            except (IOError, OSError) as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"File search failed, retrying... ({e})")
                    time.sleep(1.0 * (2 ** attempt))
                else:
                    logger.error(f"File search failed after {self.max_retries} attempts")
                    raise
```

### 3. 分析节点保持原样（不需要重试，纯 LLM）

```python
# askany/workflow/AnalysisRelated.py
def analyze_relevance_and_completeness(...):
    """分析相关性（纯 LLM 调用，不需要工具，不需要重试）"""
    # 你的现有实现
    completion = client.chat.completions.parse(...)
    # ...
```

---

## 总结

1. **不要使用 `create_agent` + `ToolRetryMiddleware`**：会污染提示词
2. **在工具类内部实现重试**：简单、直接、不污染提示词
3. **分析节点保持原样**：纯 LLM 调用，不需要工具，不需要重试
4. **每个工具独立控制重试逻辑**：灵活、可控

这样既实现了重试功能，又不会污染提示词，也不会让 LLM 在不需要工具的节点误调用工具。
