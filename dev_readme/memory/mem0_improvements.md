# Mem0 功能改进建议

## 代码审查总结

### 一、现有实现分析

当前的 mem0 实现位于 `askany/memory/mem0_adapter.py`，提供以下功能：
- `search()`: 搜索用户记忆
- `save_turn()`: 保存问答对
- `save_turn_async()`: 异步保存问答对
- `format_memories_as_system_text()`: 格式化记忆为系统提示文本

### 二、发现的问题

#### 1. 同步阻塞问题 (server.py:480)

```python
# 当前实现 - 同步调用
if user_id and mem0_adapter:
    memories = mem0_adapter.search(user_query, user_id)  # 同步阻塞
```

**问题**: 在异步请求处理中调用同步的 mem0 search，可能阻塞事件循环。

**建议**: 使用 `asyncio.get_event_loop().run_in_executor()` 包装为异步调用。

#### 2. 类型安全问题 (mem0_adapter.py:80)

```python
results: List[Dict[str, Any]] = raw_results  # type: ignore[assignment]
```

**问题**: 使用 `type: ignore` 隐藏类型检查。

**建议**: 移除 type ignore，添加正确的类型断言。

#### 3. 错误处理过于宽泛

```python
except Exception:
    logger.exception("Mem0 search failed for user_id=%s", user_id)
    return []
```

**问题**: 捕获所有异常并返回空列表，可能隐藏真正的错误。

**建议**: 
- 区分可恢复错误和不可恢复错误
- 添加更详细的日志记录

### 三、功能改进建议

#### 1. 添加异步搜索方法

```python
async def search_async(self, query: str, user_id: str, ...) -> List[Dict[str, Any]]:
    """Async wrapper for search."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, self.search, query, user_id, ...)
```

#### 2. 添加记忆管理功能

- `delete_memory(memory_id, user_id)`: 删除单条记忆
- `list_memories(user_id)`: 列出用户所有记忆
- `clear_memories(user_id)`: 清除用户所有记忆

#### 3. 添加记忆持久化选项

当前所有记忆存储在 PostgreSQL 中。建议添加：
- 内存缓存层 (LRU cache)
- 批量写入优化

#### 4. 改进配置灵活性

- 支持自定义记忆提取提示词
- 支持记忆过期策略
- 支持记忆重要性评分阈值动态调整

### 四、使用场景优化

#### 1. 记忆上下文注入优化

当前实现将记忆作为 system message 注入，可考虑：
- 根据记忆相关性动态调整注入数量
- 添加记忆时效性权重

#### 2. 与工作流集成优化

- 在特定工作流节点条件触发记忆检索
- 支持记忆增强的查询重写

### 五、安全考虑

1. 用户隔离：确保用户只能访问自己的记忆
2. 敏感信息过滤：添加自动过滤敏感信息的功能
3. 速率限制：防止滥用记忆存储

### 六、测试建议

1. 添加集成测试 (需要 PostgreSQL + pgvector)
2. 添加性能基准测试
3. 添加并发访问测试

---

## 总结

当前 mem0 实现基本满足需求，但存在一些可优化的地方：
- **高优先级**: 解决同步阻塞问题
- **中优先级**: 添加更多记忆管理功能
- **低优先级**: 改进配置灵活性和安全特性
