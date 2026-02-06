# Text_search_config 的影响
_Exported on 2025/12/7 at GMT+8 17:17:28 from Cursor (2.1.49)_

---

**User**

@askany/ingest/vector_store.py:184-195 这里的text_search_config最终会影响什么 文本是中文

---

**Cursor**


查看 `text_search_config` 的作用，特别是在中文文本场景下。



查看 `text_search_config` 在 PGVectorStore 中的具体使用：



继续查看 `text_search_config` 在创建表和全文搜索索引时的具体使用：



查找 `get_data_model` 函数，它应该使用 `text_search_config` 创建表结构：



查看 `text_search_config` 在查询时的使用：



总结 `text_search_config` 的作用：

## `text_search_config` 的作用

`text_search_config` 影响 PostgreSQL 全文搜索的两个环节：

### 1. **表创建时**（第 175-180 行）

```python
text_search_tsv = Column(
    TSVector(),
    Computed(
        "to_tsvector('%s', text)" % text_search_config, persisted=True
    ),
)
```

- 创建 `text_search_tsv` 计算列
- 使用 `to_tsvector(text_search_config, text)` 将文本转换为 tsvector
- 决定分词器、停用词和词干提取规则

### 2. **查询时**（第 930-933 行）

```python
ts_query = func.to_tsquery(
    type_coerce(self.text_search_config, REGCONFIG),
    query_str,
)
```

- 用于 `to_tsquery()` 将查询字符串转换为 tsquery
- 用于全文搜索匹配（`@@`）和排名（`ts_rank`）

### 对中文的影响

当前配置为 `"english"`（见 `config.py` 第 52 行），对中文文本的影响：

1. 无法正确分词：英文配置无法识别中文词汇边界
2. 搜索效果差：中文被当作单个字符或无法识别
3. 混合搜索失效：如果启用了 `hybrid_search=True`，全文搜索部分对中文基本无效

### 建议

对于中文文本，建议：

1. 使用 `"simple"`：不进行语言特定处理，按空格和标点分词
   ```python
   text_search_config: str = "simple"
   ```

2. 安装中文分词扩展（如 `zhparser`），然后使用中文配置：
   ```python
   text_search_config: str = "zhparser"  # 需要先安装 zhparser 扩展
   ```

3. 如果主要使用向量搜索而非全文搜索，且未启用 `hybrid_search`，该配置影响较小

需要我帮你修改配置吗？

```python
text_search_tsv = Column(
    TSVector(),
    Computed(
        "to_tsvector('%s', text)" % text_search_config, persisted=True
    ),
)
```

```python
ts_query = func.to_tsquery(
    type_coerce(self.text_search_config, REGCONFIG),
    query_str,
)
```

```python
   text_search_config: str = "simple"
```

```python
   text_search_config: str = "zhparser"  # 需要先安装 zhparser 扩展
```

---

**User**

如果主要使用向量搜索而非全文搜索，且未启用 hybrid_search，该配置影响较小
为什么这么说, 代码里有用全文搜索吗 hybrid_search又是什么?@askany/ingest/vector_store.py 

---

**Cursor**


检查代码中是否启用了 `hybrid_search`，以及全文搜索的使用情况：



检查查询引擎的实现，确认是否使用了混合搜索：



查看 VectorStoreIndex 的默认查询模式：



总结代码中的情况：



## 代码分析

### 1. **`hybrid_search` 是什么？**

`hybrid_search` 是 PGVectorStore 的一个参数，控制是否启用混合搜索（向量搜索 + PostgreSQL 全文搜索）。

在你的代码中（第84、144、184行），创建 PGVectorStore 时没有传递 `hybrid_search` 参数：

```python
self.docs_vector_store = PGVectorStore.from_params(
    ...
    text_search_config=settings.text_search_config,
    # 注意：没有传递 hybrid_search 参数
)
```

查看 `from_params` 的默认值（第421行）：
```python
hybrid_search: bool = False,  # 默认是 False
```

因此，你的代码中 `hybrid_search=False`。

### 2. **`hybrid_search=False` 时会发生什么？**

查看表创建代码（第166-224行）：

- 当 `hybrid_search=True` 时：
  - 创建 `text_search_tsv` 列（第175-180行）
  - 使用 `to_tsvector(text_search_config, text)` 生成全文搜索向量
  - 创建 GIN 索引用于全文搜索（第191-195行）

- 当 `hybrid_search=False` 时（你的情况）：
  - 不创建 `text_search_tsv` 列
  - 只创建基本列：`id`, `text`, `metadata_`, `node_id`, `embedding`
  - 没有全文搜索相关的索引

### 3. **查询时如何使用？**

查看查询代码（第1104-1126行）：

```python
def query(self, query: VectorStoreQuery, **kwargs: Any):
    if query.mode == VectorStoreQueryMode.HYBRID:
        # 混合搜索：向量 + 全文搜索
        results = self._hybrid_query(query, **kwargs)
    elif query.mode in [VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH]:
        # 纯全文搜索（需要 hybrid_search=True 才能工作）
        results = self._sparse_query_with_rank(...)
    elif query.mode == VectorStoreQueryMode.DEFAULT:
        # 纯向量搜索（默认模式）
        results = self._query_with_score(...)
```

`VectorStoreIndex` 默认使用 `DEFAULT` 模式，即纯向量搜索。

### 4. **为什么说 `text_search_config` 影响较小？**

因为：
1. `hybrid_search=False`，不会创建 `text_search_tsv` 列
2. 查询使用 `DEFAULT` 模式（纯向量搜索），不会调用全文搜索代码
3. `text_search_config` 只在创建 `text_search_tsv` 列时使用，而该列不存在

### 5. **但是有一个问题**

即使 `hybrid_search=False`，`text_search_config` 参数仍然会被传递并存储（第93、153、193行）。虽然当前不影响功能，但：
- 如果将来启用 `hybrid_search=True`，错误的配置会影响中文全文搜索
- 代码中传递了这个参数，说明可能计划使用

### 总结

- 当前代码未使用全文搜索：`hybrid_search=False`，查询使用 `DEFAULT` 模式（纯向量搜索）
- `text_search_config` 当前不影响功能：因为没有创建全文搜索相关的列和索引
- 建议：如果将来可能启用混合搜索，现在就应该将 `text_search_config` 改为 `"simple"` 或配置中文分词器

需要我帮你修改配置吗？

```python
self.docs_vector_store = PGVectorStore.from_params(
    ...
    text_search_config=settings.text_search_config,
    # 注意：没有传递 hybrid_search 参数
)
```

```python
hybrid_search: bool = False,  # 默认是 False
```

```python
def query(self, query: VectorStoreQuery, **kwargs: Any):
    if query.mode == VectorStoreQueryMode.HYBRID:
        # 混合搜索：向量 + 全文搜索
        results = self._hybrid_query(query, **kwargs)
    elif query.mode in [VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH]:
        # 纯全文搜索（需要 hybrid_search=True 才能工作）
        results = self._sparse_query_with_rank(...)
    elif query.mode == VectorStoreQueryMode.DEFAULT:
        # 纯向量搜索（默认模式）
        results = self._query_with_score(...)
```

