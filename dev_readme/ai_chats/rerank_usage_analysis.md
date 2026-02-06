# Rerank 使用方式分析

## 当前实现

### 工作流程

1. **检索阶段** (Ensemble Retriever):
   - Keyword Retriever: 检索 `similarity_top_k` 个节点
   - Vector Retriever: 检索 `similarity_top_k` 个节点
   - QueryFusionRetriever: 融合两个检索器的结果，输出 `similarity_top_k` 个节点

2. **Rerank 阶段** (SentenceTransformerRerank):
   - 接收 Ensemble Retriever 的输出（`similarity_top_k` 个节点）
   - 使用交叉编码器模型对 query-document 对进行深度语义匹配
   - 重新排序并选择 `top_n` 个节点（当前 `top_n = similarity_top_k`）

### 代码位置

```python
# askany/rag/faq_query_engine.py:81-88
reranker = self._create_reranker(
    reranker_model=reranker_model,
    top_n=similarity_top_k,  # 当前使用相同的值
)
```

## 潜在问题

### 问题 1: Reranker 没有真正发挥筛选作用

**当前情况**:
- Retriever 检索: 5 个节点 (`similarity_top_k=5`)
- Reranker 选择: 5 个节点 (`top_n=5`)
- **结果**: Reranker 只是重新排序，没有筛选

**最佳实践**:
- Retriever 应该检索更多节点（如 20 个）
- Reranker 从中选择 top N（如 5 个）
- 这样可以：
  1. 提高召回率（检索更多候选）
  2. 提高准确率（reranker 筛选最相关的）

### 问题 2: 配置不够灵活

当前 `top_n` 硬编码为 `similarity_top_k`，无法独立配置。

## 建议的改进方案

### 方案 1: 增加 rerank_top_k 配置（推荐）

```python
# config.py
similarity_top_k: int = 5  # 最终返回的节点数
rerank_top_k: int = 20  # reranker 从多少个候选中选择

# faq_query_engine.py
ensemble_retriever = QueryFusionRetriever(
    similarity_top_k=rerank_top_k,  # 检索更多节点
)

reranker = self._create_reranker(
    reranker_model=reranker_model,
    top_n=similarity_top_k,  # 最终选择 top N
)
```

### 方案 2: 使用倍数关系

```python
# 检索 3-4 倍于最终需要的节点数
rerank_candidate_k = similarity_top_k * 3  # 15-20 个候选

ensemble_retriever = QueryFusionRetriever(
    similarity_top_k=rerank_candidate_k,
)

reranker = self._create_reranker(
    top_n=similarity_top_k,  # 最终选择 5 个
)
```

## Reranker 的工作原理

### SentenceTransformerRerank

1. **输入**: 查询字符串 + 候选节点列表
2. **处理**: 
   - 对每个 query-document 对使用交叉编码器模型
   - 计算相关性分数（0-1 之间）
   - 按分数降序排序
3. **输出**: Top N 个最相关的节点

### 为什么需要更多候选？

- **检索器**（keyword/vector）基于相似度分数，可能遗漏一些语义相关但分数不高的文档
- **Reranker** 使用交叉编码器进行深度语义匹配，能发现检索器遗漏的相关文档
- 如果候选池太小，reranker 无法发挥优势

## 性能考虑

### 当前实现（检索 5，rerank 5）
- ✅ 速度快（只处理 5 个节点）
- ❌ 召回率可能不足
- ❌ Reranker 优势未充分发挥

### 改进实现（检索 20，rerank 5）
- ✅ 召回率更高
- ✅ Reranker 能筛选出最相关的
- ⚠️ 速度稍慢（需要处理 20 个节点）

**建议**: 对于 FAQ 场景，检索 15-20 个候选，rerank 到 5 个是合理的平衡。

