# Explaining QueryFusionRetriever and fusion modes
_Exported on 2025/11/26 at GMT+8 0:58:58 from Cursor (2.1.20)_

---

**User**

@.venv/lib64/python3.11/site-packages/llama_index/core/retrievers/fusion_retriever.py:1-303 @askany/rag/faq_query_engine.py:105-111 查看代码解释QueryFusionRetriever如何工作，不同的FUSION_MODES模式以及如何将不同检索结果合并返回的

---

**Cursor**


查看相关代码，解释 `QueryFusionRetriever` 的工作原理。



解释 `QueryFusionRetriever` 的工作原理、融合模式和结果合并方式。

## QueryFusionRetriever 工作原理

`QueryFusionRetriever` 是一个融合检索器，将多个检索器的结果合并，提升检索质量。

### 1. 整体工作流程

```261:282:.venv/lib64/python3.11/site-packages/llama_index/core/retrievers/fusion_retriever.py
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.RELATIVE_SCORE:
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.DIST_BASED_SCORE:
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == FUSION_MODES.SIMPLE:
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")
```

流程：
1. 查询扩展：若 `num_queries > 1`，用 LLM 生成多个相关查询
2. 并行检索：对每个查询，用每个检索器检索
3. 结果融合：按选定模式合并结果
4. 返回 Top-K：返回前 `similarity_top_k` 个结果

### 2. 查询生成

```81:96:.venv/lib64/python3.11/site-packages/llama_index/core/retrievers/fusion_retriever.py
    def _get_queries(self, original_query: str) -> List[QueryBundle]:
        prompt_str = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )
        response = self._llm.complete(prompt_str)

        # Strip code block and assume LLM properly put each query on a newline
        queries = response.text.strip("`").split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        if self._verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list.
        return [QueryBundle(q) for q in queries[: self.num_queries - 1]]
```

使用 LLM 生成多个相关查询，增加检索覆盖。

### 3. 四种融合模式详解

#### 模式 1: RECIPROCAL_RANK（倒数排名融合）

```98:133:.venv/lib64/python3.11/site-packages/llama_index/core/retrievers/fusion_retriever.py
    def _reciprocal_rerank_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """
        Apply reciprocal rank fusion.

        The original paper uses k=60 for best results:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        """
        k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
        fused_scores = {}
        hash_to_node = {}

        # compute reciprocal rank scores
        for nodes_with_scores in results.values():
            for rank, node_with_score in enumerate(
                sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
            ):
                hash = node_with_score.node.hash
                hash_to_node[hash] = node_with_score
                if hash not in fused_scores:
                    fused_scores[hash] = 0.0
                fused_scores[hash] += 1.0 / (rank + k)

        # sort results
        reranked_results = dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # adjust node scores
        reranked_nodes: List[NodeWithScore] = []
        for hash, score in reranked_results.items():
            reranked_nodes.append(hash_to_node[hash])
            reranked_nodes[-1].score = score

        return reranked_nodes
```

原理：
- 公式：`RRF_score = Σ(1 / (rank + k))`，k=60
- 特点：不依赖原始分数，只依赖排名；多个检索器都排在前面的节点得分更高
- 适用：不同检索器分数不可比时

#### 模式 2: RELATIVE_SCORE（相对分数融合，支持权重）

```135:196:.venv/lib64/python3.11/site-packages/llama_index/core/retrievers/fusion_retriever.py
    def _relative_score_fusion(
        self,
        results: Dict[Tuple[str, int], List[NodeWithScore]],
        dist_based: Optional[bool] = False,
    ) -> List[NodeWithScore]:
        """Apply relative score fusion."""
        # MinMax scale scores of each result set (highest value becomes 1, lowest becomes 0)
        # then scale by the weight of the retriever
        min_max_scores = {}
        for query_tuple, nodes_with_scores in results.items():
            if not nodes_with_scores:
                min_max_scores[query_tuple] = (0.0, 0.0)
                continue
            scores = [
                node_with_score.score or 0.0 for node_with_score in nodes_with_scores
            ]
            if dist_based:
                # Set min and max based on mean and std dev
                mean_score = sum(scores) / len(scores)
                std_dev = (
                    sum((x - mean_score) ** 2 for x in scores) / len(scores)
                ) ** 0.5
                min_score = mean_score - 3 * std_dev
                max_score = mean_score + 3 * std_dev
            else:
                min_score = min(scores)
                max_score = max(scores)
            min_max_scores[query_tuple] = (min_score, max_score)

        for query_tuple, nodes_with_scores in results.items():
            for node_with_score in nodes_with_scores:
                min_score, max_score = min_max_scores[query_tuple]
                # Scale the score to be between 0 and 1
                if max_score == min_score:
                    node_with_score.score = 1.0 if max_score > 0 else 0.0
                else:
                    node_with_score.score = (node_with_score.score - min_score) / (
                        max_score - min_score
                    )
                # Scale by the weight of the retriever
                retriever_idx = query_tuple[1]
                existing_score = node_with_score.score or 0.0
                node_with_score.score = (
                    existing_score * self._retriever_weights[retriever_idx]
                )
                # Divide by the number of queries
                node_with_score.score /= self.num_queries

        # Use a dict to de-duplicate nodes
        all_nodes: Dict[str, NodeWithScore] = {}

        # Sum scores for each node
        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    cur_score = all_nodes[hash].score or 0.0
                    all_nodes[hash].score = cur_score + (node_with_score.score or 0.0)
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)
```

原理：
1. MinMax 归一化：将每个检索器的分数归一化到 [0, 1]
2. 应用权重：乘以对应检索器权重
3. 除以查询数：平均化多查询的影响
4. 合并去重：相同节点分数累加
5. 排序返回：按最终分数降序

适用：需要为不同检索器设置权重时（如你的代码中 keyword 和 vector 的权重）。

#### 模式 3: DIST_BASED_SCORE（基于距离的分数融合）

使用 `_relative_score_fusion`，但 `dist_based=True`：

```151:158:.venv/lib64/python3.11/site-packages/llama_index/core/retrievers/fusion_retriever.py
            if dist_based:
                # Set min and max based on mean and std dev
                mean_score = sum(scores) / len(scores)
                std_dev = (
                    sum((x - mean_score) ** 2 for x in scores) / len(scores)
                ) ** 0.5
                min_score = mean_score - 3 * std_dev
                max_score = mean_score + 3 * std_dev
```

原理：
- 使用均值 ± 3 倍标准差作为归一化范围，减少异常值影响
- 适用：分数分布不均匀、存在异常值

#### 模式 4: SIMPLE（简单融合）

```198:215:.venv/lib64/python3.11/site-packages/llama_index/core/retrievers/fusion_retriever.py
    def _simple_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """Apply simple fusion."""
        # Use a dict to de-duplicate nodes
        all_nodes: Dict[str, NodeWithScore] = {}
        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    max_score = max(
                        node_with_score.score or 0.0, all_nodes[hash].score or 0.0
                    )
                    all_nodes[hash].score = max_score
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)
```

原理：
- 去重：相同节点保留最高分
- 排序：按分数降序
- 适用：快速合并，不关心权重

### 4. 在你的代码中的应用

```105:111:askany/rag/faq_query_engine.py
        ensemble_retriever = QueryFusionRetriever(
            retrievers=[keyword_retriever, vector_retriever],
            retriever_weights=ensemble_weights,
            mode=FUSION_MODES.RELATIVE_SCORE,  # Supports weighted fusion
            similarity_top_k=rerank_candidate_k,  # Retrieve more candidates for reranking
            use_async=False,  # Use sync mode for simplicity
        )
```

配置说明：
- 两个检索器：keyword（精确匹配）和 vector（语义相似）
- 使用 `RELATIVE_SCORE`：支持通过 `ensemble_weights` 控制权重
- 权重示例：`[0.5, 0.5]` 或 `[0.7, 0.3]`（更偏 keyword）

### 5. 结果合并示例

假设：
- 查询："如何重置密码"
- 生成 3 个查询：原查询 + 2 个扩展查询
- 2 个检索器：keyword、vector

合并过程（RELATIVE_SCORE）：
1. 每个查询 × 每个检索器 = 6 组结果
2. 每组结果 MinMax 归一化
3. 应用权重（如 keyword 0.7，vector 0.3）
4. 除以查询数（3）
5. 相同节点分数累加
6. 按最终分数排序，返回 Top-K

### 总结

- RECIPROCAL_RANK：基于排名，不依赖分数
- RELATIVE_SCORE：支持权重，适合混合检索器
- DIST_BASED_SCORE：对异常值更稳健
- SIMPLE：快速去重合并

你的场景使用 `RELATIVE_SCORE` 是合适的选择，可以灵活调整 keyword 和 vector 的权重。

```python
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.RELATIVE_SCORE:
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.DIST_BASED_SCORE:
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == FUSION_MODES.SIMPLE:
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")
```

```python
    def _get_queries(self, original_query: str) -> List[QueryBundle]:
        prompt_str = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )
        response = self._llm.complete(prompt_str)

        # Strip code block and assume LLM properly put each query on a newline
        queries = response.text.strip("`").split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        if self._verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list.
        return [QueryBundle(q) for q in queries[: self.num_queries - 1]]
```

```python
    def _reciprocal_rerank_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """
        Apply reciprocal rank fusion.

        The original paper uses k=60 for best results:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        """
        k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
        fused_scores = {}
        hash_to_node = {}

        # compute reciprocal rank scores
        for nodes_with_scores in results.values():
            for rank, node_with_score in enumerate(
                sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
            ):
                hash = node_with_score.node.hash
                hash_to_node[hash] = node_with_score
                if hash not in fused_scores:
                    fused_scores[hash] = 0.0
                fused_scores[hash] += 1.0 / (rank + k)

        # sort results
        reranked_results = dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # adjust node scores
        reranked_nodes: List[NodeWithScore] = []
        for hash, score in reranked_results.items():
            reranked_nodes.append(hash_to_node[hash])
            reranked_nodes[-1].score = score

        return reranked_nodes
```

```python
    def _relative_score_fusion(
        self,
        results: Dict[Tuple[str, int], List[NodeWithScore]],
        dist_based: Optional[bool] = False,
    ) -> List[NodeWithScore]:
        """Apply relative score fusion."""
        # MinMax scale scores of each result set (highest value becomes 1, lowest becomes 0)
        # then scale by the weight of the retriever
        min_max_scores = {}
        for query_tuple, nodes_with_scores in results.items():
            if not nodes_with_scores:
                min_max_scores[query_tuple] = (0.0, 0.0)
                continue
            scores = [
                node_with_score.score or 0.0 for node_with_score in nodes_with_scores
            ]
            if dist_based:
                # Set min and max based on mean and std dev
                mean_score = sum(scores) / len(scores)
                std_dev = (
                    sum((x - mean_score) ** 2 for x in scores) / len(scores)
                ) ** 0.5
                min_score = mean_score - 3 * std_dev
                max_score = mean_score + 3 * std_dev
            else:
                min_score = min(scores)
                max_score = max(scores)
            min_max_scores[query_tuple] = (min_score, max_score)

        for query_tuple, nodes_with_scores in results.items():
            for node_with_score in nodes_with_scores:
                min_score, max_score = min_max_scores[query_tuple]
                # Scale the score to be between 0 and 1
                if max_score == min_score:
                    node_with_score.score = 1.0 if max_score > 0 else 0.0
                else:
                    node_with_score.score = (node_with_score.score - min_score) / (
                        max_score - min_score
                    )
                # Scale by the weight of the retriever
                retriever_idx = query_tuple[1]
                existing_score = node_with_score.score or 0.0
                node_with_score.score = (
                    existing_score * self._retriever_weights[retriever_idx]
                )
                # Divide by the number of queries
                node_with_score.score /= self.num_queries

        # Use a dict to de-duplicate nodes
        all_nodes: Dict[str, NodeWithScore] = {}

        # Sum scores for each node
        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    cur_score = all_nodes[hash].score or 0.0
                    all_nodes[hash].score = cur_score + (node_with_score.score or 0.0)
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)
```

```python
            if dist_based:
                # Set min and max based on mean and std dev
                mean_score = sum(scores) / len(scores)
                std_dev = (
                    sum((x - mean_score) ** 2 for x in scores) / len(scores)
                ) ** 0.5
                min_score = mean_score - 3 * std_dev
                max_score = mean_score + 3 * std_dev
```

```python
    def _simple_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """Apply simple fusion."""
        # Use a dict to de-duplicate nodes
        all_nodes: Dict[str, NodeWithScore] = {}
        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    max_score = max(
                        node_with_score.score or 0.0, all_nodes[hash].score or 0.0
                    )
                    all_nodes[hash].score = max_score
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)
```

```python
        ensemble_retriever = QueryFusionRetriever(
            retrievers=[keyword_retriever, vector_retriever],
            retriever_weights=ensemble_weights,
            mode=FUSION_MODES.RELATIVE_SCORE,  # Supports weighted fusion
            similarity_top_k=rerank_candidate_k,  # Retrieve more candidates for reranking
            use_async=False,  # Use sync mode for simplicity
        )
```

---

**User**

这个模式是最推荐的吗，还有别的模式可用吗

---

**Cursor**


分析各融合模式的适用场景，并给出推荐。



分析各融合模式的适用场景，并给出推荐。



分析各融合模式的适用场景，并给出推荐。

## 四种融合模式对比

### 1. RELATIVE_SCORE（当前使用）

优点：
- 支持权重，可控制 keyword 和 vector 的贡献
- 通过 MinMax 归一化处理不同分数范围
- 适合混合检索器场景

缺点：
- 计算相对复杂
- 对异常值敏感（除非用 DIST_BASED_SCORE）

适用场景：
- 需要精确控制检索器权重
- 不同检索器分数范围差异较大
- 你的场景（keyword + vector 混合）

### 2. RECIPROCAL_RANK（RRF）

优点：
- 不依赖原始分数，只依赖排名
- 对分数分布不敏感
- 经典且稳定

缺点：
- 不支持权重
- 可能忽略高分但排名靠后的结果

适用场景：
- 不同检索器分数不可比
- 不需要权重控制
- 更关注排名一致性

### 3. DIST_BASED_SCORE

优点：
- 基于统计（均值±3σ）归一化，对异常值更稳健
- 支持权重
- 适合分数分布不均匀的情况

缺点：
- 计算稍复杂
- 可能过度平滑极端值

适用场景：
- 分数分布不均匀
- 存在异常值
- 需要权重控制且对稳健性要求高

### 4. SIMPLE

优点：
- 实现简单，速度快
- 去重并保留最高分

缺点：
- 不支持权重
- 不考虑多查询的影响
- 可能丢失信息

适用场景：
- 快速原型
- 不需要复杂融合
- 性能优先

## 推荐建议

在你的场景（FAQ + 有 Reranker）下：

### 推荐 1：RELATIVE_SCORE（当前选择，适合大多数情况）

理由：
- 支持权重，可灵活调整 keyword/vector 比例
- 你已有 reranker，融合阶段主要做初步合并
- 代码注释也提到支持权重融合

```python
# 当前配置 - 推荐保持
mode=FUSION_MODES.RELATIVE_SCORE,
retriever_weights=ensemble_weights,  # [0.5, 0.5] 或 [0.7, 0.3] 等
```

### 推荐 2：RECIPROCAL_RANK（如果分数不可比）

如果你的 keyword 和 vector 检索器分数范围差异很大，且不需要权重控制：

```python
mode=FUSION_MODES.RECIPROCAL_RANK,
# 注意：retriever_weights 在此模式下无效
```

### 推荐 3：DIST_BASED_SCORE（如果分数分布异常）

如果发现分数分布不均匀或存在异常值：

```python
mode=FUSION_MODES.DIST_BASED_SCORE,
retriever_weights=ensemble_weights,
```

## 实际建议

由于你使用了 Reranker，融合阶段主要作用是：
1. 合并多个检索器的结果
2. 初步去重和排序
3. 为 Reranker 提供候选集

建议：
- 保持 RELATIVE_SCORE：支持权重，适合你的混合检索场景
- 如果不需要权重：可尝试 RECIPROCAL_RANK，更简单且稳定
- 如果遇到异常值问题：切换到 DIST_BASED_SCORE

## 可选改进：让模式可配置

可以让融合模式可配置，便于测试和调优：


```python
# 当前配置 - 推荐保持
mode=FUSION_MODES.RELATIVE_SCORE,
retriever_weights=ensemble_weights,  # [0.5, 0.5] 或 [0.7, 0.3] 等
```

```python
mode=FUSION_MODES.RECIPROCAL_RANK,
# 注意：retriever_weights 在此模式下无效
```

```python
mode=FUSION_MODES.DIST_BASED_SCORE,
retriever_weights=ensemble_weights,
```


建议添加配置项，使融合模式可配置：

```python
# 在 config.py 中添加
faq_fusion_mode: str = "relative_score"  # Options: "relative_score", "reciprocal_rank", "dist_based_score", "simple"
```

然后在 `faq_query_engine.py` 中使用：

```python
from askany.config import settings

# 根据配置选择模式
fusion_mode_map = {
    "relative_score": FUSION_MODES.RELATIVE_SCORE,
    "reciprocal_rank": FUSION_MODES.RECIPROCAL_RANK,
    "dist_based_score": FUSION_MODES.DIST_BASED_SCORE,
    "simple": FUSION_MODES.SIMPLE,
}
selected_mode = fusion_mode_map.get(
    getattr(settings, "faq_fusion_mode", "relative_score"),
    FUSION_MODES.RELATIVE_SCORE
)

ensemble_retriever = QueryFusionRetriever(
    retrievers=[keyword_retriever, vector_retriever],
    retriever_weights=ensemble_weights,
    mode=selected_mode,
    similarity_top_k=rerank_candidate_k,
    use_async=False,
)
```

## 总结

- 当前使用 RELATIVE_SCORE 是合理的选择，适合需要权重控制的混合检索场景。
- 其他可用模式：
  - RECIPROCAL_RANK：不需要权重时更简单稳定
  - DIST_BASED_SCORE：需要权重且对异常值更稳健
  - SIMPLE：快速简单，但功能有限
- 建议：保持 RELATIVE_SCORE，如需测试其他模式，可添加配置项以便切换。

需要我帮你实现可配置的融合模式吗？

```python
# 在 config.py 中添加
faq_fusion_mode: str = "relative_score"  # Options: "relative_score", "reciprocal_rank", "dist_based_score", "simple"
```

```python
from askany.config import settings

# 根据配置选择模式
fusion_mode_map = {
    "relative_score": FUSION_MODES.RELATIVE_SCORE,
    "reciprocal_rank": FUSION_MODES.RECIPROCAL_RANK,
    "dist_based_score": FUSION_MODES.DIST_BASED_SCORE,
    "simple": FUSION_MODES.SIMPLE,
}
selected_mode = fusion_mode_map.get(
    getattr(settings, "faq_fusion_mode", "relative_score"),
    FUSION_MODES.RELATIVE_SCORE
)

ensemble_retriever = QueryFusionRetriever(
    retrievers=[keyword_retriever, vector_retriever],
    retriever_weights=ensemble_weights,
    mode=selected_mode,
    similarity_top_k=rerank_candidate_k,
    use_async=False,
)
```

```python
# 在 config.py 中添加
faq_fusion_mode: str = "relative_score"  # Options: "relative_score", "reciprocal_rank", "dist_based_score", "simple"
```

```python
from askany.config import settings

# 根据配置选择模式
fusion_mode_map = {
    "relative_score": FUSION_MODES.RELATIVE_SCORE,
    "reciprocal_rank": FUSION_MODES.RECIPROCAL_RANK,
    "dist_based_score": FUSION_MODES.DIST_BASED_SCORE,
    "simple": FUSION_MODES.SIMPLE,
}
selected_mode = fusion_mode_map.get(
    getattr(settings, "faq_fusion_mode", "relative_score"),
    FUSION_MODES.RELATIVE_SCORE
)

ensemble_retriever = QueryFusionRetriever(
    retrievers=[keyword_retriever, vector_retriever],
    retriever_weights=ensemble_weights,
    mode=selected_mode,
    similarity_top_k=rerank_candidate_k,
    use_async=False,
)
```

