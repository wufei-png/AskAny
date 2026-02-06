# 解释三个query_engine的工作方式
_Exported on 2025/11/27 at GMT+8 0:38:50 from Cursor (2.1.26)_

---

**User**

@.venv/lib/python3.11/site-packages/llama_index/core/query_engine/multistep_query_engine.py @.venv/lib/python3.11/site-packages/llama_index/core/query_engine/retry_query_engine.py @.venv/lib/python3.11/site-packages/llama_index/core/query_engine/retry_source_query_engine.py 解释这三个query_engine的工作方式

---

**Cursor**


分析这三个 query engine 的工作方式。



这三个 query engine 的工作方式如下：

## 1. MultiStepQueryEngine（多步骤查询引擎）

将复杂查询分解为多个步骤，逐步求解。

### 核心工作流程：

```126:178:.venv/lib/python3.11/site-packages/llama_index/core/query_engine/multistep_query_engine.py
    def _query_multistep(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore], Dict[str, Any]]:
        """Run query combiner."""
        prev_reasoning = ""
        cur_response = None
        should_stop = False
        cur_steps = 0

        # use response
        final_response_metadata: Dict[str, Any] = {"sub_qa": []}

        text_chunks = []
        source_nodes = []
        while not should_stop:
            if self._num_steps is not None and cur_steps >= self._num_steps:
                should_stop = True
                break
            elif should_stop:
                break

            updated_query_bundle = self._combine_queries(query_bundle, prev_reasoning)

            # TODO: make stop logic better
            stop_dict = {"query_bundle": updated_query_bundle}
            if self._stop_fn(stop_dict):
                should_stop = True
                break

            cur_response = self._query_engine.query(updated_query_bundle)

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Answer: {cur_response!s}"
            )
            text_chunks.append(cur_qa_text)
            for source_node in cur_response.source_nodes:
                source_nodes.append(source_node)
            # update metadata
            final_response_metadata["sub_qa"].append(
                (updated_query_bundle.query_str, cur_response)
            )

            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n- {cur_response!s}\n"
            )
            cur_steps += 1

        nodes = [
            NodeWithScore(node=TextNode(text=text_chunk)) for text_chunk in text_chunks
        ]
        return nodes, source_nodes, final_response_metadata
```

**工作步骤：**
1. 初始化：`prev_reasoning = ""`，记录历史推理
2. 循环执行（最多 `num_steps` 次）：
   - 使用 `_combine_queries` 结合原始查询与历史推理生成新查询
   - 检查停止条件（`stop_fn` 或达到最大步数）
   - 用基础 query engine 执行查询
   - 将问答对加入 `text_chunks`，更新 `prev_reasoning`
3. 最终合成：用 `response_synthesizer` 合并所有步骤的结果

**特点：**
- 支持早停（`early_stopping=True`）
- 默认停止条件：查询中包含 "none"
- 适合需要多步推理的复杂问题

---

## 2. RetryQueryEngine 和 RetryGuidelineQueryEngine（重试查询引擎）

基于评估结果自动重试，改进查询。

### RetryQueryEngine 工作流程：

```50:71:.venv/lib/python3.11/site-packages/llama_index/core/query_engine/retry_query_engine.py
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        assert not isinstance(response, AsyncStreamingResponse)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryQueryEngine(
                self._query_engine, self._evaluator, self.max_retries - 1
            )
            query_transformer = FeedbackQueryTransformation()
            new_query = query_transformer.run(query_bundle, {"evaluation": eval})
            return new_query_engine.query(new_query)
```

**工作步骤：**
1. 执行查询：调用底层 query engine
2. 评估结果：用 `evaluator` 评估响应
3. 判断：
   - 通过：返回响应
   - 未通过：使用 `FeedbackQueryTransformation` 基于评估反馈改写查询，创建新的 `RetryQueryEngine`（`max_retries - 1`），递归重试

### RetryGuidelineQueryEngine：

```117:142:.venv/lib/python3.11/site-packages/llama_index/core/query_engine/retry_query_engine.py
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        assert not isinstance(response, AsyncStreamingResponse)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._guideline_evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryGuidelineQueryEngine(
                self._query_engine,
                self._guideline_evaluator,
                self.resynthesize_query,
                self.max_retries - 1,
                self.callback_manager,
            )
            new_query = self.query_transformer.run(query_bundle, {"evaluation": eval})
            logger.debug("New query: %s", new_query.query_str)
            return new_query_engine.query(new_query)
```

**区别：**
- 使用 `GuidelineEvaluator`（基于特定指南评估）
- 支持 `resynthesize_query` 选项，可重新合成查询

---

## 3. RetrySourceQueryEngine（基于源节点重试的查询引擎）

评估失败时，筛选并重用通过评估的源节点，重建索引后重试。

### 核心工作流程：

```46:88:.venv/lib/python3.11/site-packages/llama_index/core/query_engine/retry_source_query_engine.py
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        response = self._query_engine._query(query_bundle)
        assert not isinstance(response, AsyncStreamingResponse)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            # Test source nodes
            source_evals = [
                self._evaluator.evaluate(
                    query=query_str,
                    response=typed_response.response,
                    contexts=[source_node.get_content()],
                )
                for source_node in typed_response.source_nodes
            ]
            orig_nodes = typed_response.source_nodes
            assert len(source_evals) == len(orig_nodes)
            new_docs = []
            for node, eval_result in zip(orig_nodes, source_evals):
                if eval_result:
                    new_docs.append(Document(text=node.node.get_content()))
            if len(new_docs) == 0:
                raise ValueError("No source nodes passed evaluation.")
            new_index = SummaryIndex.from_documents(
                new_docs,
            )
            new_retriever_engine = RetrieverQueryEngine(new_index.as_retriever())
            new_query_engine = RetrySourceQueryEngine(
                new_retriever_engine,
                self._evaluator,
                self._llm,
                self.max_retries - 1,
            )
            return new_query_engine.query(query_bundle)
```

**工作步骤：**
1. 执行查询并评估整体响应
2. 若未通过，评估每个源节点：
   - 对每个 `source_node` 单独评估
   - 筛选通过评估的节点
3. 重建索引：
   - 用通过评估的节点创建新 `SummaryIndex`
   - 创建新的 `RetrieverQueryEngine`
4. 递归重试：用新引擎和原查询重试（`max_retries - 1`）

**特点：**
- 不修改查询，而是优化数据源
- 适合源节点质量不一致的场景
- 需要 `RetrieverQueryEngine` 作为底层引擎

---

## 总结对比

| 引擎 | 改进策略 | 适用场景 |
|------|---------|---------|
| **MultiStepQueryEngine** | 分解复杂查询为多步骤 | 需要多步推理的复杂问题 |
| **RetryQueryEngine** | 基于评估反馈改写查询 | 查询表达不准确，需要优化查询 |
| **RetryGuidelineQueryEngine** | 基于指南评估并改写查询 | 需要遵循特定指南的场景 |
| **RetrySourceQueryEngine** | 筛选优质源节点重建索引 | 源数据质量不一致，需要过滤噪声 |

这些引擎可组合使用，构建更稳健的查询系统。

```python
    def _query_multistep(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore], Dict[str, Any]]:
        """Run query combiner."""
        prev_reasoning = ""
        cur_response = None
        should_stop = False
        cur_steps = 0

        # use response
        final_response_metadata: Dict[str, Any] = {"sub_qa": []}

        text_chunks = []
        source_nodes = []
        while not should_stop:
            if self._num_steps is not None and cur_steps >= self._num_steps:
                should_stop = True
                break
            elif should_stop:
                break

            updated_query_bundle = self._combine_queries(query_bundle, prev_reasoning)

            # TODO: make stop logic better
            stop_dict = {"query_bundle": updated_query_bundle}
            if self._stop_fn(stop_dict):
                should_stop = True
                break

            cur_response = self._query_engine.query(updated_query_bundle)

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Answer: {cur_response!s}"
            )
            text_chunks.append(cur_qa_text)
            for source_node in cur_response.source_nodes:
                source_nodes.append(source_node)
            # update metadata
            final_response_metadata["sub_qa"].append(
                (updated_query_bundle.query_str, cur_response)
            )

            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n- {cur_response!s}\n"
            )
            cur_steps += 1

        nodes = [
            NodeWithScore(node=TextNode(text=text_chunk)) for text_chunk in text_chunks
        ]
        return nodes, source_nodes, final_response_metadata
```

```python
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        assert not isinstance(response, AsyncStreamingResponse)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryQueryEngine(
                self._query_engine, self._evaluator, self.max_retries - 1
            )
            query_transformer = FeedbackQueryTransformation()
            new_query = query_transformer.run(query_bundle, {"evaluation": eval})
            return new_query_engine.query(new_query)
```

```python
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        assert not isinstance(response, AsyncStreamingResponse)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._guideline_evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryGuidelineQueryEngine(
                self._query_engine,
                self._guideline_evaluator,
                self.resynthesize_query,
                self.max_retries - 1,
                self.callback_manager,
            )
            new_query = self.query_transformer.run(query_bundle, {"evaluation": eval})
            logger.debug("New query: %s", new_query.query_str)
            return new_query_engine.query(new_query)
```

```python
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        response = self._query_engine._query(query_bundle)
        assert not isinstance(response, AsyncStreamingResponse)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            # Test source nodes
            source_evals = [
                self._evaluator.evaluate(
                    query=query_str,
                    response=typed_response.response,
                    contexts=[source_node.get_content()],
                )
                for source_node in typed_response.source_nodes
            ]
            orig_nodes = typed_response.source_nodes
            assert len(source_evals) == len(orig_nodes)
            new_docs = []
            for node, eval_result in zip(orig_nodes, source_evals):
                if eval_result:
                    new_docs.append(Document(text=node.node.get_content()))
            if len(new_docs) == 0:
                raise ValueError("No source nodes passed evaluation.")
            new_index = SummaryIndex.from_documents(
                new_docs,
            )
            new_retriever_engine = RetrieverQueryEngine(new_index.as_retriever())
            new_query_engine = RetrySourceQueryEngine(
                new_retriever_engine,
                self._evaluator,
                self._llm,
                self.max_retries - 1,
            )
            return new_query_engine.query(query_bundle)
```

