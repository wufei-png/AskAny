# 查看 llama_index 模块工作方式
_Exported on 2025/11/27 at GMT+8 0:39:29 from Cursor (2.1.26)_

---

**User**

@.venv/lib/python3.11/site-packages/llama_index/core/query_engine/sub_question_query_engine.py 查看代码和文档@llama_index/docs/api_reference/api_reference/query_engine/sub_question.md 解释这个模块工作方式

---

**Cursor**


分析 `SubQuestionQueryEngine` 模块的工作方式。



## SubQuestionQueryEngine 模块工作方式

`SubQuestionQueryEngine` 将复杂查询拆分为多个子问题，并行执行后合成最终答案。

### 核心组件

1. **SubQuestionAnswerPair**：存储子问题及其答案
   - `sub_q`: 子问题对象（包含问题文本和工具名称）
   - `answer`: 子问题的答案
   - `sources`: 答案的来源节点

2. **SubQuestionQueryEngine**：主查询引擎类

### 工作流程

#### 1. 初始化阶段

```60:77:.venv/lib/python3.11/site-packages/llama_index/core/query_engine/sub_question_query_engine.py
    def __init__(
        self,
        question_gen: BaseQuestionGenerator,
        response_synthesizer: BaseSynthesizer,
        query_engine_tools: Sequence[QueryEngineTool],
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
        use_async: bool = False,
    ) -> None:
        self._question_gen = question_gen
        self._response_synthesizer = response_synthesizer
        self._metadatas = [x.metadata for x in query_engine_tools]
        self._query_engines = {
            tool.metadata.name: tool.query_engine for tool in query_engine_tools
        }
        self._verbose = verbose
        self._use_async = use_async
        super().__init__(callback_manager)
```

- 接收多个 `QueryEngineTool`，每个工具对应一个查询引擎
- 建立工具名称到查询引擎的映射

#### 2. 查询执行流程（同步版本）

```134:173:.venv/lib/python3.11/site-packages/llama_index/core/query_engine/sub_question_query_engine.py
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            sub_questions = self._question_gen.generate(self._metadatas, query_bundle)

            colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

            if self._verbose:
                print_text(f"Generated {len(sub_questions)} sub questions.\n")

            if self._use_async:
                tasks = [
                    self._aquery_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

                qa_pairs_all = run_async_tasks(tasks)
                qa_pairs_all = cast(List[Optional[SubQuestionAnswerPair]], qa_pairs_all)
            else:
                qa_pairs_all = [
                    self._query_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

            # filter out sub questions that failed
            qa_pairs: List[SubQuestionAnswerPair] = list(filter(None, qa_pairs_all))

            nodes = [self._construct_node(pair) for pair in qa_pairs]

            source_nodes = [node for qa_pair in qa_pairs for node in qa_pair.sources]
            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=source_nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
```

步骤：
1. 生成子问题：使用 `question_gen` 将复杂查询拆分为多个子问题
2. 执行子问题：
   - 同步：顺序执行每个子问题
   - 异步：并行执行所有子问题
3. 过滤失败：移除执行失败的子问题
4. 构建节点：将每个子问题的答案转换为节点
5. 合成最终答案：使用 `response_synthesizer` 将所有答案合成为最终响应

#### 3. 单个子问题执行

```249:278:.venv/lib/python3.11/site-packages/llama_index/core/query_engine/sub_question_query_engine.py
    def _query_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[SubQuestionAnswerPair]:
        try:
            with self.callback_manager.event(
                CBEventType.SUB_QUESTION,
                payload={EventPayload.SUB_QUESTION: SubQuestionAnswerPair(sub_q=sub_q)},
            ) as event:
                question = sub_q.sub_question
                query_engine = self._query_engines[sub_q.tool_name]

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

                response = query_engine.query(question)
                response_text = str(response)

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

                qa_pair = SubQuestionAnswerPair(
                    sub_q=sub_q, answer=response_text, sources=response.source_nodes
                )

                event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

            return qa_pair
        except ValueError:
            logger.warning(f"[{sub_q.tool_name}] Failed to run {question}")
            return None
```

- 根据 `tool_name` 选择对应的查询引擎
- 执行查询并获取答案
- 捕获异常，失败时返回 `None`

### 关键特性

1. 支持同步和异步执行
   - `use_async=False`：顺序执行
   - `use_async=True`：并行执行（更快）

2. 容错机制
   - 失败的子问题会被过滤，不影响其他子问题

3. 可观测性
   - `verbose=True` 时打印每个子问题和答案
   - 使用不同颜色区分不同子问题

4. 灵活的响应合成
   - 使用 `response_synthesizer` 将多个子答案合成为最终答案

### 使用场景

适用于需要多数据源或多步骤推理的复杂查询，例如：
- "比较 A 和 B 的优缺点"
- "总结 X、Y、Z 三个主题的内容"
- "分析不同数据源中的相关信息"

该模块通过分解、并行执行和智能合成，高效处理复杂查询。

```python
    def __init__(
        self,
        question_gen: BaseQuestionGenerator,
        response_synthesizer: BaseSynthesizer,
        query_engine_tools: Sequence[QueryEngineTool],
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
        use_async: bool = False,
    ) -> None:
        self._question_gen = question_gen
        self._response_synthesizer = response_synthesizer
        self._metadatas = [x.metadata for x in query_engine_tools]
        self._query_engines = {
            tool.metadata.name: tool.query_engine for tool in query_engine_tools
        }
        self._verbose = verbose
        self._use_async = use_async
        super().__init__(callback_manager)
```

```python
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            sub_questions = self._question_gen.generate(self._metadatas, query_bundle)

            colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

            if self._verbose:
                print_text(f"Generated {len(sub_questions)} sub questions.\n")

            if self._use_async:
                tasks = [
                    self._aquery_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

                qa_pairs_all = run_async_tasks(tasks)
                qa_pairs_all = cast(List[Optional[SubQuestionAnswerPair]], qa_pairs_all)
            else:
                qa_pairs_all = [
                    self._query_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

            # filter out sub questions that failed
            qa_pairs: List[SubQuestionAnswerPair] = list(filter(None, qa_pairs_all))

            nodes = [self._construct_node(pair) for pair in qa_pairs]

            source_nodes = [node for qa_pair in qa_pairs for node in qa_pair.sources]
            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=source_nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
```

```python
    def _query_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[SubQuestionAnswerPair]:
        try:
            with self.callback_manager.event(
                CBEventType.SUB_QUESTION,
                payload={EventPayload.SUB_QUESTION: SubQuestionAnswerPair(sub_q=sub_q)},
            ) as event:
                question = sub_q.sub_question
                query_engine = self._query_engines[sub_q.tool_name]

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

                response = query_engine.query(question)
                response_text = str(response)

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

                qa_pair = SubQuestionAnswerPair(
                    sub_q=sub_q, answer=response_text, sources=response.source_nodes
                )

                event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

            return qa_pair
        except ValueError:
            logger.warning(f"[{sub_q.tool_name}] Failed to run {question}")
            return None
```

---

**User**

具体是用什么分解，并行是子问题并行吗，即使第二个问题依赖第一个问题的答案？SubQuestionQueryEngine可以用 @askany/rag/router.py:124-140 这里的faq_query_engine和docs_query_engine查询实现 @askany/rag/router.py:82 一样的效果吗

---

**Cursor**


查看相关代码，回答这些问题。



回答你的三个问题：



## 1. 具体是用什么分解问题？

使用 LLM 分解问题。`SubQuestionQueryEngine` 使用 `BaseQuestionGenerator`（默认是 `LLMQuestionGenerator` 或 `OpenAIQuestionGenerator`），通过 prompt 让 LLM 将复杂查询拆分为子问题。

```67:81:.venv/lib/python3.11/site-packages/llama_index/core/question_gen/llm_generators.py
    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = self._llm.predict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output
```

Prompt 模板示例：

```19:23:.venv/lib/python3.11/site-packages/llama_index/core/question_gen/prompts.py
PREFIX = """\
Given a user question, and a list of tools, output a list of relevant sub-questions \
in json markdown that when composed can help answer the full user question:

"""
```

LLM 根据工具列表和用户问题生成子问题列表。

## 2. 并行是子问题并行吗？即使第二个问题依赖第一个问题的答案？

是的，并行执行所有子问题，且不处理依赖关系。

```145:152:.venv/lib/python3.11/site-packages/llama_index/core/query_engine/sub_question_query_engine.py
            if self._use_async:
                tasks = [
                    self._aquery_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

                qa_pairs_all = run_async_tasks(tasks)
```

在 `use_async=True` 时，所有子问题会并行执行，不会等待前一个完成。如果第二个问题依赖第一个问题的答案，并行执行会导致错误或结果不准确。

适用场景：
- 子问题相互独立（如“比较 A 和 B 的优缺点”）
- 每个子问题可以独立查询不同数据源

不适用场景：
- 子问题有依赖关系（如“先查 X，再基于 X 的结果查 Y”）

## 3. SubQuestionQueryEngine 能否实现 `_route_auto` 的效果？

不能完全替代。两者的设计目标不同：

### `_route_auto` 的逻辑（条件分支）

```95:175:askany/rag/router.py
    def _route_auto(
        self, query: str, metadata_filters: Optional[Dict[str, str]] = None
    ) -> str:
        """Smart auto routing with FAQ score checking and docs enhancement.

        Flow:
        1. Check if query is code-related (use code if yes)
        2. If not code, retrieve from FAQ
        3. If FAQ score is sufficient, return FAQ answer
        4. If FAQ score is insufficient, use FAQ answer as context and enhance with docs
        """
        # Step 1: Check if it's code-related
        if self._is_code_query(query):
            # TODO: Implement code search
            return "Code search not yet implemented"

        # Step 2: Try FAQ retrieval
        if not self.faq_query_engine:
            logger.warning("No FAQ engine found, fallback to docs for query: %s", query)
            # No FAQ engine, fallback to docs
            return self.docs_query_engine.query(query, metadata_filters)

        # Retrieve from FAQ and get top score (only retrieve once)
        faq_nodes, top_score = self.faq_query_engine.retrieve_with_scores(
            query, metadata_filters
        )

        # Step 3: Check if FAQ score is sufficient
        if top_score >= settings.faq_score_threshold and faq_nodes:
            # FAQ score is good enough, synthesize answer from already retrieved nodes
            return self.faq_query_engine.synthesize_from_nodes(
                query, faq_nodes
            )  # 压缩信息，调用llm总结

        # Step 4: FAQ score is insufficient or no results, enhance with docs

        # Retrieve docs using original query (no context added to retrieval)
        docs_nodes = self.docs_query_engine.retrieve(query, metadata_filters)

        # Merge FAQ nodes with docs nodes, but mark FAQ nodes with low reliability
        all_nodes = list(docs_nodes)
        faq_nodes_merged = False
        should_merge_low_confidence_faq = (
            faq_nodes
            and top_score < settings.faq_score_threshold
            and top_score >= settings.faq_second_score_threshold
        )
        if should_merge_low_confidence_faq:
            # Mark FAQ nodes as low reliability and merge them
            marked_faq_nodes = self._mark_faq_nodes_with_low_reliability(
                faq_nodes, top_score
            )
            # Use node_id as unique identifier (more reliable than hash)
            seen_ids = {self._get_node_id(node) for node in all_nodes}
            for faq_node in marked_faq_nodes:
                faq_node_id = self._get_node_id(faq_node)
                if faq_node_id not in seen_ids:
                    all_nodes.append(faq_node)  # Add at end (lower priority)
                    seen_ids.add(faq_node_id)
                    faq_nodes_merged = True

        # Create enhanced query with instructions about FAQ nodes
        if faq_nodes_merged:
            enhanced_query = f"""{query}

重要提示：以下内容中，标记为"[FAQ-低相关性]"的内容来自FAQ库，但这些信息的相关性分数({top_score:.2f})低于推荐阈值({settings.faq_score_threshold:.2f})。
请优先参考文档库中的信息（无标记的内容），FAQ内容仅作为补充参考，使用时请谨慎验证其准确性。"""
        else:
            enhanced_query = query

        # Synthesize final answer with merged nodes (FAQ + docs)
        return self.docs_query_engine.synthesize_from_nodes(
            query_str=enhanced_query, nodes=all_nodes, context=None
        )
```

特点：
- 顺序执行：先查 FAQ，根据分数决定是否查 docs
- 条件分支：基于分数阈值做决策
- 节点合并：合并 FAQ 和 docs 节点，并标记低相关性
- 单一查询：对同一个查询进行路由和增强

### `SubQuestionQueryEngine` 的逻辑（问题分解）

特点：
- 问题分解：将复杂问题拆分为多个子问题
- 并行执行：所有子问题并行执行（不考虑依赖）
- 答案合成：将所有子问题的答案合成为最终答案
- 多工具路由：每个子问题可以分配给不同的工具/引擎

### 为什么不能替代？

1. 执行模式不同：
   - `_route_auto`：顺序执行，基于条件分支
   - `SubQuestionQueryEngine`：并行执行所有子问题

2. 设计目标不同：
   - `_route_auto`：针对同一查询，根据分数决定使用哪个数据源
   - `SubQuestionQueryEngine`：将复杂查询拆分为多个独立子问题

3. 无法实现的功能：
   - 无法根据 FAQ 分数阈值做条件判断
   - 无法标记和合并低相关性 FAQ 节点
   - 无法实现“先查 FAQ，分数不够再查 docs”的流程

### 可能的替代方案

如果要用 `SubQuestionQueryEngine` 实现类似效果，需要：
1. 自定义 `QuestionGenerator`，让它生成类似“查询 FAQ”和“查询 docs”的子问题
2. 修改执行逻辑，支持顺序执行和条件判断
3. 实现节点合并和标记逻辑

但这样会改变 `SubQuestionQueryEngine` 的设计，不如直接使用 `_route_auto` 更合适。

总结：`SubQuestionQueryEngine` 适合将复杂查询拆分为多个独立子问题并并行执行；`_route_auto` 适合对同一查询进行条件路由和数据源合并。两者解决不同的问题。

```python
    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = self._llm.predict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output
```

```python
PREFIX = """\
Given a user question, and a list of tools, output a list of relevant sub-questions \
in json markdown that when composed can help answer the full user question:

"""
```

```python
            if self._use_async:
                tasks = [
                    self._aquery_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

                qa_pairs_all = run_async_tasks(tasks)
```

```python
    def _route_auto(
        self, query: str, metadata_filters: Optional[Dict[str, str]] = None
    ) -> str:
        """Smart auto routing with FAQ score checking and docs enhancement.

        Flow:
        1. Check if query is code-related (use code if yes)
        2. If not code, retrieve from FAQ
        3. If FAQ score is sufficient, return FAQ answer
        4. If FAQ score is insufficient, use FAQ answer as context and enhance with docs
        """
        # Step 1: Check if it's code-related
        if self._is_code_query(query):
            # TODO: Implement code search
            return "Code search not yet implemented"

        # Step 2: Try FAQ retrieval
        if not self.faq_query_engine:
            logger.warning("No FAQ engine found, fallback to docs for query: %s", query)
            # No FAQ engine, fallback to docs
            return self.docs_query_engine.query(query, metadata_filters)

        # Retrieve from FAQ and get top score (only retrieve once)
        faq_nodes, top_score = self.faq_query_engine.retrieve_with_scores(
            query, metadata_filters
        )

        # Step 3: Check if FAQ score is sufficient
        if top_score >= settings.faq_score_threshold and faq_nodes:
            # FAQ score is good enough, synthesize answer from already retrieved nodes
            return self.faq_query_engine.synthesize_from_nodes(
                query, faq_nodes
            )  # 压缩信息，调用llm总结

        # Step 4: FAQ score is insufficient or no results, enhance with docs

        # Retrieve docs using original query (no context added to retrieval)
        docs_nodes = self.docs_query_engine.retrieve(query, metadata_filters)

        # Merge FAQ nodes with docs nodes, but mark FAQ nodes with low reliability
        all_nodes = list(docs_nodes)
        faq_nodes_merged = False
        should_merge_low_confidence_faq = (
            faq_nodes
            and top_score < settings.faq_score_threshold
            and top_score >= settings.faq_second_score_threshold
        )
        if should_merge_low_confidence_faq:
            # Mark FAQ nodes as low reliability and merge them
            marked_faq_nodes = self._mark_faq_nodes_with_low_reliability(
                faq_nodes, top_score
            )
            # Use node_id as unique identifier (more reliable than hash)
            seen_ids = {self._get_node_id(node) for node in all_nodes}
            for faq_node in marked_faq_nodes:
                faq_node_id = self._get_node_id(faq_node)
                if faq_node_id not in seen_ids:
                    all_nodes.append(faq_node)  # Add at end (lower priority)
                    seen_ids.add(faq_node_id)
                    faq_nodes_merged = True

        # Create enhanced query with instructions about FAQ nodes
        if faq_nodes_merged:
            enhanced_query = f"""{query}

重要提示：以下内容中，标记为"[FAQ-低相关性]"的内容来自FAQ库，但这些信息的相关性分数({top_score:.2f})低于推荐阈值({settings.faq_score_threshold:.2f})。
请优先参考文档库中的信息（无标记的内容），FAQ内容仅作为补充参考，使用时请谨慎验证其准确性。"""
        else:
            enhanced_query = query

        # Synthesize final answer with merged nodes (FAQ + docs)
        return self.docs_query_engine.synthesize_from_nodes(
            query_str=enhanced_query, nodes=all_nodes, context=None
        )
```

