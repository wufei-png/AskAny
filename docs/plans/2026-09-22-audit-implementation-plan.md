# 工作流注释与 docstring 清理：实现计划

日期：2026-09-22。状态：已实施并验证，见执行记录。

## 目标与接手方式

修正与现有行为不符的注释、复制残留 docstring 和无明确需求的 TODO。保留运行时行为，尤其是领域词否决直接回答的规则。

本计划只涉及 `wufei-png/AskAny`，当前本地目录为 `AskAny`，核对基线为 `75b7b9cbe2cc8d7bcb9d72db42f88bff2d97b15a`，计划生成前工作区干净。先读根 `AGENTS.md`、`docs/current-runtime.md`，运行 `git status --short`、`git remote -v`、`git rev-parse HEAD`，再读目标与调用链。若代码漂移，复核事实，不覆盖其他修改。

新会话启动指令：

> 阅读 AGENTS.md、docs/current-runtime.md 和 docs/plans/2026-09-22-audit-implementation-plan.md，实施已确认的注释/docstring 清理并验证无行为变化。领域词策略已经确定为保持现状，不调整赋值、阈值、prompt、Schema 或路由，不重新讨论已定选择。

本次会话只交付计划。新会话收到实施指令后执行；提交与推送按该会话授权处理。

## 已核实事实与裁决

来源为 2026-09-22 审计 C01–C05，以下已包含独立实施所需事实。

| 项目 | 当前事实与处理 |
| --- | --- |
| C01 缺失路径 | `RelevantResult.validate_and_check_existence()` 对缺失本地路径告警，最终原样返回 `paths`；重写“则报错”注释，删除注释掉的 `raise`，不恢复异常或过滤路径 |
| C02 复制残留 | 第一阶段模块及 `DirectAnswerGenerator._format_prompt()` 实际做直接回答资格与 Web/RAG 路由判断；修正误称 SubProblemGenerator / sub-problem generation 的 docstring |
| C03 模糊 TODO | 删除 `TODO maybe return score？`；将“优化 word_freq.txt 再使用”改为当前已经运行的规则说明，不增加新需求 |
| C04 废弃字段 | 删除 `NoRelevantResult`、`NoRelevantResultWithoutSubQueries` 中注释掉的 `reasoning: str = Field(...)`；同一目标模块中 `WebOrRagAnswer` 的同类废弃定义一并清理，不恢复字段、不编造删除原因 |
| C05 路径说明 | 明确只拒绝 NUL 与空白路径；这不是目录访问授权检查，不声称已有完整的路径安全边界 |

目标文件：

- `askany/workflow/AnalysisRelated_langchain.py`，核对时 blob SHA：`956193e0c18193076eb3c3efd1e74e2a808ddaef`。
- `askany/workflow/firstStageRelevant_langchain.py`，核对时 blob SHA：`f397c7dbbe7f3458f24294dd9c74f300ddedb126`。

只读上下文：`askany/ingest/keyword_extract_from_tfidf.py`、`askany/workflow/workflow_filter.py`、`askany/config.py`。

### 已确认的领域词决策

`extract_keywords_set()` 默认只返回领域词。模型先返回 `can_direct_answer=True` 时，只要关键词非空，循环后的无条件赋值就令结果变成 False。配置 `freq_in_rag_threshold` 当前默认 80，但不能改变这一最终结果；日志中的“小于10”也不应被当作实际阈值。

用户明确选择保持现状：命中任意领域词就进入后续检索路由，本次只修注释。需要准确说明是继续交给后续 Web/RAG 路由，而不是保证执行网络搜索或固定执行 RAG。

已通过抽取当前 `generate()` 方法并替换外部依赖的隔离执行，核对无关键词、高于阈值、等于阈值、低于阈值四种情况，结果依次为 True、False、False、False。此为局部方法证据，不是项目集成测试或模型效果评估。

本次不删除循环或重复赋值，不让阈值重新生效，不改日志字符串，不调整实际 `Field(description=...)` 或 Pydantic 模型 docstring，以免顺带改变结构化输出说明。

## 实施步骤

1. 对照现有代码与上述事实，确认注释修正仍适用。
2. 在路径校验处说明：NUL/空白会拒绝，缺失本地路径仅告警并保留，目的是不中断整次回答；特殊标识符仍按现有规则处理。
3. 删除弱 TODO 与指定的注释掉的字段定义，保留类型抑制、lint pragma 和 LangChain stub/运行时差异说明。
4. 修正模块与 `_format_prompt()` docstring；将词频 TODO 改为“领域词命中会否决直接回答，交由后续路由处理”等准确说明。不要把当前冗余逻辑当作获准重构范围。
5. 检查 diff 只涉及这两个文件的注释/docstring，执行静态验证及下述 AST 比较。无需为文案增添测试。
6. 更新执行记录，说明行为未变及所有检查的真实结果。

## 验证与完成条件

开始编辑前记录当前工作树基线；若存在他人未提交修改，比较编辑前后的文件，不直接以 HEAD 代替编辑前状态。清理前后移除 AST 中 Module / ClassDef / FunctionDef / AsyncFunctionDef 的首个 docstring 节点后，`ast.dump(..., include_attributes=False)` 应完全一致。该比较允许注释/docstring 变化，却能发现误改字符串、字段、赋值和分支。

从仓库根目录按 `AGENTS.md` 执行质量检查：

```bash
uv lock --check
uv run --locked ruff check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format --check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked --all-extras pyright
uv run --locked pre-commit run --all-files
uv run --locked pytest -q test -rs
git diff --check
```

若 pre-commit 自动修改了无关文件，检查并隔离本轮工具产生的修改，不丢弃他人改动。外部数据库、模型或数据缺失必须按仓库既有约定记录跳过；编程错误不能冒充前置条件缺失。全库已有失败单独报告，不扩大本次注释清理范围。

完成标准：注释准确反映路径容错和领域词策略；没有新增数值置信度、业务字段或行为；去除 docstring 后 AST 不变；检查结果与限制有记录。不要把未执行的集成测试写成通过。

## 执行记录

- [x] C01–C05 指定清理完成。仅修改两个目标 Python 文件的注释和 docstring，未改赋值、分支、Schema、prompt 或日志字符串。
- [x] 领域词命中会否决直接回答，交由后续 Web/RAG 路由判断；原有行为保留。
- [x] 两个目标文件以编辑前的 HEAD 版本为基线，移除模块、类、函数和异步函数的首个 docstring 节点后，AST 完全一致。编辑前 HEAD 为 `75b7b9cbe2cc8d7bcb9d72db42f88bff2d97b15a`，目标文件 blob SHA 与计划记录一致；工作区原有未跟踪计划文件未覆盖。
- [x] `uv lock --check`、Ruff lint、Ruff format、Pyright、`pre-commit run --all-files` 和 `git diff --check` 均通过。pre-commit 未改写其他文件。
- [x] 原样执行 `uv run --locked pytest -q test -rs`：239 passed、19 skipped、16 failed。16 个失败均在 `test/test_faq_query_engine.py`，其初始化触发 LlamaIndex 默认 OpenAI 模型检查，当前环境缺少 `OPENAI_API_KEY`。以 `IS_TESTING=1` 启用 LlamaIndex 的模拟模型模式重跑同一套件：255 passed、19 skipped。跳过项是缺失本地 LightRAG 问题文件及未开启的 Mem0、QA cache、RAGAS 集成测试；未将其记为通过。
- [x] 已审查实际 diff：仅含上述两个 Python 文件与本执行记录；清理时补回了类定义间应有的空行并复验，无其他代码变更。
