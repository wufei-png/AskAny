目标：从llamaindex的@askany/workflow/workflow_llamaindex.py 切换为langgraph的@askany/workflow/workflow_langgraph.py 尽可能使用langchain中最合适，效果最好，新颖的方式实现。
RAG使用原llamaindex的能力。

已经对接了mcp server docs-langchain，你可用于查询langgraph的相关文档。
同时我也阅读了相关文档，将可能用到的文档放在@dev_readme/from_llamaindex_to_langchain/langchain.md @dev_readme/from_llamaindex_to_langchain/langgraph.md
tool的迁移参考 @dev_readme/from_llamaindex_to_langchain/LangGraph vs LlamaIndex.md

另外有使用了langgraph的@proxyless-llm-websearch项目参考

设计流程图如下
┌─────────────┐
│ OpenWebUI   │
│ (Chat UI)   │
└──────┬──────┘
       │ OpenAI-compatible
       ▼
┌─────────────┐
│ LangServe   │
│ (FastAPI)   │
└──────┬──────┘
       │ invoke / stream
       ▼
┌─────────────┐
│ LangGraph   │
│ (Agent FSM) │
└──────┬──────┘
       ▼
┌─────────────┐
│ LlamaIndex  │
│ (RAG)       │
└─────────────┘

观察到langchain的agent每一步的提示词都一样？只在create_agent的时候bind_tools一次？
bind_tools 的本质
是的，bind_tools 会将工具定义（名称、描述、参数 schema）转换为模型可理解的格式（通常是 JSON Schema），并作为提示词的一部分发送给模型，让模型知道有哪些工具可用以及如何调用。
我们当前迁移不需要使用bind_tools，因为我们的节点是确定性的，不是自由工具选择。

LangChain/LangGraph 中不同节点使用不同提示词
可以。在 LangGraph 中，每个节点是独立函数，可以在节点内直接调用 LLM 并传入不同的 system prompt。你的 DirectAnswerGenerator、WebOrRagAnswerGenerator 和 SubProblemGenerator 都使用不同的 system prompt，迁移到 LangGraph 时可以直接保留这些逻辑。