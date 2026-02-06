目标：使用LangServe的api接口替代当前@askany/api

已经对接了mcp server docs-langchain，你可用于查询langgraph LangServe的相关文档。
同时我也阅读了相关文档，将可能用到的文档放在@dev_readme/from_llamaindex_to_langchain/langserve.md

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
