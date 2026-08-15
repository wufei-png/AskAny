# Archived Python material

This directory preserves Python files that are not part of AskAny's supported
runtime or validation surface. Archived files are reference material only: they
are excluded from Ruff, Pyright, pytest collection, packaging, and documented
run commands.

`legacy_workflows/` contains the superseded LlamaIndex workflow/client/server
chain and its diagnostic test. The chain is no longer reachable from
`askany.main`, and it references modules or APIs that no longer exist. The two
supported query paths are `askany/workflow/workflow_langgraph.py` and
`askany/workflow/min_langchain_agent.py`.

`manual_checks/` contains one-off experiments that were previously collected
as automated tests even though they require unsupported external packages or
manual interpretation.

Current standalone tools such as the LightRAG ingestion CLI, the LangGraph
visualizer, `askany_mcp`, and the independent scripts under `tool/` remain in
their original locations. Those standalone tools are outside the two-mode
static-quality scope but are not classified as legacy. The shared helpers
`tool/keyword_utils.py` and `tool/langdetect.py` are imported by the two
supported query paths, so they are included in the Ruff, Pyright, and CI gates.
