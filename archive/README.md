# Archived material

Everything under `archive/` is reference material, not part of the current
AskAny contract. Read the current root documentation and the code first. Do
not use archived commands, defaults, endpoint names, or performance claims as
operational instructions.

`docs/` contains historical design reviews, research notes, and environment
snapshots. These files are retained for provenance only and are intentionally
outside the normal documentation path.

The Python subdirectories preserve files that are not part of AskAny's
supported runtime or validation surface. Archived Python files are excluded
from Ruff, Pyright, pytest collection, packaging, and documented run commands.

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
