# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project overview

AskAny is a Chinese-optimized RAG assistant. LlamaIndex provides retrieval,
LangGraph and LangChain provide the two supported agent paths, and PostgreSQL +
pgvector stores embeddings.

The current runtime contract is documented in
[`docs/current-runtime.md`](docs/current-runtime.md). Code is authoritative
when this file and the implementation disagree.

## Setup

```bash
uv python install 3.11
uv python pin 3.11
uv sync
cp .env.example .env
```

`uv sync` installs core dependencies. Optional integrations are installed with
`uv sync --extra lightrag`, `uv sync --extra observability`, or
`uv sync --all-extras`.

PostgreSQL must have the `vector` extension. The repository development
container can be started with:

```bash
docker compose -f docker-compose.dev.yml up -d postgres
```

See [`SETUP_POSTGRESQL.md`](SETUP_POSTGRESQL.md) for host installation and
[`dev_readme/docker-compose.md`](dev_readme/docker-compose.md) for container
details. There is no `setup_postgresql.sh` in this repository.

## Supported commands

```bash
uv run --locked python -m askany.main --serve
uv run --locked python -m askany.main --ingest
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany.main --create-index
uv run --locked python -m askany.main \
  --query --query-text "question" --query-type AUTO
```

`--ingest` currently inserts Markdown nodes into the docs vector store. It
parses FAQ JSON, but the FAQ vector insertion block is disabled in
`askany/ingest/ingest.py`; do not describe this command as complete FAQ
ingestion. FAQ hot updates use `POST /v1/update_faqs`.

## Quality and tests

```bash
uv lock --check
uv run --locked ruff check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format --check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked --all-extras pyright
uv run --locked pre-commit run --all-files
uv run --locked pytest -q test -rs
```

Ruff covers the supported production files, tests, and shared helper modules;
Pyright covers `askany` and the two shared helpers. Standalone tools,
`askany_mcp`, LightRAG ingestion, visualization code, and archived Python are
outside this gate. See [`archive/README.md`](archive/README.md).

When changing code, preserve the distinction between external prerequisite
skips and real failures: missing service/model/data prerequisites may skip an
integration test, while malformed input and programming/provider regressions
must fail.

## Runtime architecture

The API selects the runtime by model name:

- names ending in `-deepsearch` use
  `askany/workflow/workflow_langgraph.py`;
- all other model names use `askany/workflow/min_langchain_agent.py`.

Both paths can use RAG, local-file search, and web search. LightRAG augmentation
defaults to enabled (`enable_lightrag=True`) but requires its optional package
and separately ingested data; unavailable optional prerequisites fall back to
the base path.

`WorkflowFilter` is used by the deepsearch single-user-message path. The
automatic agent selects tools directly.

## Key components

| Directory | Purpose |
|---|---|
| `askany/api/` | FastAPI server and OpenAI-compatible endpoints |
| `askany/config.py` | Pydantic settings and defaults |
| `askany/ingest/` | JSON/Markdown parsing and vector-store management |
| `askany/rag/` | Query routing, retrieval, reranking, and LightRAG adapter |
| `askany/workflow/` | LangGraph and LangChain agent paths |
| `askany/metrics/` | Prometheus instrumentation |
| `askany/observability/` | Optional Langfuse and RAGAS integration |
| `askany/memory/` | Optional Mem0 integration |
| `askany/prompts/` | Language-aware prompt management |
| `askany_mcp/` | Standalone MCP transports, outside the static gate |

## API endpoints

- `GET /health`: HTTP 200 with `ok` only when both workflow globals are ready;
  otherwise HTTP 503 with `degraded`.
- `GET /metrics`: Prometheus registry; there is no Prometheus enable flag or
  port setting.
- `GET /v1/models`: configured model list and possible `-deepsearch` variants.
- `POST /v1/chat/completions`: OpenAI-compatible chat and SSE streaming.
- `POST /v1/update_faqs`: base64-encoded JSON FAQ hot update.
- `GET /v1/cache/stats` and `POST /v1/cache/clear`: QA-cache operations.

## Documentation policy

Current root documentation and code are normative. Historical notes under
`archive/docs/` and `dev_readme/ai_chats/` are not instructions and must not be
used to infer current defaults or commands.
