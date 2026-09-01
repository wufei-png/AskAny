# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project overview

AskAny is a Chinese-optimized RAG assistant using LlamaIndex for retrieval,
LangGraph and LangChain for the supported agent paths, and PostgreSQL +
pgvector for vector storage. The implementation and
[`docs/current-runtime.md`](docs/current-runtime.md) are authoritative.

## Setup

```bash
uv python install 3.11
uv python pin 3.11
uv sync
cp .env.example .env
```

Install optional integrations as needed:

```bash
uv sync --extra lightrag
uv sync --extra observability
uv sync --all-extras
```

PostgreSQL requires the `vector` extension. Start the repository development
database with:

```bash
docker compose -f docker-compose.dev.yml up -d postgres
```

There is no `setup_postgresql.sh`; see `SETUP_POSTGRESQL.md` for host setup.

## Supported commands

```bash
uv run --locked python -m askany.main --serve
uv run --locked python -m askany.main --ingest
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany.main --create-index
uv run --locked python -m askany.main \
  --query --query-text "question" --query-type AUTO
```

The main `--ingest` command currently writes Markdown nodes to the docs vector
store. It parses FAQ JSON, but FAQ vector insertion is disabled in
`askany/ingest/ingest.py`; FAQ updates use `POST /v1/update_faqs`.

## Quality and tests

```bash
uv lock --check
uv run --locked ruff check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format --check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked --all-extras pyright
uv run --locked pre-commit run --all-files
uv run --locked pytest -q test -rs
```

The supported static surface is the reachable API runtime, tests, and the
shared helpers `tool/keyword_utils.py` and `tool/langdetect.py`. Standalone
tools, `askany_mcp`, LightRAG ingestion, visualization code, and archived
Python are outside this gate.

## Runtime routing

- A requested model name ending in `-deepsearch` selects
  `askany/workflow/workflow_langgraph.py`.
- All other model names select `askany/workflow/min_langchain_agent.py`.
- `WorkflowFilter` is used by the deepsearch single-user-message path; the
  automatic agent selects its tools directly.
- LightRAG augmentation defaults to enabled, but its optional dependency and
  separately ingested data are required for it to contribute results.

## API endpoints

- `GET /health`: HTTP 200 with `ok` when both workflows are ready; otherwise
  HTTP 503 with `degraded`.
- `GET /metrics`: Prometheus registry; no enable flag or custom port setting.
- `GET /v1/models`: configured models and possible `-deepsearch` variants.
- `POST /v1/chat/completions`: OpenAI-compatible chat and SSE streaming.
- `POST /v1/update_faqs`: base64-encoded JSON FAQ hot update.
- `GET /v1/cache/stats` and `POST /v1/cache/clear`: QA-cache operations.

## Documentation policy

Current root documentation and code are normative. `archive/docs/` and
`dev_readme/ai_chats/` contain historical material only and must not be used
to infer current defaults or commands.
