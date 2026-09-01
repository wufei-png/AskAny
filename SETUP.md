# AskAny setup guide

This guide describes the current repository setup. Use the code and
[`docs/current-runtime.md`](docs/current-runtime.md) as the final authority
when defaults change.

## Prerequisites

- Python `>=3.11,<3.12`;
- `uv`;
- PostgreSQL with the `vector` extension;
- an OpenAI-compatible LLM endpoint;
- access to the configured embedding and reranker models.

## Install

```bash
uv python install 3.11
uv python pin 3.11
uv sync
cp .env.example .env
```

The example is commented so this copy does not override `askany/config.py`.
Uncomment only values that should differ, using the field names from
`askany/config.py` (`POSTGRES_USER`, `OPENAI_API_BASE`, `OPENAI_MODEL`,
`EMBEDDING_MODEL`, `RERANKER_MODEL`). The repository defaults use PostgreSQL
user `wufei`, a local vLLM endpoint at `http://127.0.0.1:8081/v1`, BAAI/bge-m3
embeddings, and a local Qwen model path; those defaults are not a hosted
service configuration.

Optional dependencies:

```bash
uv sync --extra lightrag
uv sync --extra observability
uv sync --all-extras
```

## PostgreSQL

The repository development database can be started with:

```bash
docker compose -f docker-compose.dev.yml up -d postgres
```

The compose service uses `root`/`123456`. The host-side application defaults
use `wufei`; set `POSTGRES_USER` and `POSTGRES_PASSWORD` in `.env` to match the
database you actually use.

For an existing PostgreSQL installation:

```bash
createdb askany
psql -d askany -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

See [`SETUP_POSTGRESQL.md`](SETUP_POSTGRESQL.md) for host installation details
and [`dev_readme/docker-compose.md`](dev_readme/docker-compose.md) for the
development container.

## Prepare data

Put Markdown documents in `data/markdown/`. FAQ JSON files belong in
`data/json/`; each file may contain one object or a list of objects with
`question` and `answer` fields.

Run:

```bash
uv run --locked python -m askany.main --ingest
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany.main --create-index
```

Important: the current `--ingest` implementation parses FAQ JSON but its FAQ
vector insertion block is disabled. It inserts Markdown nodes into the docs
vector store. FAQ vector data is updated through `POST /v1/update_faqs` after
the server initializes the FAQ store.

LightRAG ingestion is separate:

```bash
uv run --locked python -m askany.rag.lightrag_ingest \
  --ingest-markdown --ingest-json
```

It requires the `lightrag` extra and a reachable configured LLM.

## Start and verify the API

```bash
uv run --locked python -m askany.main --serve
```

The default address is `http://0.0.0.0:8000`. In another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/v1/models
```

`/health` returns HTTP 200 with `status: "ok"` only when both supported agent
paths are ready. It returns HTTP 503 with `status: "degraded"` otherwise.

For an OpenAI-compatible chat client, use `/v1/chat/completions`. A model name
ending in `-deepsearch` selects the manual LangGraph workflow; other model names
select the automatic LangChain agent. `stream: true` uses SSE.

## OpenWebUI

Configure an OpenAI-compatible connection pointing to the API base URL, for
example `http://localhost:8000/v1`. The custom OpenAPI document is available at
`http://localhost:8000/openapi.json` when an integration specifically needs it.

## Troubleshooting

### PostgreSQL connection failure

Check that PostgreSQL is running, `POSTGRES_*` values match the server, and the
`vector` extension exists:

```bash
pg_isready
psql -d askany -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

### Vector dimension mismatch

The default embedding is `BAAI/bge-m3` with dimension `1024`. If you select a
different embedding model, set `VECTOR_DIMENSION` to its output dimension and
use a compatible vector table.

### LightRAG is unavailable

Install `uv sync --extra lightrag`, run the separate LightRAG ingestion command,
and verify `ENABLE_LIGHTRAG=true`. If the optional package or data is missing,
the main application falls back to the base LlamaIndex retrieval path.

### FAQ results are empty after `--ingest`

This is expected with the current main ingestion path because FAQ vector
insertion is disabled. Use `/v1/update_faqs` or update the implementation before
describing `--ingest` as a complete FAQ pipeline.
