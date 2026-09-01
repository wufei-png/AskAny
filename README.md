# AskAny

![AskAny Cover](images/cover/cover1.png)

[中文版](README_CN.md) | English

AskAny is a Chinese-optimized RAG (Retrieval-Augmented Generation) assistant
for operations, development, and testing teams. It combines LlamaIndex
retrieval, LangGraph/LangChain agent workflows, PostgreSQL + pgvector, and an
OpenAI-compatible FastAPI API.

> The code is the source of truth. For the current runtime boundary, see
> [docs/current-runtime.md](docs/current-runtime.md). Historical design notes
> are retained under [archive/docs/](archive/docs/) and are not operational
> documentation.

## Current architecture

```text
OpenWebUI or another OpenAI-compatible client
                       |
                       v
FastAPI: /v1/chat/completions
                       |
       model name ends with -deepsearch?
                 /                    \
                yes                    no
                 |                      |
workflow_langgraph.py          min_langchain_agent.py
  manual LangGraph path          automatic LangChain agent
                 \                    /
                  +-- RAG / web / local-file tools --+
                                      |
                         LlamaIndex FAQ/docs retrieval
                         + optional LightRAG augmentation
                                      |
                         answer and source references
```

The `-deepsearch` suffix selects the manual LangGraph path. Other model names
select the automatic agent. `WorkflowFilter` is part of the deepsearch
single-user-message path; the automatic agent selects its tools directly.

## Features

- FAQ and documentation retrieval backed by PostgreSQL + pgvector;
- keyword/vector retrieval and reranking where configured;
- LangGraph orchestration for deepsearch and LangChain automatic tool use;
- local-file search, optional web search, and source/provenance metadata;
- LightRAG knowledge-graph augmentation, enabled by default but dependent on
  the optional package and separately ingested data;
- optional Mem0 cross-session user memory;
- optional Langfuse tracing and asynchronous RAGAS evaluation;
- QA semantic cache and always-available Prometheus `/metrics` endpoint;
- OpenAI-compatible non-streaming and SSE streaming chat responses;
- standalone MCP servers in `askany_mcp/`.

## Requirements and installation

- Python `>=3.11,<3.12`;
- PostgreSQL with the `vector` extension;
- an OpenAI-compatible LLM endpoint;
- a SentenceTransformers embedding model and reranker, unless API/local
  alternatives are configured.

Install the locked core environment:

```bash
uv python install 3.11
uv python pin 3.11
uv sync
cp .env.example .env
```

Install optional integrations only when needed:

```bash
uv sync --extra lightrag
uv sync --extra observability
uv sync --all-extras
```

`.env.example` uses the actual `Settings` field names, such as
`OPENAI_API_BASE`, `OPENAI_MODEL`, `POSTGRES_USER`, `EMBEDDING_MODEL`, and
`RERANKER_MODEL`. Assignments in the template are commented, so copying it
does not override `askany/config.py`. Uncomment only the values that should
differ; the repository defaults point at a local vLLM endpoint and the local
model path in `askany/config.py`.

## Database setup

For the repository's development container:

```bash
docker compose -f docker-compose.dev.yml up -d postgres
```

The compose service uses PostgreSQL 17 with pgvector and the credentials
`root`/`123456`. The host-side defaults in `askany/config.py` use user `wufei`,
so either configure the service through `.env` or use the same credentials in
your local PostgreSQL installation. See
[SETUP_POSTGRESQL.md](SETUP_POSTGRESQL.md) and
[dev_readme/docker-compose.md](dev_readme/docker-compose.md) for the two setup
styles.

For an existing PostgreSQL installation, create the database and extension
with a user that matches `.env`:

```bash
createdb askany
psql -d askany -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Data and ingestion

Runtime data directories are ignored by Git:

- `data/json/`: FAQ JSON files;
- `data/markdown/`: Markdown documentation;
- `data/stopwords/`: optional tokenizer resources.

FAQ entries can be a single object or a list of objects, for example:

```json
{
  "question": "What is the default API port?",
  "answer": "The default API port is 8000.",
  "metadata": { "category": "configuration" }
}
```

Run the main ingestion command with:

```bash
uv run --locked python -m askany.main --ingest
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany.main --create-index
```

Current limitation: `--ingest` parses FAQ JSON but the FAQ vector insertion
block in `askany/ingest/ingest.py` is disabled. The command currently inserts
Markdown nodes into the docs vector store; it must not be described as a
complete FAQ-plus-docs ingestion pipeline. FAQ vector data is updated through
`POST /v1/update_faqs` after the server initializes the FAQ store.

LightRAG uses a separate ingestion command and storage path:

```bash
uv sync --extra lightrag
uv run --locked python -m askany.rag.lightrag_ingest \
  --ingest-markdown --ingest-json
```

## Run the API

```bash
uv run --locked python -m askany.main --serve
```

The default bind address is `0.0.0.0:8000`. Useful checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/v1/models
```

The health response is `status: "ok"` with HTTP 200 only when both supported
workflow globals are ready. Otherwise it returns `status: "degraded"` with
HTTP 503.

### API endpoints

| Method | Endpoint               | Description                                                          |
| ------ | ---------------------- | -------------------------------------------------------------------- |
| `GET`  | `/health`              | Runtime readiness check; returns `ok` or `degraded`.                 |
| `GET`  | `/metrics`             | Prometheus metrics. No feature-toggle or custom port setting exists. |
| `GET`  | `/v1/models`           | Configured models plus `-deepsearch` variants when available.        |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat; `stream: true` uses SSE.                     |
| `POST` | `/v1/update_faqs`      | Base64-encoded JSON FAQ hot update. Clears the QA cache afterward.   |
| `GET`  | `/v1/cache/stats`      | QA-cache statistics.                                                 |
| `POST` | `/v1/cache/clear`      | Clear the QA cache.                                                  |
| `GET`  | `/openapi.json`        | Custom OpenAPI document used by the integration.                     |

For chat requests, a model ending in `-deepsearch` selects the manual workflow;
all other model names select the automatic agent. Streaming ends with a stop
chunk and `data: [DONE]`.

Direct query testing is also available:

```bash
uv run --locked python -m askany.main \
  --query --query-text "your question" --query-type AUTO
```

The accepted query types are `AUTO`, `FAQ`, `DOCS`, and `CODE`. `CODE` is
currently an explicit not-implemented route.

## Configuration highlights

All settings live in `askany/config.py` and can be overridden by matching
uppercase environment variables. Important defaults are:

| Setting                            | Default                    |
| ---------------------------------- | -------------------------- |
| `language`                         | `cn`                       |
| `postgres_host` / `postgres_port`  | `localhost` / `5432`       |
| `postgres_user` / `postgres_db`    | `wufei` / `askany`         |
| `openai_api_base`                  | `http://127.0.0.1:8081/v1` |
| `embedding_model`                  | `BAAI/bge-m3`              |
| `vector_dimension`                 | `1024`                     |
| `reranker_model`                   | `BAAI/bge-reranker-v2-m3`  |
| `enable_lightrag`                  | `True`                     |
| `enable_mem0`                      | `False`                    |
| `enable_langfuse` / `enable_ragas` | `False` / `False`          |
| `enable_qa_cache`                  | `True`                     |
| `using_docs_keyword_index`         | `False`                    |

The full runtime boundary, derived defaults, and validation limitations are in
[docs/current-runtime.md](docs/current-runtime.md).

## Project structure

```text
askany/
├── api/              # FastAPI application and OpenAI-compatible endpoints
├── config.py         # Pydantic settings
├── ingest/           # Parsers and PostgreSQL/pgvector storage
├── memory/           # Optional Mem0 integration
├── metrics/          # Prometheus instrumentation
├── observability/    # Optional Langfuse and RAGAS integration
├── prompts/          # Chinese/English prompt management
├── rag/              # Routing, retrieval, reranking, and LightRAG adapter
└── workflow/         # LangGraph and LangChain agent paths
archive/              # Non-current Python and documentation material
askany_mcp/           # Standalone MCP transports
tool/                 # Independent operational tools
test/                 # Unit and opt-in integration tests
```

## MCP

`askany_mcp/server.py` is a standalone stdio MCP server. The HTTP/SSE variants
are `server_fastapi.py`, `server_sse.py`, and `server_http.py`. They reuse the
main project's environment and configuration; there is no separate
`askany_mcp/pyproject.toml`. See [mcp.md](mcp.md) and
[askany_mcp/README.md](askany_mcp/README.md) for current transport-specific
commands.

## Development and validation

```bash
uv lock --check
uv run --locked ruff check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format --check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked --all-extras pyright
uv run --locked pre-commit run --all-files
uv run --locked pytest -q test -rs
```

The static gate covers the supported runtime, tests, and the two shared helper
modules. Standalone tools, `askany_mcp`, LightRAG ingestion, and visualization
code have separate boundaries. Passing local checks does not prove live
PostgreSQL, model-provider, LightRAG, Mem0, Langfuse, RAGAS, or QA-cache
behavior.

## Further documentation

- [Current runtime contract](docs/current-runtime.md)
- [Setup guide](SETUP.md)
- [PostgreSQL and pgvector setup](SETUP_POSTGRESQL.md)
- [UV development setup](UV_SETUP.md)
- [LightRAG integration](dev_readme/lightrag.md)
- [Vector data operations](tool/README_vector_data.md)
- [Roadmap](roadmap.md)
