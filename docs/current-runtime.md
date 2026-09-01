# Current runtime contract

本文档记录当前 checkout 中可直接从代码确认的运行边界。它不是产品愿景，
也不替代 `askany/config.py`；配置字段和默认值发生变化时，应先修改代码，
再同步本文档。

## Supported entry points

The main application is started with the repository's Python 3.11 environment:

```bash
uv run --locked python -m askany.main --serve
uv run --locked python -m askany.main --ingest
uv run --locked python -m askany.main --query --query-text "question" --query-type AUTO
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany.main --create-index
```

`askany.main` initializes the configured LLM and embedding model before running
the selected command. The server listens on `0.0.0.0:8000` by default.

The API chooses the workflow from the requested model name:

| Requested model              | Runtime path                             | Behavior                                                                                                          |
| ---------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Name ending in `-deepsearch` | `askany/workflow/workflow_langgraph.py`  | Manual LangGraph orchestration; a single-user-message request can use `WorkflowFilter` and subproblem processing. |
| Any other model name         | `askany/workflow/min_langchain_agent.py` | LangChain agent with automatic tool selection.                                                                    |

Both paths can use RAG, local-file search and web search according to their
configuration. LightRAG augmentation is attempted when enabled and available;
an unavailable optional LightRAG installation falls back to the base path.

## API contract

| Method | Path                   | Current behavior                                                                                                                            |
| ------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/health`              | Returns HTTP 200 with `status: ok` only when both workflow globals are ready. Otherwise returns `status: degraded` and HTTP 503.            |
| `GET`  | `/metrics`             | Exposes the Prometheus registry. There is no `enable_prometheus` or `prometheus_port` setting.                                              |
| `GET`  | `/v1/models`           | Proxies the configured OpenAI-compatible model list and adds `-deepsearch` variants when possible.                                          |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint; `stream: true` returns SSE and ends with a stop chunk plus `data: [DONE]`.                                 |
| `POST` | `/v1/update_faqs`      | Accepts base64-encoded JSON FAQ data, updates the FAQ vector store, recreates its query engine, and clears the QA cache.                    |
| `GET`  | `/v1/cache/stats`      | Returns QA-cache statistics.                                                                                                                |
| `POST` | `/v1/cache/clear`      | Clears the QA cache.                                                                                                                        |
| `GET`  | `/openapi.json`        | Returns the custom OpenAPI document used by the integration. Its declared paths are intentionally narrower than the full FastAPI route set. |

`/v1/chat/completions` selects the deepsearch workflow only for a model ending
in `-deepsearch`. A multi-message deepsearch request and every non-deepsearch
request use the simple agent path. Deepsearch results and simple-agent
references are passed to the optional asynchronous RAGAS evaluator when it is
enabled.

## Data and ingestion boundary

The normal document directories are `data/json` for FAQ JSON and
`data/markdown` for Markdown documents. These runtime directories are ignored
by Git.

`python -m askany.main --ingest` currently parses FAQ JSON but its core FAQ
vector insertion block is disabled in `askany/ingest/ingest.py`; the command
does insert Markdown document nodes into the docs vector store. Do not describe
this command as a complete FAQ-plus-docs ingestion pipeline. FAQ vector data is
updated by `/v1/update_faqs` after the server has initialized the FAQ store.

LightRAG has a separate ingestion CLI and storage path:

```bash
uv sync --extra lightrag
uv run --locked python -m askany.rag.lightrag_ingest \
  --ingest-markdown --ingest-json
```

The LightRAG master switch defaults to `True`, but the optional dependency and
ingested LightRAG data must exist for augmentation to contribute results. Set
`ENABLE_LIGHTRAG=false` to disable it explicitly.

## Current defaults worth knowing

- PostgreSQL: `localhost:5432`, database `askany`, user `wufei`;
- embedding: `BAAI/bge-m3`, SentenceTransformers, vector dimension `1024`;
- reranker: `BAAI/bge-reranker-v2-m3`;
- FAQ/docs tables: `askany_faq_vectors` and `askany3_docs_vectors` (the
  LlamaIndex store adds its physical `data_` prefix);
- QA cache: enabled by default, similarity threshold `0.90`;
- docs keyword index: disabled by default, so the effective default docs
  similarity threshold is `0.55` after the configuration adjustment;
- Langfuse and RAGAS: disabled by default;
- Prometheus metrics: available whenever the API server is running.

## Validation boundaries

The supported static and test surface is the reachable `askany` runtime,
`test`, and the shared helpers `tool/keyword_utils.py` and
`tool/langdetect.py`. Standalone tools, `askany_mcp`, LightRAG ingestion, and
visualization code have separate operational boundaries.

```bash
uv lock --check
uv run --locked ruff check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format --check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked --all-extras pyright
uv run --locked pre-commit run --all-files
uv run --locked pytest -q test -rs
```

Passing local checks does not prove live PostgreSQL, model-provider, LightRAG,
Mem0, QA-cache, Langfuse, or RAGAS behavior. Missing external prerequisites
may be skipped by integration tests; malformed fixtures and programming or
provider regressions must fail.
