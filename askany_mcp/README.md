# AskAny MCP servers

`askany_mcp/` contains standalone MCP transports that reuse AskAny's main
Python environment and `askany/config.py`. They query the configured
PostgreSQL/pgvector stores directly and do not require
`python -m askany.main --serve`.

There is no separate `pyproject.toml` or dependency environment under this
directory. Run commands from the repository root with `uv run --locked`.

## Prerequisites

1. Python `>=3.11,<3.12` and the project environment (`uv sync`).
2. PostgreSQL with the `vector` extension and initialized vector stores.
3. A configured embedding model and LLM endpoint, as required by
   `askany/config.py`.

Check the database before starting a server:

```bash
uv run --locked python -m askany.main --check-db
```

The main project `.env` is the configuration source. The local
[`askany_mcp/.env.example`](.env.example) is only a field-name reminder; it is
not a second settings file.

## Available tool

Each MCP server implementation exposes one MCP tool:

```text
rag_search(query: string)
```

The MCP schema exposes only the required `query` argument. The server performs
its own automatic FAQ/docs fallback and returns formatted text containing
matching content, scores when available, file paths, and line ranges when the
stored metadata provides them. The Python helper function accepts a
`query_type` argument, but that is not exposed as an MCP tool parameter.

## Local stdio transport

Start it directly:

```bash
uv run --locked python -m askany_mcp.server
```

Use this command in an MCP client that launches local stdio processes. A
generic configuration fragment is:

```json
{
  "command": "uv",
  "args": [
    "--directory",
    "/absolute/path/to/AskAny",
    "run",
    "python",
    "-m",
    "askany_mcp.server"
  ]
}
```

The client-specific top-level key is not prescribed by this repository.

## Remote SSE transport

`server_fastapi.py` is the explicit remote entry point used by the test client:

```bash
uv run --locked python -m askany_mcp.server_fastapi \
  --host 0.0.0.0 --port 38081
```

It exposes:

- `GET /health` -> JSON `{"status":"ok"}`;
- `GET /sse` -> MCP SSE session endpoint;
- `POST /messages` -> MCP client messages.

The server initializes the RAG components before starting Uvicorn. The two
older Starlette implementations, `server_sse.py` and `server_http.py`, use the
same `/sse`, `/messages`, and `/health` shape with default port `8001`; they are
kept as separate transport implementations rather than presented as a REST
`/tools/rag_search` API.

Test the FastAPI/SSE entry point with:

```bash
uv run --locked python askany_mcp/test_fastapi_client.py
```

The client assumes `http://localhost:38081/sse`; pass a matching port when
starting the server or edit the test client for another endpoint.

## Direct test

```bash
uv run --locked python -m askany_mcp.test_server
```

This initializes the RAG components and runs two sample queries. It requires
the configured database, model, and data; it is not an offline unit test.

## Existing configuration files

- `.mcp.json` is a repository-level remote configuration placeholder whose URL
  is currently `http://ip:38081/sse`; replace it before use.
- `.mcp_sse.json` points to `http://localhost:8001/sse` for an SSE client.
- `.mcp_web.json` also points to port `8001`, but the current server
  implementations expose MCP SSE transport rather than a REST HTTP tool route;
  verify client support before using it.

These JSON files are examples/placeholders, not proof that a particular MCP
client accepts the configuration shape.

## Security

The remote servers bind to `0.0.0.0` by default and the code does not provide
authentication. Expose them only on a trusted network or put them behind an
authenticated, access-controlled proxy.
