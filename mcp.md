# MCP integration

AskAny exposes its direct RAG search as an MCP tool through the standalone
servers in `askany_mcp/`. This integration is independent of the main FastAPI
chat server and reuses the repository's Python environment and settings.

## Prerequisites

From the repository root:

```bash
uv sync
uv run --locked python -m askany.main --check-db
```

The configured PostgreSQL vector stores, embedding model, and LLM endpoint must
be available. The MCP servers do not ingest data themselves.

## Supported transports

| Transport | Command | Default endpoint |
|---|---|---|
| Local stdio | `uv run --locked python -m askany_mcp.server` | launched process stdin/stdout |
| Remote MCP SSE | `uv run --locked python -m askany_mcp.server_fastapi` | `http://localhost:38081/sse` |
| Alternate SSE | `uv run --locked python -m askany_mcp.server_sse` | `http://localhost:8001/sse` |
| Alternate SSE | `uv run --locked python -m askany_mcp.server_http` | `http://localhost:8001/sse` |

The `server_fastapi.py` entry point is the one used by
`askany_mcp/test_fastapi_client.py` and is the recommended remote entry point.
It exposes `GET /health`, `GET /sse`, and `POST /messages`. The two alternate
servers expose the same MCP paths and return plain-text `OK` from `/health`.

## Local stdio client configuration

Use a client-specific MCP configuration with this command and replace the
directory with the actual absolute repository path:

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

The server exposes one tool:

```text
rag_search(query: string)
```

The MCP schema exposes only `query`. Automatic FAQ/docs fallback happens inside
the server. The helper function's optional `query_type` argument is not a
client-visible MCP parameter.

## Remote SSE client configuration

Start the recommended remote server:

```bash
uv run --locked python -m askany_mcp.server_fastapi \
  --host 0.0.0.0 --port 38081
```

Configure the MCP client with:

```text
http://localhost:38081/sse
```

Verify it with:

```bash
curl http://localhost:38081/health
uv run --locked python askany_mcp/test_fastapi_client.py
```

## Repository JSON examples

- `.mcp.json` currently contains `http://ip:38081/sse`, which is a placeholder
  and is not ready to use without editing;
- `.mcp_sse.json` points to the port-8001 alternate SSE server;
- `.mcp_web.json` labels a port-8001 HTTP transport, but the current Python
  implementations expose MCP SSE endpoints rather than a REST tool route.

These files are examples/placeholders. Client configuration schemas vary, so a
file's presence does not prove that a particular client accepts it.

## OpenCode

This repository does not pin or ship an OpenCode fork, web application, or
authentication layer. If OpenCode is used, configure its MCP integration with
one of the current stdio/SSE endpoints above and follow the OpenCode version's
own configuration schema. Do not rely on old custom-branch instructions.

## Security

Remote servers bind to `0.0.0.0` by default and the implementation has no
authentication. Keep the service on a trusted network or place it behind an
authenticated, access-controlled proxy.
