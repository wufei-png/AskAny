# AskAny MCP over HTTP/SSE

The remote MCP implementations use the MCP SDK's Server-Sent Events transport.
They expose MCP protocol endpoints, not a REST endpoint such as
`POST /tools/rag_search`.

## Recommended entry point

`server_fastapi.py` is the remote entry point used by the repository test
client. Start it from the repository root:

```bash
uv run --locked python -m askany_mcp.server_fastapi \
  --host 0.0.0.0 --port 38081
```

Defaults:

- bind address: `0.0.0.0`;
- port: `38081`;
- SSE endpoint: `GET /sse`;
- client-message endpoint: `POST /messages`;
- health endpoint: `GET /health`, returning JSON `{"status":"ok"}`.

The FastAPI entry point initializes RAG components before starting Uvicorn, so
startup requires the configured database, embedding model, and LLM endpoint.

## Alternate Starlette implementations

The repository also contains two older implementations:

```bash
uv run --locked python -m askany_mcp.server_sse \
  --host 0.0.0.0 --port 8001

uv run --locked python -m askany_mcp.server_http \
  --host 0.0.0.0 --port 8001
```

Both use `GET /sse`, `POST /messages`, and `GET /health` on port `8001` by
default. Their health response is plain-text `OK`. They are separate legacy
implementations, not two additional API contracts; use `server_fastapi.py`
unless a client specifically requires one of them.

## Client configuration examples

The repository includes two port-8001 examples:

- `.mcp_sse.json`: URL `http://localhost:8001/sse` with SSE transport;
- `.mcp_web.json`: URL `http://localhost:8001` with an HTTP transport label.

The current Python servers implement SSE protocol endpoints, so verify that a
client's HTTP transport mode is compatible before using `.mcp_web.json`.
`.mcp.json` at the repository root is a remote placeholder containing
`http://ip:38081/sse`; it must be edited before use.

Client configuration schemas differ. For a generic SSE client, use the URL:

```text
http://localhost:38081/sse
```

## Verify the recommended server

In another terminal:

```bash
curl http://localhost:38081/health
uv run --locked python askany_mcp/test_fastapi_client.py
```

The test client calls the `rag_search` MCP tool and assumes port `38081`.

## Security

All remote implementations bind to `0.0.0.0` by default and do not implement
authentication. Keep them on a trusted network or place them behind an
authenticated and access-controlled proxy. Do not expose a default instance
directly to the public internet.
