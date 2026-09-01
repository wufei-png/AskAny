# AskAny MCP quick start

Run these commands from the repository root. MCP uses the main AskAny
environment; do not run `uv sync` inside `askany_mcp/`.

## Local stdio

```bash
uv sync
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany_mcp.server
```

Register the last command in your MCP client as a local stdio server. Replace
`/absolute/path/to/AskAny` with the actual repository path:

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

## Remote SSE

Start the FastAPI/SSE implementation:

```bash
uv run --locked python -m askany_mcp.server_fastapi \
  --host 0.0.0.0 --port 38081
```

Configure the client with the server's SSE URL:

```text
http://localhost:38081/sse
```

The repository test client checks the same URL:

```bash
uv run --locked python askany_mcp/test_fastapi_client.py
```

The MCP tool is `rag_search` with one required argument, `query`. It performs
automatic FAQ/docs fallback; `query_type` is not an exposed MCP argument.

The root `.mcp.json` currently contains a placeholder host (`ip`) and is not
ready to use without editing. See [`README.md`](README.md) for the full
transport and security notes.
