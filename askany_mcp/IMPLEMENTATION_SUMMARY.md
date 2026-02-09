# AskAny RAG MCP Server - Implementation Summary

## What Was Built

A standalone MCP (Model Context Protocol) server that exposes AskAny's RAG search capabilities as tools for Claude Code.

## Key Features

1. **Direct RAG Access**: Queries vector store directly without FastAPI server
2. **Simple Implementation**: Uses only lines 172-208 logic from min_langchain_agent.py
3. **No Keyword Extraction**: Pure vector similarity search
4. **Structured Output**: Returns content, scores, file paths, line numbers

## Files Created

```
askany_mcp/
├── server.py              # Main MCP server (150 lines)
├── pyproject.toml         # Dependencies (mcp, python-dotenv)
├── README.md              # Complete documentation
├── .env.example           # Configuration template
└── test_server.py         # Test script
```

## Architecture

```
Claude Code CLI
    ↓ MCP Protocol
server.py (askany_mcp/)
    ↓ Direct Python imports
askany.rag.router (QueryRouter)
    ↓
PostgreSQL + pgvector
```

## Configuration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "askany-rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/wufei/github.com/wufei-png/AskAny/askany_mcp",
        "run",
        "server.py"
      ]
    }
  }
}
```

## Testing

### 1. Test Server Directly

```bash
cd /home/wufei/github.com/wufei-png/AskAny/askany_mcp
uv run test_server.py
```

### 2. Test with Claude Code

```bash
claude "What is AskAny?"
# Should invoke rag_search tool automatically
```

## Next Steps

1. **Install dependencies**:
   ```bash
   cd /home/wufei/github.com/wufei-png/AskAny/askany_mcp
   uv sync
   ```

2. **Configure Claude Code**: Add MCP server to `~/.claude.json`

3. **Test**: Run `uv run test_server.py`

4. **Use**: Start Claude Code and ask questions

## Implementation Notes

- **No FastAPI dependency**: Direct vector store access
- **No keyword extraction**: Pure vector similarity (lines 172-208)
- **Reuses config**: Uses `askany.config.settings`
- **Structured output**: JSON with content, score, file_path, line numbers
