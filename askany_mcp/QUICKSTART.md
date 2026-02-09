# AskAny RAG MCP Server - Quick Start

## What Changed

✅ **Uses main project's uv environment** (no separate dependencies)
✅ **Project-level MCP config** (`.mcp.json` in repo root)
✅ **Part of the repo** (askany_mcp/ module)

## Quick Start

### 1. Install Dependencies

```bash
cd /home/wufei/github.com/wufei-png/AskAny
uv sync
```

### 2. Test the Server

```bash
uv run python -m askany_mcp.test_server
```

Expected output:
```
Initializing RAG components...
✓ RAG components initialized

Query: What is AskAny? (type: auto)
------------------------------------------------------------
[Result 1]
Score: 0.856
File: README.md
Lines: 1-20
Content: AskAny is a Chinese-optimized RAG...
```

### 3. Use with Claude Code

The MCP server is already configured in `.mcp.json`:

```bash
cd /home/wufei/github.com/wufei-png/AskAny
claude
```

Then ask questions:
```
You: What is AskAny?
Claude: [Automatically calls rag_search tool]
```

## Configuration

**File:** `/home/wufei/github.com/wufei-png/AskAny/.mcp.json`

```json
{
  "mcpServers": {
    "askany-rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/wufei/github.com/wufei-png/AskAny",
        "run",
        "python",
        "-m",
        "askany_mcp.server"
      ]
    }
  }
}
```
