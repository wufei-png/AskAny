# AskAny RAG MCP Server

MCP (Model Context Protocol) server that exposes AskAny's RAG search capabilities as tools for Claude Code.

## Overview

This is a **standalone MCP server** that provides direct access to AskAny's RAG system without requiring the FastAPI server (`python -m askany.main --serve`). It's a new deployment method for the FAQ system.

## Features

- **Direct RAG Query**: Queries vector store directly (PostgreSQL + pgvector)
- **Hybrid Search**: Combines vector similarity and keyword matching
- **Structured Results**: Returns content, similarity scores, file paths, and line numbers
- **Auto Routing**: Automatically routes between FAQ and docs based on query

## Architecture

```
Claude Code CLI
    ↓ (MCP Protocol)
MCP Server (askany_mcp/server.py)
    ↓ (Direct Python imports)
RAG Components (router, vector store, embedding model)
    ↓
PostgreSQL + pgvector
```

## Prerequisites

1. **PostgreSQL with pgvector** installed and running
2. **Data ingested** into vector store
3. **Python 3.11+** with uv
4. **Environment configured** (.env file)

### Verify Prerequisites

```bash
# Check PostgreSQL
psql -d askany -c "SELECT COUNT(*) FROM askany_faq_vectors;"

# Check data ingestion
cd /home/wufei/github.com/wufei-png/AskAny
python -m askany.main --check-db
```

## Installation

### 1. Install Dependencies

The MCP server uses the main project's uv environment:

```bash
cd /home/wufei/github.com/wufei-png/AskAny
uv sync
```

### 2. Configuration

The MCP server is configured via project-level `.mcp.json` (already included in repo):

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

No additional configuration needed!

## Usage

### Tool: rag_search

Search local documents and FAQ using RAG.

**Parameters:**
- `query` (string, required): Search query
- `query_type` (string, optional): "auto" (default), "faq", or "docs"

**Returns:**
- Content from matching documents
- Similarity scores
- File paths with line numbers

### Example Queries

```
User: "What is the AskAny deployment process?"
Claude: [Calls rag_search tool]
Result: Content from README.md (lines 45-67) with score 0.856

User: "How do I configure the database?"
Claude: [Calls rag_search tool with query_type="docs"]
Result: Content from CLAUDE.md (lines 12-28) with score 0.923
```

## Testing

### Test MCP Server Directly

```bash
cd /home/wufei/github.com/wufei-png/AskAny
uv run python -m askany_mcp.test_server
```

### Test with Claude Code

```bash
# Navigate to the AskAny project directory
cd /home/wufei/github.com/wufei-png/AskAny

# Start Claude Code CLI
claude

# Ask a question that triggers RAG search
"What is AskAny?"
```

## Troubleshooting

### Error: "RAG components not initialized"

**Cause:** Database connection failed or data not ingested

**Solution:**
```bash
# Check database
psql -d askany -c "\dt"

# Re-ingest data
cd /home/wufei/github.com/wufei-png/AskAny
python -m askany.main --ingest
```

### Error: "No results found"

**Cause:** Query doesn't match ingested content

**Solution:**
- Try different keywords
- Check if data is ingested: `python -m askany.main --check-db`
- Try query_type="faq" or "docs" explicitly

## Configuration

The MCP server uses the same configuration as AskAny main application.

**Config file:** `/home/wufei/github.com/wufei-png/AskAny/askany/config.py`

**Key settings:**
- `postgres_*`: Database connection
- `embedding_model`: BAAI/bge-m3
- `faq_score_threshold`: Threshold for FAQ vs docs routing

## File Structure

```
AskAny/
├── .mcp.json              # Project-level MCP configuration
├── askany_mcp/
│   ├── __init__.py        # Module initialization
│   ├── server.py          # MCP server implementation
│   ├── test_server.py     # Test script
│   └── README.md          # This file
└── askany/                # Main AskAny package
    └── config.py          # Shared configuration
```

