# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AskAny is a Chinese-optimized RAG (Retrieval-Augmented Generation) Q&A assistant combining keyword search and vector search with reranking. It uses LangGraph for workflow orchestration and LlamaIndex for retrieval.

## Commands

### Setup
```bash
uv python install 3.11 && uv python pin 3.11
uv sync
cp .env.example .env  # Configure database and API credentials
```

### Database
```bash
sudo bash setup_postgresql.sh  # Quick setup
# Or manual: createdb askany && psql -d askany -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Running
```bash
python -m askany.main --serve              # Start API server (port 8000)
python -m askany.main --ingest             # Ingest documents from data/json and data/markdown
python -m askany.main --check-db           # Verify ingested data
python -m askany.main --query --query-text "question" --query-type AUTO
```

### Code Quality
```bash
ruff format .        # Format
ruff check .         # Lint
ruff check --fix .   # Auto-fix
```

### Testing
```bash
python -m pytest test/test_workflow_client_call.py -v                    # Run test file
python -m pytest test/test_workflow_client_call.py::test_basic_query -v  # Run single test
python -m pytest test/ --cov=askany --cov-report=html                    # With coverage
```

## Architecture

```
User Query → FastAPI Server → Workflow Mode Selection → QueryRouter → RAG Engines → Response
                                    ↓
              ┌─────────────────────┴─────────────────────┐
              │                                           │
    workflow_langgraph.py                      min_langchain_agent.py
    (Manual LangGraph state machine)           (LangChain auto agent)
    - Explicit orchestration                   - Auto tool selection
    - Iterative context expansion              - Faster (~30s)
    - More stable (~60s)
```

### Key Components

| Directory | Purpose |
|-----------|---------|
| `askany/workflow/` | LangGraph state machine (`workflow_langgraph.py`) and LangChain agent (`min_langchain_agent.py`) |
| `askany/rag/` | Query routing (`router.py`), FAQ hybrid retrieval (`faq_query_engine.py`), docs retrieval (`rag_query_engine.py`) |
| `askany/ingest/` | Document ingestion, vector store management (`vector_store.py`), keyword extraction |
| `askany/prompts/` | Language-aware prompts (`prompts_cn.py`, `prompts_en.py`, `prompt_manager.py`) |
| `askany/api/server.py` | FastAPI server with OpenAI-compatible endpoints |
| `askany/config.py` | Centralized settings (Settings class with env var support) |

### Data Flow

1. **Ingestion**: JSON FAQs (`data/json/`) and Markdown docs (`data/markdown/`) → Vector embeddings (BAAI/bge-m3) → PostgreSQL + pgvector
2. **Query**: User query → WorkflowFilter → QueryRouter (FAQ/DOCS/AUTO) → Hybrid retrieval (keyword + vector) → Reranking (BAAI/bge-reranker-v2-m3) → LLM response

### Query Types (rag/router.py)

- `FAQ`: Routes to FAQ-specific hybrid retrieval
- `DOCS`: Routes to documentation retrieval
- `AUTO`: Smart routing based on query analysis

## Configuration

All settings in `askany/config.py` can be overridden via `.env`:
- `openai_api_base`, `openai_api_key`, `openai_model` - LLM configuration
- `postgres_*` - Database connection
- `embedding_model`, `reranker_model` - Model selection
- `faq_similarity_top_k`, `docs_similarity_top_k` - Retrieval parameters

## API Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat
- `POST /v1/update_faqs` - Hot update FAQ entries
- `GET /health` - Health check
