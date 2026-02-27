# LightRAG Integration

LightRAG adds **knowledge-graph (KG) augmented retrieval** to AskAny. It extracts entities and relationships from documents during ingestion, then uses graph traversal + vector search at query time to surface connections that pure vector similarity misses.

The integration is **supplementary** — LightRAG results are merged with the existing LlamaIndex FAQ/DOCS pipeline. Toggle it with a single config flag.

## Architecture

```
User Query
    ↓
min_langchain_agent.py → rag_search tool
    ├── LlamaIndex (vector + keyword hybrid)  ← always on
    └── LightRAG  (KG + vector hybrid)        ← enable_lightrag=True
            ↓
        LightRAGAdapter.retrieve_async()
            ↓
        LightRAG.aquery_data(only_need_context=True)
            ↓
        entities + relations + chunks → NodeWithScore[]
    ↓
Merged results → LLM synthesis
```

Key files:

| File | Purpose |
|------|---------|
| `askany/rag/lightrag_adapter.py` | Adapter: wraps LightRAG, converts results to LlamaIndex `NodeWithScore` |
| `askany/rag/lightrag_ingest.py` | CLI tool for ingesting docs into LightRAG's KG |
| `askany/workflow/min_langchain_agent.py` | Agent integration: calls adapter in `rag_search` tool |
| `askany/workflow/question.py` | `lightrag_questions` array (13 test questions) |
| `test/test_lightrag_retrieval.py` | Standalone retrieval test |
| `test/test_lightrag_e2e_comparison.py` | End-to-end comparison (LightRAG ON vs OFF) |

## Prerequisites

1. **PostgreSQL** running with pgvector extension (same DB as AskAny)
2. **lightrag-hku** package installed:
   ```bash
   uv add lightrag-hku
   ```
3. **Embedding model** (BAAI/bge-m3) — uses the same local SentenceTransformer as AskAny, loaded automatically
4. **LLM endpoint** reachable (configured in `.env` / `config.py`)

> **Note**: Apache AGE (PostgreSQL graph extension) is **not required**. The adapter defaults to `NetworkXStorage` for graph storage, which uses a local `.graphml` file.

## Ingestion

Ingest documents into LightRAG's knowledge graph using the CLI:

```bash
# Ingest all markdown docs (from settings.markdown_dir)
python -m askany.rag.lightrag_ingest --ingest-markdown

# Ingest a specific directory
python -m askany.rag.lightrag_ingest --ingest-markdown --markdown-dir data/markdown/viper-open/viper-devops-docs/viper-v5.5

# Ingest all JSON FAQs
python -m askany.rag.lightrag_ingest --ingest-json

# Ingest a single file
python -m askany.rag.lightrag_ingest --file path/to/doc.md

# Ingest both markdown and JSON
python -m askany.rag.lightrag_ingest --ingest-markdown --ingest-json

# Custom batch size (default: 10)
python -m askany.rag.lightrag_ingest --batch-size 5 --ingest-markdown
```

### What ingestion does

1. Reads markdown/JSON files
2. Splits by `## ` headings (markdown) or Q&A pairs (JSON)
3. LightRAG extracts **entities** and **relationships** via LLM calls
4. Stores everything in PostgreSQL tables (11 tables, prefixed `LIGHTRAG_`)
5. Builds HNSW vector indexes on `vdb_chunks`, `vdb_entity`, `vdb_relation`

### Verify ingestion

```bash
PGPASSWORD=123456 psql -h localhost -U wufei -d askany -c "
  SELECT 'doc_status' as tbl, count(*) FROM lightrag_doc_status
  UNION ALL SELECT 'doc_chunks', count(*) FROM lightrag_doc_chunks
  UNION ALL SELECT 'vdb_chunks', count(*) FROM lightrag_vdb_chunks
  UNION ALL SELECT 'vdb_entity', count(*) FROM lightrag_vdb_entity
  UNION ALL SELECT 'vdb_relation', count(*) FROM lightrag_vdb_relation;
"
```

## Testing

### 1. Standalone retrieval test

Tests that `LightRAGAdapter` can initialize, query the KG, and return well-formed `NodeWithScore` objects.

```bash
# Direct execution (no pytest needed)
python test/test_lightrag_retrieval.py

# Via pytest
python -m pytest test/test_lightrag_retrieval.py -v -s
```

This runs all 13 questions from `lightrag_questions` and prints results (chunks, entities, relations counts) for each. Requires PostgreSQL + ingested data + LLM endpoint.

### 2. End-to-end comparison test

Runs the full `min_langchain_agent` on 5 questions twice — with LightRAG disabled (baseline) and enabled (augmented) — then writes a side-by-side comparison.

```bash
python test/test_lightrag_e2e_comparison.py
```

Results are saved to `test/e2e_comparison_results.json`. Requires the full AskAny stack (PostgreSQL, LlamaIndex vector tables, LightRAG tables, LLM endpoint).

## Configuration

All settings are in `askany/config.py` and can be overridden via `.env`.

### Master switch

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_lightrag` | `False` | Master switch. Set to `True` after ingestion. |

### Query parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `lightrag_query_mode` | `"mix"` | Query mode: `local` (entity neighbors), `global` (community summaries), `hybrid` (local+global), `naive` (vector only), `mix` (all combined). `mix` gives best recall. |
| `lightrag_top_k` | `60` | Number of entities/relations retrieved from KG per query. |
| `lightrag_chunk_top_k` | `10` | Number of text chunks retrieved via vector search. |

### Ingestion parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `lightrag_chunk_token_size` | `800` | Tokens per chunk for LightRAG's internal splitter. Smaller = cleaner entity extraction on dense Chinese docs. 800 balances extraction quality vs context loss. |
| `lightrag_chunk_overlap_token_size` | `150` | Overlap between consecutive chunks. Preserves context at heading boundaries. ~19% of chunk size. |
| `lightrag_entity_extract_max_gleaning` | `1` | Extra LLM passes for entity extraction. 0 = single pass, 1 = one extra pass to catch missed entities. Higher values cost more LLM tokens. |
| `lightrag_summary_max_tokens` | `1500` | Max tokens for entity/community summaries. Higher values avoid truncation of verbose Chinese technical descriptions. |

### LLM / Embedding overrides

These default to AskAny's main LLM/embedding settings. Override only if you want LightRAG to use a different model.

| Setting | Default | Description |
|---------|---------|-------------|
| `lightrag_api_base` | `None` (→ `openai_api_base`) | LLM API endpoint for LightRAG |
| `lightrag_api_key` | `None` (→ `openai_api_key`) | LLM API key |
| `lightrag_llm_model` | `None` (→ `openai_model`) | LLM model name |
| `lightrag_embedding_model` | `None` (→ `embedding_model`) | Embedding model name |
| `lightrag_embedding_dim` | `1024` | Must match embedding model output dimension |

### Storage

| Setting | Default | Description |
|---------|---------|-------------|
| `lightrag_working_dir` | `"lightrag_data"` | Directory for cache, graph files |

Graph storage uses `NetworkXStorage` (file-based `.graphml`). KV, vector, and doc-status storage use PostgreSQL (same DB as AskAny).

## Enabling LightRAG in production

1. **Ingest your documents**:
   ```bash
   python -m askany.rag.lightrag_ingest --ingest-markdown --ingest-json
   ```

2. **Verify ingestion** (see command above)

3. **Enable the flag** in `.env`:
   ```
   enable_lightrag=True
   ```

4. **Restart the server**:
   ```bash
   python -m askany.main --serve
   ```

LightRAG results will now be merged into `rag_search` tool responses in the agent workflow.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `lightrag-hku package not installed` warning | Missing dependency | `uv add lightrag-hku` |
| `PipelineNotInitializedError` | Adapter not fully initialized | Ensure `await adapter.initialize()` is called |
| Empty retrieval results | No data ingested | Run ingestion commands above |
| `ModuleNotFoundError: pytest` | pytest not installed | `uv add --dev pytest pytest-asyncio` |
| Embedding errors / `TypeError: NoneType` | Wrong embedding setup | Adapter uses local SentenceTransformer, not API. Check BAAI/bge-m3 is downloaded. |
| `LIGHTRAG_*` tables missing | First run without ingestion | Tables are auto-created on first `adapter.initialize()` |
