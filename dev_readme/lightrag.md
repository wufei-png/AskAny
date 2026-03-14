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
| `enable_lightrag` | `True` | Master switch. Set to `False` to disable LightRAG augmentation. |

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

3. **Optional: override the flag** in `.env` if you need to disable LightRAG:
   ```
   enable_lightrag=False
   ```

4. **Restart the server**:
   ```bash
   python -m askany.main --serve
   ```

LightRAG results will now be merged into `rag_search` tool responses in the agent workflow.

## Research: LightRAG input — extract itself vs use LlamaIndex chunks

**Question:** Should LightRAG do its own chunking and extraction (current), or consume chunks produced by the LlamaIndex pipeline?

### Current behavior (LightRAG extracts itself)

1. **lightrag_ingest.py** reads raw files and passes text to the adapter:
   - Markdown: `insert_async(texts, file_paths=paths, split_by_character="\n## ")` — coarse split by H2.
   - JSON FAQs: one "问题: Q\n答案: A" string per entry, no `split_by_character`.
2. **LightRAG `ainsert()`** then:
   - Optionally splits each item on `split_by_character`.
   - Runs its **internal token-based chunker** (`chunk_token_size=800`, `chunk_overlap_token_size=150` from config).
   - Runs entity/relationship extraction (LLM) on those chunks.
   - Writes to `LIGHTRAG_*` tables and builds HNSW on `vdb_chunks`, `vdb_entity`, `vdb_relation`.

So “extract itself” means: we do a coarse split (H2 or per-FAQ); LightRAG does **fine-grained chunking + LLM extraction**.

### Alternative: use LlamaIndex chunks

- **LlamaIndex** produces nodes via `MarkdownParser` (`markdown` / `semantic` / `hybrid`) and stores them in the docs vector store (and optional cache).
- LightRAG exposes **`ainsert_custom_chunks(self, full_text, text_chunks, doc_id=None)`**: you pass pre-chunked strings; LightRAG skips its chunker and only runs entity extraction on `text_chunks`.

So we could:
- Run the main ingest (or load cached docs nodes), then call `ainsert_custom_chunks(full_text, [node.get_content() for node in nodes], doc_id=...)` in batches so LightRAG uses the same chunks as the main RAG.

### Tradeoffs

| Aspect | LightRAG extracts itself (current) | Use LlamaIndex chunks |
|--------|-------------------------------------|------------------------|
| **Chunk boundaries** | 800-token chunks tuned for entity extraction (one concept per chunk; see config comments). | LlamaIndex: structure/semantic/hybrid — often larger sections or semantic segments, tuned for retrieval. |
| **Consistency** | Two pipelines: different chunk boundaries for vector retrieval vs KG. | One source of chunks for both retrieval and KG. |
| **Entity quality** | Small chunks reduce mixed topics per chunk → cleaner entities on dense Chinese docs. | Larger or retrieval-oriented chunks can mix topics → noisier entity extraction. |
| **Ingestion** | Two ingest commands (main + `lightrag_ingest`); same files read twice. | Single parse; could feed same nodes to both stores (or ingest from cache). |
| **API** | Standard `ainsert()` is the main, supported API. | `insert_custom_chunks` / `ainsert_custom_chunks` may be deprecated (see upstream issues). |

### Recommendation

**Keep LightRAG extracting itself (current design).**

1. **Entity extraction quality:** Config and comments explicitly use 800-token chunks to avoid mixing multiple H2 sections and to get cleaner entities. LlamaIndex chunking is tuned for retrieval (sections/semantic), not for “one concept per chunk.”
2. **Different goals:** LlamaIndex optimizes for vector/keyword retrieval; LightRAG optimizes for KG extraction. Forcing the same chunks on both would likely hurt one of the two.
3. **API stability:** Relying on `ainsert_custom_chunks` is riskier if it is deprecated; `ainsert()` is the supported path.
4. **Clarity:** Two ingest commands (main vs LightRAG) are straightforward; a single pipeline would require refactoring to produce nodes once and fan-out to both stores and to maintain `full_text` per doc for custom chunks.

If you later want a “single pipeline” experiment, you could add an optional path that loads the same cached docs nodes used by the main ingest and feeds their text to `ainsert_custom_chunks` in batches, accepting the tradeoffs above (and checking upstream deprecation status for custom chunks).

---

## Data Management

### Delete all LightRAG data

To clear the knowledge graph and start fresh:

```bash
rm -rf lightrag_data/
```

This removes the local graph file (`graph_chunk_entity_relation.graphml`) and cache. PostgreSQL tables (`LIGHTRAG_*`) remain — to drop them:

```bash
PGPASSWORD=123456 psql -h localhost -U wufei -d askany -c "
  DROP TABLE IF EXISTS lightrag_doc_status, lightrag_doc_chunks,
    lightrag_vdb_chunks, lightrag_vdb_entity, lightrag_vdb_relation CASCADE;
"
```

After deletion, re-ingest:
```bash
python -m askany.rag.lightrag_ingest --ingest-markdown --ingest-json
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `lightrag-hku package not installed` warning | Missing dependency | `uv add lightrag-hku` |
| `PipelineNotInitializedError` | Adapter not fully initialized | Ensure `await adapter.initialize()` is called |
| Empty retrieval results | No data ingested | Run ingestion commands above |
| `ModuleNotFoundError: pytest` | pytest not installed | `uv add --dev pytest pytest-asyncio` |
| Embedding errors / `TypeError: NoneType` | Wrong embedding setup | Adapter uses local SentenceTransformer, not API. Check BAAI/bge-m3 is downloaded. |
| `LIGHTRAG_*` tables missing | First run without ingestion | Tables are auto-created on first `adapter.initialize()` |
