# Roadmap

This list contains work that is not currently complete in the implementation.
It is intentionally short; completed capabilities belong in the current
documentation, not on this list.

## Data and retrieval

- Complete the main FAQ ingestion path so `python -m askany.main --ingest`
  writes FAQ vectors as well as Markdown document vectors, with regression
  tests for both stores.
- Implement the `CODE` query route and document its tool/index contract.
- Decide whether the docs keyword index should become the default, based on
  measured retrieval quality and latency.
- Add parsers and ingestion contracts for additional formats such as PDF,
  DOCX, HTML, and YAML when a concrete source requires them.
- Implement safe document update/deletion. `MarkdownParser.delete_file_documents`
  currently raises `NotImplementedError`.

## Evaluation and operations

- Add opt-in end-to-end evaluation against real PostgreSQL, model providers,
  and LightRAG data, while preserving the existing skip/fail boundary.
- Add an operator-facing summary for RAGAS and Prometheus data if aggregated
  metrics are needed beyond `/metrics`.
- Define an explicit FAQ review/feedback workflow before adding human-in-the-
  loop state or answer correction storage.
- Standardize client-facing mode selection beyond the current `-deepsearch`
  model-name convention.

## Deferred ideas

- Investigate a single ingestion fan-out that reuses LlamaIndex chunks for
  LightRAG only after measuring entity extraction quality against the current
  separate pipeline.
- Reassess the standalone MCP transport implementations and consolidate them
  only if a supported client requires it.

## Not roadmap items

The following are already implemented and should not be re-added as planned
work: the two supported agent paths, SSE chat streaming, PostgreSQL/pgvector
storage with HNSW configuration, Mem0 integration, LightRAG adapter support,
Langfuse/RAGAS lifecycle hooks, and Prometheus instrumentation.
