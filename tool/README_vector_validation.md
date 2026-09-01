# Vector export/import validation

This is an operator workflow for validating a vector-data backup and restore.
It changes database state by clearing tables; use a disposable database or an
approved maintenance window and confirm a separate backup first.

The scripts use the current settings from `askany/config.py`. By default the
physical tables are `data_askany_faq_vectors` and
`data_askany3_docs_vectors`; table names change if the corresponding settings
are overridden.

## Tools

| Script | Purpose |
|---|---|
| `export_vector_data.py` | Export configured vector tables as custom dumps or schema/CSV files. |
| `backup_and_clear.py` | Copy FAQ/docs tables to `_test` tables, then clear the originals. Destructive. |
| `import_vector_data.py` | Restore tables from `vector_data`. `--drop-existing` is destructive. |
| `compare_table_data.py` | Compare each original table with its `_test` copy, including vector text. |
| `ingest_check.py` | Inspect current ingestion/index data. |

## Recommended validation sequence

Run from the repository root:

```bash
uv run --locked python tool/export_vector_data.py \
  --output-dir vector_data --format full
```

Inspect `vector_data/metadata.json` and confirm the target database before
continuing. Then:

```bash
uv run --locked python tool/backup_and_clear.py
uv run --locked python tool/import_vector_data.py \
  --input-dir vector_data
uv run --locked python tool/compare_table_data.py
```

`backup_and_clear.py` creates `_test` copies for the configured FAQ and docs
tables. `compare_table_data.py` compares the restored originals with those
copies. A successful comparison means the script-observed rows match; it is not
a live query-quality or model-provider test.

## Separate-format transfer

Use schema and CSV files instead of custom dumps when required by the target
environment:

```bash
uv run --locked python tool/export_vector_data.py \
  --output-dir vector_data --format separate
uv run --locked python tool/import_vector_data.py \
  --input-dir vector_data --drop-existing
```

Confirm PostgreSQL client tools, pgvector, permissions, and the embedding
dimension before importing. `--drop-existing` may remove existing tables.

## Recovery and cleanup

If the comparison fails, stop application writes, inspect the tool logs and
`metadata.json`, and restore from the verified backup. Do not delete `_test`
tables or source backup files until recovery is complete. The scripts have no
sample row-count success criteria because database contents vary by checkout
and environment.
