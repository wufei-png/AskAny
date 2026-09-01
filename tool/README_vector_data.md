# Vector data export and import

`export_vector_data.py` and `import_vector_data.py` operate on the vector
tables configured in `askany/config.py`. They read the same `POSTGRES_*`, table,
and vector-dimension settings as the application.

## Export

```bash
uv run --locked python tool/export_vector_data.py
uv run --locked python tool/export_vector_data.py \
  --output-dir vector_data --format full
uv run --locked python tool/export_vector_data.py \
  --output-dir vector_data --format separate
```

The default output directory is `vector_data`; the default format is `full`.
The `full` format uses PostgreSQL custom dumps. The `separate` format writes
schema SQL and CSV data. Both formats also write `metadata.json` and sequence
information when available.

The configured logical table names default to `askany_faq_vectors` and
`askany3_docs_vectors`. LlamaIndex's physical table names normally add the
`data_` prefix, so the default physical names are
`data_askany_faq_vectors` and `data_askany3_docs_vectors`. A legacy table may be
exported when it exists.

## Import

```bash
uv run --locked python tool/import_vector_data.py \
  --input-dir vector_data
uv run --locked python tool/import_vector_data.py \
  --input-dir vector_data --drop-existing
uv run --locked python tool/import_vector_data.py \
  --input-dir vector_data --skip-sequences
```

The importer prefers table names recorded in `metadata.json` and otherwise
uses the current configured defaults. It can create the `vector` extension,
but the PostgreSQL installation must already provide pgvector and the database
user must have the required privileges.

`--drop-existing` drops imported tables before restoring them. Confirm the
target database and backup first; this option is destructive.

## Verify a transfer

```bash
uv run --locked python tool/ingest_check.py
```

For the explicit backup/clear/restore comparison workflow, see
[`README_vector_validation.md`](README_vector_validation.md). Do not run that
workflow against production without an approved backup and maintenance window.

## Requirements and limitations

- PostgreSQL client tools (`pg_dump` and `pg_restore`) are required for full
  dump export/import;
- `psycopg2` and the project's locked environment are required;
- the embedding model's output dimension must match `vector_dimension` and the
  target table schema;
- table names, row counts, and index state are environment-specific and should
  be read from `metadata.json` or the target database, not assumed from an
  example.
