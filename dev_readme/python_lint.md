# Python quality checks

The locked quality surface is the supported `askany` runtime, `test`, and the
shared helpers `tool/keyword_utils.py` and `tool/langdetect.py`. Run from the
repository root:

```bash
uv sync --all-extras
uv lock --check
uv run --locked ruff check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format --check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked --all-extras pyright
uv run --locked pre-commit run --all-files
uv run --locked pytest -q test -rs
```

For an intentional local autofix, use:

```bash
uv run --locked ruff check --fix askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format askany test tool/keyword_utils.py tool/langdetect.py
```

Standalone tools, `askany_mcp`, LightRAG ingestion, visualization code, and
archived Python are outside this static gate. External-service integration
checks may skip when prerequisites are absent; malformed input and code or
provider regressions must fail.
