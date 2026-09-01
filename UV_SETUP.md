# UV development setup

AskAny declares Python `>=3.11,<3.12` in `pyproject.toml`. `uv` creates and
uses the project environment from the lockfile; the commands below work on
macOS, Linux, and WSL without hard-coded user paths.

## First setup

From the repository root:

```bash
uv python install 3.11
uv python pin 3.11
uv sync
cp .env.example .env
```

The copied `.env` starts fully commented and does not override
`askany/config.py`. Uncomment values only when they should differ; see
[`SETUP.md`](SETUP.md).

Install optional integrations when needed:

```bash
uv sync --extra lightrag
uv sync --extra observability
uv sync --all-extras
```

## Run commands

Prefer `uv run --locked` so commands use the repository environment and lock:

```bash
uv run --locked python -m askany.main --serve
uv run --locked python -m askany.main --check-db
uv run --locked pytest -q test -rs
```

Activation is optional. If a shell or IDE requires an activated environment:

```bash
source .venv/bin/activate
python --version
```

The environment path is always `<repository>/.venv`; its absolute location is
machine-specific and should not be committed to editor settings.

## Editor configuration

Point VS Code, Cursor, or another Python editor at:

```text
<repository>/.venv/bin/python
```

Do not copy a Linux-specific interpreter path into a macOS or WSL workspace.
The repository does not require a committed `.vscode/settings.json` file.

## Dependency changes

Add a runtime dependency with:

```bash
uv add package-name
```

After editing `pyproject.toml` manually, synchronize the lockfile and
environment:

```bash
uv lock
uv sync
```

Before delivery, verify the lockfile and supported quality surface:

```bash
uv lock --check
uv run --locked ruff check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format --check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked --all-extras pyright
uv run --locked pre-commit run --all-files
```
