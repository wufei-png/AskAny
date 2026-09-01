# PostgreSQL and pgvector setup

AskAny requires PostgreSQL with the `vector` extension. Choose either the
repository development container or an existing host installation. Configure
the application with the `POSTGRES_*` variables in `.env`.

## Option A: development container

From the repository root:

```bash
docker compose -f docker-compose.dev.yml up -d postgres
docker compose -f docker-compose.dev.yml exec -T postgres \
  psql -U root -d askany \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

The compose service uses `pgvector/pgvector:pg17-bookworm`, publishes port
`5432`, and persists data in the `postgres_data` volume. Its configured
credentials are:

```text
POSTGRES_HOST=localhost       # from the host; use postgres inside dev
POSTGRES_PORT=5432
POSTGRES_USER=root
POSTGRES_PASSWORD=123456
POSTGRES_DB=askany
```

If the API runs inside the `dev` service, use `POSTGRES_HOST=postgres`. If it
runs on the host, use `POSTGRES_HOST=localhost`. The application defaults use
user `wufei`, so set the values explicitly when using this compose database.

Start the optional development container with:

```bash
make build_dev_image
docker compose -f docker-compose.dev.yml up -d dev
docker compose -f docker-compose.dev.yml exec dev bash
```

The `dev` service requests NVIDIA GPU resources. Run only the `postgres`
service when GPU container support is not available.

## Option B: existing host PostgreSQL

Install PostgreSQL and the pgvector extension using your operating system's
package manager or the official pgvector instructions. Match the major
PostgreSQL version and extension package installed on the host.

For macOS with Homebrew, for example:

```bash
brew install postgresql@17
brew services start postgresql@17
```

For Debian/Ubuntu, install PostgreSQL and the pgvector package corresponding to
the installed PostgreSQL major version, then start the service with the normal
system service manager.

Create a database and application user as an administrator. Do not grant
superuser privileges unless your local policy requires them:

```sql
CREATE USER wufei WITH PASSWORD 'replace-with-a-local-password';
ALTER USER wufei CREATEDB;
CREATE DATABASE askany OWNER wufei;
```

Then enable the extension:

```bash
psql -h localhost -U wufei -d askany \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Set matching values in `.env`:

```text
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=wufei
POSTGRES_PASSWORD=replace-with-a-local-password
POSTGRES_DB=askany
```

## Verify the connection

```bash
pg_isready -h localhost -p 5432
psql -h localhost -U wufei -d askany \
  -c "SELECT version();"
psql -h localhost -U wufei -d askany \
  -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

After the database is available, run the application checks:

```bash
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany.main --create-index
```

## Cleanup

To stop the compose services without removing data:

```bash
docker compose -f docker-compose.dev.yml stop
```

To remove the compose containers and their named volumes, only after confirming
that the data is disposable:

```bash
docker compose -f docker-compose.dev.yml down -v
```

There is no repository `setup_postgresql.sh`; do not rely on an old command
that refers to that file.
