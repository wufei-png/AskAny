# Docker 开发环境

`docker-compose.dev.yml` 定义了 PostgreSQL 17 + pgvector 数据库和一个可选
的开发容器。以下命令均从仓库根目录执行。

## 启动 PostgreSQL

```bash
docker compose -f docker-compose.dev.yml up -d postgres
docker compose -f docker-compose.dev.yml exec -T postgres \
  psql -U root -d askany \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Compose 使用镜像 `pgvector/pgvector:pg17-bookworm`、主机端口 `5432` 和
命名卷 `postgres_data`。数据库凭据由 compose 文件固定为：

```text
POSTGRES_USER=root
POSTGRES_PASSWORD=123456
POSTGRES_DB=askany
```

容器内的数据库主机名是 `postgres`；从宿主机连接时使用 `localhost`。
应用默认用户是 `wufei`，因此运行应用前要在 `.env` 中将
`POSTGRES_USER`/`POSTGRES_PASSWORD` 配置为 compose 的凭据，或改用已有的
本机数据库。

## 启动开发容器

```bash
make build_dev_image
docker compose -f docker-compose.dev.yml up -d dev
docker compose -f docker-compose.dev.yml exec dev bash
```

`make build_dev_image` 默认构建标签为 `askany-dev:v2.0`；可以通过
`DEV_IMAGE_NAME` 覆盖。Compose 可以通过 `ASKANY_DEV_IMAGE` 指定要运行的
镜像，默认值同样是 `askany-dev:v2.0`。

开发容器将当前仓库挂载到 `/workspace`，并在 compose 文件中请求 NVIDIA
GPU 资源。没有 NVIDIA Container Toolkit 时，只启动 `postgres` 服务，或
自行调整开发容器的 GPU 配置。

## 在容器中运行项目

```bash
docker compose -f docker-compose.dev.yml exec dev bash
cd /workspace
uv sync
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany.main --ingest
uv run --locked python -m askany.main --serve
```

主 `--ingest` 命令当前真正写入 Markdown 文档节点；FAQ JSON 虽会解析，FAQ
向量写入代码目前被禁用。需要 FAQ 向量时使用运行中的 API 的
`POST /v1/update_faqs`。

## 日志与停止

```bash
docker compose -f docker-compose.dev.yml logs -f postgres
docker compose -f docker-compose.dev.yml logs -f dev
docker compose -f docker-compose.dev.yml stop
```

如果确认数据库和缓存数据都可以删除，再执行：

```bash
docker compose -f docker-compose.dev.yml down -v
```

不要使用未解析的全局卷名手工删除；Compose 会根据当前项目正确处理卷。
