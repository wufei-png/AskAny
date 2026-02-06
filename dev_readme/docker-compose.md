## Docker 开发环境（Ubuntu 22.04 + PostgreSQL 17 + pgvector 0.8.1）

项目提供了开箱即用的容器化开发环境，适用于本地开发与调试：

```bash
# 0. 构建 Ubuntu 22.04 开发镜像（首次或 Dockerfile.dev 变更后）
make build_dev_image

# 1. 启动 PostgreSQL 17（包含 pgvector 0.8.1）
docker compose -f docker-compose.dev.yml up -d postgres

# 2. 启动 Ubuntu 22.04 开发容器（包括 GPU 支持）
docker compose -f docker-compose.dev.yml up -d dev

# 3. 进入开发容器（可在其中运行 uv、python、make 等命令）
docker compose -f docker-compose.dev.yml exec dev zsh

# 注意：如需使用 GPU（CUDA），确保已安装 NVIDIA Container Toolkit：
# - Ubuntu/Debian: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
# - 容器启动后，可在容器内运行 `nvidia-smi` 验证 GPU 是否可用
# - 项目会自动检测并使用 GPU（embedding/reranker 模型）
```

说明：

- `pgvector/pgvector:pg17-bookworm` 官方镜像预装 PostgreSQL 17 与 pgvector 0.8.1，数据目录通过 `postgres_data` 卷持久化
- 首次启动时 `docker/initdb.d/01_enable_vector.sql` 会自动启用 `vector` 扩展
- `askany-dev` 镜像通过 `make build_dev_image` 构建（默认标签 `askany-dev:latest`，可通过 `DEV_IMAGE_NAME` 覆盖），容器挂载当前仓库到 `/workspace`
- `docker-compose.dev.yml` 通过 `ASKANY_DEV_IMAGE` 环境变量指定镜像名，未设置时使用 `askany-dev:latest`
- 可以根据需要在容器内运行 `uv sync`、`python -m askany.main --ingest` 等命令

在容器内连接数据库：

```bash
# 进入容器后，使用 postgres 作为主机名（docker-compose 服务名）
PGPASSWORD=123456 psql -h postgres -U root -d askany

# 或者使用环境变量（已在容器内设置）
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB

# 检查数据量示例
PGPASSWORD=123456 psql -h postgres -U root -d askany -c "SELECT 'data_askany_faq_vectors' as table_name, COUNT(*) as row_count FROM data_askany_faq_vectors UNION ALL SELECT 'data_askany_docs_vectors', COUNT(*) FROM data_askany_docs_vectors;"

# 查看数据示例
PGPASSWORD=123456 psql -h postgres -U root -d askany -c "SELECT id, LEFT(text, 200) as text_preview, metadata_, node_id FROM data_askany_docs_vectors LIMIT 1;"
```

**注意**：在容器内连接数据库时，主机名应使用 `postgres`（docker-compose 服务名），而不是 `localhost`。

查看日志：

```bash
# 查看 PostgreSQL 容器日志
docker compose -f docker-compose.dev.yml logs -f postgres

# 查看开发容器日志
docker compose -f docker-compose.dev.yml logs -f dev
```

停止服务：

```bash
# 停止指定服务（例如 dev 或 postgres）
docker compose -f docker-compose.dev.yml stop dev
docker compose -f docker-compose.dev.yml stop postgres
```

完全清理环境：

```bash
docker compose -f docker-compose.dev.yml down
docker volume rm postgres_data || true
docker volume rm pip_cache || true
```
