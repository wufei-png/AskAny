# AskAny

![AskAny 封面](images/cover/cover1.png)

[English](README.md) | 中文版

AskAny 是面向运维、开发和测试团队的中文优化 RAG（检索增强生成）助手，
组合使用 LlamaIndex 检索、LangGraph/LangChain Agent、PostgreSQL + pgvector
以及 OpenAI 兼容的 FastAPI 接口。

> 代码是事实基准。当前运行边界请看
> [docs/current-runtime.md](docs/current-runtime.md)。历史设计资料位于
> [archive/docs/](archive/docs/)，不作为运行文档。

## 当前架构

```text
OpenWebUI 或其他 OpenAI 兼容客户端
                 |
                 v
FastAPI: /v1/chat/completions
                 |
       请求模型名是否以 -deepsearch 结尾？
             /                         \
            是                          否
             |                           |
workflow_langgraph.py          min_langchain_agent.py
  手工编排的 LangGraph             自动选择工具的 LangChain Agent
             \                         /
              +-- RAG / Web / 本地文件工具 --+
                            |
                 LlamaIndex FAQ/文档检索
                 + 可选 LightRAG 图谱增强
                            |
                       答案与来源引用
```

模型名以 `-deepsearch` 结尾时使用手工 LangGraph 路径，其他模型名使用自动
Agent。`WorkflowFilter` 只参与 deepsearch 的单条用户消息路径；自动 Agent
自行选择工具。

## 功能

- 基于 PostgreSQL + pgvector 的 FAQ 和文档检索；
- 按配置启用关键词/向量检索和重排序；
- LangGraph 深度检索编排与 LangChain 自动工具调用；
- 本地文件搜索、可选 Web 搜索、来源和 provenance 元数据；
- LightRAG 知识图谱增强：默认打开，但需要可选依赖和单独入库的数据；
- 可选 Mem0 跨会话用户记忆；
- 可选 Langfuse 追踪和异步 RAGAS 评估；
- QA 语义缓存以及始终可用的 Prometheus `/metrics` 接口；
- OpenAI 兼容的普通响应和 SSE 流式响应；
- `askany_mcp/` 下的独立 MCP 服务。

## 环境与安装

- Python `>=3.11,<3.12`；
- 带 `vector` 扩展的 PostgreSQL；
- OpenAI 兼容的 LLM 端点；
- SentenceTransformers embedding 和 reranker，除非配置了其他 API/本地模型。

安装锁定的核心环境：

```bash
uv python install 3.11
uv python pin 3.11
uv sync
cp .env.example .env
```

按需安装可选集成：

```bash
uv sync --extra lightrag
uv sync --extra observability
uv sync --all-extras
```

`.env.example` 使用代码中真实的 `Settings` 字段名，例如
`OPENAI_API_BASE`、`OPENAI_MODEL`、`POSTGRES_USER`、`EMBEDDING_MODEL` 和
`RERANKER_MODEL`。模板中的赋值为注释，直接拷贝不会覆盖 `askany/config.py`。
只需取消注释需要改动的项；代码默认值指向本地 vLLM 端点和本地模型路径。

## 数据库

使用仓库提供的开发容器：

```bash
docker compose -f docker-compose.dev.yml up -d postgres
```

Compose 服务使用 PostgreSQL 17 + pgvector，凭据为 `root`/`123456`。而
`askany/config.py` 的主机环境默认用户是 `wufei`，因此需要通过 `.env` 与
容器凭据对齐，或在本机 PostgreSQL 中创建同名用户。详见
[SETUP_POSTGRESQL.md](SETUP_POSTGRESQL.md) 和
[dev_readme/docker-compose.md](dev_readme/docker-compose.md)。

已有 PostgreSQL 时，使用与 `.env` 一致的用户创建数据库和扩展：

```bash
createdb askany
psql -d askany -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 数据与入库

运行时数据目录被 Git 忽略：

- `data/json/`：FAQ JSON；
- `data/markdown/`：Markdown 文档；
- `data/stopwords/`：可选分词资源。

FAQ 可以是单个对象或对象列表，例如：

```json
{
  "question": "API 默认端口是多少？",
  "answer": "API 默认端口是 8000。",
  "metadata": { "category": "configuration" }
}
```

主入库命令：

```bash
uv run --locked python -m askany.main --ingest
uv run --locked python -m askany.main --check-db
uv run --locked python -m askany.main --create-index
```

当前限制：`--ingest` 会解析 FAQ JSON，但
`askany/ingest/ingest.py` 中的 FAQ 向量写入代码处于禁用状态；该命令当前
真正写入的是 Markdown 文档节点。不能把它描述成完整的 FAQ+文档入库流程。
FAQ 向量数据应在服务初始化 FAQ 存储后，通过 `POST /v1/update_faqs` 更新。

LightRAG 使用独立的入库命令和存储路径：

```bash
uv sync --extra lightrag
uv run --locked python -m askany.rag.lightrag_ingest \
  --ingest-markdown --ingest-json
```

## 启动 API

```bash
uv run --locked python -m askany.main --serve
```

默认监听 `0.0.0.0:8000`：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:8000/v1/models
```

仅当两个受支持的 workflow 全局对象都已就绪时，`/health` 才返回 HTTP 200
和 `status: "ok"`；否则返回 HTTP 503 和 `status: "degraded"`。

### API 接口

| 方法   | 接口                   | 说明                                                         |
| ------ | ---------------------- | ------------------------------------------------------------ |
| `GET`  | `/health`              | 运行就绪检查，返回 `ok` 或 `degraded`。                      |
| `GET`  | `/metrics`             | Prometheus 指标；没有 `enable_prometheus` 或自定义端口配置。 |
| `GET`  | `/v1/models`           | 返回配置的模型，并尽可能追加 `-deepsearch` 变体。            |
| `POST` | `/v1/chat/completions` | OpenAI 兼容聊天接口，`stream: true` 使用 SSE。               |
| `POST` | `/v1/update_faqs`      | 接收 base64 编码的 JSON FAQ 并热更新，随后清理 QA 缓存。     |
| `GET`  | `/v1/cache/stats`      | QA 缓存统计。                                                |
| `POST` | `/v1/cache/clear`      | 清理 QA 缓存。                                               |
| `GET`  | `/openapi.json`        | 集成使用的自定义 OpenAPI 文档。                              |

聊天请求的模型名以 `-deepsearch` 结尾时选择手工 workflow，其他模型名选择
自动 Agent。流式响应以 stop chunk 和 `data: [DONE]` 结束。

直接查询：

```bash
uv run --locked python -m askany.main \
  --query --query-text "你的问题" --query-type AUTO
```

查询类型为 `AUTO`、`FAQ`、`DOCS` 和 `CODE`；其中 `CODE` 当前明确返回未实现。

## 重要配置默认值

所有配置位于 `askany/config.py`，可用对应的大写环境变量覆盖：

| 配置                               | 默认值                     |
| ---------------------------------- | -------------------------- |
| `language`                         | `cn`                       |
| `postgres_host` / `postgres_port`  | `localhost` / `5432`       |
| `postgres_user` / `postgres_db`    | `wufei` / `askany`         |
| `openai_api_base`                  | `http://127.0.0.1:8081/v1` |
| `embedding_model`                  | `BAAI/bge-m3`              |
| `vector_dimension`                 | `1024`                     |
| `reranker_model`                   | `BAAI/bge-reranker-v2-m3`  |
| `enable_lightrag`                  | `True`                     |
| `enable_mem0`                      | `False`                    |
| `enable_langfuse` / `enable_ragas` | `False` / `False`          |
| `enable_qa_cache`                  | `True`                     |
| `using_docs_keyword_index`         | `False`                    |

完整运行边界、派生默认值和验证限制见
[docs/current-runtime.md](docs/current-runtime.md)。

## 项目结构

```text
askany/
├── api/              # FastAPI 与 OpenAI 兼容接口
├── config.py         # Pydantic 配置
├── ingest/           # 解析器和 PostgreSQL/pgvector 存储
├── memory/           # 可选 Mem0 集成
├── metrics/          # Prometheus 埋点
├── observability/    # 可选 Langfuse/RAGAS
├── prompts/          # 中英文提示词
├── rag/              # 路由、检索、重排序、LightRAG 适配器
└── workflow/         # LangGraph 和 LangChain Agent
archive/              # 非当前 Python 与文档资料
askany_mcp/           # 独立 MCP 传输
tool/                 # 独立运维工具
test/                 # 单元测试和 opt-in 集成测试
```

## MCP

`askany_mcp/server.py` 是独立 stdio MCP 服务；HTTP/SSE 变体为
`server_fastapi.py`、`server_sse.py` 和 `server_http.py`。它们复用主项目
环境和配置，`askany_mcp/` 没有独立的 `pyproject.toml`。详见
[mcp.md](mcp.md) 和 [askany_mcp/README.md](askany_mcp/README.md)。

## 开发与验证

```bash
uv lock --check
uv run --locked ruff check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked ruff format --check askany test tool/keyword_utils.py tool/langdetect.py
uv run --locked --all-extras pyright
uv run --locked pre-commit run --all-files
uv run --locked pytest -q test -rs
```

静态检查覆盖受支持运行路径、测试和两个共享工具模块；独立工具、
`askany_mcp`、LightRAG 入库和可视化代码有独立边界。本地检查通过不代表
真实 PostgreSQL、模型服务、LightRAG、Mem0、Langfuse、RAGAS 或 QA 缓存已经
完成在线验证。

## 相关文档

- [当前运行契约](docs/current-runtime.md)
- [设置指南](SETUP.md)
- [PostgreSQL 与 pgvector](SETUP_POSTGRESQL.md)
- [UV 开发环境](UV_SETUP.md)
- [LightRAG 集成](dev_readme/lightrag.md)
- [向量数据操作](tool/README_vector_data.md)
- [路线图](roadmap.md)
