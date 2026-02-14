# AskAny

[English](README.md) | 中文版

面向运维、开发、测试团队的中文优化 RAG（检索增强生成）问答助手。

> 📖 **博客文章**：[AskAny：为运维测试团队打造的智能问答助手](https://zhuanlan.zhihu.com/p/2004009140419310603) - 详细介绍项目背景、技术选型和架构设计

## 架构概览

### 人工设计工作流模式

![人工设计工作流架构](images/workflow.png)

**OpenWebUI 中的响应示例：**

![人工设计工作流示例](images/quest2.png)

### 自动 Agent 模式

![自动 Agent 架构](images/general_agent.png)

**OpenWebUI 中的响应示例：**

![自动 Agent 示例](images/quest1.png)

### 输出结构说明

AskAny 在 OpenWebUI 中的响应包含以下三个核心部分：

1. **工具调用流程**
   - 展示 Agent 执行过程中的工具调用序列
   - 显示每个工具（RAG 检索、网络搜索、本地文件搜索等）的调用情况
   - 帮助用户理解系统如何逐步获取信息来回答问题

2. **总结回答**
   - 基于检索到的信息生成的最终答案
   - 整合多轮检索和工具调用的结果
   - 提供清晰、准确的回答内容

3. **参考文档**
   - 列出用于生成回答的参考文档来源
   - 包含文档标题、来源路径等元数据
   - 支持用户追溯答案来源，验证信息可靠性

这种结构化的输出方式确保了答案的可追溯性和透明度，用户可以清楚地了解系统如何获取信息并生成答案。

## 功能特性

- **工作流 Agent**：基于 LangGraph 的多阶段 Agent 工作流，降低幻觉
- **混合检索**：结合关键词搜索和向量搜索，配合重排序
- **多模态支持**：通过 OpenWebUI 集成支持文本 + 图像
- **中文优化**：HanLP TF-IDF、PostgreSQL zhparser
- **双工作流模式**：人工设计的工作流（更稳定详细，约60秒）和自动 Agent（更快速，约30秒）
- **FAQ 标签过滤**：支持 @tag 元数据过滤
- **vLLM 集成**：OpenAI 兼容 API 的快速推理

## 技术栈

| 组件 | 技术 |
|------|------|
| 工作流引擎 | LangGraph（Agent 编排状态机） |
| RAG 框架 | LlamaIndex（检索和查询引擎） |
| API 层 | FastAPI + LangServe（OpenAI 兼容端点） |
| 向量存储 | PostgreSQL + pgvector（HNSW 索引） |
| 嵌入模型 | BAAI/bge-m3（多语言，1024维） |
| 重排序模型 | BAAI/bge-reranker-v2-m3（多语言） |
| LLM | vLLM 服务（默认：Qwen） |
| 前端 | OpenWebUI（聊天界面） |

## 架构

```
用户查询 (OpenWebUI)
    ↓
FastAPI 服务器 (askany/api/server.py)
    ↓
┌─────────────────────────────────────────────────────┐
│  工作流模式选择                                      │
│  ├── min_langchain_agent.py (LangChain Agent)      │
│  └── workflow_langgraph.py (LangGraph 状态机)       │
└─────────────────────────────────────────────────────┘
    ↓
WorkflowFilter (workflow_filter.py) - 预过滤简单查询
    ↓
QueryRouter (rag/router.py) - 路由到 FAQ/DOCS 引擎
    ↓
┌─────────────────────────────────────────────────────┐
│  RAG 引擎                                           │
│  ├── FAQQueryEngine (混合检索: 关键词 + 向量)       │
│  └── RAGQueryEngine (文档检索)                      │
└─────────────────────────────────────────────────────┘
    ↓
VectorStoreManager (PostgreSQL + pgvector)
    ↓
带引用的响应
```

### 工作流模式

**1. 人工设计工作流 (`workflow_langgraph.py`)**
- 人工设计的 LangGraph 状态机，显式编排检索流程
- 迭代式上下文扩展，可控的检索轮次
- 子问题分解处理复杂查询
- 更稳定详细的响应（平均约60秒）
- 适用场景：需要深入分析和多轮检索的复杂查询

**2. 自动 Agent (`min_langchain_agent.py`)**
- LangChain Agent 自动选择工具（RAG、网络搜索、本地文件搜索）
- 灵活自适应的工具使用策略
- 更快的响应时间（平均约30秒）
- 适用场景：需要多种信息源和快速响应的查询

## 快速开始

### 安装

```bash
# 安装 Python 3.11 和依赖
uv python install 3.11
uv python pin 3.11
uv tool install ruff
uv sync

# 配置环境
cp .env.example .env
# 编辑 .env 配置数据库凭据和 API 端点
```

### 数据库设置

```bash
# 快速设置（推荐）
sudo bash setup_postgresql.sh

# 或手动设置
createdb askany
psql -d askany -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 文档入库

AskAny 支持两种文档格式：

**1. FAQ 格式 (JSON)**

将 FAQ 文件放在 `data/json/` 目录下。每个 FAQ 条目应遵循以下结构：

```json
{
  "question": "API 服务器的默认端口是多少？",
  "answer": "默认端口是 8000。您可以通过在 config.py 中设置 api_port 或设置 API_PORT 环境变量来更改它。",
  "metadata": {
    "tag": "配置",
    "category": "api"
  }
}
```

**2. Markdown 格式**

将文档文件放在 `data/markdown/` 目录下。支持标准 Markdown 文件

**入库命令：**

```bash
# 从 data/json 和 data/markdown 入库所有文档
python -m askany.main --ingest

# 检查入库数据
python -m askany.main --check-db
```

### 启动服务器

```bash
# 启动 API 服务器（默认：http://0.0.0.0:8000）
python -m askany.main --serve

# 服务器提供：
# - OpenAI 兼容聊天端点：POST /v1/chat/completions
# - OpenAPI 规范：GET /openapi.json
# - FAQ 热更新：POST /v1/update_faqs
# - 健康检查：GET /health
```

### 测试查询

```bash
# 直接查询测试
python -m askany.main --query --query-text "你的问题" --query-type AUTO

# 查询类型：AUTO（智能路由）、FAQ、DOCS
```

## 项目结构

```
askany/
├── api/                    # FastAPI 服务器和端点
│   └── server.py          # 主 API 服务器
├── config.py              # 集中配置管理
├── ingest/                # 文档入库
│   ├── vector_store.py    # PostgreSQL + pgvector 管理
│   ├── json_parser.py     # FAQ JSON 入库
│   └── markdown_parser.py # Markdown 文档解析
├── prompts/               # 集中提示词管理
│   ├── prompts_cn.py      # 中文提示词
│   ├── prompts_en.py      # 英文提示词
│   └── prompt_manager.py  # 语言感知提示词管理器
├── rag/                   # RAG 组件
│   ├── router.py          # 查询路由逻辑
│   ├── faq_query_engine.py    # FAQ 混合检索
│   └── rag_query_engine.py    # 文档检索
├── workflow/              # Agent 工作流
│   ├── workflow_langgraph.py  # LangGraph 状态机
│   ├── min_langchain_agent.py # LangChain agent
│   ├── workflow_filter.py     # 简单查询预过滤
│   └── ...                    # 支持模块
└── main.py                # 入口点
```

### 工具脚本和测试文件

项目包含 `tool/` 目录下的工具脚本和 `test/` 目录下的测试文件：

**工具脚本 (`tool/`):**
- `export_vector_data.py` - 从 PostgreSQL 导出向量数据到文件（支持 pg_dump 自定义格式或分离的表结构/数据文件）
- `import_vector_data.py` - 从文件导入向量数据到 PostgreSQL
- `ingest_check.py` - 验证数据库中的入库数据
- `query_test.py` - 直接测试查询
- 其他用于数据迁移、关键词导出和 HNSW 索引检查的工具脚本

**测试文件 (`test/`):**
- `test_workflow_client_call.py` - 工作流客户端测试
- 各种组件和集成的测试脚本

**常用命令：**

查看 `dev_readme/script.sh` 获取常用命令，包括：
- 文档入库（带正确编码）
- 服务器启动
- 数据库数据检查
- HNSW 索引检查
- 向量数据导入/导出

使用示例：
```bash
# 导出向量数据
python tool/export_vector_data.py --output-dir vector_data --format full

# 导入向量数据
python tool/import_vector_data.py --input-dir vector_data --drop-existing

# 检查入库数据
python tool/ingest_check.py
```

## Configuration

All settings can be configured in `askany/config.py` or via environment variables in `.env`.

### Core Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `language` | Prompt language (cn/en) | cn |
| `device` | Compute device (cuda/cpu) | cuda |
| `log_level` | Logging level | DEBUG |

### LLM Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `openai_api_base` | LLM API endpoint | dashscope.aliyuncs.com |
| `openai_api_key` | API key (set in .env) | - |
| `openai_model` | LLM model name | qwen-plus |
| `temperature` | Generation temperature | 0.2 |
| `top_p` | Top-p sampling | 0.8 |
| `llm_max_tokens` | Max tokens for LLM | 40000 |
| `llm_timeout` | LLM timeout (seconds) | 700 |

### Database Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `postgres_host` | PostgreSQL host | localhost |
| `postgres_port` | PostgreSQL port | 5432 |
| `postgres_user` | Database user | root |
| `postgres_password` | Database password | 123456 |
| `postgres_db` | Database name | askany |
| `faq_vector_table_name` | FAQ vector table | askany_faq_vectors |
| `docs_vector_table_name` | Docs vector table | askany3_docs_vectors |

### Embedding & Reranker

| Setting | Description | Default |
|---------|-------------|---------|
| `embedding_model` | Embedding model | BAAI/bge-m3 |
| `embedding_model_type` | Model type | sentence_transformer |
| `embedding_local_files_only` | Offline mode | False |
| `vector_dimension` | Vector dimension | 1024 |
| `reranker_model` | Reranker model | BAAI/bge-reranker-v2-m3 |
| `reranker_local_files_only` | Offline mode | False |

### Retrieval Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `faq_similarity_top_k` | FAQ top-k results | 3 |
| `faq_rerank_candidate_k` | FAQ rerank candidates | 10 |
| `faq_score_threshold` | FAQ score threshold | 0.75 |
| `faq_ensemble_weights` | [keyword, vector] weights | [0.5, 0.5] |
| `docs_similarity_top_k` | Docs top-k results | 5 |
| `docs_rerank_candidate_k` | Docs rerank candidates | 10 |
| `docs_similarity_threshold` | Docs score threshold | 0.6 |
| `docs_ensemble_weights` | [keyword, vector] weights | [0.5, 0.5] |

### HNSW Index Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `enable_hnsw` | Enable HNSW index | True |
| `hnsw_m` | Bi-directional links | 16 |
| `hnsw_ef_construction` | Construction candidate list | 128 |
| `hnsw_ef_search` | Search candidate list | 40 |
| `hnsw_dist_method` | Distance method | vector_cosine_ops |

### Agent Workflow Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `agent_max_iterations` | Max workflow iterations | 3 |
| `enable_web_search` | Enable web search | True |
| `web_search_api_url` | Web search API endpoint URL | http://localhost:8800/search |
| `query_rewrite_bool` | Enable query rewriting | True |
| `expand_context_ratio` | Context expansion ratio | 1.0 |

### Data Paths

| Setting | Description | Default |
|---------|-------------|---------|
| `json_dir` | FAQ JSON directory | data/json |
| `markdown_dir` | Markdown docs directory | data/markdown |
| `local_file_search_dir` | Local search directory | data/markdown |
| `storage_dir` | Keyword index storage | key_word_storage |

### Customizing Prompts for Your Knowledge Base

The prompts in `askany/prompts/prompts_cn.py` (Chinese) and `askany/prompts/prompts_en.py` (English) contain `TODO` placeholders that should be customized for each knowledge base deployment.


## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/openapi.json` | GET | OpenAPI 规范 |
| `/v1/chat/completions` | POST | OpenAI 兼容聊天 |
| `/v1/update_faqs` | POST | FAQ 热更新 |

## 开发

### 代码质量

```bash
# 使用 ruff 格式化代码
ruff format .

# 代码检查
ruff check .

# 自动修复问题
ruff check --fix .
```

### 常用命令

常用命令请参考 `dev_readme/script.sh`。主要操作包括：

```bash
# 文档入库（带正确编码）
PYTHONIOENCODING=utf-8 LC_ALL=en_US.UTF-8 NO_COLOR=1 TERM=dumb python -u -m askany.main --ingest > ingest.log 2>&1

# 启动服务器
PYTHONIOENCODING=utf-8 LC_ALL=en_US.UTF-8 NO_COLOR=1 TERM=dumb python -u -m askany.main --serve > serve.log 2>&1

# 检查数据库数据量
PGPASSWORD=123456 psql -h localhost -U root -d askany -c "SELECT 'data_askany_faq_vectors' as table_name, COUNT(*) as row_count FROM data_askany_faq_vectors UNION ALL SELECT 'data_askany_docs_vectors', COUNT(*) FROM data_askany_docs_vectors;"

# 导出/导入向量数据
python tool/export_vector_data.py --output-dir vector_data
python tool/import_vector_data.py --input-dir vector_data --drop-existing
```

### 添加新文档类型

1. 在 `askany/ingest/` 中创建解析器
2. 更新 `askany/ingest/ingest.py` 中的 `ingest_documents()`
3. 考虑用于过滤的元数据模式

## 路线图

查看 [roadmap.md](roadmap.md) 了解计划中的功能和改进。

## 部署

### OpenCode 集成（推荐）

AskAny 可以与 OpenCode 集成，通过 MCP（Model Context Protocol）协议，在利用 OpenCode 原生 grep 和本地文件搜索能力的同时，使用 AskAny 的 RAG 功能。

**核心特性：**
- 通过 OpenCode 提供基于 GUI 的 Web 界面
- 原生 grep 和本地文件搜索工具
- MCP 集成以访问 AskAny RAG 能力
- 文件夹级别的权限控制
- **最推荐的解决方案** - 结合两者优势

**设置步骤：**

详细说明请参考 [mcp.md](mcp.md)。

**快速开始：**

1. 克隆 OpenCode 仓库：
```bash
git clone https://github.com/wufei-png/opencode.git
cd opencode
git checkout wf/dev
```

2. 构建并部署 OpenCode（使用您的网络 IP）：
```bash
# 终端 1: 启动服务器
bun dev serve --hostname YOUR_IP --port 4096

# 终端 2: 启动 Web 应用
VITE_HOSTNAME=YOUR_IP bun run --cwd packages/app dev
```

3. 启动 AskAny MCP 服务器：
```bash
uv run python -m askany_mcp.server_fastapi
```

4. 配置 OpenCode：在 `~/.config/opencode/opencode.json` 中添加 MCP 服务器配置，并设置 `allowed_folders` 进行权限控制。

5. 测试：使用 `askany_mcp/test_fastapi_client.py` 验证集成。

### OpenWebUI 集成

AskAny 可以与 OpenWebUI 集成，提供基于 Web 的聊天界面。

**设置步骤：**

1. 克隆 OpenWebUI 仓库：
```bash
git clone https://github.com/wufei-png/open-webui.git
cd open-webui
git checkout wf/dev  # 使用自定义分支
```

然后按照说明部署 OpenWebUI，参考 [OpenWebUI 开发文档](https://docs.openwebui.com/getting-started/development)。

我的部署操作：
- 前端：```npm run dev```
- 后端（在 conda 或 uv 环境中）：```cd backend && ENABLE_OLLAMA_API=False BYPASS_EMBEDDING_AND_RETRIEVAL=true ./dev.sh```

### Web 搜索服务

AskAny 通过 [Proxyless LLM WebSearch](https://github.com/wufei-png/proxyless-llm-websearch/tree/wf/company) 服务支持网络搜索功能。

**设置步骤：**

1. 克隆 Web 搜索服务仓库：
```bash
git clone https://github.com/wufei-png/proxyless-llm-websearch.git
cd proxyless-llm-websearch
git checkout wf/company  # 使用自定义分支
```

2. 配置环境（如需要）：
```bash
cp .env.example .env
# 编辑 .env 配置搜索引擎 API 密钥等
```

4. 启动 Web 搜索 API 服务器：
```bash
# 启动 API 服务器
python agent/api_serve.py
```

服务默认在 `http://localhost:8800` 启动。搜索端点为 `http://localhost:8800/search`。

6. 配置 AskAny 使用 Web 搜索服务：
   - **默认**：Web 搜索 API URL 在 `askany/config.py` 中配置为 `web_search_api_url`（默认：`http://localhost:8800/search`）

**注意**：确保在启动 AskAny 之前 Web 搜索服务已运行，否则网络搜索功能将不可用。

## 相关资源

- [OpenCode 集成指南](mcp.md) - 详细的 OpenCode MCP 集成说明
- [Proxyless LLM WebSearch](https://github.com/wufei-png/proxyless-llm-websearch/tree/wf/company)
- [OpenWebUI](https://github.com/wufei-png/open-webui/tree/wf/dev)
- [OpenCode 仓库](https://github.com/wufei-png/opencode/tree/wf/dev)

## 许可证

MIT
