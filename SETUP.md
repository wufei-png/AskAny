# AskAny 设置指南

## 前置要求

1. **Python 3.11** (使用uv管理)
2. **PostgreSQL** (需要安装pgvector扩展)
3. **vLLM或OpenAI API** (用于LLM和embedding)

## 安装步骤

### 1. 安装Python依赖

```bash
uv python install 3.11
uv python pin 3.11
uv sync
```

### 2. 配置PostgreSQL

```bash
# 创建数据库
createdb askany

# 连接到数据库并安装pgvector扩展
psql -d askany -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. 配置环境变量

复制 `.env.example` 到 `.env` 并编辑：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置：
- PostgreSQL连接信息
- OpenAI API密钥（或vLLM地址）
- 其他配置项

### 4. 准备数据

确保以下目录存在并包含数据：
- `data/json/` - FAQ JSON文件
- `data/markdown/` - Markdown文档

### 5. 文档入库

```bash
python -m askany.main --ingest
```

这将：
- 解析所有JSON FAQ文件
- 解析所有Markdown文档
- 生成embeddings并存储到PGVector

### 6. 启动API服务器

```bash
python -m askany.main --serve
```

服务器将在 `http://0.0.0.0:8000` 启动。

### 7. 配置OpenWebUI

1. 打开OpenWebUI设置
2. 添加自定义后端：
   - **URL**: `http://localhost:8000`
   - **OpenAPI URL**: `http://localhost:8000/openapi.json`
3. OpenWebUI会自动从 `/openapi.json` 获取API规范

## 验证

### 检查API健康状态

```bash
curl http://localhost:8000/health
```

### 检查OpenAPI规范

```bash
curl http://localhost:8000/openapi.json
```

### 测试聊天接口

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

## 故障排除

### PostgreSQL连接错误

- 检查PostgreSQL是否运行：`pg_isready`
- 验证 `.env` 中的数据库配置
- 确保pgvector扩展已安装

### 向量维度不匹配

- 默认使用OpenAI text-embedding-ada-002 (1536维)
- 如果使用其他embedding模型，需要在 `.env` 中设置 `VECTOR_DIMENSION`

### API服务器无法启动

- 检查端口8000是否被占用
- 验证所有依赖已安装：`uv pip list`
- 查看错误日志

