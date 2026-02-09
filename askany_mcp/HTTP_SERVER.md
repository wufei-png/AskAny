# AskAny MCP HTTP Server

## 概述

提供两种远程访问方式：
1. **FastAPI HTTP** - 简单的 REST API（推荐）
2. **SSE (Server-Sent Events)** - 标准 MCP SSE 传输协议

## 安装依赖

```bash
uv sync
```

新增依赖：
- `starlette>=0.27.0` - ASGI 框架
- `sse-starlette>=1.6.5` - SSE 支持

## 启动服务器

### 方式 1: FastAPI HTTP（推荐）

```bash
python -m askany_mcp.server_fastapi --host 0.0.0.0 --port 8001
# 或使用 uv
uv run python -m askany_mcp.server_fastapi --host 0.0.0.0 --port 8001
```

### 方式 2: SSE 传输（标准 MCP 协议）

```bash
python -m askany_mcp.server_sse --host 0.0.0.0 --port 8001
# 或使用 uv
uv run python -m askany_mcp.server_sse --host 0.0.0.0 --port 8001
```

参数说明：
- `--host`: 绑定的 IP 地址（默认 0.0.0.0，允许外部访问）
- `--port`: 端口号（默认 8001）

## API 端点

### 健康检查
```bash
GET http://localhost:8001/health
```

### RAG 搜索
```bash
POST http://localhost:8001/tools/rag_search
Content-Type: application/json

{
  "query": "你的搜索问题"
}
```

响应示例：
```json
{
  "results": [
    {
      "index": 1,
      "source": "data/docs/example.md (lines 10-20)",
      "content": "相关内容...",
      "score": 0.85
    }
  ]
}
```

## 配置文件

### 本地 stdio 模式（`.mcp.json`）

用于本地进程间通信：

```json
{
  "mcpServers": {
    "askany-rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/wufei/github.com/wufei-png/AskAny",
        "run",
        "python",
        "-m",
        "askany_mcp.server"
      ]
    }
  }
}
```

### FastAPI HTTP 模式（`.mcp_web.json`）

用于 FastAPI HTTP 访问：

```json
{
  "mcpServers": {
    "askany-rag": {
      "url": "http://localhost:8001",
      "transport": "http"
    }
  }
}
```

### SSE 传输模式（`.mcp_sse.json`）

用于标准 MCP SSE 协议：

```json
{
  "mcpServers": {
    "askany-rag": {
      "url": "http://localhost:8001/sse",
      "transport": "sse"
    }
  }
}
```

**远程访问时，将 `localhost` 替换为服务器 IP 地址：**

```json
{
  "url": "http://192.168.1.100:8001"
}
```
