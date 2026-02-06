# 离线模型加载指南

本文档说明如何配置 AskAny 使用本地模型文件，实现完全离线加载，避免访问 HuggingFace。

## 支持的模型

- **Embedding 模型**: `BAAI/bge-m3`, `BAAI/bge-large-zh-v1.5` 等
- **Reranker 模型**: `BAAI/bge-reranker-v2-m3`, `BAAI/bge-reranker-base` 等

## 方法一：使用本地路径（推荐）

### 1. 下载模型到本地

#### 方法 A: 使用 huggingface-cli（推荐）

```bash
# 安装 huggingface-hub
pip install huggingface-hub

# 下载 embedding 模型
huggingface-cli download BAAI/bge-m3 --local-dir /path/to/local/bge-m3

# 下载 reranker 模型
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir /path/to/local/bge-reranker-v2-m3
```

#### 方法 B: 使用 Python 脚本

```python
from sentence_transformers import SentenceTransformer

# 下载并保存 embedding 模型
model = SentenceTransformer("BAAI/bge-m3")
model.save("/path/to/local/bge-m3")

# 下载并保存 reranker 模型（使用 CrossEncoder）
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
reranker.save("/path/to/local/bge-reranker-v2-m3")
```

#### 方法 C: 从已有缓存复制

如果模型已经下载到 HuggingFace 缓存目录（`~/.cache/huggingface/hub/`），可以直接复制：

```bash
# 查找缓存目录
ls ~/.cache/huggingface/hub/models--BAAI--bge-m3/

# 复制到目标目录
cp -r ~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<commit-hash>/* /path/to/local/bge-m3/
```

### 2. 配置本地路径

在 `.env` 文件中配置：

```bash
# Embedding 模型本地路径（使用绝对路径）
EMBEDDING_MODEL=/path/to/local/bge-m3

# Reranker 模型本地路径（使用绝对路径）
RERANKER_MODEL=/path/to/local/bge-reranker-v2-m3

# 可选：强制离线模式（如果路径是本地路径，会自动启用）
EMBEDDING_LOCAL_FILES_ONLY=true
RERANKER_LOCAL_FILES_ONLY=true
```

或者在 `config.py` 中直接修改：

```python
embedding_model: str = "/path/to/local/bge-m3"
reranker_model: str = "/path/to/local/bge-reranker-v2-m3"
embedding_local_files_only: bool = True
reranker_local_files_only: bool = True
```

### 3. 验证配置

代码会自动检测：
- 如果 `model_name` 是绝对路径且存在，会自动使用 `local_files_only=True`
- 如果设置了 `embedding_local_files_only=True` 或 `reranker_local_files_only=True`，会强制离线模式

## 方法二：使用环境变量强制离线

即使使用 HuggingFace 模型名称，也可以通过环境变量强制离线模式：

```bash
# 设置 HuggingFace 离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 然后在 .env 中设置
EMBEDDING_LOCAL_FILES_ONLY=true
RERANKER_LOCAL_FILES_ONLY=true
```

这样会从本地缓存加载模型，如果缓存不存在会报错。

## 模型文件要求

本地模型目录必须包含以下文件：

### Embedding 模型必需文件：
- `config.json` - 模型配置
- `pytorch_model.bin` 或 `model.safetensors` - 模型权重
- `tokenizer_config.json` - Tokenizer 配置
- `vocab.txt` 或 `vocab.json` - 词汇表
- `modules.json` - SentenceTransformer 模块配置（如果使用 SBERT 架构）

### Reranker 模型必需文件：
- `config.json` - 模型配置
- `pytorch_model.bin` 或 `model.safetensors` - 模型权重
- `tokenizer_config.json` - Tokenizer 配置
- `vocab.txt` 或 `vocab.json` - 词汇表

## 验证离线加载

运行以下命令验证模型是否从本地加载：

```bash
# 测试 embedding 模型
python -m askany.main --query --query-text "测试" --query-type AUTO

# 查看日志，应该看到：
# INFO: Using SentenceTransformer embedding model: /path/to/local/bge-m3 (dimension: 1024)
# 而不是访问 huggingface.co 的网络请求
```

## 常见问题

### 1. 模型加载失败

**错误**: `OSError: Can't load config for '/path/to/model'`

**解决**: 确保模型目录包含所有必需文件，特别是 `config.json`。

### 2. 仍然访问 HuggingFace

**原因**: 
- 路径不是绝对路径
- 路径不存在
- `local_files_only` 未设置为 `True`

**解决**: 
- 使用绝对路径
- 检查路径是否存在
- 在配置中设置 `embedding_local_files_only=True`

### 3. Tokenizer 文件缺失

**错误**: `OSError: Can't load tokenizer`

**解决**: 确保模型目录包含完整的 tokenizer 文件（`tokenizer_config.json`, `vocab.txt` 等）。

## 示例配置

完整的 `.env` 配置示例：

```bash
# 使用本地模型路径
EMBEDDING_MODEL=/net/ai02/data/xlzhong/wufei/models/bge-m3
RERANKER_MODEL=/net/ai02/data/xlzhong/wufei/models/bge-reranker-v2-m3

# 强制离线模式
EMBEDDING_LOCAL_FILES_ONLY=true
RERANKER_LOCAL_FILES_ONLY=true

# 其他配置...
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
# ...
```

## 性能优化

使用本地模型的好处：
1. **更快的加载速度** - 无需网络下载
2. **完全离线** - 适合内网环境
3. **版本控制** - 可以精确控制使用的模型版本
4. **减少网络流量** - 适合带宽受限环境

## 相关文档

- [Embedding 模型使用说明](./embedding_model.md)
- [Reranker 模型选择指南](./rerank.md)
- [配置说明](../askany/config.py)



wget https://file.hankcs.com/hanlp/tok/coarse_electra_small_20220616_012050.zip