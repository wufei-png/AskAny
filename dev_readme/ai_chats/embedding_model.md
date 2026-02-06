# BGE Embedding 模型使用说明

## 模型下载

### 自动下载（推荐）

**不需要提前下载**，`SentenceTransformer` 会在首次使用时自动从 HuggingFace 下载模型。

首次运行时会：
1. 自动下载模型文件（约 1.3GB）
2. 缓存到 `~/.cache/huggingface/hub/` 目录
3. 后续使用直接从缓存加载，无需重新下载

### 手动下载（可选）

如果需要提前下载或遇到网络问题，可以手动下载：

```bash
# 方法1: 使用 Python 脚本
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"

# 方法2: 使用 huggingface-cli
pip install huggingface-hub
huggingface-cli download BAAI/bge-large-zh-v1.5

# 方法3: 使用测试脚本
python tool/test_embedding_simple.py
```

### 模型信息

- **模型名称**: `BAAI/bge-large-zh-v1.5`
- **模型大小**: ~1.3GB
- **向量维度**: 1024
- **语言支持**: 中文（专为中文优化）
- **缓存位置**: `~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/`

## 验证模型加载

### 快速测试

```bash
# 运行测试脚本
python tool/test_embedding_simple.py
```

### 在代码中验证

```python
from askany.main import initialize_llm

# 初始化（会自动加载模型）
llm, embed_model = initialize_llm()

# 检查模型类型和维度
print(f"Model type: {type(embed_model).__name__}")
print(f"Dimension: {embed_model.dimension}")

# 测试编码
test_embedding = embed_model.get_query_embedding("测试文本")
print(f"Embedding length: {len(test_embedding)}")
```

## 常见问题

### 1. 下载速度慢

- 使用国内镜像（如果可用）
- 设置环境变量：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

### 2. 磁盘空间不足

- 模型需要约 1.3GB 空间
- 检查 `~/.cache/huggingface/` 目录空间

### 3. 网络连接问题

- 检查是否能访问 `huggingface.co`
- 考虑使用代理或镜像

### 4. CUDA 内存不足

- 如果 GPU 内存不足，会自动回退到 CPU
- 或使用更小的模型：`BAAI/bge-base-zh-v1.5` 或 `BAAI/bge-small-zh-v1.5`

## 其他模型选项

如果 `bge-large-zh-v1.5` 太大或太慢，可以考虑：

1. **BAAI/bge-base-zh-v1.5** (更快，质量良好)
   - 维度: 1024
   - 速度: 比 large 快约 2-3 倍

2. **BAAI/bge-small-zh-v1.5** (最快，质量稍低)
   - 维度: 512
   - 速度: 比 large 快约 5-6 倍

3. **BAAI/bge-m3** (多语言支持)
   - 维度: 1024
   - 支持: 中文 + 英文 + 其他语言

修改 `askany/config.py` 中的 `embedding_model` 配置即可切换。

