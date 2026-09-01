# HNSW 索引

AskAny 使用 PostgreSQL + pgvector 存储 embedding。HNSW 默认启用，索引参数
来自 `askany/config.py`：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `enable_hnsw` | `True` | 是否使用 HNSW 索引 |
| `hnsw_m` | `16` | 每个节点的连接数；越大通常越占内存、构建越慢 |
| `hnsw_ef_construction` | `128` | 构建索引时的候选数量 |
| `hnsw_ef_search` | `40` | 查询时的候选数量 |
| `hnsw_dist_method` | `vector_cosine_ops` | 默认使用 cosine 距离操作符 |

当前使用两个主要向量表：

- 配置名 `askany_faq_vectors`，FAQ 物理表通常带 LlamaIndex 的 `data_` 前缀；
- 配置名 `askany3_docs_vectors`，文档物理表通常带同样的 `data_` 前缀。

实际表名、索引是否已经创建以及行数都取决于当前数据库，不能从本文件
推断。使用代码提供的命令创建索引：

```bash
uv run --locked python -m askany.main --create-index
```

检查当前数据库结构可使用：

```bash
uv run --locked python tool/query_hnsw_structure.py \
  --table data_askany3_docs_vectors
```

如果表名或参数已通过 `.env` 覆盖，请以实际配置和数据库查询结果为准。批量
写入时，`VectorStoreManager.add_docs_nodes()` 默认不会自动创建索引，除非调用
方明确启用 `auto_create_index=True`；主 CLI 的 `--create-index` 用于显式创建。
