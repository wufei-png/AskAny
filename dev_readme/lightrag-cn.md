---
项目检索架构总览
整体架构：双层混合检索
用户查询
    ↓
┌─────────────────────────────────────────────────────────┐
│  QueryRouter (rag/router.py)                            │
│  - FAQ 路由 / DOCS 路由 / AUTO 智能路由                  │
└─────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────┐    ┌──────────────────────────────┐
│   LlamaIndex 检索    │    │     LightRAG 检索           │
│   (向量 + 关键词)     │    │   (知识图谱 + 向量)           │
│                      │    │                              │
│ • FAQQueryEngine    │    │ • lightrag_chunk (向量)      │
│ • RAGQueryEngine    │    │ • lightrag_entity (实体)     │
│                      │    │ • lightrag_relation (关系)   │
└──────────────────────┘    └──────────────────────────────┘
    ↓                              ↓
┌──────────────────────────────────────────────────────────┐
│  Rerank: BAAI/bge-reranker-v2-m3                        │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│  merge_lightrag_with_llamaindex()                        │
│  (LightRAG 结果与 LlamaIndex 结果合并)                    │
└──────────────────────────────────────────────────────────┘
    ↓
                    LLM 生成答案
---
1. LlamaIndex 检索层（基础层）
向量存储
- 技术: PostgreSQL + pgvector (HNSW 索引)
- 类: VectorStoreManager (askany/ingest/vector_store.py)
- 嵌入模型: BAAI/bge-m3 (多语言, 1024维)
混合检索实现
FAQ 检索 (faq_query_engine.py):
# 使用 QueryFusionRetriever 融合关键词 + 向量
retriever = QueryFusionRetriever(
    retrievers=[keyword_retriever, vector_retriever],
    similarity_top_k=faq_similarity_top_k,
    num_queries=4,  # 重写查询生成多个查询
    mode="reciprocal_rerank",
    weights=faq_ensemble_weights,  # [0.5, 0.5] 关键词:向量
)
文档检索 (rag_query_engine.py):
# 使用自定义的 KeywordVectorAppendRetriever
retriever = KeywordVectorAppendRetriever(
    keyword_retriever=keyword_retriever,
    vector_retriever=vector_retriever,
    similarity_top_k=docs_similarity_top_k,
    weights=docs_ensemble_weights,  # [0.5, 0.5]
)
重排序
- 模型: BAAI/bge-reranker-v2-m3
- 类: SafeReranker (rerank.py) - 安全包装器，防止节点丢失
---
2. LightRAG 检索层（图谱增强）
核心概念
概念	说明	存储
Entity (实体)	从文档中抽取的关键实体	lightrag_vdb_entity (向量)
Relation (关系)	实体之间的关系	lightrag_vdb_relation (向量)
Chunk (块)	文本切块	lightrag_vdb_chunks (向量)
Community (社区)	实体聚类后的摘要	全局检索用
检索模式
param = QueryParam(
    mode="mix",              # local/global/hybrid/naive/mix/bypass
    only_need_context=True, # 只返回上下文，不返回 LLM 答案
    top_k=60,               # KG 实体/关系数量
    chunk_top_k=10,         # 向量块数量
)
result = await self._rag.aquery_data(query, param=param)
各模式区别：
- local: 基于实体邻居检索（1度关系）
- global: 基于社区摘要检索
- hybrid: local + global
- naive: 纯向量检索（类似传统 RAG）
- mix: 全部组合 → 默认，召回率最高
- bypass: 跳过 KG，仅用向量
入库流程
python -m askany.rag.lightrag_ingest --ingest-markdown --ingest-json
入库时：
1. 按 ##  标题切分 Markdown
2. LightRAG 内部再 token 切分 (800 tokens/chunk, 150 overlap)
3. LLM 抽取实体 + 关系
4. 存入 PostgreSQL + 构建 HNSW 索引
---
3. 结果合并
lightrag_merge.py 中的 merge_lightrag_with_llamaindex() 函数：
- 将 LightRAG 的 entity/relation/chunk 与 LlamaIndex 结果合并
- 通过 related_lightrag_entities / related_lightrag_relations 字段关联
- 去重后统一重排序
---
总结
特性	LlamaIndex (基础)	LightRAG (增强)
向量检索	✅ 关键词 + 向量融合	✅ chunk 向量检索
知识图谱	❌	✅ 实体 + 关系
检索目标	相似文档	实体 + 关联信息
启用方式	默认始终开启	enable_lightrag=True
当前配置: enable_lightrag=True, lightrag_query_mode="mix"
这意味着系统同时使用两种检索，通过 mix 模式最大化召回率，然后统一重排序。