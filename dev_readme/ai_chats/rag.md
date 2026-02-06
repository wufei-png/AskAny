## FAQ

### Split/Parser

  - json: faq不切分 84字符/120-135 tokens 适合作为单个chunk
常见chunk大小：100-500 tokens，当前在范围内

JsonParser,SimpleNodeParser
FAQ 文档通过 _documents_to_nodes() 直接转换为节点，不进行切分。
JSONNodeParser 的适用场景
JSONNodeParser 主要用于：
需要将大型 JSON 文档拆分为多个节点
需要按 JSON 结构智能切分
处理复杂的嵌套 JSON 结构


 KeywordTableIndex VectorStoreIndex PGVectorStore EnsembleRetriever 

### markdown
MarkdownNodeParser -> SemanticSplitterNodeParser
- VectorStoreIndex + PGVectorStore