

LlamaIndex 更多支持“检索融合（RRF / Hybrid Search）”
你的场景非常依赖混合检索：
文档语义检索
文档关键词检索（ILIKE）
文档全文检索（pg_trgm）
代码 grep
代码语义 embedding
代码符号搜索

LlamaIndex 有成熟的 Multi-Index Query Engine + Fusion Retriever，
可以直接做 RRF（Reciprocal Rank Fusion）、weighted search。

LangChain 的 fusion 支持弱得多。


6. Response Synthesizer（回答合成器）
用于把多个 Node 的内容合并成答案。
模式：
simple concat
summarize
refine
tree aggregation
比如：
第一段回答不全 → refine 策略让下一段继续补充。
这是 LlamaIndex 比 LangChain 更强的一点：
处理多文档、多段信息的能力比较优秀。



对于qwen2.5-7b-instruct模型而言，已知用户问题是中文， 那么前面的提示词应该用英文还是中文效果更好？
Request options: {'method': 'post', 'url': '/chat/completions', 'files': None, 'idempotency_key': 'stainless-python-retry-f6bcebc2-af91-4e87-8b2e-2c66623f9ff5', 'json_data': {'messages': [{'role': 'user', 'content': 'You are a helpful assistant that generates multiple search queries based on a single input query. Generate 3 search queries, one on each line, related to the following input query:\nQuery: cassandra组件的concurrent_reads有什么用？\nQueries:\n'}], 'model': '/net/ai02/data/xlzhong/wufei/models/qwen2.5-7b-instruct/', 'stream': False, 'temperature': 0.7}}
