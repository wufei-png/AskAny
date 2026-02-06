QueryFusionRetriever 的内部"rerank" vs 外部 Reranker
1. QueryFusionRetriever 内部的"rerank"（实际是融合）
QueryFusionRetriever 内部不是语义 rerank，而是结果融合（Fusion）：
作用：合并多个检索器的结果并统一排序
方法：
RECIPROCAL_RANK：RRF（Reciprocal Rank Fusion），基于排名合并
RELATIVE_SCORE：相对分数融合（MinMax 归一化 + 加权）
DIST_BASED_SCORE：基于距离的分数融合
SIMPLE：简单去重和排序
特点：
不进行语义理解
基于分数和排名
速度快
用于合并 keyword 和 vector 的结果
2. 外部 Reranker（SentenceTransformerRerank）
作用：对已合并的结果进行语义重排序
方法：使用交叉编码器（cross-encoder）模型
特点：
进行 query-document 对的深度语义匹配
更准确但更慢
在融合后的结果上进一步优化
3. 两者的区别




（自带的subquestion不合适
并行是子问题并行吗？即使第二个问题依赖第一个问题的答案？
是的，并行执行所有子问题，且不处理依赖关系。
            if self._use_async:
                tasks = [
                    self._aquery_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

                qa_pairs_all = run_async_tasks(tasks)
                ）