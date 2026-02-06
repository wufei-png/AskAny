agent设计：
在一个rag系统（llamaindex和pgvector构建）中，已经有了对markdown的向量索引和keyword索引（关键字倒排）的混合检索, 拿到结果后，想要将这个结果作为上下文输入给agent
这个agent做这样的事情：已知markdown文档根路径以及文件搜索工具，根据用户问题和检索结果做是否有足够的信息判断，如果有,llm判断结束，总结返回。如果没有，从用户问题和rag检索结果 提取关键字 调用文件搜索tool搜索原始文档片段，如果llm认为不足够，按照llm想法继续搜索，超过搜索次数返回文档没有已知结果，否则总结返回。


考虑到一次rag的耗时很长，agent的工具目前设计为 文件搜索工具，不要在用rag工具搜索，重点依赖原始文档的grep（参考cursor的agent设计）

第一次rag的keyword缓存，给agent使用

问题1： 假如前几次没有搜索到，随着循环的逐渐增加，调用chatcomplete接口的msg列表应该始终用一个，不断丰富它，还是多轮对话的形式？



目前的流程为，对用户问题进行混合rag检索@askany/rag/router.py
利用llamaindex的workflow功能，对目前功能进行丰富。


现状是startevent直接rag检索生成RAGRetrievalEvent,现在需要增加以下流程,中间插入一个事件 DirectAnswerEvent 利用@firstStageRelevant.py 判断问题是否可以直接回答,如果可以直接让模型回答, stopevent
否则判断Need web search 以及 Need rag search 如果web为true rag为false,调用网络搜索工具, 然后回答 stopevent.
如果都为false warning日志警告(因为之前已经判断过不能直接回答),并直接让模型回答.
如果web为false rag为true,或者两者都为true 调用rag工具检索,可能调用web搜索, merge并rerank 并发送RAGRetrievalEvent事件(当前仅支持rag检索) 给到analyze_relevance

你还需要实现网络搜索工具，参考@llama_index/llama-index-integrations/tools/llama-index-tools-duckduckgo

不依赖外部知识库内容,直接回答,或者通过网络搜索直接回答
首先有一个相关性判断工具,"你是一个运维助手,判断问题是否可以不依赖外部知识库内容,直接回答,或者通过网络搜索直接回答 
SubProblemGenerator class 用llm对用户问题数量进行判断，分三种情况
- 如果只有一个问题，直接进行后续workflow;
- 如果有多个问题且问题不相关，并行处理每个问题，处理所有问题后，返回的格式为
问题1: 答案1
问题2: 答案2
...
- 如果有多个问题且相关，对问题处理的逻辑顺序进行排序，串行处理，区别是上一个问题处理完成后，将问题和答案附加到下一个问题上作为上下文输入给后续workflow


后续的workflow为：

- 首先调用@askany/rag/router.py 进行混合rag检索（这一步和现在一样），拿到检索出的内容片段和问题关键字列表，此时llm会判断两个条件，相关和完整。如果已有信息相关且完整，总结返回

否则：如果都不相关，llm要根据问题重新生成关键字和老的关键字列表合并，用本地文件搜索工具搜索包含这些关键字的文件路径和命中的那一行内容到llm上下文中
如果相关但不完整，用本地文件搜索工具搜索该片段位于哪里，并扩大该部分的内容输入llm

后续的循环一直在其中循环，直到超过askany.config.settings.agent_max_iterations次或者认为相关和完整

Workflow:
    Step 1: RAG 检索
    Step 2: LLM 判断 相关 / 完整
        - if 相关 & 完整 → Step 6: 总结返回
        - if 都不相关 → Step 3
        - if 相关但不完整 → Step 4
    Step 3: 生成关键词 → 本地搜索 → Step 2
    Step 4: 找到 chunk 所在位置 → 扩展上下文 → Step 2
    Step 6: 输出答案

目前SubProblemGenerator（需要参考@test/vllm_structured_chat2.py设计结构化输出二级问题[[str]]，第一级为并行执行的，第二级为前后相关的问题）

LocalFileSearchTool的设计参考@LocalFileSearchTool.md

模型判断相关性和完整性需要参考@test/vllm_structured_chat2.py（已经可以用） 结构化输出 如果relevant_file_paths为空，则认为不相关，如果is_complete为False，则认为不完整，如果missing_info_keywords不为空，则为模型生成的关键词，和之前的rag关键词以及之前循环提取的关键词merge。

最后输出也是用结构化输出，输入的node用最后一次模型认为的参考node列表，参考@test/vllm_structured_chat3.py的设计结构化输出 


