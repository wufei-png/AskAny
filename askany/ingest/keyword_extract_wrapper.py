"""Wrapper for combining LLM and TF-IDF keyword extraction."""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings  # noqa: E402
from askany.ingest.keyword_extract_from_llm import KeywordExtractorFromLLM  # noqa: E402
from askany.ingest.keyword_extract_from_tfidf import (  # noqa: E402
    KeywordExtractorFromTFIDF,
)
from askany.ingest.custom_keyword import KNOWLEDGE_EMBEDDING_KEYWORDS  # noqa: E402
from cachetools import LRUCache, cachedmethod
from tool.langdetect import contains_chinese  # noqa: E402
import numpy as np  # noqa: E402

try:
    from llama_index.core.embeddings import BaseEmbedding
except ImportError:
    BaseEmbedding = None  # type: ignore


class KeywordExtractorWrapper:
    """Wrapper class that combines LLM and TF-IDF keyword extraction."""

    def __init__(
        self,
        llm_client: Optional = None,
        llm_max_keywords: int = 2,
        tfidf_extractor: Optional[KeywordExtractorFromTFIDF] = None,
        priority: str = "llm",
        topn: Optional[int] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        """初始化 KeywordExtractorWrapper

        Args:
            llm_client: OpenAI 客户端，如果为 None 则从配置创建
            llm_max_keywords: LLM 提取的最大关键词数量，默认为 3
            tfidf_extractor: TF-IDF 提取器实例，如果为 None 则创建新实例
            priority: 合并策略优先级，"llm" 表示 LLM 优先（LLM在前，TF-IDF去重），
                     "tfidf" 表示 TF-IDF 优先（TF-IDF在前，LLM去重），默认为 "llm"
            topn: 最终输出的最大关键词数量，如果为 None 则使用配置中的 max_keywords_for_docs
            embed_model: Embedding 模型实例，用于计算关键词与 custom 关键词的相似度
        """
        self.llm_extractor = KeywordExtractorFromLLM(
            client=llm_client, max_keywords=llm_max_keywords
        )
        self.tfidf_extractor = (
            tfidf_extractor
            if tfidf_extractor is not None
            else KeywordExtractorFromTFIDF()
        )
        self.priority = priority
        self.topn = topn if topn is not None else settings.max_keywords_for_docs
        self.cache = LRUCache(maxsize=1024)
        self.embed_model = embed_model
        # Cache for custom keywords embeddings
        self._custom_embeddings_cache: Optional[np.ndarray] = None

    def GetKeywordExtractorFromTFIDF(self) -> KeywordExtractorFromTFIDF:
        return self.tfidf_extractor

    def GetKeywordExtractorFromLLM(self) -> KeywordExtractorFromLLM:
        return self.llm_extractor

    def _get_custom_embeddings(self) -> np.ndarray:
        """获取 custom 关键词的 embedding 向量（带缓存）

        Returns:
            shape (len(KNOWLEDGE_EMBEDDING_KEYWORDS), dimension) 的 numpy 数组
        """
        if self._custom_embeddings_cache is not None:
            return self._custom_embeddings_cache

        if self.embed_model is None:
            # 如果没有 embedding 模型，返回空数组
            self._custom_embeddings_cache = np.array([])
            return self._custom_embeddings_cache

        # 批量计算所有 custom 关键词的 embedding
        embeddings = self.embed_model._get_text_embeddings(KNOWLEDGE_EMBEDDING_KEYWORDS)
        self._custom_embeddings_cache = np.array(embeddings)
        return self._custom_embeddings_cache

    def _filter_by_custom_similarity(
        self, keywords: List[str], query: str
    ) -> Tuple[List[str], List[str]]:
        """根据与 custom 关键词的相似度过滤关键词

        如果关键词与 KNOWLEDGE_EMBEDDING_KEYWORDS 中任意一个的相似度超过阈值，
        则直接保留，不经过 LLM 判断。

        Args:
            keywords: 待过滤的关键词列表
            query: 用户查询（用于上下文，当前未使用）

        Returns:
            (matched_keywords, remaining_keywords) 元组
            - matched_keywords: 与 custom 关键词相似度超过阈值的关键词
            - remaining_keywords: 剩余需要经过 LLM 判断的关键词
        """
        if not keywords or self.embed_model is None:
            return [], keywords

        # 获取相似度阈值
        threshold = settings.custom_keyword_similarity_threshold

        # 获取 custom 关键词的 embedding
        custom_embeddings = self._get_custom_embeddings()
        if custom_embeddings.size == 0:
            return [], keywords

        # 批量计算所有关键词的 embedding
        keyword_embeddings = np.array(self.embed_model._get_text_embeddings(keywords))

        # 计算相似度矩阵：keywords x custom_keywords
        # 由于 embedding 已经 normalize，余弦相似度 = 点积
        similarity_matrix = np.dot(keyword_embeddings, custom_embeddings.T)

        # 对每个关键词，找到与所有 custom 关键词的最大相似度
        max_similarities = np.max(similarity_matrix, axis=1)

        # 分离匹配和未匹配的关键词
        matched_keywords = []
        remaining_keywords = []

        for i, keyword in enumerate(keywords):
            max_sim = max_similarities[i]
            if max_sim >= threshold:
                matched_keywords.append(keyword)
            else:
                remaining_keywords.append(keyword)

        return matched_keywords, remaining_keywords

    def _filter_duplicate_and_substring_keywords(
        self, keywords: List[str]
    ) -> List[str]:
        """过滤关键词列表：先去除小写重复，再去除子串

        1. 首先过滤掉小写后一样的字符，只保留小写的
        2. 然后过滤掉包含在长字符串中的子字符串

        例如：['IPS', 'start failed', 'failed', 'ips', 'start']
        - 第一步：小写去重 -> ['ips', 'start failed', 'failed', 'start']
        - 第二步：过滤子串 -> ['start failed', 'ips'] (过滤掉 'failed' 和 'start')

        Args:
            keywords: 待过滤的关键词列表

        Returns:
            过滤后的关键词列表
        """
        if not keywords:
            return []

        # 第一步：小写去重，保留小写版本
        seen_lower = {}
        lowercase_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen_lower:
                seen_lower[kw_lower] = kw
                lowercase_keywords.append(kw_lower)

        # 第二步：过滤子串
        # 按长度从长到短排序，这样长的字符串会先被处理
        sorted_keywords = sorted(lowercase_keywords, key=len, reverse=True)
        filtered = []

        for kw in sorted_keywords:
            # 检查当前关键词是否是已保留关键词的子串
            is_substring = False
            for kept in filtered:
                if kw in kept and kw != kept:
                    is_substring = True
                    break
            if not is_substring:
                filtered.append(kw)

        return filtered

    def _expand_keywords_with_separators(self, keywords: List[str]) -> List[str]:
        """处理关键词，对非中文的关键词生成分隔符变体

        - 如果包含空格，生成用-和_拼接的新词
        - 如果包含-且没有_，生成用_替换-的词
        - 如果包含_且没有-，生成用-替换_的词

        Args:
            keywords: 关键词列表

        Returns:
            扩展后的关键词列表
        """
        expanded = []

        for keyword in keywords:
            expanded.append(keyword)

            # 只处理非中文的关键词
            if contains_chinese(keyword):
                continue
            print("keyword", keyword)
            # 如果包含空格，生成用-和_拼接的新词
            if " " in keyword:
                parts = keyword.split()
                if len(parts) > 1:
                    # 生成用下划线拼接的词
                    underscore_keyword = "_".join(parts)
                    expanded.append(underscore_keyword)
                    # 生成用连字符拼接的词
                    hyphen_keyword = "-".join(parts)
                    expanded.append(hyphen_keyword)

                    nospace_keyword = "".join(parts)
                    expanded.append(nospace_keyword)

            # 如果包含-且没有_，生成用_替换-的词
            if "-" in keyword and "_" not in keyword:
                underscore_keyword = keyword.replace("-", "_")
                expanded.append(underscore_keyword)

                nospace_keyword = keyword.replace("-", "")
                expanded.append(nospace_keyword)

            # 如果包含_且没有-，生成用-替换_的词
            if "_" in keyword and "-" not in keyword:
                hyphen_keyword = keyword.replace("_", "-")
                expanded.append(hyphen_keyword)

                nospace_keyword = keyword.replace("_", "")
                expanded.append(nospace_keyword)
        return list(set(expanded))

    def _expand_keywords_with_numbers(self, keywords: List[str]) -> List[str]:
        """处理关键词，对包含数字或中文数字的关键词生成数字转换变体

        - 如果包含阿拉伯数字（0-9），生成将数字转换为中文数字的变体
        - 如果包含中文数字（零-九），生成将中文数字转换为阿拉伯数字的变体
        - 保留原词

        Args:
            keywords: 关键词列表

        Returns:
            扩展后的关键词列表
        """
        # 数字到中文数字的映射
        digit_to_chinese = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }

        # 中文数字到数字的映射
        chinese_to_digit = {v: k for k, v in digit_to_chinese.items()}

        expanded = []

        for keyword in keywords:
            expanded.append(keyword)  # 保留原词

            # 检测是否包含阿拉伯数字
            has_digit = bool(re.search(r"\d", keyword))

            # 检测是否包含中文数字
            chinese_digits = "零一二三四五六七八九"
            has_chinese_digit = any(c in keyword for c in chinese_digits)

            # 如果包含阿拉伯数字，转换为中文数字
            if has_digit:
                chinese_keyword = keyword
                for digit, chinese in digit_to_chinese.items():
                    chinese_keyword = chinese_keyword.replace(digit, chinese)
                if chinese_keyword != keyword:
                    expanded.append(chinese_keyword)

            # 如果包含中文数字，转换为阿拉伯数字
            if has_chinese_digit:
                digit_keyword = keyword
                for chinese, digit in chinese_to_digit.items():
                    digit_keyword = digit_keyword.replace(chinese, digit)
                if digit_keyword != keyword:
                    expanded.append(digit_keyword)

        return list(set(expanded))

    @cachedmethod(lambda self: self.cache)
    def extract_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """从查询中提取关键词，合并 LLM 和 TF-IDF 的结果

        Args:
            query: 用户查询字符串

        Returns:
            合并后的关键词列表，根据 priority 参数决定顺序
        """
        # Extract keywords using LLM
        llm_keywords = self.llm_extractor.extract_keywords(query)
        print(f"LLM keywords: {llm_keywords}")
        # Extract keywords using TF-IDF
        tfidf_keywords_set = self.tfidf_extractor.extract_keywords_set(query)
        print(f"TF-IDF keywords: {tfidf_keywords_set}")

        if self.priority == "llm":
            # LLM 优先策略：LLM 关键词在前，TF-IDF 去重
            llm_keywords_set = set(llm_keywords)
            # Remove common keywords from TF-IDF set
            tfidf_keywords_filtered = list(tfidf_keywords_set - llm_keywords_set)
            # Combine: LLM keywords first, then TF-IDF keywords (without common ones)
            merged_keywords = llm_keywords + tfidf_keywords_filtered
        else:
            # TF-IDF 优先策略：TF-IDF 关键词在前，LLM 去重
            tfidf_keywords_list = list(tfidf_keywords_set)
            # Remove common keywords from LLM list
            llm_keywords_filtered = [
                kw for kw in llm_keywords if kw not in tfidf_keywords_set
            ]
            # Combine: TF-IDF keywords first, then LLM keywords (without common ones)
            merged_keywords = tfidf_keywords_list + llm_keywords_filtered
        print("test1")

        # 先根据与 custom 关键词的相似度过滤
        # 相似度超过阈值的关键词直接保留，不经过 LLM 判断
        custom_matched, remaining_keywords = self._filter_by_custom_similarity(
            merged_keywords, query
        )
        print(
            f"custom matched keywords (similarity >= {settings.custom_keyword_similarity_threshold}): {custom_matched}"
        )
        print(f"Remaining keywords for LLM filtering: {remaining_keywords}")
        # 过滤重复和子串关键词
        remaining_keywords = self._filter_duplicate_and_substring_keywords(
            remaining_keywords
        )
        print(
            f"Filtered remaining keywords (after duplicate and substring removal): {remaining_keywords}"
        )
        # 对剩余关键词进行 LLM 过滤
        if remaining_keywords:
            filtered_keywords = self.llm_extractor.filter_keywords(
                remaining_keywords, query
            )
        else:
            filtered_keywords = []
        print("test2")

        # 合并结果：custom 匹配的关键词 + LLM 过滤后的关键词
        filtered_keywords = custom_matched + filtered_keywords

        filtered_keywords = filtered_keywords[: self.topn]
        # Limit output to topn keywords using max_keywords_for_docs

        # Calculate the difference: keywords that were filtered out
        filtered_keywords_set = set(filtered_keywords)

        keywords_other = [
            kw for kw in merged_keywords if kw not in filtered_keywords_set
        ]
        print("test3")
        # Split space and use - and _ to generate more keywords
        filtered_keywords = self._expand_keywords_with_separators(filtered_keywords)
        keywords_other = self._expand_keywords_with_separators(keywords_other)

        # Expand keywords with number conversions (数字 <-> 中文数字)
        filtered_keywords = self._expand_keywords_with_numbers(filtered_keywords)
        keywords_other = self._expand_keywords_with_numbers(keywords_other)

        return filtered_keywords, keywords_other


if __name__ == "__main__":
    # Test KeywordExtractorWrapper
    print("Testing KeywordExtractorWrapper...")
    print("-" * 80)

    # Initialize embedding model for similarity filtering
    from askany.main import initialize_llm

    _, embed_model = initialize_llm()
    wrapper = KeywordExtractorWrapper(embed_model=embed_model)

    # Test cases for similarity filtering
    # print("\n=== 测试相似度过滤功能 ===\n")

    # # Test case 1: 应该匹配的关键词（与 custom 关键词相似）
    # test_keywords_similar = [
    #     "custom",  # 应该匹配 "custom"
    #     "cassandra",  # 可能匹配（如果相似度足够高）
    # ]

    # # Test case 2: 不应该匹配的关键词（与 custom 关键词不相似）
    # test_keywords_dissimilar = [
    #     "激活率",  # 不相关
    #     "投资",  # 不相关
    #     "美股",  # 不相关
    #     "机器人",  # 可能不相关
    #     "门",  # 完全不相关
    # ]

    # print("测试相似的关键词（应该被直接保留，不经过 LLM）：")
    # matched, remaining = wrapper._filter_by_custom_similarity(
    #     test_keywords_similar, "test query"
    # )
    # print(f"  输入关键词: {test_keywords_similar}")
    # print(f"  匹配的关键词（相似度 >= {settings.custom_keyword_similarity_threshold}）: {matched}")
    # print(f"  剩余关键词（需要 LLM 判断）: {remaining}")
    # print()

    # print("测试不相似的关键词（应该全部经过 LLM 判断）：")
    # matched, remaining = wrapper._filter_by_custom_similarity(
    #     test_keywords_dissimilar, "test query"
    # )
    # print(f"  输入关键词: {test_keywords_dissimilar}")
    # print(f"  匹配的关键词（相似度 >= {settings.custom_keyword_similarity_threshold}）: {matched}")
    # print(f"  剩余关键词（需要 LLM 判断）: {remaining}")
    # print()

    # print("=" * 80)
    # print("\n=== 测试完整的关键词提取流程 ===\n")

    # queries = [
    # ]

    # for query in queries:
    #     keywords, keywords_other = wrapper.extract_keywords(query)
    #     print(f"Query: {query}")
    #     print(f"Filtered Keywords: {keywords}")
    #     print(f"Other Keywords: {keywords_other}")
    #     print("-" * 80)

    # raise Exception("Stop here")
    queries = [
        #         """
        # 初始选择：你面前有三扇关闭的门 (门 1, 门 2, 门 3)，一扇后面是汽车，另外两扇后面是山羊。你随机选择一扇门（例如门 1），但先不打开。
        # 主持人操作：主持人知道汽车在哪。他会打开你没选的另外两扇门中（门 2 或门 3）的一扇，且这扇门后一定是山羊。
        # 关键抉择：主持人问你：“要不要换到剩下那扇未打开的门？”"""
        # "如何重启解析的多模态applet 服务?"
    ]
    for query in queries:
        keywords, _ = wrapper.extract_keywords(query)
        print(f"Query: {query}")
        print(f"Merged Keywords: {keywords}")
        print("-" * 80)
    # # Test query
    # query = "query？"
    # keywords = wrapper.extract_keywords(query)

    # print(f"Query: {query}")
    # print(f"Merged Keywords: {keywords}")
    # print("-" * 80)
