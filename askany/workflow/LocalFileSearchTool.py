"""Local file search tool for finding and extracting content from files."""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core.node_parser import MarkdownNodeParser

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from askany.config import settings

from askany.ingest.keyword_extract_from_tfidf import KeywordExtractorFromTFIDF
from askany.workflow.token_control import estimate_tokens


class LocalFileSearchTool:
    """Tool for searching and extracting content from local files."""

    def __init__(
        self,
        base_path: Optional[str] = None,
        keyword_extractor: Optional[KeywordExtractorFromTFIDF] = None,
    ):
        """Initialize LocalFileSearchTool.

        Args:
            base_path: Base path for file search (default: current working directory)
        """
        if base_path:
            self.base_path = Path(base_path).resolve()
        else:
            self.base_path = Path.cwd()
        print(f"Base path: {self.base_path}")
        self.markdown_parser = MarkdownNodeParser.from_defaults()
        # 查找所有markdown文件
        self.markdown_files = self._find_markdown_files(self.base_path)
        print(f"Found {len(self.markdown_files)} markdown files")
        self.expand_context_ratio = settings.expand_context_ratio
        self.keyword_expand_ratio = settings.keyword_expand_ratio

        if keyword_extractor is not None and isinstance(
            keyword_extractor, KeywordExtractorFromTFIDF
        ):
            self.keyword_extractor = keyword_extractor
        else:
            self.keyword_extractor = KeywordExtractorFromTFIDF()

    def find_text_line_range(
        self, text: str, file_path: str
    ) -> Optional[Tuple[int, int]]:
        """
        函数1: 输入一段文本和本地文件路径，提取该文本在该文件的第几行到第几行。

        实现方式：提取文本第一行，查看第一次匹配的行号，以及最后一行内容，
        查看匹配中最后一次匹配的行号，这样即使有多个匹配，确保包含最大的信息量。

        Args:
            text: 要查找的文本
            file_path: 文件路径

        Returns:
            (start_line, end_line) 行号范围，如果未找到返回None
        """
        full_path = self._resolve_path(file_path)
        if not full_path.exists() or not full_path.is_file():
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 提取文本的第一行和最后一行
            text_lines = text.strip().split("\n")
            if not text_lines:
                return None

            first_line_text = text_lines[0].strip()
            last_line_text = (
                text_lines[-1].strip() if len(text_lines) > 1 else first_line_text
            )

            # 查找第一次匹配的行号
            start_line = None
            for i, line in enumerate(lines, start=1):
                if first_line_text in line:
                    start_line = i
                    break

            if start_line is None:
                return None

            # 查找最后一次匹配的行号（从后往前搜索）
            end_line = start_line
            for i in range(len(lines), start_line - 1, -1):
                if last_line_text in lines[i - 1]:
                    end_line = i
                    break

            return (start_line, end_line)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def _expand_context_by_line_num(
        self,
        file_path: str,
        line_num: int,
        expand_mode: str = settings.expand_context_mode,
        expand_ratio: float = settings.expand_context_ratio,
        expand_lines: Optional[int] = None,
    ) -> Optional[Dict[str, any]]:
        """
        直接使用已知的行号来扩展上下文，而不需要重新查找文本。

        Args:
            file_path: 文件路径
            line_num: 已知的行号
            expand_mode: 扩展模式，"ratio"（按比例扩展）或 "markdown"（提取markdown块）
            expand_ratio: 扩展比例（仅在expand_mode="ratio"时使用）
            expand_lines: 固定扩展行数（仅在expand_mode="ratio"时使用，优先级高于expand_ratio）

        Returns:
            {file_path: str, start_line: int, end_line: int, content: str} 或 None
        """
        full_path = self._resolve_path(file_path)
        if not full_path.exists() or not full_path.is_file():
            return None

        start_line = line_num
        end_line = line_num

        if expand_mode == "markdown":
            # 提取markdown块
            return self._expand_markdown_block(file_path, start_line, end_line)
        else:
            # 按比例或固定行数扩展
            if expand_lines is not None:
                expand_size = expand_lines
            else:
                expand_size = int((end_line - start_line + 1) * expand_ratio)

            # 扩展行号范围
            new_start_line = max(1, start_line - expand_size)
            new_end_line = end_line + expand_size

            # 检查并修正 end_line 是否超过文件行数
            file_line_count = self.get_file_line_count(file_path)
            if file_line_count is not None:
                new_end_line = min(new_end_line, file_line_count)

            # 获取扩展后的内容
            content = self.get_file_content_by_lines(
                file_path, new_start_line, new_end_line
            )
            if content is None:
                return None

            return {
                "file_path": file_path,
                "start_line": new_start_line,
                "end_line": new_end_line,
                "content": content,
            }

    def get_file_line_count(self, file_path: str) -> Optional[int]:
        """获取文件的总行数。

        Args:
            file_path: 文件路径

        Returns:
            文件行数，如果文件不存在或读取失败返回None
        """
        full_path = self._resolve_path(file_path)
        if not full_path.exists() or not full_path.is_file():
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return len(lines)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def get_file_content_by_lines(
        self, file_path: str, start_line: int, end_line: int
    ) -> Optional[str]:
        """
        函数2: 根据文件路径和第一行和最后一行行号，返回对应内容。

        Args:
            file_path: 文件路径
            start_line: 起始行号（从1开始）
            end_line: 结束行号（从1开始），如果为-1则返回从start_line到文件末尾的所有内容

        Returns:
            文件内容，如果文件不存在或读取失败返回None
        """
        full_path = self._resolve_path(file_path)
        if not full_path.exists() or not full_path.is_file():
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 确保行号在有效范围内
            start_line = max(1, min(start_line, len(lines)))

            # 如果end_line为-1，返回从start_line到文件末尾的所有内容
            if end_line == -1:
                end_line = len(lines)
            else:
                end_line = max(start_line, min(end_line, len(lines)))

            # 提取指定范围的内容
            content_lines = lines[start_line - 1 : end_line]
            return "".join(content_lines)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def _filter_results_by_tokens(
        self, results: List[Dict[str, any]], max_tokens: int
    ) -> List[Dict[str, any]]:
        """根据 token 限制过滤结果列表。

        Args:
            results: 结果列表，每个结果包含 content 字段
            max_tokens: 最大 token 数

        Returns:
            过滤后的结果列表
        """
        if not results:
            return results

        filtered_results = []
        total_tokens = 0

        for result in results:
            content = result.get("content", "")
            if not content:
                continue

            content_tokens = estimate_tokens(content)

            # 如果添加这个结果会超过限制，停止添加
            if total_tokens + content_tokens > max_tokens:
                break

            filtered_results.append(result)
            total_tokens += content_tokens

        if len(filtered_results) < len(results):
            print(
                f"Filtered results due to token limit: original={len(results)} results, "
                f"kept={len(filtered_results)} results, total_tokens={total_tokens}/{max_tokens}"
            )

        return filtered_results

    def _filter_all_results_by_tokens(
        self, all_results: Dict[str, List[Dict[str, any]]], max_tokens: int
    ) -> Dict[str, List[Dict[str, any]]]:
        """根据 token 限制过滤所有关键字的结果（整体估算）。

        Args:
            all_results: 所有关键字的结果字典 {keyword: [results]}
            max_tokens: 最大 token 数（整体限制）

        Returns:
            过滤后的结果字典
        """
        if not all_results:
            return all_results

        # 将所有结果展平为一个列表，同时记录每个结果属于哪个关键字
        flat_results = []
        for keyword, results in all_results.items():
            for result in results:
                flat_results.append((keyword, result))

        # 按顺序过滤，累计 token 数
        filtered_flat_results = []
        total_tokens = 0

        for keyword, result in flat_results:
            content = result.get("content", "")
            if not content:
                continue

            content_tokens = estimate_tokens(content)

            # 如果添加这个结果会超过限制，停止添加
            if total_tokens + content_tokens > max_tokens:
                break

            filtered_flat_results.append((keyword, result))
            total_tokens += content_tokens

        # 重新组织为字典格式
        filtered_results = {keyword: [] for keyword in all_results.keys()}
        for keyword, result in filtered_flat_results:
            filtered_results[keyword].append(result)

        # 移除空的关键字
        filtered_results = {
            keyword: results for keyword, results in filtered_results.items() if results
        }

        total_original = sum(len(results) for results in all_results.values())
        total_filtered = sum(len(results) for results in filtered_results.values())
        if total_filtered < total_original:
            print(
                f"Filtered all results due to token limit: original={total_original} results, "
                f"kept={total_filtered} results, total_tokens={total_tokens}/{max_tokens}"
            )

        return filtered_results

    def _filter_keywords_by_limits(
        self, all_results: Dict[str, List[Dict[str, any]]]
    ) -> Dict[str, List[Dict[str, any]]]:
        """根据文件数量和匹配数量限制过滤关键词结果。

        如果某个关键词命中了超过 one_keyword_max_file_num 的文件，
        或者命中的数量超过了 one_keyword_max_matches_num，
        则过滤掉这个关键词的所有命中。

        Args:
            all_results: 所有关键字的结果字典 {keyword: [results]}

        Returns:
            过滤后的结果字典
        """
        if not all_results:
            return all_results

        max_file_num = settings.one_keyword_max_file_num
        max_matches_num = settings.one_keyword_max_matches_num

        filtered_results = {}
        filtered_keywords = []

        for keyword, results in all_results.items():
            if not results:
                continue

            # 统计命中的文件数量（通过 file_path 去重）
            unique_files = set()
            for result in results:
                file_path = result.get("file_path")
                if file_path:
                    unique_files.add(file_path)

            file_count = len(unique_files)
            matches_count = len(results)

            # 如果超过限制，过滤掉这个关键词的所有命中
            if file_count > max_file_num or matches_count > max_matches_num:
                filtered_keywords.append(keyword)
            else:
                filtered_results[keyword] = results

        if filtered_keywords:
            print(
                f"Filtered {len(filtered_keywords)} keywords due to limits: {filtered_keywords}"
            )

        return filtered_results

    def expand_context(
        self,
        text: str,
        file_path: str,
        expand_mode: str = settings.expand_context_mode,
        expand_ratio: float = settings.expand_context_ratio,
        expand_lines: Optional[int] = None,
    ) -> Optional[Dict[str, any]]:
        """
        函数3: 找到 chunk 所在位置 → 扩展上下文。

        利用函数1和函数2，如果设置的是上下扩展行号的比例，用这段文本的start和end，
        end-start*比例，得到上下扩展的行号，然后调用函数2返回扩展后的内容。
        如果设置的是提取文本所在的markdown块，需要具备markdown解析能力，
        拿到该段文本所在的markdown完整信息块。

        Args:
            text: 要查找的文本
            file_path: 文件路径
            expand_mode: 扩展模式，"ratio"（按比例扩展）或 "markdown"（提取markdown块）
            expand_ratio: 扩展比例（仅在expand_mode="ratio"时使用）
            expand_lines: 固定扩展行数（仅在expand_mode="ratio"时使用，优先级高于expand_ratio）

        Returns:
            {file_path: str, start_line: int, end_line: int, content: str} 或 None
        """
        full_path = self._resolve_path(file_path)
        if not full_path.exists() or not full_path.is_file():
            return None

        # 首先找到文本所在的行号范围
        line_range = self.find_text_line_range(text, file_path)
        if line_range is None:
            return None

        start_line, end_line = line_range

        if expand_mode == "markdown":
            # 提取markdown块
            return self._expand_markdown_block(file_path, start_line, end_line)
        else:
            # 按比例或固定行数扩展
            if expand_lines is not None:
                expand_size = expand_lines
            else:
                expand_size = int((end_line - start_line + 1) * expand_ratio)

            # 扩展行号范围
            new_start_line = max(1, start_line - expand_size)
            new_end_line = end_line + expand_size

            # 检查并修正 end_line 是否超过文件行数
            file_line_count = self.get_file_line_count(file_path)
            if file_line_count is not None:
                new_end_line = min(new_end_line, file_line_count)

            # 获取扩展后的内容
            content = self.get_file_content_by_lines(
                file_path, new_start_line, new_end_line
            )
            if content is None:
                return None

            return {
                "file_path": file_path,
                "start_line": new_start_line,
                "end_line": new_end_line,
                "content": content,
            }

    ##TODO cache the search results
    def search_by_keywords(
        self,
        keywords: List[str],
        expand_mode: str = settings.expand_context_mode,
        expand_ratio: float = settings.expand_context_ratio,
        expand_lines: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        函数4: 根据关键字本地搜索。

        输入[string] 查找每个关键字在哪些文件以及对应的行号。
        返回格式为：
        {keyword :[
          {file_path: str, start_line: int, end_line: int, content: str}
          {file_path: str, start_line: int, end_line: int, content: str}
        ]}
        一个关键字可以匹配多个文件，同样这里返回的行数也有两种参数，
        一种是固定上下扩展n行，一种是找到该段文本所在的markdown完整信息块。

        Args:
            keywords: 关键字列表
            expand_mode: 扩展模式，"ratio"（按比例扩展）或 "markdown"（提取markdown块）
            expand_ratio: 扩展比例（仅在expand_mode="ratio"时使用）
            expand_lines: 固定扩展行数（仅在expand_mode="ratio"时使用，优先级高于expand_ratio）

        Returns:
            {keyword: [{file_path, start_line, end_line, content}, ...]}
        """
        if not keywords:
            return {}

        # 初始化结果字典
        results = {keyword: [] for keyword in keywords}

        # 构建包含所有关键字的正则表达式（用于快速过滤）
        escaped_keywords = [re.escape(keyword) for keyword in keywords]
        combined_pattern = re.compile("|".join(escaped_keywords), re.IGNORECASE)

        # 为每个关键字创建单独的正则表达式，用于确定具体匹配的关键字
        # 这样可以在匹配的行中准确找到所有匹配的关键字
        keyword_patterns = {
            keyword: re.compile(re.escape(keyword), re.IGNORECASE)
            for keyword in keywords
        }

        # 只遍历所有文件一次
        # 用于统计每个keyword在所有文件中的总出现次数
        keyword_total_frequency: Dict[str, int] = {keyword: 0 for keyword in keywords}

        for file_path in self.markdown_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # 查找包含任何关键字的行及其匹配的关键字
                # 使用字典存储：行号 -> 匹配的关键字列表
                matching_lines_keywords: Dict[int, List[str]] = {}
                file_path_str = str(file_path)

                for i, line in enumerate(lines, start=1):
                    # 快速检查是否包含任何关键字
                    if combined_pattern.search(line):
                        # 检查具体是哪些关键字匹配了这一行（可能多个）
                        matched_keywords = []
                        for keyword, pattern in keyword_patterns.items():
                            if pattern.search(line):
                                matched_keywords.append(keyword)
                                # 统计词频：每个keyword在所有文件中的总出现次数
                                keyword_total_frequency[keyword] += 1

                        if matched_keywords:
                            # print(f"matched_keywords: {matched_keywords}")
                            matching_lines_keywords[i] = matched_keywords

                if not matching_lines_keywords:
                    continue

                # 对每个匹配的行，扩展上下文
                for line_num, matched_keywords in matching_lines_keywords.items():
                    # 获取单行内容作为文本
                    line_text = lines[line_num - 1].strip()
                    if not line_text:
                        continue

                    # 扩展上下文（只需要扩展一次，然后添加到所有匹配的关键字结果中）
                    expanded = self.expand_context(
                        line_text,
                        file_path_str,
                        expand_mode=expand_mode,
                        expand_ratio=expand_ratio,
                        expand_lines=expand_lines,
                    )

                    if not expanded:
                        print(
                            f"DEBUG: expand_context returned None for line_text='{line_text[:50]}...' at line_num={line_num} in file={file_path_str}"
                        )
                        # 尝试直接使用已知的行号来扩展上下文
                        expanded = self._expand_context_by_line_num(
                            file_path_str,
                            line_num,
                            expand_mode=expand_mode,
                            expand_ratio=expand_ratio,
                            expand_lines=expand_lines,
                        )
                        if expanded:
                            print(
                                f"DEBUG: _expand_context_by_line_num succeeded for line_num={line_num}"
                            )

                    if expanded:
                        # 为每个匹配的keyword添加词频信息（在所有文件中的总词频）
                        for keyword in matched_keywords:
                            # 添加该keyword在所有文件中的总词频
                            expanded["keyword_frequency"] = keyword_total_frequency[
                                keyword
                            ]
                            # 将结果添加到所有匹配的关键字中
                            results[keyword].append(expanded)

            except Exception as e:
                print(f"Error searching in file {file_path}: {e}")
                continue

        # 对每个关键字的结果，合并同一文件中重叠的结果
        for keyword in keywords:
            results[keyword] = self._merge_overlapping_results(results[keyword])

        # 根据文件数量和匹配数量限制过滤关键词结果
        results = self._filter_keywords_by_limits(results)

        # 更新 keywords 列表，只保留过滤后仍然存在的结果中的关键词
        keywords = list(results.keys())

        # 对整体结果进行 token 限制过滤（所有关键字的结果一起估算）
        # max_tokens = settings.llm_max_tokens // 2
        max_tokens = settings.llm_max_tokens
        results = self._filter_all_results_by_tokens(results, max_tokens)

        return results

    def _search_by_sliding_windows(
        self, tokens: List[str]
    ) -> Optional[Dict[str, List[Dict[str, any]]]]:
        """
        核心函数：使用滑动窗口搜索关键字。

        生成所有窗口大小>=1的滑动窗口，收集所有keywords，调用一次search_by_keywords，
        然后找到最长的有结果的窗口并返回其结果。

        Args:
            tokens: 分词后的token列表

        Returns:
            Dict[str, List[Dict[str, any]]]: 最长窗口的搜索结果，如果没找到返回None
        """
        # if len(tokens) < 2:
        #     return None

        # 生成所有窗口大小>=2的滑动窗口，并记录每个keyword字符串对应的最大窗口大小
        all_keywords = []  # 所有窗口的keywords列表（用于一次性搜索）
        keyword_to_max_window = {}  # keyword_str -> max_window_size

        # 窗口大小从2到len(tokens)
        for window_size in range(1, len(tokens) + 1):
            # 步长为1，尝试所有可能的滑动窗口位置
            for start_idx in range(len(tokens) - window_size + 1):
                window_tokens = tokens[start_idx : start_idx + window_size]
                # 将窗口tokens转换为字符串作为keyword
                # TODO use - and _ to join?
                keyword_str = " ".join(window_tokens)

                # 记录每个keyword对应的最大窗口大小
                if keyword_str not in keyword_to_max_window:
                    keyword_to_max_window[keyword_str] = window_size
                    all_keywords.append(keyword_str)
                else:
                    # 更新最大窗口大小
                    if window_size > keyword_to_max_window[keyword_str]:
                        keyword_to_max_window[keyword_str] = window_size

        if not all_keywords:
            return None

        # 一次性调用search_by_keywords，传入所有keywords（去重后的）
        search_results = self.search_by_keywords(all_keywords)

        # 检查是否有结果
        if not any(results for results in search_results.values()):
            return None

        # 找到所有具有最大window_size的keywords
        best_window_size = 0
        best_keyword_strs = []

        for keyword_str, results in search_results.items():
            if not results:
                continue

            # 获取这个keyword对应的窗口大小
            window_size = keyword_to_max_window.get(keyword_str, 0)
            if window_size > best_window_size:
                best_window_size = window_size
                best_keyword_strs = [keyword_str]
            elif window_size == best_window_size:
                best_keyword_strs.append(keyword_str)

        if not best_keyword_strs:
            return None

        # 如果只有一个keyword有最大window_size，直接返回
        if len(best_keyword_strs) == 1:
            return {best_keyword_strs[0]: search_results[best_keyword_strs[0]]}

        # 如果有多个keyword，需要排序
        # 收集所有最大window_size的keyword的结果，并统计每个file_path的命中情况
        # file_path -> {matched_keywords: set, total_frequency: int, results_by_keyword: {keyword: result}}
        file_path_info = {}  # file_path -> {matched_keywords: set, total_frequency: int, results_by_keyword: dict}

        # 收集所有最大window_size的keyword的结果
        for keyword_str in best_keyword_strs:
            for result in search_results[keyword_str]:
                file_path = result.get("file_path")
                if file_path not in file_path_info:
                    file_path_info[file_path] = {
                        "matched_keywords": set(),
                        "total_frequency": 0,
                        "results_by_keyword": {},
                    }
                # 添加这个keyword的结果
                file_path_info[file_path]["results_by_keyword"][keyword_str] = result
                # 添加这个keyword到匹配集合（只统计best_keyword_strs中的keywords）
                file_path_info[file_path]["matched_keywords"].add(keyword_str)

        # 计算每个file_path的所有命中keyword的词频总和
        # 每个keyword在所有文件中的总词频是固定的，可以从任意一个结果中获取
        for file_path, info in file_path_info.items():
            total_freq = 0
            for keyword_str in info["matched_keywords"]:
                # 从search_results中获取该keyword的词频（所有文件中的总词频）
                # 由于词频是固定的，可以从第一个结果中获取
                keyword_results = search_results.get(keyword_str, [])
                if keyword_results:
                    keyword_freq = keyword_results[0].get("keyword_frequency", 0)
                    total_freq += keyword_freq
            info["total_frequency"] = total_freq

        # 排序：先按命中keyword数量降序，再按词频总和升序（少的在前）
        sorted_file_paths = sorted(
            file_path_info.items(),
            key=lambda x: (-len(x[1]["matched_keywords"]), x[1]["total_frequency"]),
        )

        # 重新构建结果字典，保持原有的格式 {keyword: [results]}
        # 按照排序后的顺序组织结果
        final_results = {}
        max_num = settings.docs_similarity_top_k
        for file_path, info in sorted_file_paths:
            # 只返回最大window_size的keywords的结果
            for keyword_str in best_keyword_strs:
                if keyword_str in info["results_by_keyword"]:
                    if keyword_str not in final_results:
                        final_results[keyword_str] = []
                    # 添加这个结果（按排序后的顺序）
                    result = info["results_by_keyword"][keyword_str]
                    if result not in final_results[keyword_str]:
                        final_results[keyword_str].append(result)
                        max_num -= 1
                    if max_num <= 0:
                        return final_results
        return final_results

    def search_keyword_using_binary_algorithm(
        self, keywords: List[str]
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        函数5: 使用滑动窗口算法搜索关键字。

        使用传入的keywords列表，使用滑动窗口搜索（窗口大小>=2）。

        Args:
            keywords: 关键词列表

        Returns:
            Dict[str, List[Dict[str, any]]]: 关键词 -> 结果列表
        """
        if not keywords:
            return {}

        # 使用滑动窗口搜索
        window_results = self._search_by_sliding_windows(keywords)
        if window_results is not None:
            # 辅助函数：检查搜索结果是否为空
            def _has_results(search_results: Dict[str, List[Dict[str, any]]]) -> bool:
                """检查搜索结果是否非空"""
                return any(results for results in search_results.values())

            if _has_results(window_results):
                # 对整体结果进行 token 限制过滤（所有关键字的结果一起估算）
                # max_tokens = settings.llm_max_tokens // 2
                max_tokens = settings.llm_max_tokens
                window_results = self._filter_all_results_by_tokens(
                    window_results, max_tokens
                )
                return window_results

        # 如果没找到结果，返回空字典
        return {}

    def _merge_overlapping_results(
        self, results: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """合并同一文件中重叠的结果。

        Args:
            results: 结果列表，每个结果包含 {file_path, start_line, end_line, content}

        Returns:
            合并后的结果列表
        """
        if not results:
            return results

        # 按文件路径分组
        file_groups: Dict[str, List[Dict[str, any]]] = {}
        for result in results:
            file_path = result.get("file_path")
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(result)

        merged_results = []

        # 对每个文件的结果进行合并
        for file_path, file_results in file_groups.items():
            if len(file_results) == 1:
                # 只有一个结果，直接添加
                merged_results.append(file_results[0])
                continue

            # 按 start_line 排序
            file_results.sort(key=lambda x: x.get("start_line", 0))

            # 合并重叠的结果
            merged = []
            current = file_results[0].copy()

            for next_result in file_results[1:]:
                current_start = current.get("start_line", 0)
                current_end = current.get("end_line", 0)
                next_start = next_result.get("start_line", 0)
                next_end = next_result.get("end_line", 0)

                # 检查是否重叠：当前结果的结束行 >= 下一个结果的开始行
                if current_end >= next_start:
                    # 有重叠，合并：取最小 start_line 和最大 end_line
                    merged_start = min(current_start, next_start)
                    merged_end = max(current_end, next_end)

                    # 重新获取合并后的内容
                    merged_content = self.get_file_content_by_lines(
                        file_path, merged_start, merged_end
                    )
                    if merged_content:
                        current = {
                            "file_path": file_path,
                            "start_line": merged_start,
                            "end_line": merged_end,
                            "content": merged_content,
                        }
                    else:
                        # 如果获取内容失败，使用当前结果
                        current["start_line"] = merged_start
                        current["end_line"] = merged_end
                else:
                    # 没有重叠，保存当前结果，开始新的合并
                    merged.append(current)
                    current = next_result.copy()

            # 添加最后一个结果
            merged.append(current)
            merged_results.extend(merged)

        # 去除相同文件名和相同内容的结果（保留第一个）
        deduplicated_results = self._deduplicate_by_filename_and_content(merged_results)

        return deduplicated_results

    def _deduplicate_by_filename_and_content(
        self, results: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """去除相同文件名和相同内容的结果，只保留第一个。

        Args:
            results: 结果列表，每个结果包含 {file_path, start_line, end_line, content}

        Returns:
            去重后的结果列表
        """
        if not results:
            return results

        seen = {}  # key: (filename, content), value: result
        deduplicated = []

        for result in results:
            file_path = result.get("file_path", "")
            content = result.get("content", "")

            # 提取文件名（不包含路径）
            filename = Path(file_path).name

            # 使用文件名和内容的组合作为唯一标识
            key = (filename, content)

            # 如果这个组合还没有出现过，保留它
            if key not in seen:
                seen[key] = result
                deduplicated.append(result)

        return deduplicated

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to base_path.

        Args:
            file_path: File path (absolute, relative, or virtual path starting with /)

        Returns:
            Resolved Path object
        """
        path = Path(file_path)

        # If path is already absolute, check if it exists
        # If it exists, return it directly (regardless of base_path)
        # If it doesn't exist but is absolute, still return it (might be a valid path)
        if path.is_absolute():
            # For absolute paths, always return as-is
            # This handles cases where markdown_files contains absolute paths
            try:
                resolved_abs = path.resolve()
                return resolved_abs
            except (OSError, RuntimeError):
                # Even if resolve() fails, return the absolute path as-is
                return path

        # Handle virtual paths (starting with / but relative to base_path)
        # These are typically returned by glob_search/grep_search tools
        # Only treat as virtual if it's not an absolute path (already handled above)
        if file_path.startswith("/") and not path.is_absolute():
            # This is a virtual path (starts with / but not absolute after Path() conversion)
            # Remove leading / and treat as relative to base_path
            file_path = file_path.lstrip("/")
            path = Path(file_path)

        # If the path is relative, try to resolve it relative to base_path
        # But first check if it already contains base_path as a prefix
        try:
            # Try to make path absolute first
            abs_path = path.resolve()
            # Check if the absolute path is within base_path
            try:
                abs_path.relative_to(self.base_path.resolve())
                return abs_path
            except ValueError:
                # Path is not within base_path, but it's an absolute path
                # Return it as-is instead of treating as relative
                return abs_path
        except (OSError, RuntimeError):
            # Path doesn't exist or can't be resolved, resolve relative to base_path
            pass

        # Resolve relative to base_path
        resolved = self.base_path.resolve() / path
        # If the resolved path doesn't exist, try to find it relative to cwd
        if not resolved.exists():
            cwd_path = Path.cwd() / path
            if cwd_path.exists():
                return cwd_path.resolve()
        return resolved

    def _expand_markdown_block(
        self, file_path: str, start_line: int, end_line: int
    ) -> Optional[Dict[str, any]]:
        """扩展markdown块：找到包含指定行的完整markdown块。

        Args:
            file_path: 文件路径
            start_line: 起始行号
            end_line: 结束行号

        Returns:
            {file_path: str, start_line: int, end_line: int, content: str} 或 None
        """
        full_path = self._resolve_path(file_path)
        if not full_path.exists() or not full_path.is_file():
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            # 使用MarkdownNodeParser解析文件
            from llama_index.core import Document

            doc = Document(text=content, metadata={"file_path": str(file_path)})
            nodes = self.markdown_parser.get_nodes_from_documents([doc])

            # 找到包含指定行的节点
            target_node = None
            for node in nodes:
                # 获取节点在原始文件中的行号范围
                node_start, node_end = self._get_node_line_range(node, lines)
                if node_start is None or node_end is None:
                    continue

                # 检查目标行是否在这个节点内
                if (
                    node_start <= start_line <= node_end
                    or node_start <= end_line <= node_end
                ):
                    # 如果当前节点包含目标行，且还没有找到节点，或者当前节点更大
                    if target_node is None or (
                        node_start <= target_node["start"]
                        and node_end >= target_node["end"]
                    ):
                        target_node = {
                            "node": node,
                            "start": node_start,
                            "end": node_end,
                        }

            if target_node is None:
                # 如果找不到markdown块，使用原始行号范围
                # 检查并修正 end_line 是否超过文件行数
                file_line_count = self.get_file_line_count(file_path)
                if file_line_count is not None:
                    end_line = min(end_line, file_line_count)

                content_text = self.get_file_content_by_lines(
                    file_path, start_line, end_line
                )
                if content_text is None:
                    return None
                return {
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": content_text,
                }

            # 返回完整的markdown块
            # 检查并修正 end_line 是否超过文件行数
            target_end_line = target_node["end"]
            file_line_count = self.get_file_line_count(file_path)
            if file_line_count is not None:
                target_end_line = min(target_end_line, file_line_count)

            block_content = self.get_file_content_by_lines(
                file_path, target_node["start"], target_end_line
            )
            if block_content is None:
                return None

            return {
                "file_path": file_path,
                "start_line": target_node["start"],
                "end_line": target_end_line,
                "content": block_content,
            }
        except Exception as e:
            print(f"Error expanding markdown block in {file_path}: {e}")
            # 如果出错，回退到原始行号范围
            # 检查并修正 end_line 是否超过文件行数
            file_line_count = self.get_file_line_count(file_path)
            if file_line_count is not None:
                end_line = min(end_line, file_line_count)

            content_text = self.get_file_content_by_lines(
                file_path, start_line, end_line
            )
            if content_text is None:
                return None
            return {
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "content": content_text,
            }

    def _get_node_line_range(
        self, node, lines: List[str]
    ) -> Tuple[Optional[int], Optional[int]]:
        """获取节点在原始文件中的行号范围。

        Args:
            node: Markdown节点
            lines: 文件的所有行

        Returns:
            (start_line, end_line) 或 (None, None)
        """
        node_text = node.get_content() if hasattr(node, "get_content") else node.text
        if not node_text:
            return (None, None)

        # 获取节点的第一行和最后一行
        node_lines = node_text.strip().split("\n")
        if not node_lines:
            return (None, None)

        first_line_text = node_lines[0].strip()
        last_line_text = (
            node_lines[-1].strip() if len(node_lines) > 1 else first_line_text
        )

        # 查找第一次匹配
        start_line = None
        for i, line in enumerate(lines, start=1):
            if first_line_text in line:
                start_line = i
                break

        if start_line is None:
            return (None, None)

        # 查找最后一次匹配
        end_line = start_line
        for i in range(len(lines), start_line - 1, -1):
            if last_line_text in lines[i - 1]:
                end_line = i
                break

        return (start_line, end_line)

    def _find_markdown_files(self, directory: Path) -> List[Path]:
        """查找目录下所有markdown文件。

        Args:
            directory: 目录路径

        Returns:
            Markdown文件路径列表
        """
        markdown_files = []
        if not directory.exists():
            return markdown_files

        try:
            # 如果目录是符号链接，先解析它
            if directory.is_symlink():
                directory = directory.resolve()

            if not directory.is_dir():
                return markdown_files

            # 查找所有.md和.markdown文件
            # 使用 walk 方式遍历，这样可以处理符号链接
            for root, dirs, files in os.walk(directory, followlinks=True):
                root_path = Path(root)
                for file in files:
                    if file.endswith((".md", ".markdown")):
                        file_path = root_path / file
                        # 确保是文件而不是目录，并且不在隐藏目录中
                        if file_path.is_file() and not any(
                            part.startswith(".") for part in file_path.parts[1:]
                        ):
                            markdown_files.append(file_path)

            # 去重并排序
            markdown_files = list(set(markdown_files))
            markdown_files.sort()
        except Exception as e:
            print(f"Error finding markdown files in {directory}: {e}")

        return markdown_files


if __name__ == "__main__":
    # Test LocalFileSearchTool
    tool = LocalFileSearchTool(settings.local_file_search_dir)
    print("Testing LocalFileSearchTool")
    print("-" * 80)

    # Test search_by_keywords
    # print("Test 1: search_by_keywords")
    # keywords = ["后向兼容"]
    # result = tool.search_by_keywords(keywords, expand_mode="ratio", expand_ratio=5)
    # print(f"Search results for keywords {keywords}:")
    # for keyword, matches in result.items():
    #     print(f"  Keyword: {keyword}")
    #     print(f"  Matches: {len(matches)}")
    #     for match in matches:  # Show first 2 matches
    #         print(f"    File: {match['file_path']}")
    #         print(f"    Lines: {match['start_line']}-{match['end_line']}")
    #         print(f"    Content preview: {match['content'][:-1]}")
    # print("-" * 80)

    # print("Test 1: search_by_keywords")
    # keywords = ["ips start failed"]
    # result = tool.search_by_keywords(keywords, expand_mode="ratio", expand_ratio=5)
    # print(f"Search results for keywords {keywords}:")
    # for keyword, matches in result.items():
    #     print(f"  Keyword: {keyword}")
    #     print(f"  Matches: {len(matches)}")
    #     for match in matches:  # Show first 2 matches
    #         print(f"    File: {match['file_path']}")
    #         print(f"    Lines: {match['start_line']}-{match['end_line']}")
    #         print(f"    Content preview: {match['content'][:-1]}")
    # print("-" * 80)

    # print("Test 1: search_by_keywords")
    # keywords = ["ips start failed","检查一下ips的配置文件"]
    # result = tool.search_by_keywords(keywords, expand_mode="ratio", expand_ratio=5)
    # print(f"Search results for keywords {keywords}:")
    # for keyword, matches in result.items():
    #     print(f"  Keyword: {keyword}")
    #     print(f"  Matches: {len(matches)}")
    #     for match in matches:  # Show first 2 matches
    #         print(f"    File: {match['file_path']}")
    #         print(f"    Lines: {match['start_line']}-{match['end_line']}")
    #         print(f"    Content preview: {match['content'][:-1]}")
    # print("-" * 80)

    # print("Test 1: search_by_keywords by markdown mode")
    # # keywords = ['ips', 'start', 'failed']
    # keywords = ['Cassandra', 'concurrent_reads']
    # # keywords = ['DockerRegistry']
    # result = tool.search_keyword_using_binary_algorithm(keywords)
    # print(f"search_keyword_using_binary_algorithm Search results for keywords {keywords}:")
    # for keyword, matches in result.items():
    #     print(f"  Keyword: {keyword}")
    #     print(f"  Matches: {len(matches)}")
    #     for match in matches:  # Show first 2 matches
    #         print(f"    File: {match['file_path']}")
    #         print(f"    Lines: {match['start_line']}-{match['end_line']}")
    #         print(f"    Content preview: {match['content'][:-1]}")
    # print("-" * 80)
    # raise Exception("Stop here")

    print("Test 1: search_by_keywords by markdown mode")
    keywords = ["五分量"]
    result = tool.search_by_keywords(keywords)
    print(f"Search results for keywords {keywords}:")
    for keyword, matches in result.items():
        print(f"  Keyword: {keyword}")
        print(f"  Matches: {len(matches)}")
        for match in matches:  # Show first 2 matches
            print(f"    File: {match['file_path']}")
            print(f"    Lines: {match['start_line']}-{match['end_line']}")
            # print(f"    Content preview: {match['content'][:-1]}")
    raise Exception("Stop here")

    print("Test 1: search_by_keywords by markdown mode")
    keywords = ["ips start failed"]
    result = tool.search_by_keywords(keywords, expand_mode="markdown")
    print(f"Search results for keywords {keywords}:")
    for keyword, matches in result.items():
        print(f"  Keyword: {keyword}")
        print(f"  Matches: {len(matches)}")
        for match in matches:  # Show first 2 matches
            print(f"    File: {match['file_path']}")
            print(f"    Lines: {match['start_line']}-{match['end_line']}")
            print(f"    Content preview: {match['content'][:-1]}")
    print("-" * 80)

    # Test find_text_line_range (if we have a test file)
    print("Test 2: find_text_line_range")
    # This would require an actual file to test, so we'll skip if file doesn't exist
    test_file = "data/markdown/xxx-FAQ-1-break-changes.md"
    test_text = "xxx 的重大改动"
    line_range = tool.find_text_line_range(test_text, test_file)
    if line_range:
        print(f"Found text '{test_text}' at lines {line_range[0]}-{line_range[1]}")
    else:
        print(f"Text '{test_text}' not found in file")
    print("-" * 80)

    # Test expand_context with markdown mode
    print("Test 3: expand_context with markdown mode")
    test_file = "data/markdown/break-changes.md"
    test_text = "此更新会添加jobs表的一个字段"
    line_range = tool.expand_context(test_text, test_file, expand_mode="markdown")
    if line_range:
        print(
            f"Found text '{test_text}' at lines {line_range['start_line']}-{line_range['end_line']}"
        )
        print(f"Content preview: {line_range['content'][:-1]}")
    else:
        print(f"Text '{test_text}' not found in file")
    print("-" * 80)

    # Test expand_context with ratio mode
    print("Test 4: expand_context with ratio mode")
    test_file = "data/markdown/break-changes.md"
    test_text = "此更新会添加jobs表的一个字段"
    line_range = tool.expand_context(
        test_text, test_file, expand_mode="ratio", expand_ratio=5
    )
    if line_range:
        print(
            f"Found text '{test_text}' at lines {line_range['start_line']}-{line_range['end_line']}"
        )
        print(f"Content preview: {line_range['content'][:-1]}")
    else:
        print(f"Text '{test_text}' not found in file")
