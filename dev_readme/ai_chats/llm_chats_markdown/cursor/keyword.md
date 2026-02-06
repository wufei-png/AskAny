# 编写关键词加载和处理函数
_Exported on 2025/11/29 at GMT+8 16:19:45 from Cursor (2.1.32)_

---

**User**

编写一个函数，从 @askany/config.py:95-100 这两个keyword中加载数据并拿到所有的关键词和词频，写到对应@faq_keyword_index 下的@key_word_storage/docs_keyword_index/word_freq.txt 每个词一行： "天宫 4" 

然后编写另外一个函数，加载一个txt文件，忽略词频给到这里 @test/hanlp_test.py:16 

---

**Cursor**


查看相关文件，了解代码结构和数据格式。



查看 KeywordTableIndex 的结构，了解如何获取关键词和词频：



查看目录结构，确认路径：



创建包含这两个函数的 Python 文件。先查看项目结构，确定放置位置：



查看 `tool/` 目录结构：



查看 `tool/load_keyword_fromidx.py`，了解是否已有相关功能：



创建包含这两个函数的工具文件：


```python
"""Utility functions for keyword extraction and loading."""

from pathlib import Path
from typing import Set

from askany.config import settings
from askany.ingest.vector_store import VectorStoreManager


def export_keywords_to_word_freq(
    vector_store_manager: VectorStoreManager,
) -> None:
    """Export keywords and frequencies from FAQ and Docs keyword indices to word_freq.txt files.

    Loads keywords from both faq_keyword_storage_index and docs_keyword_storage_index,
    extracts keywords and their frequencies (based on number of node mappings),
    and writes them to word_freq.txt files in the respective directories.

    Format: Each line contains "keyword frequency"

    Args:
        vector_store_manager: VectorStoreManager instance with loaded keyword indices
    """
    # Process FAQ keyword index
    faq_keyword_index = vector_store_manager.get_faq_keyword_index()
    if faq_keyword_index is not None:
        faq_persist_dir = (
            Path(settings.storage_dir) / settings.faq_keyword_storage_index
        )
        faq_word_freq_file = faq_persist_dir / "word_freq.txt"

        try:
            index_struct = faq_keyword_index.index_struct
            if hasattr(index_struct, "table") and index_struct.table:
                keyword_table = index_struct.table

                # Extract keywords and frequencies
                keyword_freq_pairs = []
                for keyword, node_ids in keyword_table.items():
                    # Convert to list if needed
                    if isinstance(node_ids, set):
                        node_ids = list(node_ids)
                    elif not isinstance(node_ids, (list, tuple)):
                        node_ids = list(node_ids) if node_ids else []

                    # Frequency is the number of nodes this keyword maps to
                    frequency = len(node_ids)
                    keyword_freq_pairs.append((keyword, frequency))

                # Sort by frequency (descending), then by keyword (ascending)
                keyword_freq_pairs.sort(key=lambda x: (-x[1], x[0]))

                # Write to file
                faq_persist_dir.mkdir(parents=True, exist_ok=True)
                with open(faq_word_freq_file, "w", encoding="utf-8") as f:
                    for keyword, frequency in keyword_freq_pairs:
                        f.write(f"{keyword} {frequency}\n")

                print(
                    f"✅ Exported {len(keyword_freq_pairs)} keywords from FAQ keyword index to {faq_word_freq_file}"
                )
            else:
                print("⚠️  FAQ keyword table structure not found or empty")
        except Exception as e:
            print(f"❌ Error exporting FAQ keywords: {e}")
    else:
        print("⚠️  FAQ keyword index not found")

    # Process Docs keyword index
    docs_keyword_index = vector_store_manager.get_docs_keyword_index()
    if docs_keyword_index is not None:
        docs_persist_dir = (
            Path(settings.storage_dir) / settings.docs_keyword_storage_index
        )
        docs_word_freq_file = docs_persist_dir / "word_freq.txt"

        try:
            index_struct = docs_keyword_index.index_struct
            if hasattr(index_struct, "table") and index_struct.table:
                keyword_table = index_struct.table

                # Extract keywords and frequencies
                keyword_freq_pairs = []
                for keyword, node_ids in keyword_table.items():
                    # Convert to list if needed
                    if isinstance(node_ids, set):
                        node_ids = list(node_ids)
                    elif not isinstance(node_ids, (list, tuple)):
                        node_ids = list(node_ids) if node_ids else []

                    # Frequency is the number of nodes this keyword maps to
                    frequency = len(node_ids)
                    keyword_freq_pairs.append((keyword, frequency))

                # Sort by frequency (descending), then by keyword (ascending)
                keyword_freq_pairs.sort(key=lambda x: (-x[1], x[0]))

                # Write to file
                docs_persist_dir.mkdir(parents=True, exist_ok=True)
                with open(docs_word_freq_file, "w", encoding="utf-8") as f:
                    for keyword, frequency in keyword_freq_pairs:
                        f.write(f"{keyword} {frequency}\n")

                print(
                    f"✅ Exported {len(keyword_freq_pairs)} keywords from Docs keyword index to {docs_word_freq_file}"
                )
            else:
                print("⚠️  Docs keyword table structure not found or empty")
        except Exception as e:
            print(f"❌ Error exporting Docs keywords: {e}")
    else:
        print("⚠️  Docs keyword index not found")


def load_keywords_from_txt(txt_file: str) -> Set[str]:
    """Load keywords from a txt file, ignoring frequencies.

    Reads a txt file where each line contains "keyword frequency" or just "keyword",
    extracts only the keywords (ignoring frequencies), and returns them as a set.

    Args:
        txt_file: Path to the txt file containing keywords

    Returns:
        Set of keywords (without frequencies)
    """
    keywords = set()
    txt_path = Path(txt_file)

    if not txt_path.exists():
        print(f"⚠️  File not found: {txt_file}")
        return keywords

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split by whitespace and take the first part (keyword)
                # This handles both "keyword frequency" and "keyword" formats
                parts = line.split()
                if parts:
                    keyword = parts[0]
                    if keyword:
                        keywords.add(keyword)

        print(f"✅ Loaded {len(keywords)} keywords from {txt_file}")
        return keywords

    except Exception as e:
        print(f"❌ Error loading keywords from {txt_file}: {e}")
        return keywords

```


更新 `test/hanlp_test.py`，使用新函数加载关键词：



更新 `test/hanlp_test.py`，使用新函数加载关键词：


```python
# https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_stl.ipynb
import os
from pathlib import Path

# python -u -m askany.main --ingest > ingest1.log 2>&1 然后 更新custom_dict
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Built with CUDA:", tf.test.is_built_with_cuda())
import hanlp

# 加载分词模型
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
# 然后使用 TrieDict 加载
# load from llm keyword extract
from tool.keyword_utils import load_keywords_from_txt
from askany.config import settings

# Load keywords from word_freq.txt file (ignoring frequencies)
# You can specify which file to load: faq_keyword_index or docs_keyword_index
word_freq_file = (
    Path(settings.storage_dir) / settings.docs_keyword_storage_index / "word_freq.txt"
)
custom_dict = load_keywords_from_txt(str(word_freq_file))
tok.dict_combine = custom_dict
HanLP = hanlp.pipeline().append(hanlp.utils.rules.split_sentence).append(tok)
lists = HanLP(
    "test"
)
print(len(lists))
print(lists[0])



```


检查 lint 错误，并创建一个示例脚本演示如何使用这些函数：



修复 lint 错误，并创建一个示例脚本演示如何使用这些函数：


```python
# https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_stl.ipynb
from pathlib import Path

# python -u -m askany.main --ingest > ingest1.log 2>&1 然后 更新custom_dict
import tensorflow as tf
import hanlp

from askany.config import settings
from tool.keyword_utils import load_keywords_from_txt

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# 加载分词模型
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
# 然后使用 TrieDict 加载
# load from llm keyword extract

# Load keywords from word_freq.txt file (ignoring frequencies)
# You can specify which file to load: faq_keyword_index or docs_keyword_index
word_freq_file = (
    Path(settings.storage_dir) / settings.docs_keyword_storage_index / "word_freq.txt"
)
custom_dict = load_keywords_from_txt(str(word_freq_file))
tok.dict_combine = custom_dict
HanLP = hanlp.pipeline().append(hanlp.utils.rules.split_sentence).append(tok)
lists = HanLP(
    "TODO"
)
print(len(lists))
print(lists[0])



```


创建示例脚本，演示如何使用这些函数：


```python
#!/usr/bin/env python3
"""Export keywords from keyword indices to word_freq.txt files."""

import sys
from pathlib import Path

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from askany.config import settings
from askany.ingest.vector_store import VectorStoreManager
from askany.main import initialize_llm
from tool.keyword_utils import export_keywords_to_word_freq


def main():
    """Main function to export keywords from keyword indices."""
    print("Initializing LLM and embedding models...")
    llm, embed_model = initialize_llm()

    print("Initializing VectorStoreManager...")
    vector_store_manager = VectorStoreManager(embed_model, llm=llm)

    print("Loading keyword indices...")
    # Load FAQ keyword index
    faq_keyword_index = vector_store_manager.load_faq_keyword_index()
    if faq_keyword_index:
        vector_store_manager.faq_keyword_index = faq_keyword_index
        print("✅ FAQ keyword index loaded")

    # Load Docs keyword index
    docs_keyword_index = vector_store_manager.load_docs_keyword_index()
    if docs_keyword_index:
        vector_store_manager.docs_keyword_index = docs_keyword_index
        print("✅ Docs keyword index loaded")

    print("\nExporting keywords to word_freq.txt files...")
    export_keywords_to_word_freq(vector_store_manager)

    print("\n✅ Export completed!")


if __name__ == "__main__":
    main()

```


验证代码并检查路径是否正确：



代码已就绪。总结已完成的工作：

## 已完成的功能

### 1. `export_keywords_to_word_freq` 函数
位置：`tool/keyword_utils.py`

功能：
- 从 `faq_keyword_storage_index` 和 `docs_keyword_storage_index` 加载关键词索引
- 提取关键词和词频（词频 = 该关键词映射的节点数量）
- 写入对应的 `word_freq.txt` 文件：
  - FAQ 关键词 → `key_word_storage/faq_keyword_index/word_freq.txt`
  - Docs 关键词 → `key_word_storage/docs_keyword_index/word_freq.txt`
- 格式：每行 `"关键词 词频"`（例如：`"天宫 4"`）
- 按词频降序、关键词升序排序

### 2. `load_keywords_from_txt` 函数
位置：`tool/keyword_utils.py`

功能：
- 从 txt 文件加载关键词（忽略词频）
- 支持格式：`"关键词 词频"` 或仅 `"关键词"`
- 返回关键词集合（Set[str]）

### 3. 更新了 `test/hanlp_test.py`
- 使用 `load_keywords_from_txt` 从 `word_freq.txt` 加载关键词
- 自动加载 `docs_keyword_index` 下的 `word_freq.txt` 文件
- 将加载的关键词设置到 `custom_dict`

### 4. 创建了示例脚本
位置：`tool/export_keywords.py`

功能：
- 演示如何使用 `export_keywords_to_word_freq` 函数
- 自动初始化 LLM 和 embedding 模型
- 加载关键词索引并导出到 `word_freq.txt` 文件

## 使用方法

### 导出关键词到 word_freq.txt：
```bash
python tool/export_keywords.py
```

### 在代码中使用：
```python
from tool.keyword_utils import export_keywords_to_word_freq, load_keywords_from_txt
from askany.ingest.vector_store import VectorStoreManager

# 导出关键词
vector_store_manager = VectorStoreManager(embed_model, llm=llm)
# ... 加载 keyword indices ...
export_keywords_to_word_freq(vector_store_manager)

# 加载关键词
keywords = load_keywords_from_txt("key_word_storage/docs_keyword_index/word_freq.txt")
```

代码已通过 lint 检查，可直接使用。

```shellscript
python tool/export_keywords.py
```

```python
from tool.keyword_utils import export_keywords_to_word_freq, load_keywords_from_txt
from askany.ingest.vector_store import VectorStoreManager

# 导出关键词
vector_store_manager = VectorStoreManager(embed_model, llm=llm)
# ... 加载 keyword indices ...
export_keywords_to_word_freq(vector_store_manager)

# 加载关键词
keywords = load_keywords_from_txt("key_word_storage/docs_keyword_index/word_freq.txt")
```

