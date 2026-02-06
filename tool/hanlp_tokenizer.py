# https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_stl.ipynb
import sys
from pathlib import Path

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))
import hanlp

# python -u -m askany.main --ingest > ingest1.log 2>&1 然后 更新custom_dict
import tensorflow as tf

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
# print(custom_dict)
tok.dict_combine = custom_dict
HanLP = hanlp.pipeline().append(hanlp.utils.rules.split_sentence).append(tok)
lists = HanLP("激活率是什么意思？")
print(len(lists))
for list in lists:
    print(list)
    print("-" * 10)

lists = HanLP("ips start failed")
print(len(lists))
for list in lists:
    print(list)
    print("-" * 10)


lists = HanLP("k8s如何重启deployment？")
print(len(lists))
for list in lists:
    print(list)
    print("-" * 10)


lists = HanLP("中美当前的关系下，普通人如何投资美股？")
print(len(lists))
for list in lists:
    print(list)
    print("-" * 10)
