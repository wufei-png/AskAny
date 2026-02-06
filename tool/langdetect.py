# from fast_langdetect import detect


# def is_chinese(text):
#     return detect(text, model="auto", k=1)[0]['lang'] == "zh"

import re

def contains_chinese(s: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', s))

# if __name__ == "__main__":
#   querys=["Hello, world!", "解析结果成功写到kafka中了，但没有在属性库里找到，怎么办？", "activate rate", "I am wufei", "Hello, world!", "Hello 世界 こんにちは", "Hello 世界 こんにちは","我是wufei"]
#   for query in querys:
#     print(f"Query: {query}")
#     print(f"Is Chinese: {contains_chinese(query)}")
#     print("-" * 80)
#   for query in querys:
#     print(f"Query: {query}")
#     print(f"Is Chinese: {is_chinese(query)}")
#     print("-" * 80)
