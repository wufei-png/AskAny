import re


def contains_chinese(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))
