"""Chinese prompt overrides for llama_index defaults."""

from llama_index.core.prompts import default_prompts

# Chinese version of DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL.
CHINESE_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
    "下面给出一个问题。请根据该问题从文本中提取最多 {max_keywords} 个关键词。"
    "优先挑选能够帮助我们检索答案的关键信息，并避免常见停用词。\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "请使用如下逗号分隔格式返回：'KEYWORDS: <关键词列表>'\n"
)


def apply_chinese_prompts() -> None:
    """Override llama_index default prompts with Chinese templates."""

    default_prompts.DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
        CHINESE_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL
    )
