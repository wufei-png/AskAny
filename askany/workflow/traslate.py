import deepl


def translation(strs: str) -> str:
    auth_key = settings.deepl_auth_key  # use DeepL free API
    target_language = "ZH"  # "EN-US"
    # 调用deepl
    translator = deepl.Translator(auth_key)  # input the auth_key
    result = translator.translate_text(strs, target_lang=target_language)
    return result.text
