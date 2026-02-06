"""Prompt manager for AskAny workflow.

Provides centralized prompt management with multi-language support (cn/en).
"""

from typing import Literal

from askany.prompts import prompts_cn, prompts_en

# Supported languages
LanguageType = Literal["cn", "en"]


class PromptManager:
    """Manager for accessing prompts in different languages."""

    def __init__(self, language: LanguageType = "cn"):
        """Initialize PromptManager.

        Args:
            language: Language code, "cn" for Chinese, "en" for English.
        """
        self.language = language
        self._prompts = prompts_cn if language == "cn" else prompts_en

    def get(self, prompt_name: str) -> str:
        """Get a prompt by name.

        Args:
            prompt_name: Name of the prompt constant.

        Returns:
            The prompt string.

        Raises:
            AttributeError: If prompt_name doesn't exist.
        """
        return getattr(self._prompts, prompt_name)

    # Agent System Prompt
    @property
    def agent_system_prompt(self) -> str:
        return self._prompts.AGENT_SYSTEM_PROMPT

    # Tool Descriptions
    @property
    def rag_search_description(self) -> str:
        return self._prompts.RAG_SEARCH_DESCRIPTION

    @property
    def web_search_description(self) -> str:
        return self._prompts.WEB_SEARCH_DESCRIPTION

    @property
    def local_file_search_description(self) -> str:
        return self._prompts.LOCAL_FILE_SEARCH_DESCRIPTION

    @property
    def get_file_content_description(self) -> str:
        return self._prompts.GET_FILE_CONTENT_DESCRIPTION

    # Relevance Analysis
    @property
    def relevance_analysis_system(self) -> str:
        return self._prompts.RELEVANCE_ANALYSIS_SYSTEM

    @property
    def relevance_analysis_task(self) -> str:
        return self._prompts.RELEVANCE_ANALYSIS_TASK

    @property
    def no_relevant_system(self) -> str:
        return self._prompts.NO_RELEVANT_SYSTEM

    @property
    def no_relevant_task(self) -> str:
        return self._prompts.NO_RELEVANT_TASK

    @property
    def no_relevant_without_sub_system(self) -> str:
        return self._prompts.NO_RELEVANT_WITHOUT_SUB_SYSTEM

    @property
    def no_relevant_without_sub_task(self) -> str:
        return self._prompts.NO_RELEVANT_WITHOUT_SUB_TASK

    @property
    def simple_keywords_system(self) -> str:
        return self._prompts.SIMPLE_KEYWORDS_SYSTEM

    @property
    def simple_keywords_task(self) -> str:
        return self._prompts.SIMPLE_KEYWORDS_TASK

    # Direct Answer & Web/RAG Routing
    @property
    def direct_answer_system(self) -> str:
        return self._prompts.DIRECT_ANSWER_SYSTEM

    @property
    def direct_answer_task(self) -> str:
        return self._prompts.DIRECT_ANSWER_TASK

    @property
    def web_or_rag_system(self) -> str:
        return self._prompts.WEB_OR_RAG_SYSTEM

    @property
    def web_or_rag_task(self) -> str:
        return self._prompts.WEB_OR_RAG_TASK

    # Final Answer Generation
    @property
    def final_answer_system(self) -> str:
        return self._prompts.FINAL_ANSWER_SYSTEM

    @property
    def final_answer_task(self) -> str:
        return self._prompts.FINAL_ANSWER_TASK

    @property
    def final_answer_no_context_task(self) -> str:
        return self._prompts.FINAL_ANSWER_NO_CONTEXT_TASK

    @property
    def not_complete_answer_system(self) -> str:
        return self._prompts.NOT_COMPLETE_ANSWER_SYSTEM

    @property
    def not_complete_answer_task(self) -> str:
        return self._prompts.NOT_COMPLETE_ANSWER_TASK

    # Sub-problem Decomposition
    @property
    def sub_problem_system(self) -> str:
        return self._prompts.SUB_PROBLEM_SYSTEM

    @property
    def sub_problem_task(self) -> str:
        return self._prompts.SUB_PROBLEM_TASK


# Global prompt manager instance (lazy initialization)
_prompt_manager: PromptManager | None = None


def get_prompts(language: LanguageType | None = None) -> PromptManager:
    """Get the prompt manager instance.

    Args:
        language: Language code. If None, uses settings.language.

    Returns:
        PromptManager instance.
    """
    global _prompt_manager

    if language is None:
        from askany.config import settings
        language = getattr(settings, "language", "cn")

    if _prompt_manager is None or _prompt_manager.language != language:
        _prompt_manager = PromptManager(language)

    return _prompt_manager
