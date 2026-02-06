"""Prompts module for AskAny.

This module provides centralized prompt management with multi-language support.
All prompts used in the workflow are defined here for easy maintenance.

Usage:
    from askany.prompts import get_prompts
    prompts = get_prompts()  # Uses language from settings
    # or
    prompts = get_prompts("en")  # Force English
"""

from askany.prompts.prompt_manager import PromptManager, get_prompts

__all__ = ["PromptManager", "get_prompts"]
