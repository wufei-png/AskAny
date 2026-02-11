"""Custom keywords for knowledge embedding matching.

This module defines custom keywords that are used for similarity matching
in keyword extraction. Keywords matching these custom keywords will be
directly retained without LLM judgment.
"""

# List of custom keywords for knowledge embedding matching
# These keywords will be used to filter extracted keywords based on similarity
# If a keyword matches any of these custom keywords (above threshold),
# it will be retained directly without LLM judgment
KNOWLEDGE_EMBEDDING_KEYWORDS: list[str] = []
