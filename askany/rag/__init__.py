"""RAG (Retrieval-Augmented Generation) module."""

from .faq_query_engine import FAQQueryEngine
from .rag_query_engine import RAGQueryEngine
from .router import QueryRouter, create_query_router

__all__ = ["RAGQueryEngine", "FAQQueryEngine", "QueryRouter", "create_query_router"]

# QA Cache Manager singleton
from askany.cache.qa_cache import QACacheManager

_qa_cache_manager: QACacheManager | None = None


def get_qa_cache_manager() -> QACacheManager | None:
    """Get the global QA cache manager instance."""
    return _qa_cache_manager


def set_qa_cache_manager(manager: QACacheManager) -> None:
    """Set the global QA cache manager instance."""
    global _qa_cache_manager
    _qa_cache_manager = manager
