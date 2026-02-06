"""Custom KeywordTableIndex and KeywordTableGPTRetriever with KeywordExtractorWrapper support."""

from typing import Any, List, Optional, Set, Union

from llama_index.core import KeywordTableIndex
from llama_index.core.indices.keyword_table.base import BaseKeywordTableIndex
from llama_index.core.indices.keyword_table.retrievers import (
    KeywordTableGPTRetriever,
)
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate

from askany.ingest.keyword_extract_wrapper import KeywordExtractorWrapper
from askany.config import settings


# Global KeywordExtractorWrapper instance (singleton pattern)
_global_keyword_extractor: Optional[KeywordExtractorWrapper] = None


def get_global_keyword_extractor() -> Optional[KeywordExtractorWrapper]:
    """Get the global KeywordExtractorWrapper instance."""
    return _global_keyword_extractor


def set_global_keyword_extractor(extractor: KeywordExtractorWrapper) -> None:
    """Set the global KeywordExtractorWrapper instance."""
    global _global_keyword_extractor
    _global_keyword_extractor = extractor


class CustomKeywordTableIndex(KeywordTableIndex):
    """Custom KeywordTableIndex that uses KeywordExtractorWrapper for keyword extraction."""

    def __init__(
        self,
        nodes: Optional[list] = None,
        objects: Optional[list] = None,
        index_struct: Optional[Any] = None,
        llm: Optional[LLM] = None,
        keyword_extract_template: Optional[BasePromptTemplate] = None,
        max_keywords_per_chunk: int = 10,
        use_async: bool = False,
        show_progress: bool = False,
        keyword_extractor: Optional[KeywordExtractorWrapper] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CustomKeywordTableIndex.

        Args:
            keyword_extractor: KeywordExtractorWrapper instance. If None, uses global instance.
            Other args: Same as KeywordTableIndex.
        """
        # Store keyword extractor (use provided one, or global, or create new)
        if keyword_extractor is not None:
            self._keyword_extractor = keyword_extractor
        elif _global_keyword_extractor is not None:
            self._keyword_extractor = _global_keyword_extractor
        else:
            # Fallback: create a new one (should not happen in normal usage)
            self._keyword_extractor = KeywordExtractorWrapper(
                priority=settings.keyword_extractor_priority
            )
            set_global_keyword_extractor(self._keyword_extractor)
        # Call parent constructor
        super().__init__(
            nodes=nodes,
            objects=objects,
            index_struct=index_struct,
            llm=llm,
            keyword_extract_template=keyword_extract_template,
            max_keywords_per_chunk=max_keywords_per_chunk,
            use_async=use_async,
            show_progress=show_progress,
            **kwargs,
        )
        print(
            f"CustomKeywordTableIndex initialized with keyword extractor: {self._keyword_extractor}"
        )

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text using KeywordExtractorWrapper."""
        keywords, _ = self._keyword_extractor.extract_keywords(text)
        return set(keywords)

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text using KeywordExtractorWrapper (async)."""
        # KeywordExtractorWrapper.extract_keywords is sync, so we just call it
        keywords, _ = self._keyword_extractor.extract_keywords(text)
        return set(keywords)

    def as_retriever(
        self,
        retriever_mode: Union[str, Any] = "default",
        **kwargs: Any,
    ) -> Any:
        """Create a retriever from this index.

        Override to use CustomKeywordTableGPTRetriever for default mode.
        """
        from llama_index.core.indices.keyword_table.base import (
            KeywordTableRetrieverMode,
        )

        if (
            retriever_mode == KeywordTableRetrieverMode.DEFAULT
            or retriever_mode == "default"
        ):
            # Use custom retriever with keyword extractor
            return CustomKeywordTableGPTRetriever(
                self,
                object_map=self._object_map,
                llm=self._llm,
                keyword_extractor=self._keyword_extractor,
                **kwargs,
            )
        else:
            # For other modes, use parent implementation
            return super().as_retriever(retriever_mode=retriever_mode, **kwargs)


class CustomKeywordTableGPTRetriever(KeywordTableGPTRetriever):
    """Custom KeywordTableGPTRetriever that uses KeywordExtractorWrapper for keyword extraction."""

    def __init__(
        self,
        index: BaseKeywordTableIndex,
        keyword_extract_template: Optional[BasePromptTemplate] = None,
        query_keyword_extract_template: Optional[BasePromptTemplate] = None,
        max_keywords_per_query: int = 3,
        num_chunks_per_query: int = 10,
        llm: Optional[LLM] = None,
        keyword_extractor: Optional[KeywordExtractorWrapper] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CustomKeywordTableGPTRetriever.

        Args:
            keyword_extractor: KeywordExtractorWrapper instance. If None, uses global instance.
            Other args: Same as KeywordTableGPTRetriever.
        """
        # Store keyword extractor (use provided one, or global, or create new)
        if keyword_extractor is not None:
            self._keyword_extractor = keyword_extractor
        elif _global_keyword_extractor is not None:
            self._keyword_extractor = _global_keyword_extractor
        else:
            # Fallback: create a new one (should not happen in normal usage)
            self._keyword_extractor = KeywordExtractorWrapper(
                priority=settings.keyword_extractor_priority
            )
            set_global_keyword_extractor(self._keyword_extractor)
        # Call parent constructor
        super().__init__(
            index=index,
            keyword_extract_template=keyword_extract_template,
            query_keyword_extract_template=query_keyword_extract_template,
            max_keywords_per_query=max_keywords_per_query,
            num_chunks_per_query=num_chunks_per_query,
            llm=llm,
            **kwargs,
        )

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords from query using KeywordExtractorWrapper."""
        keywords, _ = self._keyword_extractor.extract_keywords(query_str)
        # Limit to max_keywords_per_query
        return keywords[: self.max_keywords_per_query]
