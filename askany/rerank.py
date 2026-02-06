"""Safe reranker wrapper that handles edge cases and prevents node loss."""

import logging
from typing import List, Optional

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


class SafeReranker(BaseNodePostprocessor):
    """Safe wrapper for reranker that prevents node loss.

    This wrapper:
    1. Checks if input node count is less than reranker's top_n before reranking
    2. Falls back to original nodes if reranker returns empty results
    3. Provides detailed logging for debugging
    4. Inherits from BaseNodePostprocessor to be compatible with RetrieverQueryEngine
    """

    reranker: Optional[BaseNodePostprocessor] = None
    rerank_top_n: Optional[int] = None

    def __init__(self, reranker: Optional[BaseNodePostprocessor]):
        """Initialize SafeReranker.

        Args:
            reranker: The underlying reranker instance to wrap
        """
        super().__init__()  # Initialize BaseNodePostprocessor
        self.reranker = reranker
        if reranker:
            self.rerank_top_n = getattr(reranker, "top_n", None)
            logger.debug(
                "SafeReranker initialized - rerank_top_n: %s", self.rerank_top_n
            )
        else:
            self.rerank_top_n = None

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Safely postprocess nodes with reranker.

        This is the internal implementation called by BaseNodePostprocessor.postprocess_nodes().

        Args:
            nodes: List of nodes to rerank
            query_bundle: Query bundle for reranking

        Returns:
            Reranked nodes, or original nodes if reranking is skipped or fails
        """
        if not self.reranker or not nodes:
            return nodes

        input_node_count = len(nodes)

        # Check if we should skip reranking
        # If reranker's top_n is set and input nodes are fewer, skip reranking
        # This prevents reranker from filtering out all nodes when top_n > input_count
        if (self.rerank_top_n < 0) or (
            self.rerank_top_n is not None
            and self.rerank_top_n > 0
            and input_node_count <= self.rerank_top_n
        ):
            logger.debug(
                "跳过reranker - 输入节点数(%d) < reranker top_n(%d)，直接使用原始结果",
                input_node_count,
                self.rerank_top_n,
            )
            return nodes

        # Perform reranking
        logger.debug("使用reranker对节点重新排序")
        logger.debug(
            "Reranker配置 - top_n: %s, 输入节点数: %d",
            self.rerank_top_n,
            input_node_count,
        )

        # Log input node details for debugging
        for idx, node in enumerate(nodes):
            node_text_preview = (
                node.node.text[:100] if hasattr(node.node, "text") else "N/A"
            )
            logger.debug(
                "输入节点 %d: score=%.4f, text_preview=%s",
                idx,
                node.score,
                node_text_preview,
            )

        original_nodes = nodes  # Save original nodes as fallback
        try:
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
            logger.debug("Reranker重排序完成 - 节点数: %d", len(reranked_nodes))

            # Log reranked node details
            if reranked_nodes:
                for idx, node in enumerate(reranked_nodes):
                    node_text_preview = (
                        node.node.text[:100] if hasattr(node.node, "text") else "N/A"
                    )
                    logger.debug(
                        "重排序后节点 %d: score=%.4f, text_preview=%s",
                        idx,
                        node.score,
                        node_text_preview,
                    )

            # If reranker returns empty nodes, fall back to original nodes
            if reranked_nodes:
                return reranked_nodes
            else:
                logger.warning(
                    "Reranker返回空节点，回退到原始节点 - 原始节点数: %d, reranker_top_n: %s",
                    len(original_nodes),
                    self.rerank_top_n,
                )
                return original_nodes
        except Exception as e:
            logger.error(
                "Reranker处理失败，回退到原始节点 - 错误: %s, 原始节点数: %d",
                str(e),
                len(original_nodes),
                exc_info=True,
            )
            return original_nodes

    def __bool__(self) -> bool:
        """Check if reranker is available."""
        return self.reranker is not None

    @classmethod
    def create(
        cls,
        top_n: int = -1,
        device: Optional[str] = None,
        reranker_model: Optional[str] = None,
    ) -> "SafeReranker":
        """Create a SafeReranker instance with underlying reranker.

        Args:
            top_n: Number of top results to return. Use -1 to return all nodes.
            device: Device to use ("cuda" or "cpu"). If None, auto-detect.
            reranker_model: Model name for reranker (e.g., "BAAI/bge-reranker-v2-m3") or
                          local path to model directory (e.g., "/path/to/bge-reranker-v2-m3").
                          If local path is provided, model will be loaded offline.
                          If None, uses settings.reranker_model.

        Returns:
            SafeReranker instance wrapping the created reranker
        """
        # Import settings here to avoid circular import
        from askany.config import settings

        try:
            reranker_model = reranker_model or settings.reranker_model
            reranker_type = getattr(settings, "reranker_type", "sentence_transformer")

            # Auto-detect device if not provided
            if device is None:
                try:
                    import torch

                    if torch.cuda.is_available():
                        device = "cuda"
                        logger.info("CUDA available, using GPU for reranker")
                    else:
                        device = "cpu"
                        logger.info("Using CPU for reranker")
                except ImportError:
                    device = "cpu"
                    logger.info("torch not available, using CPU for reranker")

            base_reranker: Optional[BaseNodePostprocessor] = None

            if reranker_type == "flag_embedding":
                # Try to use FlagEmbeddingReranker if available
                try:
                    try:
                        from llama_index.postprocessor.flag_embedding_reranker import (
                            FlagEmbeddingReranker,
                        )
                    except ImportError:
                        from llama_index.core.postprocessor.flag_embedding_reranker import (
                            FlagEmbeddingReranker,
                        )

                    base_reranker = FlagEmbeddingReranker(
                        model=reranker_model,
                        top_n=top_n,
                        device=device,
                    )
                    logger.info("FlagEmbeddingReranker created successfully")
                except ImportError:
                    logger.warning(
                        "FlagEmbeddingReranker not available, "
                        "falling back to SentenceTransformerRerank"
                    )

            # Default: Use SentenceTransformerRerank
            if base_reranker is None:
                try:
                    # Check if local_files_only should be used
                    local_files_only = getattr(
                        settings, "reranker_local_files_only", False
                    )
                    # Auto-detect: if reranker_model is an absolute path, use local_files_only
                    import os

                    if os.path.isabs(reranker_model) and os.path.exists(reranker_model):
                        local_files_only = True

                    logger.info(
                        f"Creating SentenceTransformerRerank with model={reranker_model}, device={device}, top_n={top_n}, local_files_only={local_files_only}"
                    )
                    if not local_files_only:
                        logger.info(
                            "This may take a while if the model needs to be downloaded or loaded..."
                        )
                    base_reranker = SentenceTransformerRerank(
                        model=reranker_model,
                        top_n=top_n,
                        trust_remote_code=True,
                        device=device,
                        local_files_only=local_files_only,
                    )
                    logger.info("SentenceTransformerRerank created successfully")
                except TypeError:
                    # If device or local_files_only parameter is not supported
                    logger.warning(
                        "SentenceTransformerRerank does not support device/local_files_only parameter, "
                        "using auto-detection"
                    )
                    base_reranker = SentenceTransformerRerank(
                        model=reranker_model,
                        top_n=top_n,
                        trust_remote_code=True,
                    )
                    logger.info("SentenceTransformerRerank created successfully")

            return cls(base_reranker)
        except Exception as e:
            logger.error("Failed to create reranker: %s", str(e), exc_info=True)
            return cls(None)
