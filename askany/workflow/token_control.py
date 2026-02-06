"""Token control utilities for LLM API calls."""

import logging
from typing import List

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

# Rough estimation: 1 token ≈ 4 characters for Chinese/English mixed text
# This is a conservative estimate (actual may be 3-5 chars per token)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // CHARS_PER_TOKEN + 1


def truncate_nodes_by_tokens(
    nodes: List[NodeWithScore],
    max_tokens: int,
    reserve_for_prompt: int = 1000,
) -> tuple[List[NodeWithScore], int, bool]:
    """Truncate nodes list to fit within token limit.

    Args:
        nodes: List of nodes to truncate
        max_tokens: Maximum token limit
        reserve_for_prompt: Tokens to reserve for prompt template (query, system message, etc.)

    Returns:
        Tuple of (truncated_nodes, total_tokens, was_truncated)
    """
    available_tokens = max_tokens - reserve_for_prompt
    if available_tokens <= 0:
        logger.warning(
            "Token limit too low: max_tokens=%d, reserve_for_prompt=%d",
            max_tokens,
            reserve_for_prompt,
        )
        return [], 0, False

    # Flatten nodes if they contain nested lists
    flat_nodes = []
    for node in nodes:
        if isinstance(node, list):
            # If node is a list, extend with its items
            flat_nodes.extend(node)
        elif isinstance(node, NodeWithScore):
            flat_nodes.append(node)
        else:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unexpected node type in truncate_nodes_by_tokens: {type(node)}, skipping"
            )
    nodes = flat_nodes

    truncated_nodes = []
    total_tokens = 0
    was_truncated = False

    for node in nodes:
        # Ensure node is NodeWithScore
        if not isinstance(node, NodeWithScore):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Skipping invalid node type: {type(node)}")
            continue

        content = (
            node.node.get_content()
            if hasattr(node.node, "get_content")
            else node.node.text
        )
        if not content:
            continue

        node_tokens = estimate_tokens(content)

        # If adding this node would exceed limit, truncate it
        if total_tokens + node_tokens > available_tokens:
            # Try to truncate the node content to fit
            remaining_tokens = available_tokens - total_tokens
            if remaining_tokens > 100:  # Only truncate if we have meaningful space
                # Truncate content to fit remaining tokens
                max_chars = remaining_tokens * CHARS_PER_TOKEN
                truncated_content = content[:max_chars]
                # Create a truncated copy
                from llama_index.core.schema import TextNode

                truncated_node = TextNode(
                    text=truncated_content + "\n[内容已截断...]",
                    metadata=(
                        node.node.metadata.copy()
                        if hasattr(node.node, "metadata")
                        else {}
                    ),
                )
                truncated_nodes.append(
                    NodeWithScore(node=truncated_node, score=node.score)
                )
                total_tokens += estimate_tokens(truncated_content)
                was_truncated = True
            # Stop adding more nodes
            break
        else:
            truncated_nodes.append(node)
            total_tokens += node_tokens

    if was_truncated or len(truncated_nodes) < len(nodes):
        logger.warning(
            "Nodes truncated due to token limit: original=%d nodes, kept=%d nodes, "
            "total_tokens=%d/%d (reserved %d for prompt)",
            len(nodes),
            len(truncated_nodes),
            total_tokens,
            available_tokens,
            reserve_for_prompt,
        )

    return truncated_nodes, total_tokens, was_truncated


def truncate_text_by_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to fit within token limit.

    Args:
        text: Input text
        max_tokens: Maximum token limit

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    text_tokens = estimate_tokens(text)
    if text_tokens <= max_tokens:
        return text, False

    # Truncate to fit
    max_chars = max_tokens * CHARS_PER_TOKEN
    truncated = text[:max_chars] + "\n[内容已截断...]"
    logger.warning(
        "Text truncated due to token limit: original_tokens=%d, truncated_tokens=%d/%d",
        text_tokens,
        max_tokens,
        max_tokens,
    )
    return truncated, True


def check_and_truncate_messages(
    messages: List[dict],
    max_tokens: int,
) -> tuple[List[dict], int, bool]:
    """Check and truncate messages to fit within token limit.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum token limit

    Returns:
        Tuple of (truncated_messages, total_tokens, was_truncated)
    """
    total_tokens = 0
    truncated_messages = []
    was_truncated = False

    # System messages are usually small, keep them
    # User messages might be large, truncate from the end
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            truncated_messages.append(msg)
            continue

        msg_tokens = estimate_tokens(content)

        # For system messages, keep as is (usually small)
        if msg.get("role") == "system":
            truncated_messages.append(msg)
            total_tokens += msg_tokens
            continue

        # For user messages, check if we need to truncate
        if total_tokens + msg_tokens > max_tokens:
            # Truncate this message
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 50:  # Only if we have meaningful space
                truncated_content, truncated = truncate_text_by_tokens(
                    content, remaining_tokens
                )
                truncated_messages.append(
                    {
                        "role": msg["role"],
                        "content": truncated_content,
                    }
                )
                total_tokens += estimate_tokens(truncated_content)
                was_truncated = truncated or was_truncated
            # Stop processing more messages
            break
        else:
            truncated_messages.append(msg)
            total_tokens += msg_tokens

    if was_truncated:
        logger.warning(
            "Messages truncated due to token limit: total_tokens=%d/%d",
            total_tokens,
            max_tokens,
        )

    return truncated_messages, total_tokens, was_truncated
