"""Summary generator for long content based on token limits."""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class SummaryMode(str, Enum):
    """Summary mode enumeration."""

    SIMPLE = "simple"  # Mode 1: Direct compression without splitting
    SEGMENT = "segment"  # Mode 2: Split into segments for large content
    AUTO = "auto"  # Mode 3: Try simple first, fallback to segment on error


class SummaryResponse(BaseModel):
    """Summary response structure."""

    summary: str = Field(
        description="压缩后的总结内容，需要保持原意和关键信息。",
    )


class SummaryFromLlm:
    """Generator for summarizing content based on token limits."""

    def __init__(
        self, llm: Optional[ChatOpenAI] = None, mode: SummaryMode = SummaryMode.AUTO
    ):
        """Initialize SummaryFromLlm.

        Args:
            llm: LangChain ChatOpenAI instance. If None, creates one from settings.
            mode: Summary mode. Options:
                - SIMPLE: Direct compression without splitting (Mode 1)
                - SEGMENT: Split into segments for large content (Mode 2)
                - AUTO: Try simple first, fallback to segment on error (Mode 3, default)
        """
        if llm is None:
            api_base = settings.openai_api_base
            api_key = settings.openai_api_key if settings.openai_api_key else None
            model = settings.openai_model

            # Create ChatOpenAI client from configuration
            client_api_key = api_key if api_key else ""
            self.llm = ChatOpenAI(
                model=model,
                api_key=client_api_key,
                base_url=api_base,
                temperature=settings.temperature,
                max_tokens=settings.output_tokens,
            )
        else:
            self.llm = llm

        self.mode = mode

        # 不再使用 structured output，直接使用 completion 接口
        # 这样可以避免 structured output 可能带来的问题（如换行符问题）

    def _split_by_newlines(self, text: str) -> list[str]:
        """Split text by newlines into paragraphs.

        Args:
            text: Input text to split

        Returns:
            List of paragraphs
        """
        paragraphs = text.split("\n")
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _find_split_points(self, paragraphs: list[str], num_segments: int) -> list[int]:
        """Find split points in paragraphs for dividing into segments based on character count.

        Args:
            paragraphs: List of paragraphs
            num_segments: Number of segments to create

        Returns:
            List of paragraph indices where to split (exclusive end indices)
        """
        if num_segments <= 1:
            return [len(paragraphs)]

        total_paragraphs = len(paragraphs)
        if total_paragraphs == 0:
            return []

        # Calculate character count for each paragraph
        paragraph_chars = [len(p) for p in paragraphs]
        total_chars = sum(paragraph_chars)

        if total_chars == 0:
            return [len(paragraphs)]

        # Calculate target characters per segment
        target_chars_per_segment = total_chars / num_segments

        split_points = []
        current_chars = 0
        current_segment = 0

        for i, para_chars in enumerate(paragraph_chars):
            current_chars += para_chars

            # Check if we've reached the target for current segment
            # Use >= to ensure we don't create empty segments
            if current_chars >= target_chars_per_segment * (current_segment + 1):
                split_points.append(i + 1)  # Exclusive end index
                current_segment += 1

                # If we've created enough segments, break
                if current_segment >= num_segments - 1:
                    break

        # Add final split point to include all remaining paragraphs
        split_points.append(total_paragraphs)

        return split_points

    def _create_segments(
        self, paragraphs: list[str], split_points: list[int]
    ) -> list[str]:
        """Create text segments based on split points.

        Args:
            paragraphs: List of paragraphs
            split_points: List of paragraph indices where to split

        Returns:
            List of text segments
        """
        segments = []
        start_idx = 0

        for end_idx in split_points:
            segment_paragraphs = paragraphs[start_idx:end_idx]
            segment_text = "\n".join(segment_paragraphs)
            segments.append(segment_text)
            start_idx = end_idx

        return segments

    def _summarize_segment(
        self, segment: str, compression_ratio: float
    ) -> Tuple[str, Dict]:
        """Summarize a single segment with specified compression ratio.

        Args:
            segment: Text segment to summarize
            compression_ratio: Target compression ratio (0-1)

        Returns:
            Tuple of (summary_text, token_usage_dict)
        """
        compression_percentage = int(compression_ratio * 100)

        prompt = f"""请将以下内容压缩到原来的约 {compression_percentage}%，保持原意和关键信息，使用自然流畅的语言：

{segment}

请确保压缩后的内容保持原意，不要遗漏重要信息。"""

        messages = [
            SystemMessage(
                content="你是一个专业的文本总结助手，擅长压缩长文本同时保持关键信息。"
            ),
            HumanMessage(content=prompt),
        ]

        # 使用普通的 completion 接口，不使用 structured output
        try:
            logger.info(f"Summarizing segment: {len(segment)} characters")
            response = self.llm.invoke(messages)
            logger.info(
                f"Response received: {len(response.content) if hasattr(response, 'content') else 0} characters"
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error summarizing segment: {error_msg}")
            if "length limit" in error_msg.lower() or "LengthFinishReasonError" in str(
                type(e)
            ):
                logger.error(
                    "LLM response hit token limit. This may indicate the model is generating "
                    "excessively long responses. Error: %s",
                    error_msg,
                )
                # Return original segment as fallback
                return segment, {}
            elif "timeout" in error_msg.lower() or "Timeout" in str(type(e)):
                logger.error(
                    "LLM request timed out. Error: %s",
                    error_msg,
                )
                # Return original segment as fallback
                return segment, {}
            else:
                # Re-raise other exceptions
                raise

        # 直接从 response.content 获取响应内容
        summary_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # 获取 token usage（如果可用）
        token_usage = {}
        if hasattr(response, "response_metadata"):
            token_usage = response.response_metadata.get("token_usage", {})
            if token_usage:
                logger.debug(f"Token Usage: {token_usage}")
        else:
            logger.warning("Token usage not available in response_metadata")

        return summary_text, token_usage

    def _summarize_simple(
        self, content: str, completion_tokens: int, target_tokens: int
    ) -> Tuple[str, Dict]:
        """Mode 1: Simple mode - directly compress content without splitting.

        Args:
            content: Original content to summarize
            completion_tokens: Original completion tokens count
            target_tokens: Target token limit

        Returns:
            Tuple of (summarized_content, total_token_usage_dict)
        """
        if completion_tokens <= target_tokens:
            logger.info(
                f"Content already within token limit: {completion_tokens} <= {target_tokens}"
            )
            return content, {}

        # Calculate compression ratio
        compression_ratio = target_tokens / completion_tokens
        logger.info(
            f"Mode 1 (Simple): Summarizing content: {completion_tokens} tokens -> "
            f"{target_tokens} tokens (compression ratio: {compression_ratio:.2%})"
        )

        # Directly summarize the entire content
        summary, token_usage = self._summarize_segment(content, compression_ratio)

        # Check if we need further compression
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters for Chinese)
        estimated_tokens = len(summary) / 4

        max_iterations = 3
        iteration = 0
        total_token_usage = (
            token_usage.copy()
            if token_usage
            else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        )

        while estimated_tokens > target_tokens and iteration < max_iterations:
            iteration += 1
            logger.info(
                f"Iteration {iteration}: Estimated tokens {estimated_tokens:.0f} > "
                f"target {target_tokens}, further compressing..."
            )

            # Further compress with adjusted ratio
            remaining_ratio = target_tokens / estimated_tokens
            new_compression_ratio = compression_ratio * remaining_ratio

            summary, token_usage = self._summarize_segment(
                summary, new_compression_ratio
            )

            # Accumulate token usage
            if token_usage:
                total_token_usage["prompt_tokens"] += token_usage.get(
                    "prompt_tokens", 0
                )
                total_token_usage["completion_tokens"] += token_usage.get(
                    "completion_tokens", 0
                )
                total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

            estimated_tokens = len(summary) / 4

        logger.info(
            f"Mode 1 (Simple) complete: {completion_tokens} tokens -> "
            f"~{estimated_tokens:.0f} tokens (estimated)"
        )

        return summary, total_token_usage

    def _summarize_segment_mode(
        self, content: str, completion_tokens: int, target_tokens: int
    ) -> Tuple[str, Dict]:
        """Mode 2: Segment mode - split content into segments for large content.

        Args:
            content: Original content to summarize
            completion_tokens: Original completion tokens count
            target_tokens: Target token limit

        Returns:
            Tuple of (summarized_content, total_token_usage_dict)
        """
        if completion_tokens <= target_tokens:
            logger.info(
                f"Content already within token limit: {completion_tokens} <= {target_tokens}"
            )
            return content, {}

        # Calculate compression ratio
        compression_ratio = target_tokens / completion_tokens
        logger.info(
            f"Mode 2 (Segment): Summarizing content: {completion_tokens} tokens -> "
            f"{target_tokens} tokens (compression ratio: {compression_ratio:.2%})"
        )

        # Calculate number of segments needed
        # Use a heuristic: if compression ratio is very low, split into more segments
        if compression_ratio < 0.3:
            num_segments = max(2, int(1 / compression_ratio))
        elif compression_ratio < 0.5:
            num_segments = 2
        else:
            num_segments = 1

        logger.info(f"Dividing content into {num_segments} segments for summarization")

        # Split content by newlines
        paragraphs = self._split_by_newlines(content)

        if not paragraphs:
            logger.warning("No paragraphs found in content")
            return content, {}

        # Find split points
        split_points = self._find_split_points(paragraphs, num_segments)

        # Create segments
        segments = self._create_segments(paragraphs, split_points)

        logger.info(f"Created {len(segments)} segments for summarization")

        # Summarize each segment
        summarized_segments = []
        total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for i, segment in enumerate(segments):
            logger.debug(f"Summarizing segment {i + 1}/{len(segments)}")
            summary, token_usage = self._summarize_segment(segment, compression_ratio)

            # Accumulate token usage
            if token_usage:
                total_token_usage["prompt_tokens"] += token_usage.get(
                    "prompt_tokens", 0
                )
                total_token_usage["completion_tokens"] += token_usage.get(
                    "completion_tokens", 0
                )
                total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

            summarized_segments.append(summary)

        # Combine summarized segments
        summarized_content = "\n\n".join(summarized_segments)

        # Check if we need further compression
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters for Chinese)
        estimated_tokens = len(summarized_content) / 4

        max_iterations = 3
        iteration = 0

        while estimated_tokens > target_tokens and iteration < max_iterations:
            iteration += 1
            logger.info(
                f"Iteration {iteration}: Estimated tokens {estimated_tokens:.0f} > "
                f"target {target_tokens}, further compressing..."
            )

            # Further compress with adjusted ratio
            remaining_ratio = target_tokens / estimated_tokens
            new_compression_ratio = compression_ratio * remaining_ratio

            summary, token_usage = self._summarize_segment(
                summarized_content, new_compression_ratio
            )

            # Accumulate token usage
            if token_usage:
                total_token_usage["prompt_tokens"] += token_usage.get(
                    "prompt_tokens", 0
                )
                total_token_usage["completion_tokens"] += token_usage.get(
                    "completion_tokens", 0
                )
                total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

            summarized_content = summary
            estimated_tokens = len(summarized_content) / 4

        logger.info(
            f"Mode 2 (Segment) complete: {completion_tokens} tokens -> "
            f"~{estimated_tokens:.0f} tokens (estimated)"
        )

        return summarized_content, total_token_usage

    def summarize(
        self, content: str, completion_tokens: int, target_tokens: int
    ) -> Tuple[str, Dict]:
        """Summarize content to meet token limit.

        Supports three modes:
        - SIMPLE (Mode 1): Direct compression without splitting
        - SEGMENT (Mode 2): Split into segments for large content
        - AUTO (Mode 3): Try simple first, fallback to segment on error (default)

        Args:
            content: Original content to summarize
            completion_tokens: Original completion tokens count
            target_tokens: Target token limit

        Returns:
            Tuple of (summarized_content, total_token_usage_dict)
        """
        if completion_tokens <= target_tokens:
            logger.info(
                f"Content already within token limit: {completion_tokens} <= {target_tokens}"
            )
            return content, {}

        # Mode 3: Auto mode - try simple first, fallback to segment on error
        if self.mode == SummaryMode.AUTO:
            logger.info("Mode 3 (Auto): Attempting simple mode first...")
            try:
                return self._summarize_simple(content, completion_tokens, target_tokens)
            except Exception as e:
                logger.warning(
                    f"Mode 1 (Simple) failed: {e}. Falling back to Mode 2 (Segment)..."
                )
                return self._summarize_segment_mode(
                    content, completion_tokens, target_tokens
                )

        # Mode 1: Simple mode
        elif self.mode == SummaryMode.SIMPLE:
            return self._summarize_simple(content, completion_tokens, target_tokens)

        # Mode 2: Segment mode
        elif self.mode == SummaryMode.SEGMENT:
            return self._summarize_segment_mode(
                content, completion_tokens, target_tokens
            )

        else:
            raise ValueError(f"Unknown summary mode: {self.mode}")
