"""Web search tool using proxyless-llm-websearch API."""

import logging
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from llama_index.core.schema import NodeWithScore, TextNode

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from askany.config import settings
from askany.workflow.SummaryFromLlm import SummaryFromLlm

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Tool for searching the web using proxyless-llm-websearch API."""

    def __init__(
        self,
        max_results: int = 5,
        api_url: Optional[str] = None,
        engine: str = "bing",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        rerank_top_k: Optional[int] = None,
        timeout: int = settings.llm_timeout,
    ):
        """Initialize WebSearchTool.

        Args:
            max_results: Maximum number of search results to return (used for rerank_top_k)
            api_url: URL of the web search API endpoint. Defaults to settings.web_search_api_url (configurable in config.py or .env)
            engine: Search engine to use (bing, quark, baidu, sougou). Defaults to "bing"
            chunk_size: Chunk size for text splitting. Defaults to 512
            chunk_overlap: Chunk overlap for text splitting. Defaults to 128
            rerank_top_k: Top K results after reranking. Defaults to None (uses max_results)
            timeout: Request timeout in seconds. Defaults to 30
        """
        self.max_results = max_results
        self.api_url = api_url or settings.web_search_api_url
        self.engine = engine
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Use max_results if rerank_top_k is not explicitly provided
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else max_results
        self.timeout = timeout
        self.use_proxy = True
        # Initialize summary generator for token limit handling
        self.summary_generator = SummaryFromLlm()

    def search(self, query: str) -> List[NodeWithScore]:
        """Search the web for the given query.

        Args:
            query: Search query string

        Returns:
            List of NodeWithScore objects containing search results
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to web search")
            return []

        logger.debug("Web search started - query: %s, engine: %s", query, self.engine)

        # Prepare request data
        data = {
            "question": query,
            "engine": self.engine,
            "split": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            "rerank": {"top_k": self.rerank_top_k},
        }

        try:
            # Make API request
            # For localhost connections, always disable proxy to avoid connection issues
            # Check if API URL is localhost or 127.0.0.1
            parsed_url = urlparse(self.api_url)
            is_localhost = parsed_url.hostname in ("localhost", "127.0.0.1", "::1")

            # If use_proxy is False or target is localhost, disable proxy
            if self.use_proxy and not is_localhost:
                response = requests.post(self.api_url, json=data, timeout=self.timeout)
            else:
                # Disable proxy completely by creating a session with trust_env=False
                # This ignores all environment variables (http_proxy, https_proxy, all_proxy)
                session = requests.Session()
                session.trust_env = False
                response = session.post(
                    self.api_url,
                    json=data,
                    timeout=self.timeout,
                    proxies={"http": None, "https": None, "all": None},
                )

            if response.status_code != 200:
                logger.error(
                    "Web search API request failed - status_code: %d, response: %s",
                    response.status_code,
                    response.text,
                )
                return []

            # Parse response
            result = response.json()
            if "data" not in result:
                logger.error(
                    "Web search API response missing 'data' field - response: %s",
                    result,
                )
                return []

            answer_text = result["data"]
            if not answer_text or not answer_text.strip():
                logger.warning("Web search API returned empty answer")
                return []

            # Extract token usage information
            token_usage = result.get("token_usage", {})
            completion_tokens = token_usage.get("completion_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)

            logger.info(
                f"Web search token usage - Prompt: {prompt_tokens}, "
                f"Completion: {completion_tokens}, Total: {total_tokens}"
            )

            # Check if completion tokens exceed the limit
            if completion_tokens > 0 and completion_tokens > settings.web_chat_tokens:
                logger.warning(
                    f"Completion tokens ({completion_tokens}) exceed limit "
                    f"({settings.web_chat_tokens}), summarizing content..."
                )

                # Initialize summary generator if not already initialized
                if self.summary_generator is None:
                    self.summary_generator = SummaryFromLlm()

                # Summarize the content
                summarized_text, summary_token_usage = self.summary_generator.summarize(
                    content=answer_text,
                    completion_tokens=completion_tokens,
                    target_tokens=settings.web_chat_tokens,
                )

                logger.info(
                    f"Content summarized - Original: {completion_tokens} tokens, "
                    f"Target: {settings.web_chat_tokens} tokens"
                )

                if summary_token_usage:
                    logger.info(
                        f"Summary token usage - Prompt: {summary_token_usage.get('prompt_tokens', 0)}, "
                        f"Completion: {summary_token_usage.get('completion_tokens', 0)}, "
                        f"Total: {summary_token_usage.get('total_tokens', 0)}"
                    )

                # Use summarized text
                answer_text = summarized_text

            logger.debug("Web search completed - answer length: %d", len(answer_text))

            # Convert answer to NodeWithScore
            # Since the API returns a final answer string, we create a single node
            # If the API later returns structured results, we can parse them here
            node = TextNode(
                text=answer_text,
                metadata={
                    "source": f"web_search_{self.engine}",
                    "type": "web_search",
                    "engine": self.engine,
                    "query": query,
                },
            )

            # Use a default score for web search results
            # The API has already done reranking, so we use a high score
            nodes = [NodeWithScore(node=node, score=1)]

            logger.debug("Web search nodes created - count: %d", len(nodes))
            return nodes

        except requests.exceptions.Timeout:
            logger.error(
                "Web search API request timed out after %d seconds", self.timeout
            )
            return []
        except requests.exceptions.ConnectionError as e:
            logger.error(
                "Web search API connection error - url: %s, error: %s",
                self.api_url,
                str(e),
            )
            return []
        except requests.exceptions.RequestException as e:
            logger.error("Web search API request exception: %s", str(e))
            return []
        except Exception as e:
            logger.error(
                "Unexpected error during web search: %s", str(e), exc_info=True
            )
            return []


if __name__ == "__main__":
    # Configure logger with handler for console output
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    print("root")
    web_search_tool = WebSearchTool()
    nodes = web_search_tool.search("中美的关系现在如何？")
    print("root2")
    for node in nodes:
        logger.info(node.node.text)
        logger.info(node.node.metadata)
        logger.info(node.score)
        logger.info("-" * 100)
    # nodes = web_search_tool.search(
    #     "https://xueqiu.com/8244815919/327993547 在这一文中的机器人技术中，与美国合作的厂家中有什么特别的吗"
    # )
    # for node in nodes:
    #     logger.info(node.node.text)
    #     logger.info(node.node.metadata)
    #     logger.info(node.score)
    #     logger.info("-" * 100)
