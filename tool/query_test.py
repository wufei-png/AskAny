#!/usr/bin/env python3
"""Test query functionality for different query types (AUTO, FAQ, DOCS, CODE) via HTTP API."""

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse
import time

# Add parent directory to path to import askany modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)

try:
    import requests
except ImportError:
    print(
        "❌ Error: requests library not found. Please install it: pip install requests"
    )
    sys.exit(1)

from askany.config import settings
from askany.rag.router import QueryType


def load_faq_data(faq_json_path: str):
    """Load FAQ data from JSON file."""
    try:
        with open(faq_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading FAQ data: {e}")
        return []


def send_query(
    base_url: str,
    query: str,
    query_type: QueryType,
    description: str = "",
    model: str = "gpt-3.5-turbo",
    messages: list = None,
):
    """Send a query to the API server.

    Args:
        base_url: Base URL of the API server (e.g., "http://localhost:8000")
        query: User query text
        query_type: Type of query (AUTO, FAQ, DOCS, CODE)
        description: Description of the test
        model: Model name to use in the request
        messages: Optional list of previous messages for multi-turn conversations.
                  If provided, will be used as conversation history.
                  Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Returns:
        Response text or None if error
    """
    print("\n" + "=" * 80)
    if description:
        print(f"Test: {description}")
    print(f"Query Type: {query_type.value.upper()}")
    print(f"Query: {query}")
    print("=" * 80)

    # Build messages
    if messages is None:
        messages = []
        messages.append({"role": "user", "content": query})
    else:
        # Use provided messages as history (make a copy to avoid modifying original)
        messages = messages.copy()

    # Build request payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": settings.temperature,
    }

    # Send request
    url = f"{base_url}/v1/chat/completions"
    try:
        print(f"\nSending request to: {url}")
        print(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        # For localhost connections, disable proxy to avoid connection issues
        parsed_url = urlparse(url)
        is_localhost = parsed_url.hostname in ("localhost", "127.0.0.1", "::1")

        if is_localhost:
            # Disable proxy for localhost connections
            session = requests.Session()
            session.trust_env = False
            response = session.post(
                url,
                json=payload,
                proxies={"http": None, "https": None, "all": None},
            )
        else:
            response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        print("\n✅ Response received:")
        print(f"Response ID: {result.get('id', 'N/A')}")

        # Extract response text from choices
        if "choices" in result and len(result["choices"]) > 0:
            response_text = result["choices"][0]["message"]["content"]
            print(f"\nResponse Content:\n{response_text}\n")

            # Print usage if available
            if "usage" in result:
                usage = result["usage"]
                print(f"Usage: {usage}\n")

            return response_text
        else:
            print(f"⚠️  No choices in response: {result}\n")
            return None

    except requests.exceptions.RequestException as e:
        print(f"\n❌ HTTP Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                print(
                    f"Error details: {json.dumps(error_detail, ensure_ascii=False, indent=2)}"
                )
            except Exception:
                print(f"Error response: {e.response.text}")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback

        traceback.print_exc()
        return None


def check_server_health(base_url: str) -> bool:
    """Check if the server is running and healthy.

    Args:
        base_url: Base URL of the API server

    Returns:
        True if server is healthy, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        print(f"  - Unable to connect to {base_url}")
        print("  - Make sure the server is running and accessible")
        return False
    except requests.exceptions.Timeout as e:
        print(f"Timeout error: {e}")
        print(f"  - Server at {base_url} did not respond within 5 seconds")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"  - Server responded with status code: {e.response.status_code}")
        return False
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return False


def main():
    """Main function - CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test query functionality for different query types via HTTP API"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help=f"Base URL of the API server (default: http://localhost:{settings.api_port})",
    )
    parser.add_argument(
        "--faq-json",
        type=str,
        default="data/json/faq.json",
        help="Path to FAQ JSON file (default: data/json/faq.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model name to use in requests (default: gpt-3.5-turbo)",
    )

    args = parser.parse_args()

    # Determine server URL
    if args.server_url:
        base_url = args.server_url.rstrip("/")
    else:
        # Use localhost instead of 0.0.0.0 for client connections
        host = settings.api_host
        if host == "0.0.0.0":
            host = "localhost"
        base_url = "http://172.20.25.225:38081"

    # Check server health
    print("=" * 80)
    print("Checking server health...")
    print("=" * 80)
    if not check_server_health(base_url):
        print(f"❌ Server at {base_url} is not responding or not healthy.")
        print("Please make sure the server is running.")
        print("You can start it with: python -m askany.main --serve")
        return
    print(f"✅ Server at {base_url} is healthy")

    # Load FAQ data to construct FAQ query
    faq_data = load_faq_data(args.faq_json)
    faq_query = None
    if faq_data:
        # Use a question from FAQ data that should match
        # Pick a question that's likely to be in the ingested data
        # Prefer shorter, more specific questions
        for item in faq_data:
            if "question" in item and item["question"]:
                # Use a question that's likely to match well
                # "激活时间不够就告警" is a good test case
                if "激活时间" in item["question"]:
                    faq_query = item["question"]
                    break
        # If no matching question found, use the first one
        if not faq_query and faq_data:
            faq_query = faq_data[0].get("question", "")

    # If no FAQ query found, use a default
    if not faq_query:
        faq_query = "ips启动失败怎么办？"

    # Construct Docs query based on ingest_check.log content
    docs_query = "todo"
    # Test queries
    print("\n" + "=" * 80)
    print("Starting Query Tests")
    print("=" * 80)

    # Test 1: AUTO mode with FAQ query (should route to FAQ if score is high enough)
    # send_query(
    #     base_url,
    #     faq_query,
    #     QueryType.AUTO,
    #     description="AUTO mode with FAQ query (should try FAQ first)",
    #     model=args.model,
    # )

    # # Test 2: FAQ mode with FAQ query
    # send_query(
    #     base_url,
    #     faq_query,
    #     QueryType.FAQ,
    #     description="FAQ mode with FAQ query",
    #     model=args.model,
    # )

    # Test 3: AUTO mode with Docs query (should route to Docs)
    questions = [
        # "https://xueqiu.com/8244815919/327993547 在这一文中的机器人技术中，与美国合作的厂家中有什么特别的吗",
    ]
    for question in questions:
        print(f"Testing question: {question}")
        time_start = time.time()
        send_query(
            base_url,
            question,
            QueryType.AUTO,
            description="AUTO mode with Docs query (should route to Docs)",
            model=args.model,
        )
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
    # # Test 4: DOCS mode with Docs query
    # send_query(
    #     base_url,
    #     docs_query,
    #     QueryType.DOCS,
    #     description="DOCS mode with Docs query",
    #     model=args.model,
    # )

    # # Test 5: CODE mode (not implemented yet, but test it)
    # code_query = "如何查看代码中的错误处理逻辑？"
    # send_query(
    #     base_url,
    #     code_query,
    #     QueryType.CODE,
    #     description="CODE mode (may not be implemented yet)",
    #     model=args.model,
    # )

    print("\n" + "=" * 80)
    print("Query Tests Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
