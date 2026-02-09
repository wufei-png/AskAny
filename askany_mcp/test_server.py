#!/usr/bin/env python3
"""Test script for AskAny RAG MCP server."""


def test_rag_search():
    """Test RAG search functionality."""
    from askany_mcp.server import initialize_rag_components, rag_search_query

    print("Initializing RAG components...")
    initialize_rag_components()
    print("✓ RAG components initialized\n")

    # Test queries
    test_queries = [
        "acl error code = 507011",
        "方舟部署",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)

        try:
            results = rag_search_query(query, "auto")

            if not results:
                print("No results found\n")
                continue

            for i, result in enumerate(results, 1):
                print(f"\n[Result {i}]")
                print(f"File: {result['file_path']}")
                if result['start_line'] and result['end_line']:
                    print(f"Lines: {result['start_line']}-{result['end_line']}")
                print(f"Content: {result['content'][:200]}...")
                print()

        except Exception as e:
            print(f"Error: {e}\n")

    print("✓ Test completed")


if __name__ == "__main__":
    test_rag_search()
