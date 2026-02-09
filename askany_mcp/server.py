#!/usr/bin/env python3
"""MCP server for AskAny RAG system.

This server exposes RAG search capabilities as MCP tools.
It directly queries the vector store without requiring the FastAPI server.
"""

import logging
from typing import Any
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("askany-mcp")

# Global variables for RAG components
router = None
embed_model = None
llm = None


def initialize_rag_components():
    """Initialize RAG components (router, embedding model, LLM)."""
    global router, embed_model, llm

    try:
        from askany.config import settings
        from askany.main import initialize_llm, get_device
        from askany.ingest import VectorStoreManager
        from askany.rag import create_query_router

        logger.info("Initializing RAG components...")

        # Initialize LLM and embedding model
        llm, embed_model = initialize_llm()
        device = get_device()

        # Initialize vector store manager
        vector_store_manager = VectorStoreManager(embed_model, llm=llm)

        # Initialize indexes
        try:
            vector_store_manager.initialize_faq_index()
            vector_store_manager.initialize_docs_index()
            logger.info("Initialized separate FAQ and docs indexes")
        except Exception as e:
            logger.warning(f"Separate indexes not available: {e}")
            vector_store_manager.initialize()

        # Create router
        router = create_query_router(vector_store_manager, llm, embed_model, device)
        logger.info("RAG components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
        raise


def rag_search_query(query: str, query_type: str = "auto") -> list[dict[str, Any]]:
    """Execute RAG search and return structured results.

    Args:
        query: Search query string
        query_type: Query type (auto/faq/docs)

    Returns:
        List of result dictionaries with content, score, file_path, line numbers
    """
    from askany.config import settings
    from askany.rag.router import QueryType
    from askany.rag.query_parser import parse_query_filters

    if router is None:
        raise RuntimeError("RAG components not initialized")

    # Parse query filters
    cleaned_query, metadata_filters = parse_query_filters(query)

    # Convert query_type string to QueryType enum
    if query_type.lower() == "faq":
        query_type_enum = QueryType.FAQ
    elif query_type.lower() == "docs":
        query_type_enum = QueryType.DOCS
    else:
        query_type_enum = QueryType.AUTO

    # Retrieve nodes from RAG (lines 189-207 from min_langchain_agent.py)
    if query_type_enum == QueryType.FAQ and router.faq_query_engine:
        logger.info("Using FAQ engine")
        nodes = router.faq_query_engine.retrieve(cleaned_query, metadata_filters)
    elif query_type_enum == QueryType.DOCS:
        logger.info("Using DOCS engine")
        # Debug: Check if docs_query_engine and its retriever exist
        logger.info(f"Docs query engine: {router.docs_query_engine}")
        logger.info(f"Docs query engine retriever: {router.docs_query_engine.query_engine.retriever if hasattr(router.docs_query_engine, 'query_engine') else None}")
        nodes = router.docs_query_engine.retrieve(cleaned_query, metadata_filters)
    else:
        # AUTO mode: try FAQ first, then docs
        logger.info("Using AUTO mode")
        if router.faq_query_engine:
            logger.info("FAQ engine exists")
            # Check if FAQ engine has retrieve_with_scores (FAQQueryEngine)
            if hasattr(router.faq_query_engine, 'retrieve_with_scores'):
                logger.info("FAQ engine has retrieve_with_scores, using it")
                logger.info(f"Query: {cleaned_query}")
                logger.info(f"Metadata filters: {metadata_filters}")
                nodes, top_score = router.faq_query_engine.retrieve_with_scores(
                    cleaned_query, metadata_filters
                )
                logger.info(f"FAQ retrieved {len(nodes)} nodes with top_score={top_score}")
                # If FAQ score is low, try docs instead
                if top_score < settings.docs_similarity_threshold:
                    logger.info(f"FAQ score {top_score} < threshold {settings.docs_similarity_threshold}, using docs")
                    # Debug: Check if docs_query_engine and its retriever exist
                    logger.info(f"Docs query engine: {router.docs_query_engine}")
                    logger.info(f"Docs query engine retriever: {router.docs_query_engine.query_engine.retriever if hasattr(router.docs_query_engine, 'query_engine') else None}")
                    nodes = router.docs_query_engine.retrieve(
                        cleaned_query, metadata_filters
                    )
                    logger.info(f"Docs retrieved {len(nodes)} nodes")
            else:
                # FAQ engine is RAGQueryEngine, just retrieve
                logger.info("FAQ engine is RAGQueryEngine, using it directly")
                nodes = router.faq_query_engine.retrieve(cleaned_query, metadata_filters)
                logger.info(f"FAQ retrieved {len(nodes)} nodes")
                # If FAQ returns no results, fall back to docs
                if not nodes:
                    logger.info("FAQ returned no results, falling back to docs")
                    # Debug: Check if docs_query_engine and its retriever exist
                    logger.info(f"Docs query engine: {router.docs_query_engine}")
                    logger.info(f"Docs query engine retriever: {router.docs_query_engine.query_engine.retriever if hasattr(router.docs_query_engine, 'query_engine') else None}")
                    nodes = router.docs_query_engine.retrieve(cleaned_query, metadata_filters)
                    logger.info(f"Docs retrieved {len(nodes)} nodes")
        else:
            # No FAQ engine, use docs
            logger.info("No FAQ engine, using docs")
            # Debug: Check if docs_query_engine and its retriever exist
            logger.info(f"Docs query engine: {router.docs_query_engine}")
            logger.info(f"Docs query engine retriever: {router.docs_query_engine.query_engine.retriever if hasattr(router.docs_query_engine, 'query_engine') else None}")
            nodes = router.docs_query_engine.retrieve(
                cleaned_query, metadata_filters
            )
            logger.info(f"Docs retrieved {len(nodes)} nodes")

    logger.info(f"Retrieved {len(nodes)} nodes before filtering")

    # Log scores of retrieved nodes
    if nodes:
        for i, node in enumerate(nodes[:3]):  # Log first 3 nodes
            score = node.score if hasattr(node, "score") and node.score else None
            logger.info(f"Node {i+1} score: {score}")

    # Filter low quality docs nodes based on similarity threshold
    filtered_nodes = []
    for node in nodes:
        score = node.score if hasattr(node, "score") and node.score else 0.0
        logger.debug(f"Node score: {score}, threshold: {settings.docs_similarity_threshold}")
        if score >= settings.docs_similarity_threshold:
            filtered_nodes.append(node)

    logger.info(f"Filtered {len(nodes)} nodes to {len(filtered_nodes)} nodes (threshold: {settings.docs_similarity_threshold})")
    nodes = filtered_nodes

    # Format nodes into structured results
    results = []
    for node in nodes:
        content = (
            node.node.get_content()
            if hasattr(node.node, "get_content")
            else node.node.text
        )

        file_path = (
            node.node.metadata.get("file_path")
            or node.node.metadata.get("source")
            or "unknown"
        )

        start_line = node.node.metadata.get("start_line")
        end_line = node.node.metadata.get("end_line")

        score = node.score if hasattr(node, "score") else None

        result = {
            "content": content,
            "score": score,
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
        }
        results.append(result)

    return results


# Create MCP server instance
server = Server("askany-rag")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="rag_search",
            description="Search local documents and FAQ using RAG (Retrieval-Augmented Generation). "
            "Returns relevant content with file paths and line numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name != "rag_search":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query")
    if not query:
        raise ValueError("Missing required argument: query")

    try:
        results = rag_search_query(query, "auto")

        if not results:
            return [
                TextContent(
                    type="text",
                    text="No results found for your query.",
                )
            ]

        # Format results as text
        formatted_results = []
        for i, result in enumerate(results, 1):
            lines = ""
            if result["start_line"] and result["end_line"]:
                lines = f" (lines {result['start_line']}-{result['end_line']})"

            formatted_results.append(
                f"[Result {i}]\n"
                f"Source: {result['file_path']}{lines}\n"
                f"Content:\n{result['content']}\n"
            )

        return [
            TextContent(
                type="text",
                text="\n---\n".join(formatted_results),
            )
        ]

    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return [
            TextContent(
                type="text",
                text=f"Error executing RAG search: {str(e)}",
            )
        ]


async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting AskAny RAG MCP server...")

    # Initialize RAG components
    initialize_rag_components()

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="askany-rag",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
