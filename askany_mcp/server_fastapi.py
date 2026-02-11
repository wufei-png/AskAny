#!/usr/bin/env python3
"""FastAPI-based HTTP MCP server for AskAny RAG system.

Uses the official MCP Python SDK (modelcontextprotocol) with FastAPI.
Exposes RAG search as MCP tools via SSE transport for remote access.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("askany-mcp-fastapi")

# Suppress verbose output
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from fastapi import FastAPI
from fastapi.responses import Response
from starlette.requests import Request
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
import uvicorn

# Global variables for RAG components
router = None
embed_model = None
llm = None
_initialization_error = None


def _ensure_initialized():
    """Ensure RAG components are initialized (lazy initialization)."""
    global router, embed_model, llm, _initialization_error

    if router is not None:
        return

    if _initialization_error is not None:
        raise _initialization_error

    try:
        from askany.config import settings
        from askany.main import initialize_llm, get_device
        from askany.ingest import VectorStoreManager
        from askany.rag import create_query_router

        logger.info("Initializing RAG components...")

        llm, embed_model = initialize_llm()
        device = get_device()

        vector_store_manager = VectorStoreManager(embed_model, llm=llm)

        try:
            vector_store_manager.initialize_faq_index()
            vector_store_manager.initialize_docs_index()
            logger.info("Initialized separate FAQ and docs indexes")
        except Exception as e:
            logger.warning(f"Separate indexes not available: {e}")
            vector_store_manager.initialize()

        router = create_query_router(vector_store_manager, llm, embed_model, device)

        logger.info("RAG components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
        _initialization_error = e
        raise


def rag_search_query(query: str, query_type: str = "auto") -> list[dict[str, Any]]:
    """Execute RAG search and return structured results."""
    _ensure_initialized()

    from askany.config import settings
    from askany.rag.router import QueryType
    from askany.rag.query_parser import parse_query_filters

    cleaned_query, metadata_filters = parse_query_filters(query)

    if query_type.lower() == "faq":
        query_type_enum = QueryType.FAQ
    elif query_type.lower() == "docs":
        query_type_enum = QueryType.DOCS
    else:
        query_type_enum = QueryType.AUTO

    # Retrieve nodes
    if query_type_enum == QueryType.FAQ and router.faq_query_engine:
        nodes = router.faq_query_engine.retrieve(cleaned_query, metadata_filters)
    elif query_type_enum == QueryType.DOCS:
        nodes = router.docs_query_engine.retrieve(cleaned_query, metadata_filters)
    else:
        # AUTO mode
        if router.faq_query_engine:
            if hasattr(router.faq_query_engine, "retrieve_with_scores"):
                nodes, top_score = router.faq_query_engine.retrieve_with_scores(
                    cleaned_query, metadata_filters
                )
                if top_score < settings.docs_similarity_threshold:
                    nodes = router.docs_query_engine.retrieve(
                        cleaned_query, metadata_filters
                    )
            else:
                nodes = router.faq_query_engine.retrieve(
                    cleaned_query, metadata_filters
                )
                if not nodes:
                    nodes = router.docs_query_engine.retrieve(
                        cleaned_query, metadata_filters
                    )
        else:
            nodes = router.docs_query_engine.retrieve(cleaned_query, metadata_filters)

    # Filter by threshold
    filtered_nodes = [
        node
        for node in nodes
        if (node.score if hasattr(node, "score") and node.score else 0.0)
        >= settings.docs_similarity_threshold
    ]

    # Format results
    results = []
    for node in filtered_nodes:
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

        results.append(
            {
                "content": content,
                "score": node.score if hasattr(node, "score") else None,
                "file_path": file_path,
                "start_line": node.node.metadata.get("start_line"),
                "end_line": node.node.metadata.get("end_line"),
            }
        )

    return results


# Create MCP server instance
server = Server("askany-rag")

# Create SSE transport (endpoint where clients POST messages)
sse_transport = SseServerTransport("/messages")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available MCP tools."""
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
    """Handle MCP tool calls."""
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


# ASGI app for POST messages (MCP protocol)
messages_app = sse_transport.handle_post_message


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("AskAny RAG MCP FastAPI server starting")
    yield
    logger.info("AskAny RAG MCP FastAPI server shutting down")


# Create FastAPI app

app = FastAPI(
    title="AskAny RAG MCP Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/sse")
async def handle_sse(request: Request):
    """MCP SSE endpoint - clients connect here to establish MCP session."""
    logger.info("New SSE connection")
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0],
            streams[1],
            InitializationOptions(
                server_name="askany-rag",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
    # Return empty response to avoid NoneType on client disconnect (per MCP SDK)
    return Response()


# Mount MCP messages endpoint for POST (client messages)
app.mount("/messages", messages_app)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AskAny RAG MCP FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=38081, help="Port to bind to")
    args = parser.parse_args()

    logger.info(f"Starting AskAny RAG MCP FastAPI server on {args.host}:{args.port}")
    logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")
    _ensure_initialized()  # Initialize RAG components
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
