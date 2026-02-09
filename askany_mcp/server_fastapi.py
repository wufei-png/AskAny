#!/usr/bin/env python3
"""FastAPI-based HTTP MCP server for AskAny RAG system.

This server exposes RAG search capabilities via HTTP/SSE for remote access.
"""

import sys
import os
import logging
from typing import Any
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("askany-mcp-fastapi")

# Suppress verbose output
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
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
            if hasattr(router.faq_query_engine, 'retrieve_with_scores'):
                nodes, top_score = router.faq_query_engine.retrieve_with_scores(
                    cleaned_query, metadata_filters
                )
                if top_score < settings.docs_similarity_threshold:
                    nodes = router.docs_query_engine.retrieve(cleaned_query, metadata_filters)
            else:
                nodes = router.faq_query_engine.retrieve(cleaned_query, metadata_filters)
                if not nodes:
                    nodes = router.docs_query_engine.retrieve(cleaned_query, metadata_filters)
        else:
            nodes = router.docs_query_engine.retrieve(cleaned_query, metadata_filters)

    # Filter by threshold
    filtered_nodes = [
        node for node in nodes
        if (node.score if hasattr(node, "score") and node.score else 0.0) >= settings.docs_similarity_threshold
    ]

    # Format results
    results = []
    for node in filtered_nodes:
        content = node.node.get_content() if hasattr(node.node, "get_content") else node.node.text
        file_path = node.node.metadata.get("file_path") or node.node.metadata.get("source") or "unknown"

        results.append({
            "content": content,
            "score": node.score if hasattr(node, "score") else None,
            "file_path": file_path,
            "start_line": node.node.metadata.get("start_line"),
            "end_line": node.node.metadata.get("end_line"),
        })

    return results


# Create FastAPI app
app = FastAPI(title="AskAny RAG MCP Server", version="0.1.0")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/tools/rag_search")
async def rag_search_tool(request: Request):
    """MCP tool: RAG search endpoint."""
    try:
        body = await request.json()
        query = body.get("query")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required parameter: query"}
            )

        results = rag_search_query(query, "auto")

        if not results:
            return {"results": [], "message": "No results found"}

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            lines = ""
            if result["start_line"] and result["end_line"]:
                lines = f" (lines {result['start_line']}-{result['end_line']})"

            formatted_results.append({
                "index": i,
                "source": f"{result['file_path']}{lines}",
                "content": result["content"],
                "score": result["score"]
            })

        return {"results": formatted_results}

    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="AskAny RAG MCP FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()

    logger.info(f"Starting AskAny RAG MCP FastAPI server on {args.host}:{args.port}")
    logger.info(f"API endpoint: http://{args.host}:{args.port}/tools/rag_search")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
