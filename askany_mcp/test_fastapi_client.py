#!/usr/bin/env python3
"""Test client for AskAny MCP FastAPI server using MCP Python SDK."""

import asyncio
import logging
from mcp import ClientSession
from mcp.client.sse import sse_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test-client")


async def test_mcp_server():
    """Test MCP server via SSE transport."""
    server_url = "http://localhost:8001/sse"
    
    logger.info(f"Connecting to MCP server at {server_url}")
    
    try:
        # Create SSE client - returns (read_stream, write_stream)
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                logger.info("Initializing MCP session...")
                await session.initialize()
                
                # List available tools
                logger.info("Listing available tools...")
                tools = await session.list_tools()
                logger.info(f"Available tools: {[tool.name for tool in tools.tools]}")
                
                # Test RAG search
                test_queries = [
                    "acl error code = 507011",
                    "方舟部署",
                ]
                
                for query in test_queries:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Testing query: {query}")
                    logger.info(f"{'='*60}")
                    
                    try:
                        # Call the rag_search tool
                        result = await session.call_tool(
                            "rag_search",
                            arguments={"query": query}
                        )
                        
                        # Print results
                        if result.content:
                            for content in result.content:
                                if hasattr(content, "text"):
                                    print(f"\n{content.text}")
                                else:
                                    print(f"\n{content}")
                        else:
                            logger.warning("No content returned")
                            
                    except Exception as e:
                        logger.error(f"Error calling tool: {e}", exc_info=True)
                
                logger.info("\n✓ Test completed successfully")
                
    except Exception as e:
        logger.error(f"Error connecting to server: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
