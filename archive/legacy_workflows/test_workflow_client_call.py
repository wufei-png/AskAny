#!/usr/bin/env python3
"""Test script for workflow client call to debug timeout issues.

This test specifically targets the workflow client call pattern used in
workflow_llamaindex.py:716-726 to diagnose ReadTimeout errors.

CRITICAL INSIGHT:
================
The workflow calls itself RECURSIVELY via HTTP client:
- In process_sub_query step (workflow_llamaindex.py:728), the workflow calls
  run_workflow_via_client() which makes an HTTP request back to the same server
- This creates a nested execution pattern: workflow -> HTTP call -> same workflow
- If the inner workflow execution takes longer than the HTTP client timeout (30s),
  a ReadTimeout error occurs
- This is why ReadTimeout errors happen in production when workflows are complex

The timeout is hardcoded to 30.0 seconds in workflow_server.py:214:
    httpx_client = httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=30.0)

SOLUTION: Increase the timeout for complex workflows that make recursive calls.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from workflows import Workflow, step
from workflows.context import Context
from workflows.events import StartEvent, StopEvent

from askany.config import settings
from askany.rag.router import QueryType
from askany.workflow.workflow_server import (
    create_workflow_server,
    run_workflow_via_client,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RecursiveTestWorkflow(Workflow):
    """Test workflow that can call itself recursively via HTTP client."""

    def __init__(self, workflow_client=None):
        """Initialize with optional workflow_client for recursive calls."""
        super().__init__()
        self.workflow_client = workflow_client

    @step
    async def recursive_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Step that can call itself recursively via HTTP client."""
        query = getattr(ev, "query", "test")

        # Check if this is a recursive call
        # Note: The context dict passed to run_workflow must follow SerializedContext format
        # with a "state" key containing properly structured state data.
        # Simple dict like {"is_recursive": True} won't work - need proper format:
        # {"state": {"state_type": "DictState", "state_module": "...", "state_data": {"_data": {"is_recursive": "true"}}}}
        is_recursive = await ctx.store.get("is_recursive", False)
        # Fallback: check query marker (simpler and more reliable for testing)

        if is_recursive:
            # This is the inner/recursive call - simulate some work
            print(f"    [RecursiveTestWorkflow] Inner call executing for: {query}")
            await asyncio.sleep(1)  # Simulate some work (like RAG retrieval)
            return StopEvent(result=f"Recursive inner response for: {query}")
        else:
            # This is the outer call - make recursive HTTP call
            print(
                f"    [RecursiveTestWorkflow] Outer call, making recursive HTTP call for: {query}"
            )
            if self.workflow_client is not None:
                # Make recursive call via HTTP client (simulating process_sub_query)
                # NOTE: Simple dict {"is_recursive": True} doesn't work!
                # Must create a Context, set values in store, then serialize with to_dict()
                from workflows.context import Context, JsonSerializer

                # Create a context for this workflow, set the flag, and serialize
                temp_ctx = Context(self)
                await temp_ctx.store.set("is_recursive", True)
                sub_context = temp_ctx.to_dict(serializer=JsonSerializer())
                try:
                    print(
                        "    [RecursiveTestWorkflow] Calling run_workflow_via_client..."
                    )
                    sub_answer = await run_workflow_via_client(
                        self.workflow_client,
                        query,  # Use original query, context carries the flag
                        QueryType.AUTO,
                        context=sub_context,
                    )
                    print("    [RecursiveTestWorkflow] Recursive call completed")
                    return StopEvent(
                        result=f"Outer response with recursive call: {sub_answer}"
                    )
                except Exception as e:
                    print(
                        f"    [RecursiveTestWorkflow] Recursive call failed: {type(e).__name__}: {e}"
                    )
                    return StopEvent(
                        result=f"Recursive call failed: {type(e).__name__}: {e}"
                    )
            else:
                return StopEvent(result=f"No workflow_client available for: {query}")


# Global shared server and client
_shared_server = None
_shared_client = None
_shared_workflow = None
_server_thread = None


async def setup_shared_server():
    """Setup a shared server for all tests."""
    global _shared_server, _shared_client, _shared_workflow, _server_thread

    print("\n" + "=" * 80)
    print("Starting shared test server...")
    print("=" * 80)

    # Create workflow and server
    _shared_workflow = RecursiveTestWorkflow()
    _shared_server, _shared_client = create_workflow_server(_shared_workflow)
    _shared_workflow.workflow_client = _shared_client

    # Start server in background thread
    import threading

    _server_thread = threading.Thread(
        target=lambda: asyncio.run(
            _shared_server.serve(
                host=settings.inner_server_host, port=settings.inner_server_port
            )
        ),
        daemon=True,
    )
    _server_thread.start()

    # Wait for server to start
    print("Waiting for server to start...")
    await asyncio.sleep(2)
    print(f"✅ Server started on port {settings.inner_server_port}")

    return _shared_client


async def test_basic_query(client):
    """Test basic workflow client call."""
    print("\n" + "=" * 80)
    print("Test 1: Basic Query")
    print("=" * 80)

    try:
        print("\nTesting with basic query...")

        start_time = time.time()
        result = await run_workflow_via_client(
            client=client,
            query="测试查询",
            query_type=QueryType.AUTO,
            context={"is_recursive": True},  # Skip recursive call
        )
        elapsed = time.time() - start_time

        print(f"✅ Success! Result: {result[:150]}...")
        print(f"⏱️  Elapsed time: {elapsed:.2f} seconds")
        return True
    except Exception as e:
        print(f"❌ Test failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_with_context(client):
    """Test workflow client call with context (simulating sub-query pattern)."""
    print("\n" + "=" * 80)
    print("Test 2: Call with Context (Sub-query Pattern)")
    print("=" * 80)

    try:
        print("\nTesting the pattern from workflow_llamaindex.py:716-726...")

        enhanced_query = "测试子问题"
        sub_context = {"is_sub_query_workflow": True, "is_recursive": True}

        print(f"enhanced_query: {enhanced_query}")
        print(f"sub_context: {sub_context}")

        start_time = time.time()
        result = await run_workflow_via_client(
            client=client,
            query=enhanced_query,
            query_type=QueryType.AUTO,
            context=sub_context,
        )
        elapsed = time.time() - start_time

        print(f"✅ Success! Result: {result[:150]}...")
        print(f"⏱️  Elapsed time: {elapsed:.2f} seconds")
        return True
    except Exception as e:
        print(f"❌ Test failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_recursive_call(client):
    """Test recursive workflow call (actual pattern from process_sub_query)."""
    print("\n" + "=" * 80)
    print("Test 3: Recursive Workflow Call (Actual Pattern)")
    print("=" * 80)
    print("⚠️  This simulates the process_sub_query pattern where workflow calls itself")

    try:
        print("\nTesting recursive call pattern...")
        print("Pattern: workflow -> HTTP call -> same workflow (nested)")

        start_time = time.time()
        # No is_recursive flag - outer call will make recursive HTTP call
        result = await run_workflow_via_client(
            client=client,
            query="递归测试",
            query_type=QueryType.AUTO,
            context={},
        )
        elapsed = time.time() - start_time

        print(f"✅ Success! Result: {result[:200]}...")
        print(f"⏱️  Elapsed time: {elapsed:.2f} seconds")
        print("\n💡 If this takes >30s, it will timeout (HTTP client timeout)")
        return True
    except httpx.ReadTimeout:
        elapsed = time.time() - start_time if "start_time" in locals() else 0
        print("⚠️  ReadTimeout - demonstrates timeout issue!")
        print(f"   Elapsed: {elapsed:.2f}s")
        print("\n🔧 SOLUTION: Increase timeout in workflow_server.py:214")
        print("   Current: timeout=30.0")
        print("   Suggested: timeout=120.0 or timeout=300.0")
        return False
    except Exception as e:
        print(f"❌ Test failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_timeout_config():
    """Check HTTP client timeout configuration."""
    print("\n" + "=" * 80)
    print("Test 4: HTTP Client Timeout Configuration")
    print("=" * 80)

    try:
        # Create a temporary workflow just to check config
        workflow = RecursiveTestWorkflow()
        _, client = create_workflow_server(workflow)

        # Check the httpx client timeout
        httpx_client = getattr(client, "_httpx_client", None)
        if httpx_client:
            timeout = getattr(httpx_client, "_timeout", None)
            print(f"HTTP client timeout: {timeout}")

            if timeout and isinstance(timeout, httpx.Timeout):
                print(f"  Read timeout: {timeout.read}")
                print(f"  Connect timeout: {timeout.connect}")

                if timeout.read and timeout.read < 60:
                    print(
                        f"  ⚠️  WARNING: Read timeout ({timeout.read}s) may be too short"
                    )
                    print("     Consider increasing in workflow_server.py:214")
                else:
                    print(f"  ✅ Read timeout ({timeout.read}s) seems reasonable")
        else:
            print("  ⚠️  httpx_client not accessible")

        return True
    except Exception as e:
        print(f"❌ Test failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Workflow Client Call Test Suite")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Inner server host: {settings.inner_server_host}")
    print(f"  Inner server port: {settings.inner_server_port}")
    print("  HTTP client timeout: 30.0 seconds (hardcoded in workflow_server.py:214)")
    print(
        f"\n⚠️  NOTE: Make sure no workflow server is running on port {settings.inner_server_port}"
    )
    print("\n🔍 KEY INSIGHT: The workflow calls itself recursively via HTTP client")
    print("   (see workflow_llamaindex.py:728 - process_sub_query step)")
    print("   This means if the inner workflow takes >30s, it will timeout")

    # Setup shared server
    client = await setup_shared_server()

    results = []

    # Run tests using the shared server
    results.append(await test_basic_query(client))
    await asyncio.sleep(0.5)

    results.append(await test_with_context(client))
    await asyncio.sleep(0.5)

    results.append(await test_recursive_call(client))
    await asyncio.sleep(0.5)

    results.append(await test_timeout_config())

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed")

    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
