#!/usr/bin/env python3
"""Test script for WorkflowServer to diagnose 503 errors.

This script creates a simple test workflow, starts a WorkflowServer,
and tests all endpoints to diagnose issues.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from workflows import Workflow, step
from workflows.context import Context
from workflows.events import StartEvent, StopEvent
from workflows.server import WorkflowServer

from askany.config import settings


# Simple test workflow
class TestWorkflow(Workflow):
    """Simple test workflow for debugging."""

    @step
    async def greet(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Simple greeting step."""
        name = ev.get("name", "World")
        return StopEvent(result=f"Hello, {name}!")


def test_workflow_server():
    """Test WorkflowServer functionality."""
    print("=" * 80)
    print("WorkflowServer Test Script")
    print("=" * 80)

    # Configuration
    host = settings.inner_server_host
    port = settings.inner_server_port

    # For client connections, use localhost instead of 0.0.0.0
    if host == "0.0.0.0":
        client_host = "localhost"
    else:
        client_host = host

    base_url = f"http://{client_host}:{port}"
    bind_url = f"http://{host}:{port}"

    print("\n1. Configuration:")
    print(f"   Server bind: {bind_url}")
    print(f"   Client connect: {base_url}")
    print(f"   Port: {port}")

    # Create test workflow
    print("\n2. Creating test workflow...")
    test_workflow = TestWorkflow()
    print("   ✅ Test workflow created")

    # Create server
    print("\n3. Creating WorkflowServer...")
    server = WorkflowServer()
    print(f"   ✅ WorkflowServer created: {server}")

    # Add workflow
    print("\n4. Adding workflow to server...")
    server.add_workflow("test", test_workflow)
    print("   ✅ Workflow 'test' added")

    # Check registered workflows
    try:
        workflows = getattr(server, "_workflows", None)
        if workflows:
            print(f"   ✅ Registered workflows: {list(workflows.keys())}")
        else:
            print("   ⚠️  Could not inspect registered workflows")
    except Exception as e:
        print(f"   ⚠️  Error inspecting workflows: {e}")

    # Start server in background
    print("\n5. Starting server in background thread...")

    import threading

    def run_server():
        """Run server in event loop."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print(f"   [Thread] Starting server on {host}:{port}")
            loop.run_until_complete(server.serve(host=host, port=port))
        except Exception as e:
            print(f"   [Thread] ERROR: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    thread = threading.Thread(target=run_server, daemon=True, name="TestWorkflowServer")
    thread.start()
    print(f"   ✅ Thread started: {thread.name} (alive: {thread.is_alive()})")

    # Wait a bit for server to start
    print("\n6. Waiting for server to start...")
    time.sleep(2)

    if not thread.is_alive():
        print("   ❌ Thread died! Server may have crashed.")
        return False

    print(f"   ✅ Thread still alive: {thread.is_alive()}")

    # Test endpoints
    print("\n7. Testing endpoints...")

    # Check for proxy settings
    import os

    proxy_vars = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    proxies = {k: os.environ.get(k) for k in proxy_vars if os.environ.get(k)}
    if proxies:
        print("   ⚠️  Proxy environment variables detected:")
        for k, v in proxies.items():
            print(f"      {k}={v}")
    else:
        print("   ✅ No proxy environment variables")

    endpoints_to_test = [
        ("/health", "Health check"),
        ("/workflows", "List workflows"),  # Note: no trailing slash per deployment.md
    ]

    for endpoint, description in endpoints_to_test:
        url = f"{base_url}{endpoint}"
        print(f"\n   Testing {description}: {url}")
        try:
            # Create client without proxy for localhost connections
            # trust_env=False disables proxy from environment variables
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                response = client.get(url)
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ Response: {data}")
                except Exception:
                    print(f"   ✅ Response (text): {response.text[:200]}")
            else:
                print(
                    f"   ❌ Error response ({response.status_code}): {response.text[:200]}"
                )
                # Check if it's a proxy issue
                if "Proxy-Connection" in response.headers:
                    print(
                        "   ⚠️  Response contains Proxy-Connection header - may be going through proxy"
                    )
        except httpx.ConnectError as e:
            print(f"   ❌ Connection error: {e}")
            print(f"   💡 Try checking if port {port} is accessible")
        except httpx.TimeoutException as e:
            print(f"   ❌ Timeout: {e}")
        except httpx.HTTPStatusError as e:
            print(f"   ❌ HTTP error: {e.response.status_code}")
            print(f"   Response headers: {dict(e.response.headers)}")
            print(f"   Response body: {e.response.text[:200]}")
            # Check for proxy headers in error response
            if "Proxy-Connection" in e.response.headers:
                print("   ⚠️  Error response contains Proxy-Connection header")
        except Exception as e:
            print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    # Test running a workflow
    print("\n8. Testing workflow execution...")
    try:
        import json

        payload = {
            "start_event": {"name": "TestUser"},
            "context": {},
        }
        url = f"{base_url}/workflows/test/run"
        print(f"   POST {url}")
        print(f"   Payload: {json.dumps(payload, indent=2)}")

        # Use client without proxy for localhost
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            response = client.post(url, json=payload)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Workflow result: {result}")
        else:
            print(f"   ❌ Error: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Error running workflow: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Keep server running for a bit
    print("\n9. Server is running. Press Ctrl+C to stop...")
    print("   (Server will run for 3 seconds for testing)")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n   Interrupted by user")

    print("\n✅ Test completed")
    return True


if __name__ == "__main__":
    try:
        test_workflow_server()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
