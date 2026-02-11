# OpenCode Integration Mode

This document describes how to integrate AskAny with OpenCode using MCP (Model Context Protocol) to leverage OpenCode's native grep and local file search capabilities alongside AskAny's RAG functionality.

## Overview

OpenCode integration mode provides a GUI-based interface through OpenCode's web interface, combining:
- **OpenCode's native capabilities**: Powerful grep and local file search tools
- **AskAny's RAG capabilities**: Via MCP server integration using `@.mcp.json` and `@askany_mcp`

This mode has been tested and proven to be the **most recommended solution** compared to existing deployment methods, as it:
- Retains MCP capabilities through the MCP server
- Leverages OpenCode's powerful native grep and tool call features
- Provides excellent compatibility and functionality as OpenCode is a popular open-source agent

## Prerequisites

1. **OpenCode Repository**: Clone the custom OpenCode fork
2. **AskAny MCP Server**: The `askany_mcp` module must be properly configured
3. **Network Access**: Both services need to be accessible on your local network

## Setup Instructions

### 1. Clone OpenCode Repository

```bash
git clone https://github.com/wufei-png/opencode.git
cd opencode
git checkout wf/dev  # Use the custom branch
```

### 2. Build and Deploy OpenCode

OpenCode needs to be built and deployed on your local network. Use two terminals:

**Terminal 1: Start the server** (listening on your network IP)

```bash
# Replace 10.202.47.43 with your actual network IP address
bun dev serve --hostname 10.202.47.43 --port 4096
```

**Terminal 2: Start the web app** (listening on your network IP)

```bash
# Replace 10.202.47.43 with your actual network IP address
VITE_HOSTNAME=10.202.47.43 bun run --cwd packages/app dev
```

This will build and deploy OpenCode on your local network, making it accessible via the web interface.

### 3. Configure Folder Permissions

OpenCode's local startup has permission issues that may expose all files in the root directory. To restrict access:

Edit `~/.config/opencode/opencode.json` and add:

```json
{
  "allowed_folders": [
    "/path/to/your/allowed/folder1",
    "/path/to/your/allowed/folder2"
  ]
}
```

This restricts OpenCode to only access specified folders, improving security.

### 4. Start AskAny MCP Server

Start the AskAny MCP FastAPI server:

```bash
cd /path/to/AskAny
uv run python -m askany_mcp.server_fastapi
```

The server will start and be ready to accept MCP connections.

### 5. Configure OpenCode MCP Integration

Add the AskAny MCP server to OpenCode's configuration file `~/.config/opencode/opencode.json`:

```json
{
  "mcpServers": {
    "askany_mcp": {
      "type": "remote",
      "url": "http://your-server-ip:port/sse",
      "enabled": true
    }
  },
  "allowed_folders": [
    "/path/to/your/allowed/folders"
  ]
}
```

Replace `your-server-ip:port` with the actual IP and port where `askany_mcp.server_fastapi` is running.

### 6. Test the Integration

Use the test client to verify the MCP server is working:

```bash
cd /path/to/AskAny
python askany_mcp/test_fastapi_client.py
```

## Usage

Once configured, you can:

1. **Access OpenCode Web Interface**: Open your browser and navigate to the OpenCode web interface (typically `http://your-ip:port`)

2. **Use OpenCode Native Tools**: Leverage OpenCode's powerful grep and local file search capabilities directly in the GUI

3. **Use AskAny RAG via MCP**: Reference `@.mcp.json` and `@askany_mcp` in your queries to access AskAny's RAG search capabilities

4. **Combined Workflow**: Use both OpenCode's native tools and AskAny's RAG in the same session for comprehensive code and documentation search

## Advantages

This integration mode offers several advantages:

- ✅ **Best of Both Worlds**: Combines OpenCode's native grep/tool capabilities with AskAny's RAG
- ✅ **GUI Interface**: Web-based interface for easier interaction
- ✅ **MCP Protocol**: Standardized protocol ensures compatibility
- ✅ **Permission Control**: Folder-level access control for security
- ✅ **Tested Solution**: Proven to work reliably in production environments
- ✅ **Open Source**: Both OpenCode and AskAny are open source

## Troubleshooting

### MCP Server Not Connecting

- Verify the MCP server is running: `uv run python -m askany_mcp.server_fastapi`
- Check the URL in `~/.config/opencode/opencode.json` matches the server address
- Ensure network connectivity between OpenCode and the MCP server

### Permission Issues

- Check `allowed_folders` in `~/.config/opencode/opencode.json`
- Ensure paths are absolute and accessible
- Restart OpenCode after configuration changes

### Build Issues

- Ensure Node.js and Bun are properly installed
- Check network IP addresses are correct
- Verify ports are not already in use

## Related Files

- `askany_mcp/server_fastapi.py` - FastAPI MCP server implementation
- `askany_mcp/test_fastapi_client.py` - Test client for MCP server
- `.mcp.json` - Project-level MCP configuration (for reference)

## References

- [OpenCode Repository](https://github.com/wufei-png/opencode/tree/wf/dev)
- [AskAny MCP Documentation](askany_mcp/README.md)
