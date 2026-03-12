#!/usr/bin/env python3
"""
Crucible entry point — routes to HTTP or MCP server based on SERVER_MODE env var.

SERVER_MODE=http  → FastAPI HTTP server (Zone 3 pre-processing, voice notes)
SERVER_MODE=mcp   → FastMCP STDIO server (Zone 5a tool, on-demand transcription)
"""

import os
import sys

def main():
    mode = os.environ.get("SERVER_MODE", "http").lower()

    if mode == "http":
        from http_server import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
    elif mode == "mcp":
        from mcp_server import mcp
        mcp.run()
    else:
        print(f"Unknown SERVER_MODE: {mode}. Use 'http' or 'mcp'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
