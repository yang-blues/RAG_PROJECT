"""Standalone Streamable HTTP entry point for the stock-mcp MCP server.

Run:
    python server.py                 # listens on 127.0.0.1:8000
    MCP_PORT=9000 python server.py
    MCP_HOST=0.0.0.0 MCP_PORT=8000 python server.py

Environment variables:
    MCP_HOST             bind address (default 127.0.0.1)
    MCP_PORT             bind port (default 8000)
    MCP_ALLOWED_HOSTS    comma-separated public hostnames to allow when the
                         server runs behind a reverse proxy (DNS-rebinding
                         protection), e.g. "mcp.example.com,mcp.example.com:*"

The MCP endpoint is served at  /mcp  (Streamable HTTP transport).
"""

import contextlib
import inspect
import os

import uvicorn


from mcp.server.fastmcp import FastMCP  # type: ignore
_FASTMCP_VERSION = 1

from stock_server import register


def _allowed_hosts():
    env = os.environ.get("MCP_ALLOWED_HOSTS", "").strip()
    hosts = [h.strip() for h in env.split(",") if h.strip()]
    hosts.extend(["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*", "[::1]:*"])
    return hosts


def build_app():
    """Build the ASGI (Starlette) app exposing the MCP server at /mcp."""
    _path_prefix, mcp = register()

    if _FASTMCP_VERSION == 2:
        params = inspect.signature(mcp.http_app).parameters
        if "host" in params and "allowed_hosts" not in params:
            return mcp.http_app(host="/mcp")
        if "allowed_hosts" in params:
            kwargs = {"allowed_hosts": _allowed_hosts()}
            if "host" in params:
                kwargs["host"] = "/mcp"
            return mcp.http_app(**kwargs)
        return mcp.http_app()

    # fastmcp v1: configure transport security then take streamable_http_app().
    try:
        from mcp.server.transport_security import TransportSecuritySettings
        settings = getattr(mcp, "settings", None)
        if settings is not None and hasattr(settings, "transport_security"):
            settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=_allowed_hosts(),
            )
    except Exception:
        pass

    app = mcp.streamable_http_app()

    # v1: the session manager lifecycle must be driven manually
    # (uvicorn does not run the mounted app's own lifespan here).
    session_manager = getattr(mcp, "session_manager", None)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        ctx = session_manager.run() if session_manager is not None else None
        if ctx is not None:
            await ctx.__aenter__()
        try:
            yield
        finally:
            if ctx is not None:
                await ctx.__aexit__(None, None, None)

    app.router.lifespan_context = lifespan
    return app


app = build_app()


def main():
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8001"))
    print("=" * 60)
    print(f"stock-mcp MCP server starting...")
    print(f"  MCP endpoint: http://{host}:{port}/mcp")
    print(f"  fastmcp version: {_FASTMCP_VERSION}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port, log_level="info")


mcp_name = 'stock-mcp'


if __name__ == "__main__":
    main()
