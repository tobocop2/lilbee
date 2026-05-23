"""Mount the FastMCP tool server over streamable-http on the Litestar daemon."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

from litestar.handlers import asgi
from litestar.types import Receive, Scope, Send
from mcp.server.transport_security import TransportSecuritySettings

from lilbee.mcp_server import mcp

if TYPE_CHECKING:
    from litestar import Litestar
    from litestar.handlers import ASGIRouteHandler

MCP_MOUNT_PATH = "/mcp"

_Lifespan = Callable[["Litestar"], AbstractAsyncContextManager[None]]

# The daemon binds localhost; agents reach it on 127.0.0.1/localhost with a
# dynamic port. DNS-rebinding protection stays on, scoped to those hosts.
_TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
)


def build_mcp_mount() -> tuple[ASGIRouteHandler, _Lifespan]:
    """Return the MCP route handler and the session-manager lifespan.

    A fresh session manager is built per call: ``session_manager.run()`` can
    only be entered once per instance, and the app factory runs once per
    process in production but repeatedly across tests.
    """
    # FastMCP caches the session manager; clear it so each app owns its own,
    # since run() is single-use per instance.
    mcp._session_manager = None
    mcp.settings.streamable_http_path = "/"
    mcp.settings.transport_security = _TRANSPORT_SECURITY
    asgi_app = mcp.streamable_http_app()
    manager = mcp.session_manager

    async def _forward(scope: Scope, receive: Receive, send: Send) -> None:
        await asgi_app(scope, receive, send)

    handler = asgi(MCP_MOUNT_PATH, is_mount=True, copy_scope=True)(_forward)

    @asynccontextmanager
    async def _session_lifespan(app: Litestar) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return handler, _session_lifespan
