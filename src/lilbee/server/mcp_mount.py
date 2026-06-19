"""Mount the FastMCP tool server over streamable-http on the Litestar daemon."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, cast

from litestar.handlers import asgi
from litestar.types import ASGIApp, Receive, Scope, Send
from mcp.server.transport_security import TransportSecuritySettings

from lilbee.core.config import cfg
from lilbee.mcp_server import mcp

if TYPE_CHECKING:
    from litestar import Litestar
    from litestar.handlers import ASGIRouteHandler

MCP_MOUNT_PATH = "/mcp"

_Lifespan = Callable[["Litestar"], AbstractAsyncContextManager[None]]

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
_WILDCARD_BIND = "0.0.0.0"  # noqa: S104 - sentinel for comparison, not a bind address


def _transport_security() -> TransportSecuritySettings:
    """DNS-rebinding allowlist scoped to the configured bind host.

    Defaults to loopback (the usual bind). When the daemon is bound to a
    specific non-loopback host, that host is added so the mount does not
    fail closed and reject every request. A wildcard bind (0.0.0.0) cannot
    be enumerated, so only loopback is allowed there.
    """
    hosts = [f"{h}:*" for h in _LOOPBACK_HOSTS]
    origins = [f"{scheme}://{h}:*" for h in _LOOPBACK_HOSTS for scheme in ("http", "https")]
    bind = cfg.server_host
    if bind and bind not in _LOOPBACK_HOSTS and bind != _WILDCARD_BIND:
        hosts.append(f"{bind}:*")
        origins.extend(f"{scheme}://{bind}:*" for scheme in ("http", "https"))
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
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
    mcp.settings.transport_security = _transport_security()
    asgi_app = cast("ASGIApp", mcp.streamable_http_app())
    manager = mcp.session_manager

    async def _forward(scope: Scope, receive: Receive, send: Send) -> None:
        await asgi_app(scope, receive, send)

    handler = asgi(MCP_MOUNT_PATH, is_mount=True, copy_scope=True)(_forward)

    @asynccontextmanager
    async def _session_lifespan(app: Litestar) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return handler, _session_lifespan
