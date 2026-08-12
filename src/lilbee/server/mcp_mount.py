"""Mount the MCP tool server over streamable-http on the Litestar daemon."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, cast

from litestar.handlers import asgi
from litestar.types import ASGIApp, Receive, Scope, Send
from mcp.server.transport_security import TransportSecuritySettings

from lilbee.app.endpoints import MCP_PATH
from lilbee.core.config import cfg
from lilbee.mcp_server import build_mcp_server, set_http_mounted

if TYPE_CHECKING:
    from litestar import Litestar
    from litestar.handlers import ASGIRouteHandler

log = logging.getLogger(__name__)

_Lifespan = Callable[["Litestar"], AbstractAsyncContextManager[None]]

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
# Wildcard binds cannot be enumerated for a Host allowlist. Sentinels for
# comparison, not bind addresses.
_WILDCARD_BINDS = ("0.0.0.0", "::")  # noqa: S104


def _fmt_host(host: str) -> str:
    """Bracket an IPv6 literal so it matches Host/Origin header syntax."""
    return f"[{host}]" if ":" in host else host


def _transport_security() -> TransportSecuritySettings:
    """DNS-rebinding allowlist scoped to the configured bind host.

    Defaults to loopback (the usual bind). When the daemon is bound to a
    specific non-loopback host, that host is added so the mount does not
    fail closed and reject every request. A wildcard bind (0.0.0.0 / ::)
    cannot be enumerated, so only loopback is allowed there.
    """
    hosts = [f"{_fmt_host(h)}:*" for h in _LOOPBACK_HOSTS]
    origins = [
        f"{scheme}://{_fmt_host(h)}:*" for h in _LOOPBACK_HOSTS for scheme in ("http", "https")
    ]
    bind = cfg.server_host
    if bind in _WILDCARD_BINDS:
        # The REST API on this port serves LAN clients fine, so without this
        # only /mcp fails, with an opaque transport-security rejection.
        log.warning(
            "Bound to %s, which cannot be enumerated for a Host allowlist, so %s "
            "accepts loopback Host headers only. Bind to a specific address to "
            "reach the MCP endpoint from other machines.",
            bind,
            MCP_PATH,
        )
    elif bind and bind not in _LOOPBACK_HOSTS:
        hosts.append(f"{_fmt_host(bind)}:*")
        origins.extend(f"{scheme}://{_fmt_host(bind)}:*" for scheme in ("http", "https"))
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


def build_mcp_mount() -> tuple[ASGIRouteHandler, _Lifespan]:
    """Return the MCP route handler and the session-manager lifespan.

    Each mount builds its own MCP server: the SDK caches a single session
    manager per server and ``run()`` is single-use, so a shared server's
    second lifespan would raise.
    """
    # Mark MCP as served over the shared HTTP daemon so single-vault-only tools
    # (init, reset) refuse runtime vault-switch / teardown that would race
    # concurrent in-flight handlers on the process-global Services singleton.
    set_http_mounted(True)
    server = build_mcp_server()
    asgi_app = cast(
        "ASGIApp",
        server.streamable_http_app(
            streamable_http_path="/",
            transport_security=_transport_security(),
        ),
    )
    manager = server.session_manager

    async def _forward(scope: Scope, receive: Receive, send: Send) -> None:
        await asgi_app(scope, receive, send)

    handler = asgi(MCP_PATH, is_mount=True, copy_scope=True)(_forward)

    @asynccontextmanager
    async def _session_lifespan(app: Litestar) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return handler, _session_lifespan
