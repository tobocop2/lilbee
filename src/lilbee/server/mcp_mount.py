"""Mount the MCP tool server over streamable-http on the Litestar daemon."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, cast

from litestar.handlers import asgi
from litestar.types import (
    ASGIApp,
    HTTPResponseBodyEvent,
    HTTPResponseStartEvent,
    HTTPScope,
    Receive,
    Scope,
    Send,
)
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


# Message-post path of the legacy HTTP+SSE transport, inside the mount.
_SSE_MESSAGE_PATH = "/messages/"


def _header(scope: HTTPScope, name: bytes) -> bytes | None:
    """Return the first value of *name* among the request headers, if any."""
    return next((v for k, v in scope["headers"] if k == name), None)


def _is_legacy_sse_contact(scope: HTTPScope) -> bool:
    """True for a sessionless GET asking for SSE at the mount root.

    That request is the legacy HTTP+SSE (2024-11-05) handshake, which clients
    fall back to when streamable-http first contact fails. A GET carrying a
    session ID is the streamable transport's own standalone stream instead.
    """
    if scope["method"] != "GET" or scope["path"] not in ("", "/"):
        return False
    if _header(scope, b"mcp-session-id") is not None:
        return False
    accept = _header(scope, b"accept") or b""
    return b"text/event-stream" in accept


async def _method_not_allowed(send: Send) -> None:
    """Answer 405: a sessionless GET with no SSE accept has no stream to serve."""
    start: HTTPResponseStartEvent = {
        "type": "http.response.start",
        "status": 405,
        "headers": [(b"allow", b"GET, POST, DELETE")],
    }
    body: HTTPResponseBodyEvent = {
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    }
    await send(start)
    await send(body)


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
    security = _transport_security()
    streamable_app = cast(
        "ASGIApp",
        server.streamable_http_app(
            streamable_http_path="/",
            transport_security=security,
        ),
    )
    # The SDK ships the two transports as separate apps and no combined mount,
    # so this dispatcher serves both at one endpoint per the spec's
    # backwards-compatibility flow: without the legacy app a falling-back
    # client meets a 400 and reports the server as disconnected.
    sse_app = cast(
        "ASGIApp",
        server.sse_app(
            sse_path="/",
            message_path=_SSE_MESSAGE_PATH,
            transport_security=security,
        ),
    )
    manager = server.session_manager

    async def _forward(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await streamable_app(scope, receive, send)
            return
        # The union does not narrow through the "type" comparison above.
        http_scope = cast("HTTPScope", scope)
        # Starlette route matching needs the bare mount root spelled "/".
        if http_scope["path"] == "":
            http_scope["path"] = "/"
        if http_scope["path"].startswith(_SSE_MESSAGE_PATH) or _is_legacy_sse_contact(http_scope):
            # The endpoint event advertises root_path + message path, and
            # the mount strips MCP_PATH from the path it forwards.
            http_scope["root_path"] = MCP_PATH
            await sse_app(scope, receive, send)
            return
        if (
            http_scope["method"] == "GET"
            and http_scope["path"] == "/"
            and _header(http_scope, b"mcp-session-id") is None
        ):
            await _method_not_allowed(send)
            return
        await streamable_app(scope, receive, send)

    handler = asgi(MCP_PATH, is_mount=True, copy_scope=True)(_forward)

    @asynccontextmanager
    async def _session_lifespan(app: Litestar) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return handler, _session_lifespan
