"""The guard schema plugin documents HTTP handlers and leaves other route kinds alone."""

from __future__ import annotations

from litestar import Router, WebSocket, get, websocket
from litestar.config.app import AppConfig
from litestar.exceptions import NotAuthorizedException
from litestar.routes import HTTPRoute, WebSocketRoute

from lilbee.server.auth import BEARER_SCHEME, GuardSchemaPlugin


@get("/ping")
async def _ping() -> str:
    return "pong"


@websocket("/feed")
async def _feed(socket: WebSocket) -> None:
    await socket.close()


def test_a_router_with_a_websocket_route_still_documents_its_http_handlers() -> None:
    router = Router(path="/api", route_handlers=[_ping, _feed])
    GuardSchemaPlugin().on_app_init(AppConfig(route_handlers=[router]))
    http = next(r for r in router.routes if isinstance(r, HTTPRoute))
    assert http.route_handlers
    for handler in http.route_handlers:
        assert {BEARER_SCHEME: []} in (handler.security or [])
        assert NotAuthorizedException in (handler.raises or [])
    assert any(isinstance(r, WebSocketRoute) for r in router.routes)
