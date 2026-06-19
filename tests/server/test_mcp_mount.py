"""MCP-over-streamable-http mount: routing, shared auth, shared Services."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from litestar import Litestar
from litestar.middleware.base import DefineMiddleware
from litestar.testing import AsyncTestClient

from lilbee.app.services import set_services
from lilbee.mcp_server import mcp
from lilbee.server import auth as auth_mod
from lilbee.server.auth import AuthMiddleware
from lilbee.server.mcp_mount import MCP_MOUNT_PATH, build_mcp_mount

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_ACCEPT = "application/json, text/event-stream"


def test_configures_localhost_transport_security() -> None:
    build_mcp_mount()
    assert mcp.settings.streamable_http_path == "/"
    security = mcp.settings.transport_security
    assert security is not None
    assert security.enable_dns_rebinding_protection
    assert "127.0.0.1:*" in security.allowed_hosts


def test_transport_security_includes_configured_bind_host(monkeypatch) -> None:
    from lilbee.core.config import cfg
    from lilbee.server.mcp_mount import _transport_security

    monkeypatch.setattr(cfg, "server_host", "192.168.1.50")
    security = _transport_security()
    assert "192.168.1.50:*" in security.allowed_hosts
    assert "http://192.168.1.50:*" in security.allowed_origins
    # Loopback stays allowed as the default.
    assert "127.0.0.1:*" in security.allowed_hosts


def test_transport_security_brackets_ipv6_bind_host(monkeypatch) -> None:
    from lilbee.core.config import cfg
    from lilbee.server.mcp_mount import _transport_security

    monkeypatch.setattr(cfg, "server_host", "fd00::1")
    security = _transport_security()
    assert "[fd00::1]:*" in security.allowed_hosts
    assert "http://[fd00::1]:*" in security.allowed_origins
    # IPv6 loopback is bracketed and allowed by default too.
    assert "[::1]:*" in security.allowed_hosts


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::"])
def test_transport_security_loopback_only_for_wildcard_bind(monkeypatch, wildcard) -> None:
    from lilbee.core.config import cfg
    from lilbee.server.mcp_mount import _fmt_host, _transport_security

    monkeypatch.setattr(cfg, "server_host", wildcard)
    security = _transport_security()
    # The wildcard bind itself is never added as its own allowlist entry.
    assert f"{_fmt_host(wildcard)}:*" not in security.allowed_hosts
    assert "127.0.0.1:*" in security.allowed_hosts


def test_handler_mounts_at_mcp_path() -> None:
    handler, _ = build_mcp_mount()
    assert MCP_MOUNT_PATH in handler.paths


def test_fresh_session_manager_per_build() -> None:
    build_mcp_mount()
    first = mcp.session_manager
    build_mcp_mount()
    second = mcp.session_manager
    assert first is not second


@pytest.fixture
def auth_token():
    previous = auth_mod.session_manager.token
    previous_init = auth_mod.session_manager._initialized
    auth_mod.session_manager.token = "mcp-token-" + "x" * 40
    auth_mod.session_manager._initialized = True
    yield auth_mod.session_manager.token
    auth_mod.session_manager.token = previous
    auth_mod.session_manager._initialized = previous_init


@pytest.fixture
def mcp_app() -> Litestar:
    handler, lifespan = build_mcp_mount()
    return Litestar(
        route_handlers=[handler],
        middleware=[DefineMiddleware(AuthMiddleware)],
        lifespan=[lifespan],
    )


def _client(app: Litestar) -> AsyncTestClient:
    # Host must match the daemon's localhost transport-security allow-list.
    return AsyncTestClient(app, base_url="http://127.0.0.1:8000")


def _parse_sse(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:") :].strip())
    raise AssertionError(f"no SSE data frame in response: {text!r}")


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": _ACCEPT}


async def _initialize(client: AsyncTestClient, token: str) -> str:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }
    resp = await client.post(MCP_MOUNT_PATH, json=body, headers=_bearer(token))
    assert resp.status_code == _HTTP_OK
    session_id = resp.headers["mcp-session-id"]
    await client.post(
        MCP_MOUNT_PATH,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={**_bearer(token), "mcp-session-id": session_id},
    )
    return session_id


async def _call(
    client: AsyncTestClient, token: str, session_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    resp = await client.post(
        MCP_MOUNT_PATH,
        json=payload,
        headers={**_bearer(token), "mcp-session-id": session_id},
    )
    assert resp.status_code == _HTTP_OK
    return _parse_sse(resp.text)


async def test_initialize_requires_the_same_bearer_token(
    mcp_app: Litestar, auth_token: str
) -> None:
    """No bearer token is rejected by the shared AuthMiddleware."""
    async with _client(mcp_app) as client:
        resp = await client.post(
            MCP_MOUNT_PATH,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"Accept": _ACCEPT},
        )
    assert resp.status_code == _HTTP_UNAUTHORIZED


async def test_initialize_succeeds_with_the_session_token(
    mcp_app: Litestar, auth_token: str
) -> None:
    async with _client(mcp_app) as client:
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        }
        resp = await client.post(MCP_MOUNT_PATH, json=body, headers=_bearer(auth_token))
    assert resp.status_code == _HTTP_OK
    assert _parse_sse(resp.text)["result"]["serverInfo"]["name"] == "lilbee"


async def test_tools_list_exposes_search_over_http(mcp_app: Litestar, auth_token: str) -> None:
    async with _client(mcp_app) as client:
        session_id = await _initialize(client, auth_token)
        result = await _call(
            client, auth_token, session_id, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        )
    names = {tool["name"] for tool in result["result"]["tools"]}
    assert "search" in names


async def test_search_tool_uses_the_shared_services_singleton(
    mcp_app: Litestar, auth_token: str
) -> None:
    """A tool call over http reads the same get_services() the REST routes use."""
    from tests.conftest import make_mock_services

    services = make_mock_services()
    services.searcher.search = MagicMock(return_value=[])
    set_services(services)
    try:
        async with _client(mcp_app) as client:
            session_id = await _initialize(client, auth_token)
            await _call(
                client,
                auth_token,
                session_id,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "search", "arguments": {"query": "hello"}},
                },
            )
    finally:
        set_services(None)
    services.searcher.search.assert_called_once()
    assert services.searcher.search.call_args.args[0] == "hello"
