"""MCP-over-streamable-http mount: routing, shared auth, shared Services."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import anyio
import pytest
from litestar import Litestar
from litestar.middleware.base import DefineMiddleware
from litestar.testing import AsyncTestClient
from mcp.server.mcpserver import MCPServer

from lilbee.app.endpoints import MCP_PATH
from lilbee.app.services import set_services
from lilbee.server import auth as auth_mod
from lilbee.server import mcp_mount
from lilbee.server.auth import AuthMiddleware
from lilbee.server.mcp_mount import build_mcp_mount

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_ACCEPT = "application/json, text/event-stream"


def _mounted_server(monkeypatch) -> MCPServer:
    """Build a mount and hand back the server it mounted."""
    built: list[MCPServer] = []
    real = mcp_mount.build_mcp_server

    def _spy() -> MCPServer:
        server = real()
        built.append(server)
        return server

    monkeypatch.setattr(mcp_mount, "build_mcp_server", _spy)
    build_mcp_mount()
    return built[0]


def test_configures_localhost_transport_security(monkeypatch) -> None:
    # The streamable-http options are app-factory arguments rather than server
    # state, so this reads the session manager they landed on, not the call.
    # The SDK's own default already allows loopback over http with rebinding
    # protection on, so only the https origins tell the two apart.
    server = _mounted_server(monkeypatch)
    security = server.session_manager.security_settings
    assert security is not None
    assert security.enable_dns_rebinding_protection
    assert "127.0.0.1:*" in security.allowed_hosts
    assert "https://127.0.0.1:*" in security.allowed_origins


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


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::"])
def test_a_wildcard_bind_warns_that_mcp_stays_loopback_only(monkeypatch, caplog, wildcard) -> None:
    """The REST API on the same port serves LAN clients fine, so an operator
    who deliberately exposed the daemon otherwise gets a silently half-working
    server: only /mcp fails, with an opaque transport-security rejection."""
    from lilbee.core.config import cfg
    from lilbee.server.mcp_mount import _transport_security

    monkeypatch.setattr(cfg, "server_host", wildcard)
    with caplog.at_level("WARNING"):
        _transport_security()
    assert "/mcp" in caplog.text
    assert wildcard in caplog.text


def test_handler_mounts_at_mcp_path() -> None:
    handler, _ = build_mcp_mount()
    assert MCP_PATH in handler.paths


def test_fresh_session_manager_per_build(monkeypatch) -> None:
    """run() is single-use per manager, so a second app in the same process
    needs its own or its lifespan raises."""
    first = _mounted_server(monkeypatch)
    second = _mounted_server(monkeypatch)
    assert first.session_manager is not second.session_manager


def test_mount_is_stateful(monkeypatch) -> None:
    # Stateful sessions carry clientInfo across requests, which memory owner
    # derivation reads. Stateless serving is a future scale-out deployment
    # mode, not a flag on this one.
    server = _mounted_server(monkeypatch)
    assert server.session_manager.stateless is False


async def test_two_mounts_in_one_process_both_start() -> None:
    """Several apps per process is the normal test-suite shape, and the CLI can
    rebuild one. Both lifespans must enter."""
    _, first = build_mcp_mount()
    _, second = build_mcp_mount()
    app = MagicMock(spec=Litestar)
    async with first(app), second(app):
        pass


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
    resp = await client.post(MCP_PATH, json=body, headers=_bearer(token))
    assert resp.status_code == _HTTP_OK
    session_id = resp.headers["mcp-session-id"]
    await client.post(
        MCP_PATH,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={**_bearer(token), "mcp-session-id": session_id},
    )
    return session_id


async def _call(
    client: AsyncTestClient, token: str, session_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    resp = await client.post(
        MCP_PATH,
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
            MCP_PATH,
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
        resp = await client.post(MCP_PATH, json=body, headers=_bearer(auth_token))
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


_HTTP_ACCEPTED = 202
_HTTP_NOT_FOUND = 404
_HTTP_METHOD_NOT_ALLOWED = 405
_SSE_TIMEOUT_S = 10
_INITIALIZE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "legacy-test", "version": "0"},
    },
}


class _RawSseGet:
    """Drives a GET against the app at the raw ASGI level.

    The test client buffers whole response bodies, which never terminates on
    an SSE stream, so the legacy-transport tests read ASGI messages directly.
    """

    def __init__(self, app: Litestar, token: str) -> None:
        self._app = app
        self._token = token
        send, recv = anyio.create_memory_object_stream[dict](100)
        self._send_stream = send
        self._recv_stream = recv
        self._buffer = ""
        self.start_message: dict | None = None

    async def run(self) -> None:
        """Run the request until cancelled, forwarding ASGI messages to the test."""
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": MCP_PATH,
            "raw_path": MCP_PATH.encode(),
            "query_string": b"",
            "root_path": "",
            "server": ("127.0.0.1", 8000),
            "client": ("127.0.0.1", 51234),
            "headers": [
                (b"host", b"127.0.0.1:8000"),
                (b"accept", b"text/event-stream"),
                (b"authorization", f"Bearer {self._token}".encode()),
            ],
        }

        async def receive() -> dict:
            await anyio.sleep_forever()
            raise AssertionError("unreachable")

        async def send(message: dict) -> None:
            await self._send_stream.send(message)

        await self._app(scope, receive, send)

    async def next_event(self) -> tuple[str, str]:
        """Return the next (event, data) SSE frame, reading body chunks as needed."""
        with anyio.fail_after(_SSE_TIMEOUT_S):
            while True:
                frame, _, rest = self._buffer.partition("\n\n")
                if rest or (self._buffer.endswith("\n\n") and frame):
                    self._buffer = rest
                    event = data = ""
                    for line in frame.splitlines():
                        if line.startswith("event:"):
                            event = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data = line[len("data:") :].strip()
                    if data:
                        return event, data
                    continue
                message = await self._recv_stream.receive()
                if message["type"] == "http.response.start":
                    self.start_message = message
                elif message["type"] == "http.response.body":
                    # The SDK emits CRLF line endings; normalize for parsing.
                    self._buffer += message["body"].decode().replace("\r\n", "\n")


async def test_sessionless_get_opens_the_legacy_sse_stream(
    mcp_app: Litestar, auth_token: str
) -> None:
    """A streamable-http first contact that fails makes the client fall back to
    the legacy HTTP+SSE transport: a sessionless GET expecting an SSE stream
    whose first event names the message endpoint."""
    async with _client(mcp_app):  # enters the session-manager lifespan
        sse = _RawSseGet(mcp_app, auth_token)
        async with anyio.create_task_group() as tg:
            tg.start_soon(sse.run)
            event, endpoint = await sse.next_event()
            tg.cancel_scope.cancel()
    assert sse.start_message is not None
    assert sse.start_message["status"] == _HTTP_OK
    headers = dict(sse.start_message["headers"])
    assert headers[b"content-type"].startswith(b"text/event-stream")
    assert event == "endpoint"
    assert endpoint.startswith(f"{MCP_PATH}/messages/?session_id=")


async def test_legacy_sse_fallback_completes_initialize(mcp_app: Litestar, auth_token: str) -> None:
    """The legacy transport round-trips: initialize POSTed to the advertised
    endpoint is accepted and answered over the open SSE stream."""
    async with _client(mcp_app) as client:
        sse = _RawSseGet(mcp_app, auth_token)
        async with anyio.create_task_group() as tg:
            tg.start_soon(sse.run)
            _, endpoint = await sse.next_event()
            resp = await client.post(
                endpoint,
                json=_INITIALIZE_BODY,
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            assert resp.status_code == _HTTP_ACCEPTED
            event, data = await sse.next_event()
            tg.cancel_scope.cancel()
    assert event == "message"
    assert json.loads(data)["result"]["serverInfo"]["name"] == "lilbee"


async def test_sessionless_get_without_sse_accept_is_405(
    mcp_app: Litestar, auth_token: str
) -> None:
    """A sessionless GET that does not ask for SSE has no stream to serve; the
    spec's answer is 405, not the SDK's 400 that reads as a broken server."""
    async with _client(mcp_app) as client:
        resp = await client.get(MCP_PATH, headers={"Authorization": f"Bearer {auth_token}"})
    assert resp.status_code == _HTTP_METHOD_NOT_ALLOWED


async def test_get_with_a_session_id_is_the_streamable_stream_not_the_legacy_one(
    mcp_app: Litestar, auth_token: str
) -> None:
    """A GET carrying a session ID is the streamable transport's standalone
    stream: it reaches the streamable app (which rejects the unknown session),
    never the legacy SSE handshake or the 405 guard."""
    async with _client(mcp_app) as client:
        resp = await client.get(
            MCP_PATH,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Accept": "text/event-stream",
                "mcp-session-id": "0" * 32,
            },
        )
    assert resp.status_code == _HTTP_NOT_FOUND


async def test_non_http_scopes_route_to_the_streamable_app() -> None:
    """The dispatcher inspects only http scopes; anything else goes straight to
    the streamable app, which closes an unmatched websocket."""
    handler, _ = build_mcp_mount()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "websocket", "path": "/", "headers": [], "query_string": b""}
    await handler.fn(scope, receive, send)
    assert sent[-1]["type"] == "websocket.close"


async def test_bare_mount_root_path_is_normalized_to_slash() -> None:
    """Some ASGI servers hand the mount root through as an empty path; the
    dispatcher treats it exactly like "/" (here: the 405 guard answers)."""
    handler, _ = build_mcp_mount()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "", "headers": [], "query_string": b""}
    await handler.fn(scope, receive, send)
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == _HTTP_METHOD_NOT_ALLOWED


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
