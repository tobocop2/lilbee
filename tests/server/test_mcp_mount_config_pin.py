"""A tool called over a live HTTP MCP session reads the Config published when its POST ran.

The session's server task starts inside the first request (initialize). The MCP SDK
runs each message in the context of the POST that sent it, so a tool call reads that
POST's pin and not the pin of initialize. This guards against an SDK change.
"""

from __future__ import annotations

from typing import Any

from litestar import Litestar
from litestar.middleware.base import DefineMiddleware

from lilbee.app.endpoints import MCP_PATH
from lilbee.core.config import cfg
from lilbee.server import app as app_mod
from lilbee.server import mcp_mount
from lilbee.server.auth import AuthMiddleware
from tests.server.test_mcp_mount import _bearer, _client, _initialize, auth_token  # noqa: F401


class _NoCheck:
    async def take(self) -> None:
        return None


async def test_a_session_tool_reads_a_change_published_after_initialize(
    monkeypatch,
    auth_token: str,  # noqa: F811
) -> None:
    seen: list[int] = []
    built: list[Any] = []
    real = mcp_mount.build_mcp_server

    def _spy():
        server = real()
        built.append(server)
        return server

    monkeypatch.setattr(mcp_mount, "build_mcp_server", _spy)
    handler, lifespan = mcp_mount.build_mcp_mount()

    async def record(name: str, *_args: Any, **_kwargs: Any) -> str:
        seen.append(cfg.top_k)
        return "ran"

    monkeypatch.setattr(built[0]._tool_manager, "call_tool", record)
    reload_mw = app_mod._ConfigFileReload()
    reload_mw._follower = _NoCheck()
    app = Litestar(
        route_handlers=[handler],
        middleware=[DefineMiddleware(AuthMiddleware), reload_mw],
        lifespan=[lifespan],
        logging_config=None,
    )
    start = cfg.top_k
    try:
        cfg.top_k = 7
        async with _client(app) as client:
            session_id = await _initialize(client, auth_token)
            cfg.top_k = 11  # another client's write, published after initialize
            await client.post(
                MCP_PATH,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "search", "arguments": {"query": "x"}},
                },
                headers={**_bearer(auth_token), "mcp-session-id": session_id},
            )
    finally:
        cfg.top_k = start
    assert seen == [11]
