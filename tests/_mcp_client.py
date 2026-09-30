"""A real MCP ``ClientSession`` connected to the lilbee server over the SDK's memory transport."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio
from mcp.shared.memory import create_client_server_memory_streams

from lilbee.mcp_server import build_mcp_server
from mcp import ClientSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


@asynccontextmanager
async def mcp_client() -> AsyncIterator[tuple[ClientSession, Any, Callable[[], Awaitable[None]]]]:
    """A connected client session, its initialize result, and a call that drops the connection.

    Dropping closes the client's send stream, so the server reads end of stream
    with no ``notifications/cancelled`` for the calls still in flight, and returns
    once the server has stopped and every handler has unwound.
    """
    lowlevel = build_mcp_server()._lowlevel_server
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):
        served = anyio.Event()

        async def _serve() -> None:
            await lowlevel.run(
                server_streams[0],
                server_streams[1],
                lowlevel.create_initialization_options(),
                raise_exceptions=False,
            )
            served.set()

        async def _drop() -> None:
            await client_streams[1].aclose()
            await served.wait()

        tg.start_soon(_serve)
        async with ClientSession(client_streams[0], client_streams[1]) as session:
            yield session, await session.initialize(), _drop
        tg.cancel_scope.cancel()
