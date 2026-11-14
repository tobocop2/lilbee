"""A chat stream that a stopping server cuts ends with the stop error event."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Iterator

import pytest

from lilbee.app.services import answer_provider_waiters, set_server_exit_hook
from lilbee.server.chat_dispatch.concurrency import ChatSlotGuard
from lilbee.server.routes.search import _gated_stream

_STOP_MESSAGE = "The server is stopping."


@pytest.fixture(autouse=True)
def _serving() -> Iterator[None]:
    """Arm a serving run, so the stop state starts clear, and clear it afterwards."""
    set_server_exit_hook(lambda _status: None)
    yield
    set_server_exit_hook(None)


async def _tokens() -> AsyncGenerator[str, None]:
    yield "event: token\ndata: one\n\n"
    await asyncio.Event().wait()


async def _cut_after_first_token() -> list[str]:
    """Cancel the response task as the web server does once its graceful wait ends."""
    frames: list[str] = []

    async def consume() -> None:
        async for frame in _gated_stream(_tokens(), ChatSlotGuard()):
            frames.append(frame)

    task = asyncio.create_task(consume())
    while not frames:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return frames


async def test_a_cut_by_a_stopping_server_ends_with_the_stop_error() -> None:
    answer_provider_waiters()

    frames = await _cut_after_first_token()

    assert frames[0].startswith("event: token")
    assert len(frames) == 2
    last = frames[1]
    assert last.startswith("event: error")
    data = next(line for line in last.splitlines() if line.startswith("data: "))
    assert json.loads(data.removeprefix("data: "))["message"] == _STOP_MESSAGE


async def test_a_client_disconnect_on_a_running_server_adds_no_event() -> None:
    frames = await _cut_after_first_token()

    assert len(frames) == 1
    assert frames[0].startswith("event: token")
