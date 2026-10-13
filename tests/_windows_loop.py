"""The Windows proactor-loop scope for a test that needs a loop the selector loop cannot give."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from lilbee.runtime import asyncio_loop


@contextmanager
def windows_proactor_loop() -> Iterator[None]:
    """On Windows, run the block's crawl loops on the proactor loop, which can start Chromium.

    Covers loops made with ``asyncio.new_event_loop()`` inside the block and the TUI's
    background loop, which is made here so a test's own loop policy cannot replace it.
    """
    if sys.platform != "win32":
        yield
        return
    previous = asyncio.get_event_loop_policy()
    asyncio_loop.shutdown()
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio_loop.get_loop()
        yield
    finally:
        asyncio_loop.shutdown()
        asyncio.set_event_loop_policy(previous)
