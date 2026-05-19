"""Tests for the shared chat-generation concurrency gate."""

from __future__ import annotations

import asyncio
import time

import pytest

from lilbee.server.chat_dispatch.concurrency import (
    ChatBusyError,
    acquire_chat_lock_or_busy,
    chat_lock,
)


@pytest.fixture(autouse=True)
def _reset_lock():
    """Clear any held state so tests run in isolation."""
    chat_lock.cache_clear()
    yield
    lock = chat_lock()
    if lock.locked():
        lock.release()
    chat_lock.cache_clear()


def test_chat_lock_is_singleton() -> None:
    assert chat_lock() is chat_lock()


async def test_acquire_chat_lock_or_busy_immediate_when_free() -> None:
    """When the lock is free, acquire returns immediately with it held."""
    await acquire_chat_lock_or_busy(timeout=0.5)
    assert chat_lock().locked() is True


async def test_acquire_chat_lock_or_busy_waits_then_succeeds() -> None:
    """When the lock is held but released within timeout, acquire queues and
    proceeds; the previous immediate-429 behaviour is gone.
    """
    lock = chat_lock()
    await lock.acquire()

    async def _release_after(delay: float) -> None:
        await asyncio.sleep(delay)
        lock.release()

    release_task = asyncio.create_task(_release_after(0.05))
    start = time.monotonic()
    await acquire_chat_lock_or_busy(timeout=1.0)
    elapsed = time.monotonic() - start
    await release_task
    assert lock.locked() is True
    assert 0.04 < elapsed < 0.5


async def test_acquire_chat_lock_or_busy_raises_after_timeout() -> None:
    """A genuinely stuck previous request still surfaces a ``ChatBusyError``."""
    lock = chat_lock()
    await lock.acquire()
    with pytest.raises(ChatBusyError):
        await acquire_chat_lock_or_busy(timeout=0.05)
    assert lock.locked() is True


async def test_chat_busy_error_inherits_from_exception() -> None:
    """The translation layers catch ``ChatBusyError`` and emit each protocol's
    429 envelope; it must be a normal Exception so the existing handler chain
    catches it.
    """
    assert issubclass(ChatBusyError, Exception)
