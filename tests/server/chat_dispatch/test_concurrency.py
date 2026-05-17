"""Tests for the shared chat-generation concurrency gate."""

from __future__ import annotations

import pytest

from lilbee.server.chat_dispatch.concurrency import (
    ChatBusyError,
    acquire_or_raise_busy,
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


async def test_acquire_or_raise_busy_when_held() -> None:
    lock = chat_lock()
    await lock.acquire()
    with pytest.raises(ChatBusyError):
        acquire_or_raise_busy()


async def test_acquire_or_raise_busy_noop_when_free() -> None:
    # No raise; subsequent acquire must succeed and not block.
    acquire_or_raise_busy()
    lock = chat_lock()
    await lock.acquire()
    assert lock.locked() is True


async def test_chat_busy_error_inherits_from_exception() -> None:
    # The translation layers catch ``ChatBusyError`` and emit each
    # protocol's 429 envelope; it must be a normal Exception, not
    # BaseException, so the existing handler chain catches it.
    assert issubclass(ChatBusyError, Exception)
