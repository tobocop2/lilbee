"""Tests for the shared chat-generation admission gate."""

from __future__ import annotations

import asyncio
import time

import pytest

from lilbee.server.chat_dispatch.concurrency import (
    ChatBusyError,
    acquire_chat_slot_or_busy,
    chat_gate,
    release_chat_slot,
)


@pytest.fixture(autouse=True)
def _reset_gate():
    """Fresh gate per test so in-flight counts never leak between tests."""
    chat_gate.cache_clear()
    yield
    chat_gate.cache_clear()


def test_chat_gate_is_singleton() -> None:
    assert chat_gate() is chat_gate()


async def test_admits_immediately_when_free() -> None:
    await acquire_chat_slot_or_busy(1, timeout=0.5)
    assert chat_gate().in_flight == 1
    await release_chat_slot()
    assert chat_gate().in_flight == 0


async def test_admits_up_to_capacity_concurrently() -> None:
    """Capacity N lets N run at once; the N+1th has to wait (here it 429s)."""
    await acquire_chat_slot_or_busy(2, timeout=0.5)
    await acquire_chat_slot_or_busy(2, timeout=0.5)
    assert chat_gate().in_flight == 2
    with pytest.raises(ChatBusyError):
        await acquire_chat_slot_or_busy(2, timeout=0.05)
    await release_chat_slot()
    await release_chat_slot()


async def test_waits_then_succeeds_when_a_slot_frees() -> None:
    await acquire_chat_slot_or_busy(1, timeout=0.5)  # fill the single slot

    async def _release_after(delay: float) -> None:
        await asyncio.sleep(delay)
        await release_chat_slot()

    task = asyncio.create_task(_release_after(0.05))
    start = time.monotonic()
    await acquire_chat_slot_or_busy(1, timeout=1.0)
    elapsed = time.monotonic() - start
    await task
    assert chat_gate().in_flight == 1
    assert 0.04 < elapsed < 0.5
    await release_chat_slot()


async def test_raises_after_timeout_when_full() -> None:
    await acquire_chat_slot_or_busy(1, timeout=0.5)
    with pytest.raises(ChatBusyError):
        await acquire_chat_slot_or_busy(1, timeout=0.05)
    assert chat_gate().in_flight == 1
    await release_chat_slot()


async def test_raises_immediately_with_zero_timeout_when_full() -> None:
    """A zero timeout is already expired on entry, so a full gate raises without
    ever waiting (the deadline-already-passed branch, not the wait_for timeout)."""
    await acquire_chat_slot_or_busy(1, timeout=0.5)
    with pytest.raises(ChatBusyError):
        await acquire_chat_slot_or_busy(1, timeout=0.0)
    assert chat_gate().in_flight == 1
    await release_chat_slot()


async def test_capacity_floor_is_one() -> None:
    """A bogus capacity of 0 is clamped to 1, never 'no slots at all'."""
    await acquire_chat_slot_or_busy(0, timeout=0.5)
    assert chat_gate().in_flight == 1
    with pytest.raises(ChatBusyError):
        await acquire_chat_slot_or_busy(0, timeout=0.05)
    await release_chat_slot()


async def test_chat_busy_error_inherits_from_exception() -> None:
    """The translation layers catch ``ChatBusyError`` to emit each protocol's 429
    envelope; it must be a normal Exception so the existing handler chain catches it.
    """
    assert issubclass(ChatBusyError, Exception)


async def test_slot_guard_releases_exactly_once() -> None:
    """Multiple cleanup paths share one guard; only the first release frees the slot."""
    from lilbee.server.chat_dispatch.concurrency import ChatSlotGuard

    await acquire_chat_slot_or_busy(2, timeout=0.5)
    await acquire_chat_slot_or_busy(2, timeout=0.5)
    guard = ChatSlotGuard()
    assert guard.released is False
    await guard.release()
    assert guard.released is True
    assert chat_gate().in_flight == 1
    await guard.release()
    assert chat_gate().in_flight == 1
    await release_chat_slot()
