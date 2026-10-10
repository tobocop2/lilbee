"""Tests for the bounded test waits."""

from __future__ import annotations

import asyncio
import time

import pytest

from tests._async_wait import wait_until


class _Pilot:
    """A pilot whose pause yields for a short, real interval."""

    async def pause(self) -> None:
        await asyncio.sleep(0.01)


def _true_after(seconds: float):
    deadline = time.monotonic() + seconds
    return lambda: time.monotonic() >= deadline


@pytest.mark.asyncio
async def test_wait_until_keeps_pumping_for_its_time_budget() -> None:
    assert await wait_until(_Pilot(), _true_after(0.2), max_pauses=1, timeout=5.0)


@pytest.mark.asyncio
async def test_wait_until_with_no_time_budget_stops_after_its_pauses() -> None:
    assert not await wait_until(_Pilot(), _true_after(5.0), max_pauses=1)
