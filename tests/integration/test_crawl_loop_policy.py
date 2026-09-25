"""The Windows proactor-loop scope that the browser-mode crawl tests run their crawls in."""

from __future__ import annotations

import asyncio
import sys

import pytest

from lilbee.runtime import asyncio_loop
from tests.integration._crawl_site import windows_proactor_loop


class _StubProactorPolicy(asyncio.DefaultEventLoopPolicy):
    """Stands in for ``WindowsProactorEventLoopPolicy`` and records the loops it makes."""

    made: list[asyncio.AbstractEventLoop]

    def __init__(self) -> None:
        super().__init__()
        self.made = []

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        loop = super().new_event_loop()
        self.made.append(loop)
        return loop


@pytest.fixture
def fresh_background_loop():
    asyncio_loop.shutdown()
    yield
    asyncio_loop.shutdown()


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubProactorPolicy:
    """Pretend to be Windows, with a recording stand-in for the proactor policy."""
    policy = _StubProactorPolicy()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(asyncio, "WindowsProactorEventLoopPolicy", lambda: policy, raising=False)
    return policy


def test_off_windows_the_policy_and_background_loop_are_left_alone(
    monkeypatch: pytest.MonkeyPatch, fresh_background_loop: None
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    policy = asyncio.get_event_loop_policy()
    loop = asyncio_loop.get_loop()
    with windows_proactor_loop():
        assert asyncio.get_event_loop_policy() is policy
        assert asyncio_loop.get_loop() is loop
    assert asyncio.get_event_loop_policy() is policy
    assert asyncio_loop.get_loop() is loop


def test_on_windows_the_block_runs_on_the_proactor_policy_and_restores_the_previous_one(
    stub: _StubProactorPolicy, fresh_background_loop: None
) -> None:
    previous = asyncio.get_event_loop_policy()
    loop_before = asyncio_loop.get_loop()
    with windows_proactor_loop():
        assert asyncio.get_event_loop_policy() is stub
        background = asyncio_loop.get_loop()
        assert stub.made == [background]
        assert background is not loop_before
        assert loop_before.is_closed()
    assert asyncio.get_event_loop_policy() is previous
    assert background.is_closed()
    assert asyncio_loop.get_loop() not in stub.made


def test_on_windows_the_background_loop_survives_a_policy_swap_inside_the_block(
    stub: _StubProactorPolicy, fresh_background_loop: None
) -> None:
    previous = asyncio.get_event_loop_policy()
    with windows_proactor_loop():
        asyncio.set_event_loop_policy(previous)
        try:
            assert asyncio_loop.get_loop() in stub.made
        finally:
            asyncio.set_event_loop_policy(stub)
