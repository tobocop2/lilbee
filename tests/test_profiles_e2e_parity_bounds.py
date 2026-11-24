"""The TUI tests end a hang with a named failure, and their absence waits keep a floor."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import tests.test_profiles_e2e_parity as parity
from lilbee.cli.tui.screens.settings import SettingsScreen
from tests._async_wait import ABSENCE_PAUSES, ABSENCE_SECONDS

_PAUSE_SECONDS = 0.2


def test_the_timeout_ends_the_worker_instead_of_raising_in_the_loop(
    pytestconfig: pytest.Config,
) -> None:
    """A signal timeout is lost when a Textual task takes it, and the test waits on."""
    assert pytestconfig.getini("timeout_method") == "thread"


async def test_the_wait_for_an_offer_that_does_not_come_has_a_floor_and_an_end() -> None:
    """A pause lasts up to a second on a busy host: the wait keeps its pauses and its seconds."""
    pauses = 0

    async def _slow_pause() -> None:
        nonlocal pauses
        pauses += 1
        await asyncio.sleep(_PAUSE_SECONDS)

    pilot = SimpleNamespace(pause=_slow_pause)
    app = SimpleNamespace(screen=SettingsScreen())
    start = time.monotonic()
    owed = await parity._rebuild_owed_on_leaving(app, pilot)  # type: ignore[arg-type]
    elapsed = time.monotonic() - start
    assert owed is False
    assert pauses >= ABSENCE_PAUSES
    assert elapsed >= ABSENCE_SECONDS
    assert elapsed < 3.0, pauses
