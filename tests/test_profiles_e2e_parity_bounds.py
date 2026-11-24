"""The TUI tests end a hang with a named failure, and their absence waits keep a floor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tests._async_wait as async_wait
import tests.test_profiles_e2e_parity as parity
from lilbee.cli.tui.screens.settings import SettingsScreen


def test_the_timeout_ends_the_worker_instead_of_raising_in_the_loop(
    pytestconfig: pytest.Config,
) -> None:
    """A signal timeout is lost when a Textual task takes it, and the test waits on."""
    assert pytestconfig.getini("timeout_method") == "thread"


@pytest.mark.parametrize(
    ("pause_seconds", "pauses"),
    [
        pytest.param(2.0, 5, id="a-slow-pause-leaves-only-the-pause-floor"),
        pytest.param(0.125, 8, id="a-fast-pause-leaves-only-the-time-floor"),
    ],
)
async def test_the_wait_for_an_offer_that_does_not_come_keeps_both_floors(
    monkeypatch: pytest.MonkeyPatch, pause_seconds: float, pauses: int
) -> None:
    """A fake clock moves only in a pause: one floor at a time holds the wait.

    The wait ends at the later of 5 pauses and 1 second. A slow pause passes the second at the
    first pause, so 5 pauses hold the wait. A fast pause reaches 5 pauses before the second,
    so the second holds it, at 8 pauses of 0.125 s.
    """
    now = 0.0
    paused = 0

    async def _pause() -> None:
        nonlocal now, paused
        paused += 1
        now += pause_seconds

    monkeypatch.setattr(async_wait, "time", SimpleNamespace(monotonic=lambda: now))
    pilot = SimpleNamespace(pause=_pause)
    app = SimpleNamespace(screen=SettingsScreen())
    owed = await parity._rebuild_owed_on_leaving(app, pilot)  # type: ignore[arg-type]
    assert owed is False
    assert paused == pauses
