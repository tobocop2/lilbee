"""LilbeeApp test host: skips the heavyweight on_mount setup."""

from __future__ import annotations

from contextlib import contextmanager

from lilbee.cli.tui.app import LilbeeApp


class LilbeeAppHost(LilbeeApp):
    """LilbeeApp subclass that skips the heavyweight on_mount work for tests."""

    _test_skip_auto_init = True


_GATE_HANDOVER_TIMEOUT_S = 20.0
_GATE_POLL_S = 0.02
_PUMP_TIMEOUT_S = 5.0


async def pump_until(pilot, predicate, *, timeout_s: float = _PUMP_TIMEOUT_S) -> bool:
    """Pump Textual's message bus until *predicate* holds; return whether it did.

    One ``pilot.pause()`` flushes a single hop of the bus. An effect that spans
    two or more hops (focus change, the widget's blur handler, the message it
    posts, the screen's handler, the write) is not guaranteed to have landed
    after one pause, which is why a bare pause passes locally and fails on a
    loaded runner. Bounded by wall clock rather than a pump count so a slow
    runner gets more attempts rather than the same fixed few.

    Returns a bool instead of asserting so the caller keeps its own assertion,
    and with it the failure message that says what was actually expected.
    """
    import asyncio
    import time

    deadline = time.monotonic() + timeout_s
    while True:
        await pilot.pause()
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_GATE_POLL_S)


async def await_chat(app, pilot) -> object:
    """Wait for the startup gate to hand over a fully composed chat screen, and return it.

    Production pushes a blocking StartupGate before the chat screen exists, and its
    boot worker runs on a real thread, so a bare ``pilot.pause()`` no longer lands
    on chat. Waiting for the screen alone races its children: a caller that queries
    a widget the instant the screen appears can miss it on a slow runner.
    """
    import asyncio
    import time

    from lilbee.cli.tui.screens.chat import ChatScreen

    deadline = time.monotonic() + _GATE_HANDOVER_TIMEOUT_S
    while time.monotonic() < deadline:
        chat = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        if chat is not None and chat.is_mounted and chat.query("#chat-input"):
            return chat
        await pilot.pause()
        await asyncio.sleep(_GATE_POLL_S)
    raise AssertionError("the startup gate never revealed a composed chat screen")


@contextmanager
def ready_services():
    """Bind a mock container whose chat role is ready, so the startup gate releases at once.

    Also pins ``needs_setup`` False: the chat screen's setup-check worker runs
    on a real thread, so an unpinned True pushes the SetupWizard over whatever
    screen the test just pushed, on a slow runner's schedule.
    """
    from unittest.mock import MagicMock, patch

    from lilbee.app.services import set_services

    services = MagicMock()
    services.provider.role_ready.return_value = True
    set_services(services)
    try:
        with patch("lilbee.cli.tui.screens.chat.needs_setup", return_value=False):
            yield services
    finally:
        set_services(None)


def shown_footer_keys(app) -> set[str]:
    """Keys the active screen's footer row is currently advertising.

    Read off ``active_bindings`` rather than the rendered widgets so a caller
    can assert what the row claims without waiting for a repaint. Filtered on
    ``binding.show`` because that is what the footer filters on; note Textual
    drops a binding whose ``check_action`` returned False and keeps a None one
    greyed, so "absent here" means False, not None.
    """
    return {binding.key for _, binding, _, _ in app.screen.active_bindings.values() if binding.show}
