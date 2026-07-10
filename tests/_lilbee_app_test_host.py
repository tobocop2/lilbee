"""LilbeeApp test host: skips the heavyweight on_mount setup."""

from __future__ import annotations

from contextlib import contextmanager

from lilbee.cli.tui.app import LilbeeApp


class LilbeeAppHost(LilbeeApp):
    """LilbeeApp subclass that skips the heavyweight on_mount work for tests."""

    _test_skip_auto_init = True


_GATE_HANDOVER_TIMEOUT_S = 20.0
_GATE_POLL_S = 0.02


async def await_chat(app, pilot) -> object:
    """Wait for the startup gate to hand the screen over to chat, and return it.

    Production pushes a blocking StartupGate before the chat screen exists, and its
    boot worker runs on a real thread, so a bare ``pilot.pause()`` no longer lands
    on chat.
    """
    import asyncio
    import time

    from lilbee.cli.tui.screens.chat import ChatScreen

    deadline = time.monotonic() + _GATE_HANDOVER_TIMEOUT_S
    while time.monotonic() < deadline:
        chat = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        if chat is not None:
            return chat
        await pilot.pause()
        await asyncio.sleep(_GATE_POLL_S)
    raise AssertionError("the startup gate never revealed the chat screen")


@contextmanager
def ready_services():
    """Bind a mock container whose chat role is ready, so the startup gate releases at once."""
    from unittest.mock import MagicMock

    from lilbee.app.services import set_services

    services = MagicMock()
    services.provider.role_ready.return_value = True
    set_services(services)
    try:
        yield services
    finally:
        set_services(None)
