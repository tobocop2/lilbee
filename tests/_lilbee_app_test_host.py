"""LilbeeApp test host: skips the heavyweight on_mount setup."""

from __future__ import annotations

import asyncio
import functools
import weakref
from contextlib import asynccontextmanager, contextmanager

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen


class LilbeeAppHost(LilbeeApp):
    """LilbeeApp subclass that skips the heavyweight on_mount work for tests."""

    _test_skip_auto_init = True


# Chat screens whose show handler has returned. The handler moves focus to the chat
# input, so a caller that opens a widget and types before it ran loses focus to it.
_shown_chats: weakref.WeakSet[ChatScreen] = weakref.WeakSet()


def _record_show(on_show):
    @functools.wraps(on_show)
    def wrapper(self: ChatScreen) -> None:
        on_show(self)
        _shown_chats.add(self)

    return wrapper


ChatScreen.on_show = _record_show(ChatScreen.on_show)  # type: ignore[method-assign]

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
    import time

    deadline = time.monotonic() + timeout_s
    while True:
        await pilot.pause()
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_GATE_POLL_S)


@asynccontextmanager
async def saved_setting(app: LilbeeApp, key: str):
    """Run the body's action, then wait until the app publishes a change of *key*.

    The wait has no deadline: pytest-timeout ends a save that never happens.
    """
    saved = asyncio.Event()

    def on_change(change: tuple[str, object]) -> None:
        if change[0] == key:
            saved.set()

    app.settings_changed_signal.subscribe(app, on_change, immediate=True)
    yield
    await saved.wait()


async def await_chat(app, pilot) -> object:
    """Wait for the startup gate to hand over a fully composed chat screen, and return it.

    Production pushes a blocking StartupGate before the chat screen exists, and its
    boot worker runs on a real thread, so a bare ``pilot.pause()`` no longer lands
    on chat. Waiting for the screen alone races its children: a caller that queries
    a widget the instant the screen appears can miss it on a slow runner. The screen's
    show handler moves focus to the chat input, so the wait also covers that handler
    having returned: a caller that opens a drawer and types would otherwise lose focus
    to it.

    An app that exits while the wait runs (a handler raised and Textual shut it down)
    fails the wait at once. The handler's own error is raised when ``run_test`` closes.
    """
    import time

    deadline = time.monotonic() + _GATE_HANDOVER_TIMEOUT_S
    while time.monotonic() < deadline:
        if app.return_code is not None:
            raise AssertionError("the app exited before it showed a composed chat screen")
        chat = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        if (
            chat is not None
            and chat.is_mounted
            and chat.query("#chat-input")
            and chat in _shown_chats
        ):
            return chat
        await pilot.pause()
        await asyncio.sleep(_GATE_POLL_S)
    raise AssertionError("the startup gate never revealed a composed chat screen")


@contextmanager
def ready_services():
    """Bind a mock container whose chat role is ready, so the startup gate releases at once.

    Also pins ``chat_ready``/``embedding_ready`` True: the app's readiness worker runs on a real
    thread, so an unpinned False routes the test to the catalog on a slow
    runner's schedule.
    """
    from unittest.mock import MagicMock, create_autospec, patch

    from lilbee.app.services import set_services
    from lilbee.data.store import Store
    from lilbee.data.store.types import SourceSummary

    services = MagicMock()
    services.provider.role_ready.return_value = True
    services.store = create_autospec(Store, instance=True)
    services.store.source_summary.return_value = SourceSummary(0, 0, {}, [])
    services.store.get_sources.return_value = []
    services.store.count_sources.return_value = 0
    set_services(services)
    try:
        with (
            patch("lilbee.cli.tui.app.chat_ready", return_value=True),
            patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
        ):
            yield services
    finally:
        set_services(None)


def send_key_burst(app, *keys: str) -> None:
    """Queue *keys* on the app in one go, the way a terminal delivers typing ahead.

    ``pilot.press`` waits for the app to go idle after every key, so a key can
    never arrive while an earlier key's focus change is still pending. A burst
    removes that wait. Single characters only, plus named keys such as enter.
    """
    from textual import events

    for key in keys:
        app.post_message(events.Key(key, key if len(key) == 1 else None))


def shown_footer_keys(app) -> set[str]:
    """Keys the active screen's footer row is currently advertising.

    Read off ``active_bindings`` rather than the rendered widgets so a caller
    can assert what the row claims without waiting for a repaint. Filtered on
    ``binding.show`` because that is what the footer filters on; note Textual
    drops a binding whose ``check_action`` returned False and keeps a None one
    greyed, so "absent here" means False, not None.
    """
    return {binding.key for _, binding, _, _ in app.screen.active_bindings.values() if binding.show}
