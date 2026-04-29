"""T4 TUI. The chat screen renders after launch and responds to keystrokes."""

from __future__ import annotations

import pytest
from drivers.tui import TuiSession

_TUI_BOOT_TIMEOUT = 60.0


@pytest.mark.tui
def test_tui_launches_and_renders_lilbee_branding(tui: TuiSession) -> None:
    """The TUI binary boots and prints something containing 'lilbee'."""
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)


@pytest.mark.tui
def test_tui_renders_chat_screen_default(tui: TuiSession) -> None:
    """Default screen shows the chat prompt or a recognisable chat affordance."""
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    # One of these markers should appear within the boot window. The exact
    # affordance text varies by version (Send, Ask, Chat, model bar label).
    visible = tui.text().lower()
    assert any(token in visible for token in ("chat", "ask", "model", "send"))


@pytest.mark.tui
def test_tui_remains_alive_after_boot(tui: TuiSession) -> None:
    """The TUI process is still running after the screen renders.

    Counterpart to test_tui_launches_and_renders_lilbee_branding: catches
    the failure mode where the TUI prints something then immediately
    exits with a non-zero code (e.g. missing dependency at boot).
    """
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    assert tui.is_alive()
