"""User text names the machine lilbee runs on "this computer", never "this host"."""

from __future__ import annotations

import pytest

from lilbee.app.presets import RESERVE_REMOTE_NOTE
from lilbee.cli import preset as preset_cli
from lilbee.cli.tui import messages
from lilbee.providers.fleet.engines.host_gate import PLATFORM_UNREAD

RENDERED = {
    "reserve note": RESERVE_REMOTE_NOTE,
    "reserve help": preset_cli._reserve_help,
    "reserve placeholder": messages.PRESET_DIALOG_RESERVE_PLACEHOLDER,
    "unreadable platform": PLATFORM_UNREAD.reason,
}


@pytest.mark.parametrize("name", sorted(RENDERED))
def test_rendered_text_says_this_computer(name: str) -> None:
    text = RENDERED[name]
    assert "this host" not in text
    assert "this computer" in text
