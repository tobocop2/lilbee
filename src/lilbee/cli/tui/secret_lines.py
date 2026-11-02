"""The one rule for chat lines that must stay out of the input history."""

from __future__ import annotations

from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.cli.tui.preset_commands import is_preset_line

_LOGIN = "/login"
_SET = "/set"


def carries_secret(text: str) -> bool:
    """Whether the chat line *text* holds a secret, with the dispatcher's word split.

    A /preset line, a /login line with a token and a /set line for a secret setting all count.
    """
    words = text.split()
    if not words:
        return False
    command = words[0].lower()
    if is_preset_line(text):
        return True
    if command == _LOGIN:
        return len(words) > 1
    if command == _SET and len(words) > 1:
        defn = SETTINGS_MAP.get(words[1])
        return defn is not None and defn.secret
    return False
