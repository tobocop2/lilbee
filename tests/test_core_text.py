"""Control-character stripping for text a terminal shows."""

from __future__ import annotations

import pytest

from lilbee.core.text import strip_control_characters


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("plain title", "plain title"),
        ("a \x1b[31mred\x1b[0m b", "a red b"),
        ("a \x9b31m b", "a  b"),
        ("\x1b]0;owned\x07title", "title"),
        ("\x1b]8;;https://x.example\x1b\\link\x1b]8;;\x1b\\", "link"),
        ("a \x1bPdevice\x9c b", "a  b"),
        ("a \x1bc b", "a  b"),
        ("a \x1b]0;never ended", "a "),
        ("bell \x07 del \x7f nul \x00 c1 \x85", "bell  del  nul  c1 "),
        ("tab\tand\nline", "tabandline"),
    ],
)
def test_control_characters_and_escape_sequences_are_removed(raw: str, shown: str) -> None:
    assert strip_control_characters(raw) == shown
