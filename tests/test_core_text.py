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


MARKDOWN_BODY = (
    "# Title\n\nIntro text.\n\n- first\n- second\n\n```python\ndef f():\n\treturn 1\n```\n\nEnd.\n"
)


def test_keep_lines_renders_a_markdown_body_line_for_line() -> None:
    assert strip_control_characters(MARKDOWN_BODY, keep_lines=True) == MARKDOWN_BODY


def test_keep_lines_still_removes_escapes_and_controls_in_a_body() -> None:
    raw = "# Title\n\x1b[31mred\x1b[0m\n\x1b]0;owned\x07- item\x00\n\tindent\x85\n"
    assert strip_control_characters(raw, keep_lines=True) == "# Title\nred\n- item\n\tindent\n"


def test_keep_lines_turns_crlf_into_lf_and_drops_a_lone_cr() -> None:
    assert strip_control_characters("a\r\nb\rc\n", keep_lines=True) == "a\nbc\n"


def test_a_label_stays_on_one_line_by_default() -> None:
    assert strip_control_characters(MARKDOWN_BODY) == MARKDOWN_BODY.replace("\n", "").replace(
        "\t", ""
    )
