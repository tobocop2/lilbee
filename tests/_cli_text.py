"""The plain text of captured CLI output, for assertions that hold in every terminal."""

from __future__ import annotations

from rich.text import Text


def plain_cli_text(output: str) -> str:
    """Flatten captured CLI output to one space-joined line of plain text.

    Typer styles its help and error panels when GITHUB_ACTIONS, FORCE_COLOR or
    PY_COLORS is set, so a run in CI puts ANSI codes inside a phrase. Rich's own
    decoder strips the styling. The panel border and every line wrap then
    collapse to a single space, so a phrase that rich wraps still matches.
    """
    plain = Text.from_ansi(output).plain
    return " ".join(plain.replace("│", " ").split())
