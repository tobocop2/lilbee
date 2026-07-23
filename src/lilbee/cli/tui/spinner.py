"""Braille spinner frames, sourced from Rich's canonical ``dots`` spinner.

Rich already ships the cli-spinners frame data that backs its own ``Spinner``;
the task bar and the catalog loader index into these frames on their own poll
tick, so there is one definition here rather than a hand-copied braille tuple at
each call site.
"""

from __future__ import annotations

from typing import cast

from rich.spinner import SPINNERS

# Rich stores the frames as one glyph-per-character string, but types the spinner
# table loosely (the entry values are ``object``), so narrow it here.
SPINNER_FRAMES: tuple[str, ...] = tuple(cast(str, SPINNERS["dots"]["frames"]))


def spinner_frame(tick: int) -> str:
    """The braille spinner glyph for *tick*, wrapping over the frame set."""
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
