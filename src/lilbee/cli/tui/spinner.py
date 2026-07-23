"""Braille spinner frames, sourced from Rich's canonical ``dots`` spinner.

Rich already ships the cli-spinners frame data that backs its own ``Spinner``;
the task bar and the catalog loader index into these frames on their own poll
tick, so there is one definition here rather than a hand-copied braille tuple at
each call site.
"""

from __future__ import annotations

from rich.spinner import SPINNERS

# Rich stores the frames as one glyph-per-character string.
SPINNER_FRAMES: tuple[str, ...] = tuple(SPINNERS["dots"]["frames"])


def spinner_frame(tick: int) -> str:
    """The braille spinner glyph for *tick*, wrapping over the frame set."""
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
