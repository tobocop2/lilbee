"""Correct color reduction for terminals without truecolor.

Without truecolor (macOS Terminal.app; any SSH session, since OpenSSH does not
forward COLORTERM) Rich reduces colors with ``Color.downgrade``, which uses the
grey ramp only below 15% saturation and otherwise rounds each channel into the
6x6x6 cube, whose first two steps are 0 and 95.

Dark theme surfaces straddle that line. 18 of the 33 surfaces across the themes
lilbee ships sit above it and get cube-rounded, and their channels are far
enough below 95 that each rounds to 0 or up to 95; 8 come back more saturated
than they went in. On rose-pine, $background #191724 and $surface #1f1d2e both
land on slot 16 and $panel #26233a lands on slot 17, a navy in no theme.

Build vs buy: Textual owns the filter pipeline (``App.get_line_filters``) and
Rich owns a correct perceptual matcher (``Palette.match``), which ``downgrade``
does not call. This module only connects the two. A style resolved to 8-bit here
reaches Rich already 8-bit, so its reduction is a no-op.

The 256 palette holds no dark tinted colors, so surfaces land on the greyscale
ramp and the theme tint survives only in the accents. The ramp's ten-unit steps
still separate $background, $surface and $panel on most themes.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from typing import TYPE_CHECKING

from rich._palettes import EIGHT_BIT_PALETTE
from rich.color import Color, ColorType
from rich.color_triplet import ColorTriplet
from rich.segment import Segment
from rich.style import Style
from textual.filter import LineFilter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from textual.color import Color as TextualColor

# Rich's name for the color system that needs correcting. Its 16-color path
# already matches against the standard palette properly, and routing that through
# 8-bit first only adds a rounding step, so "standard" is deliberately excluded.
EIGHT_BIT_COLOR_SYSTEM = "256"
TRUECOLOR_COLOR_SYSTEM = "truecolor"

# macOS Terminal.app exports COLORTERM=truecolor but cannot render 24-bit SGR: it
# reads "48;2;r;g;b" as separate codes, so #191724 turns cyan because the trailing
# 36 lands as "foreground cyan". Measured on Terminal 453 / macOS 14.6.1, where a
# 24-bit gradient comes out alternating green and magenta while the 256-color one
# is correct. Rich believes COLORTERM, so the color system alone cannot detect it.
APPLE_TERMINAL = "Apple_Terminal"

# tmux overwrites TERM_PROGRAM with its own name, hiding the terminal underneath,
# and passes COLORTERM through, so a tmux session inside Terminal.app looks
# truecolor-capable from both signals. It keeps the original in its global
# environment, which is the only way back to the real terminal. tmux forwards
# 8-bit colors unchanged, so resolving this is enough to fix the nested case.
TMUX_TERM_PROGRAM = "tmux"
_TMUX_ENV_TIMEOUT_S = 1.0


def needs_eight_bit(color_system: str | None, term_program: str | None) -> bool:
    """Whether truecolor styles must be resolved to the 256 palette before output."""
    return color_system == EIGHT_BIT_COLOR_SYSTEM or term_program == APPLE_TERMINAL


def draws_block_glyphs(color_system: str | None, term_program: str | None) -> bool:
    """Whether the terminal can be trusted to tile partial-block border glyphs.

    Not the same question as needs_eight_bit, and not the same slope: a 16-colour
    terminal needs no colour correction, because Rich's standard-palette path is
    already nearest neighbour, but it is *less* likely to draw U+2580..U+259F
    cell-exact, not more.

    Font metrics cannot be queried, so this goes on terminal identity: the
    terminals that advertise truecolor ship a font that tiles, minus Terminal.app,
    which advertises it falsely.
    """
    return color_system == TRUECOLOR_COLOR_SYSTEM and term_program != APPLE_TERMINAL


def resolve_term_program(environ: Mapping[str, str]) -> str | None:
    """The terminal actually drawing the screen, seeing through tmux where possible.

    Reports the client that started the tmux server, so attaching one server from
    two different terminals can still get this wrong; there is no per-client answer
    to ask for.
    """
    term_program = environ.get("TERM_PROGRAM")
    if term_program != TMUX_TERM_PROGRAM:
        return term_program
    tmux = shutil.which("tmux")
    if tmux is None:
        return term_program
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, path resolved by which
            [tmux, "show-environment", "-g", "TERM_PROGRAM"],
            capture_output=True,
            encoding="utf-8",
            timeout=_TMUX_ENV_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return term_program
    if result.returncode != 0:
        return term_program
    _, _, value = result.stdout.strip().partition("=")
    return value or term_program


@lru_cache(maxsize=4096)
def nearest_eight_bit(rgb: tuple[int, int, int]) -> Color:
    """The 256-palette color closest to *rgb*, by true nearest neighbour.

    Cached because this runs per segment per repaint.
    """
    return Color("", ColorType.EIGHT_BIT, number=EIGHT_BIT_PALETTE.match(ColorTriplet(*rgb)))


def _reduce(color: Color | None) -> Color | None:
    """Resolve *color* to an explicit 8-bit color, or None if it needs no change."""
    if color is None or color.type != ColorType.TRUECOLOR:
        return None
    triplet = color.get_truecolor()
    return nearest_eight_bit((triplet.red, triplet.green, triplet.blue))


class EightBitPalette(LineFilter):
    """Resolve truecolor styles to their nearest 256-palette entry.

    Applied before Rich sees the segments, so Rich's own cube-snap never runs.
    """

    def apply(self, segments: list[Segment], background: TextualColor) -> list[Segment]:
        output: list[Segment] = []
        for segment in segments:
            style = segment.style
            if style is None:
                output.append(segment)
                continue
            color = _reduce(style.color)
            bgcolor = _reduce(style.bgcolor)
            if color is None and bgcolor is None:
                output.append(segment)
                continue
            output.append(
                Segment(
                    segment.text,
                    style + Style.from_color(color or style.color, bgcolor or style.bgcolor),
                    segment.control,
                )
            )
        return output
