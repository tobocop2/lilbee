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

from functools import lru_cache
from typing import TYPE_CHECKING

from rich._palettes import EIGHT_BIT_PALETTE
from rich.color import Color, ColorType
from rich.color_triplet import ColorTriplet
from rich.segment import Segment
from rich.style import Style
from textual.filter import LineFilter

if TYPE_CHECKING:
    from textual.color import Color as TextualColor

# Rich's name for the one color system that needs correcting. Its 16-color path
# already matches against the standard palette properly, and routing that through
# 8-bit first only adds a rounding step, so "standard" is deliberately excluded.
EIGHT_BIT_COLOR_SYSTEM = "256"


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
