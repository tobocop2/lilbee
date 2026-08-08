"""Correct color reduction for terminals without truecolor.

Terminals that advertise 256 colors rather than truecolor (macOS Terminal.app,
and any SSH session, since OpenSSH forwards LANG/LC_* but not COLORTERM) make
Rich reduce every color to the 256-color palette. Rich's ``Color.downgrade``
does that by snapping each channel into the 6x6x6 cube, whose first two steps
are 0 and 95, and only falls back to the grey ramp under 15% saturation. Dark
theme surfaces sit at roughly 20% saturation with channel values of 25-58,
inside that gap, so they either collapse to pure black or overshoot into a
saturated hue: rose-pine's $background and $surface both land on slot 16, and
$panel lands on slot 17, a navy that appears nowhere in the theme.

Build vs buy: Textual owns the filter pipeline (``App.get_line_filters``) and
Rich owns a correct perceptual matcher (``Palette.match``, redmean-weighted).
Neither needs reimplementing. This module only connects the two, because
``downgrade`` does not call ``match``. Resolving a style to 8-bit here means
Rich receives a color that is already 8-bit, so its own reduction is a no-op.

What this does not do is invent color the terminal cannot show. The 256 palette
holds no dark tinted colors at all: below the greys, its color cube jumps
straight from 0 to 95 per channel. So on such a terminal every dark surface
lands on the greyscale ramp and the theme's tint survives only in the brighter
accents. Neutral grey is the honest nearest answer, and the ramp's ten-unit
steps still separate $background, $surface and $panel on most themes.
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

# Color systems that reduce a truecolor style, and so need the correction.
REDUCING_COLOR_SYSTEMS = frozenset({"256", "eight_bit", "standard", "windows"})


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
    triplet = color.triplet
    if triplet is None:
        return None
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
