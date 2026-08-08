"""Colors survive reduction to a 256-color terminal, and no emoji reach the screen."""

from __future__ import annotations

import pathlib
import unicodedata

import pytest
from rich._palettes import EIGHT_BIT_PALETTE
from rich.cells import cell_len
from rich.color import Color, ColorSystem, ColorType
from rich.segment import Segment
from rich.style import Style
from textual.color import Color as TextualColor

from lilbee.app.themes import DARK_THEMES
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.color_compat import EightBitPalette, nearest_eight_bit
from lilbee.cli.tui.task_queue import STATUS_ICONS

_COLOR_SYSTEMS = {
    "standard": ColorSystem.STANDARD,
    "256": ColorSystem.EIGHT_BIT,
    "truecolor": ColorSystem.TRUECOLOR,
}

# The surfaces that must stay visually distinct: without them, every panel edge
# and every card lift disappears into a single flat fill.
_LAYERING_TOKENS = ("background", "surface", "panel")

_BLACK = TextualColor(0, 0, 0)


def _slot(hex_color: str) -> int:
    """The 256-palette slot this theme color reduces to under the filter."""
    triplet = Color.parse(hex_color[:7]).get_truecolor()
    return nearest_eight_bit((triplet.red, triplet.green, triplet.blue)).number


def _theme_colors(theme_name: str) -> dict[str, str]:
    from textual.theme import BUILTIN_THEMES

    return BUILTIN_THEMES[theme_name].to_color_system().generate()


def _rgb(hex_color: str) -> tuple[int, int, int]:
    triplet = Color.parse(hex_color[:7]).get_truecolor()
    return (triplet.red, triplet.green, triplet.blue)


def _distance(slot: int, rgb: tuple[int, int, int]) -> int:
    entry = EIGHT_BIT_PALETTE[slot]
    return sum((a - b) ** 2 for a, b in zip(entry, rgb, strict=True))


def _spread(rgb: tuple[int, int, int]) -> int:
    """Channel spread, a saturation proxy: 0 is neutral grey, 255 fully saturated."""
    return max(rgb) - min(rgb)


class TestLayeringSurvivesReduction:
    def test_rose_pine_surfaces_land_on_distinct_slots(self):
        """Rich's own downgrade() gives 16/16/17 here: two collapse, one turns navy."""
        colors = _theme_colors("rose-pine")
        slots = [_slot(colors[token]) for token in _LAYERING_TOKENS]
        assert slots == [234, 235, 236]

    @pytest.mark.parametrize("theme_name", sorted(DARK_THEMES))
    def test_no_surface_gains_saturation(self, theme_name: str):
        """The defect, stated directly: a muted surface must not come back vivid.

        rose-pine's $panel #26233a has a channel spread of 23 and reduces under
        downgrade() to #00005f, a spread of 95: a navy that is in no theme. Across
        the dark themes that happens to seven of 33 surfaces. Distinctness is not
        the oracle here, since dracula's surfaces land on three different slots
        under downgrade() precisely because one of them is that wrong hue.
        """
        colors = _theme_colors(theme_name)
        for token in _LAYERING_TOKENS:
            source = _rgb(colors[token])
            reduced = tuple(EIGHT_BIT_PALETTE[_slot(colors[token])])
            assert _spread(reduced) <= _spread(source), (
                f"{theme_name} ${token} {source} came back more saturated as {reduced}"
            )

    @pytest.mark.parametrize("theme_name", sorted(DARK_THEMES))
    def test_never_further_from_the_truth_than_rich(self, theme_name: str):
        """No theme is made worse; rose-pine and most others are made much better."""
        colors = _theme_colors(theme_name)
        for token in _LAYERING_TOKENS:
            rgb = _rgb(colors[token])
            rich_slot = Color.parse(colors[token][:7]).downgrade(2).number
            assert _distance(_slot(colors[token]), rgb) <= _distance(rich_slot, rgb)

    def test_identical_theme_colors_stay_identical(self):
        """solarized-dark defines $surface and $panel as the same color.

        Their sharing a slot is the theme's own doing, not a loss from reduction,
        and pinning it here stops a future reader reading it as a defect.
        """
        colors = _theme_colors("solarized-dark")
        assert colors["surface"][:7] == colors["panel"][:7]
        assert _slot(colors["surface"]) == _slot(colors["panel"])


class TestFilter:
    def test_truecolor_styles_become_eight_bit(self):
        """Resolving here is what stops Rich running its own cube-snap downgrade."""
        out = EightBitPalette().apply(
            [Segment("x", Style(bgcolor="#1f1d2e", color="#e0def4"))], _BLACK
        )
        style = out[0].style
        assert style.bgcolor.type is ColorType.EIGHT_BIT
        assert style.color.type is ColorType.EIGHT_BIT
        assert style.bgcolor.number == 235

    def test_text_and_control_are_preserved(self):
        segment = Segment("hello", Style(color="#e0def4"), None)
        out = EightBitPalette().apply([segment], _BLACK)
        assert out[0].text == "hello"
        assert out[0].control is None

    def test_segments_without_style_pass_through_untouched(self):
        segment = Segment("plain", None)
        assert EightBitPalette().apply([segment], _BLACK) == [segment]

    def test_already_eight_bit_styles_are_left_alone(self):
        """Only truecolor needs resolving; re-matching a palette color would be lossy."""
        style = Style.from_color(Color.from_ansi(42))
        out = EightBitPalette().apply([Segment("x", style)], _BLACK)
        assert out[0].style is style


class TestFilterInstallation:
    """Installed on a real app, so the wiring is covered and not just the constant."""

    @staticmethod
    async def _filters_for(color_system: str) -> list[str]:
        from tests._lilbee_app_test_host import LilbeeAppHost, ready_services

        app = LilbeeAppHost()
        with ready_services():
            async with app.run_test(size=(80, 24)) as pilot:
                app.console._color_system = _COLOR_SYSTEMS[color_system]
                await pilot.pause()
                return [type(line_filter).__name__ for line_filter in app.get_line_filters()]

    @pytest.mark.asyncio
    async def test_installed_on_a_256_color_terminal(self):
        assert EightBitPalette.__name__ in await self._filters_for("256")

    @pytest.mark.asyncio
    async def test_absent_on_truecolor(self):
        """Capable terminals must render exactly as before this change."""
        assert EightBitPalette.__name__ not in await self._filters_for("truecolor")

    @pytest.mark.asyncio
    async def test_absent_on_a_16_color_terminal(self):
        """Rich's 16-color path already matches properly; 8-bit first would only round twice."""
        assert EightBitPalette.__name__ not in await self._filters_for("standard")


class TestNoEmoji:
    def test_status_icons_are_all_one_cell_wide(self):
        """A double-width icon shifts its row out of the column the others share."""
        wide = {status: icon for status, icon in STATUS_ICONS.items() if cell_len(icon) != 1}
        assert not wide

    def test_no_emoji_in_tui_source(self):
        """Emoji need a fallback font, render in their own colors and are double-width."""
        root = pathlib.Path(__file__).parent.parent / "src" / "lilbee" / "cli" / "tui"
        offenders: list[str] = []
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".py", ".tcss"}:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                for char in line:
                    if unicodedata.east_asian_width(char) == "W" or 0x1F000 <= ord(char) <= 0x1FAFF:
                        offenders.append(f"{path.name}:{lineno} {char!r} U+{ord(char):04X}")
        assert not offenders

    def test_palette_search_icon_is_overridden(self):
        """Guards the one glyph owned by Textual, which an upgrade could move."""
        from textual.command import SearchIcon

        assert cell_len(msg.COMMAND_PALETTE_ICON) == 1
        assert SearchIcon.icon._default != msg.COMMAND_PALETTE_ICON
