"""Colors survive reduction to a 256-color terminal, and no emoji reach the screen."""

from __future__ import annotations

import os
import pathlib
import subprocess
import unicodedata
from unittest import mock

import pytest
from rich._palettes import EIGHT_BIT_PALETTE
from rich.cells import cell_len
from rich.color import Color, ColorSystem, ColorType
from rich.segment import Segment
from rich.style import Style
from textual.color import Color as TextualColor

from lilbee.app.themes import DARK_THEMES
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.color_compat import (
    EightBitPalette,
    nearest_eight_bit,
    needs_eight_bit,
    resolve_term_program,
)
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
        the dark themes that happens to 8 of 33 surfaces. Distinctness is not the
        oracle here, since dracula's surfaces land on three different slots under
        downgrade() precisely because one of them is that wrong hue.
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


class TestDetection:
    """Which terminals need the correction, decided without mounting an app."""

    def test_a_256_color_terminal_does(self):
        assert needs_eight_bit("256", "iTerm.app")

    def test_a_truecolor_terminal_does_not(self):
        """Capable terminals must render exactly as before this change."""
        assert not needs_eight_bit("truecolor", "iTerm.app")

    def test_a_16_color_terminal_does_not(self):
        """Rich's 16-color path already matches properly; 8-bit first would only round twice."""
        assert not needs_eight_bit("standard", "iTerm.app")

    def test_terminal_app_does_even_though_it_claims_truecolor(self):
        """Terminal.app exports COLORTERM=truecolor and renders 24-bit SGR as garbage.

        Verified on Terminal 453 / macOS 14.6.1: a 24-bit gradient comes out
        alternating green and magenta, and #191724 renders cyan, because the
        parameters of "48;2;25;23;36" are read as separate SGR codes. Rich believes
        COLORTERM, so without this the filter never installs where it is needed most.
        """
        assert needs_eight_bit("truecolor", "Apple_Terminal")


class TestSeeingThroughTmux:
    """tmux hides the real terminal, and a tmux inside Terminal.app hit the same bug.

    tmux overwrites TERM_PROGRAM with its own name and forwards COLORTERM, so both
    signals said truecolor and the filter stayed off: running the TUI in tmux inside
    Terminal.app rendered the same bright green page as bare Terminal.app did.
    """

    @staticmethod
    def _completed(stdout: str, returncode: int = 0):
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)

    def test_a_plain_terminal_is_returned_as_is(self):
        assert resolve_term_program({"TERM_PROGRAM": "Apple_Terminal"}) == "Apple_Terminal"

    def test_an_absent_variable_stays_absent(self):
        assert resolve_term_program({}) is None

    def test_tmux_is_resolved_to_the_terminal_underneath(self):
        with (
            mock.patch("lilbee.cli.tui.color_compat.shutil.which", return_value="/usr/bin/tmux"),
            mock.patch(
                "subprocess.run", return_value=self._completed("TERM_PROGRAM=Apple_Terminal\n")
            ),
        ):
            assert resolve_term_program({"TERM_PROGRAM": "tmux"}) == "Apple_Terminal"

    def test_a_resolved_terminal_app_turns_the_filter_on(self):
        """The whole point: the nested case must reach needs_eight_bit as Apple_Terminal."""
        with (
            mock.patch("lilbee.cli.tui.color_compat.shutil.which", return_value="/usr/bin/tmux"),
            mock.patch(
                "subprocess.run", return_value=self._completed("TERM_PROGRAM=Apple_Terminal\n")
            ),
        ):
            assert needs_eight_bit("truecolor", resolve_term_program({"TERM_PROGRAM": "tmux"}))

    def test_a_truecolor_terminal_under_tmux_stays_off(self):
        """iTerm users in tmux must not be downgraded."""
        with (
            mock.patch("lilbee.cli.tui.color_compat.shutil.which", return_value="/usr/bin/tmux"),
            mock.patch(
                "lilbee.cli.tui.color_compat.subprocess.run",
                return_value=self._completed("TERM_PROGRAM=iTerm.app\n"),
            ),
        ):
            assert not needs_eight_bit("truecolor", resolve_term_program({"TERM_PROGRAM": "tmux"}))

    def test_no_tmux_binary_falls_back_rather_than_raising(self):
        with mock.patch("lilbee.cli.tui.color_compat.shutil.which", return_value=None):
            assert resolve_term_program({"TERM_PROGRAM": "tmux"}) == "tmux"

    @pytest.mark.parametrize(
        "run_kwargs",
        [
            pytest.param(
                {"return_value": subprocess.CompletedProcess([], 1, stdout="")},
                id="nonzero-exit",
            ),
            pytest.param(
                {"return_value": subprocess.CompletedProcess([], 0, stdout="\n")},
                id="empty-output",
            ),
            pytest.param({"side_effect": OSError("boom")}, id="oserror"),
            pytest.param(
                {"side_effect": subprocess.TimeoutExpired("tmux", 1)},
                id="timeout",
            ),
        ],
    )
    def test_a_misbehaving_tmux_falls_back_rather_than_raising(self, run_kwargs: dict):
        """Startup must not die because the tmux query misbehaved."""
        with (
            mock.patch("lilbee.cli.tui.color_compat.shutil.which", return_value="/usr/bin/tmux"),
            mock.patch("lilbee.cli.tui.color_compat.subprocess.run", **run_kwargs),
        ):
            assert resolve_term_program({"TERM_PROGRAM": "tmux"}) == "tmux"


class TestFilterInstallation:
    """Installed on a real app, so the wiring is covered and not just the predicate."""

    @staticmethod
    async def _filters_for(color_system: str, term_program: str) -> list[str]:
        from tests._lilbee_app_test_host import LilbeeAppHost, ready_services

        with mock.patch.dict(os.environ, {"TERM_PROGRAM": term_program}):
            app = LilbeeAppHost()
            with ready_services():
                async with app.run_test(size=(80, 24)) as pilot:
                    app.console._color_system = _COLOR_SYSTEMS[color_system]
                    await pilot.pause()
                    return [type(line_filter).__name__ for line_filter in app.get_line_filters()]

    @pytest.mark.asyncio
    async def test_installed_on_a_256_color_terminal(self):
        assert EightBitPalette.__name__ in await self._filters_for("256", "iTerm.app")

    @pytest.mark.asyncio
    async def test_installed_on_terminal_app(self):
        assert EightBitPalette.__name__ in await self._filters_for("truecolor", "Apple_Terminal")

    @pytest.mark.asyncio
    async def test_absent_on_truecolor(self):
        assert EightBitPalette.__name__ not in await self._filters_for("truecolor", "iTerm.app")


def _block_based_border_styles() -> frozenset[str]:
    """Border styles Textual draws from block glyphs, read from its own table.

    Hardcoding the list missed six of the ten, including `outer`, which is what
    Textual's Toast uses for its severity rail. Deriving it means a new Textual
    border style is covered the day it lands.
    """
    from textual._border import BORDER_CHARS

    return frozenset(
        name
        for name, rows in BORDER_CHARS.items()
        if any(0x2580 <= ord(char) <= 0x259F for row in rows for char in row)
    )


_PARTIAL_BLOCK_BORDERS = _block_based_border_styles()
_BORDER_EDGES = ("border_top", "border_right", "border_bottom", "border_left")


def _catalog_row():
    """One installed row, the shape ModelInfoModal renders."""
    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow

    return LocalCatalogRow(
        name="Qwen3 0.6B",
        task=ModelTask.CHAT,
        params="0.6B",
        size="0.5 GB",
        quant="Q4_K_M",
        downloads="--",
        featured=False,
        installed=True,
        sort_downloads=0,
        sort_size=0.5,
        ref="Qwen/Qwen3-0.6B-GGUF",
    )


def _screen_kwargs(cls_name: str) -> dict:
    """Constructor arguments for the surfaces that take them."""
    from lilbee.cli.tui.widgets.model_bar import ModelOption

    return {
        "ConfirmDialog": {"title": "Delete model", "message": "This removes 17.3 GB from disk."},
        "NoticeDialog": {"title": "Heads up", "message": "The engine is still warming."},
        "ModelInfoModal": {"row": _catalog_row()},
        "ModelPickerModal": {
            "scope": "chat",
            "options": [ModelOption("Qwen3 0.6B", "Qwen/Qwen3-0.6B-GGUF")],
        },
    }.get(cls_name, {})


# Every screen and modal the TUI can put on the stack. Chat is pushed explicitly
# like the rest: the test host skips the on_mount that installs it, so relying on
# the default screen audits a bare Screen with no widgets on it at all.
_SCREENS = [
    pytest.param("lilbee.cli.tui.screens.chat", "ChatScreen", id="chat"),
    pytest.param("lilbee.cli.tui.screens.catalog", "CatalogScreen", id="catalog"),
    pytest.param("lilbee.cli.tui.screens.settings", "SettingsScreen", id="settings"),
    pytest.param("lilbee.cli.tui.screens.task_center", "TaskCenter", id="tasks"),
    pytest.param("lilbee.cli.tui.screens.fleet", "FleetScreen", id="fleet"),
    pytest.param("lilbee.cli.tui.screens.sessions", "SessionsScreen", id="sessions"),
    pytest.param("lilbee.cli.tui.screens.memories", "MemoriesScreen", id="memories"),
    pytest.param("lilbee.cli.tui.screens.status", "StatusScreen", id="status"),
    pytest.param("lilbee.cli.tui.screens.wiki", "WikiScreen", id="wiki"),
    pytest.param("lilbee.cli.tui.screens.wiki_drafts", "WikiDraftsScreen", id="wiki-drafts"),
    pytest.param("lilbee.cli.tui.screens.setup", "SetupWizard", id="setup"),
    pytest.param("lilbee.cli.tui.screens.startup_gate", "StartupGate", id="startup-gate"),
    pytest.param("lilbee.cli.tui.screens.command_palette", "LilbeeCommandPalette", id="palette"),
    pytest.param("lilbee.cli.tui.widgets.confirm_dialog", "ConfirmDialog", id="confirm"),
    pytest.param("lilbee.cli.tui.widgets.notice_dialog", "NoticeDialog", id="notice"),
    pytest.param("lilbee.cli.tui.widgets.crawl_dialog", "CrawlDialog", id="crawl"),
    pytest.param(
        "lilbee.cli.tui.widgets.slash_command_catalog", "SlashCommandCatalog", id="slash-catalog"
    ),
    pytest.param("lilbee.cli.tui.screens.model_info", "ModelInfoModal", id="model-info"),
    pytest.param("lilbee.cli.tui.screens.model_picker", "ModelPickerModal", id="model-picker"),
]


class TestNoPartialBlockBorders:
    """Borders must be box-drawing, which tiles in every font.

    Textual's `tall` and `thick` are built from partial block glyphs. Measured in
    macOS Terminal.app, `tall`'s left edge draws as three disconnected blocks and
    `thick`'s rails show a seam per cell, so a bordered panel reads as dashes and
    corner ticks. Checked on the mounted screen rather than by grepping the
    stylesheets, because Textual's own Input, Select and Button set `tall` in
    their DEFAULT_CSS and lilbee has to restate those in app.tcss.
    """

    def test_every_nav_view_is_covered_above(self):
        """The screen list is hand-written, so pin it against the app's own view set.

        Without this a new nav view joins the TUI and silently escapes the audit.
        """
        from lilbee.cli.tui.app import _VIEW_FACTORIES

        covered = {param.values[1] for param in _SCREENS if param.values[1]}
        missing = {
            type(factory()).__name__
            for factory in _VIEW_FACTORIES.values()
            if type(factory()).__name__ not in covered
        }
        assert not missing

    @staticmethod
    def _offenders(screen) -> list[str]:
        return [
            f"{type(widget).__name__}.{edge}={border[0]}"
            for widget in screen.query("*")
            for edge in _BORDER_EDGES
            if (border := getattr(widget.styles, edge, None))
            and border[0] in _PARTIAL_BLOCK_BORDERS
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("module", "cls_name"), _SCREENS)
    async def test_screen_has_no_partial_block_border(self, module: str, cls_name: str):
        """Every tab is activated, not just the one the screen opens on.

        Textual defers a TabPane's content until the pane is active, so auditing
        the default view of Settings sees 104 widgets where all seven tabs hold
        747. Two defects hid behind that and behind the unmounted modals: a
        Checkbox in the crawl dialog and the chat completion list.
        """
        import importlib

        from textual.widgets import TabbedContent, TabPane

        from tests._lilbee_app_test_host import LilbeeAppHost, ready_services

        offenders: list[str] = []
        # Force the reduced-glyph path: a capable terminal deliberately keeps the
        # block rails, so asserting against them there would assert the opposite
        # of what the app is for.
        with mock.patch.dict(os.environ, {"TERM_PROGRAM": "Apple_Terminal"}):
            app = LilbeeAppHost()
            with ready_services():
                async with app.run_test(size=(120, 40)) as pilot:
                    await pilot.pause()
                    cls = getattr(importlib.import_module(module), cls_name)
                    app.push_screen(cls(**_screen_kwargs(cls_name)))
                    for _ in range(4):
                        await pilot.pause()

                    assert list(app.screen.query("*")), f"{cls_name} mounted no widgets to audit"

                    tabbed = list(app.screen.query(TabbedContent))
                    panes = [pane.id for pane in app.screen.query(TabPane)]
                    for pane_id in panes:
                        tabbed[0].active = pane_id
                        for _ in range(3):
                            await pilot.pause()
                        offenders += self._offenders(app.screen)
                    if not panes:
                        offenders = self._offenders(app.screen)
        assert not offenders


class TestCapableTerminalsKeepTheBlockRails:
    """The reduced look is for terminals that need it, not for everyone.

    lilbee is drawn with `tall` rails. Converting them outright cost every capable
    terminal the look it renders perfectly well, so the style is carried in a CSS
    variable and only swapped where the glyphs do not tile.
    """

    @staticmethod
    def _rail_vars(term_program: str) -> tuple[str, str]:
        from lilbee.cli.tui.app import LilbeeApp

        with mock.patch.dict(os.environ, {"TERM_PROGRAM": term_program}):
            variables = LilbeeApp().get_css_variables()
        return variables["rail"], variables["rail-heavy"]

    def test_a_capable_terminal_keeps_the_block_rails(self):
        assert self._rail_vars("iTerm.app") == ("tall", "thick")

    def test_terminal_app_falls_back_to_box_drawing(self):
        assert self._rail_vars("Apple_Terminal") == ("solid", "heavy")

    @pytest.mark.asyncio
    async def test_the_extra_stylesheet_loads_only_where_needed(self):
        """Textual's own block borders are restated in a sheet, not in the base one."""
        from lilbee.cli.tui.app import LilbeeApp

        for term_program, wanted in (("iTerm.app", False), ("Apple_Terminal", True)):
            with mock.patch.dict(os.environ, {"TERM_PROGRAM": term_program}):
                app = LilbeeApp()
            loaded = any(str(path).endswith("app_safe.tcss") for path in app.css_path)
            assert loaded is wanted, f"{term_program} loaded={loaded}"


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
            source = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(source.splitlines(), 1):
                for char in line:
                    if unicodedata.east_asian_width(char) == "W" or 0x1F000 <= ord(char) <= 0x1FAFF:
                        offenders.append(f"{path.name}:{lineno} {char!r} U+{ord(char):04X}")
        assert not offenders

    def test_the_replacement_icon_is_one_cell_and_not_textuals(self):
        from textual.command import SearchIcon

        assert cell_len(msg.COMMAND_PALETTE_ICON) == 1
        assert SearchIcon.icon._default != msg.COMMAND_PALETTE_ICON

    @pytest.mark.asyncio
    async def test_the_open_palette_shows_the_replacement_icon(self):
        """Opens the real palette and reads the mounted icon.

        Asserting the constant alone would pass even if the override never ran,
        and this is the one glyph Textual owns, so an upgrade moving SearchIcon
        out of the palette must fail here rather than silently restore the emoji.
        """
        from textual.command import SearchIcon

        from tests._lilbee_app_test_host import LilbeeAppHost, ready_services

        app = LilbeeAppHost()
        with ready_services():
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                await pilot.press("ctrl+p")
                await pilot.pause()
                icons = list(app.screen.query(SearchIcon))
                assert icons, "the palette no longer mounts a SearchIcon"
                assert [icon.icon for icon in icons] == [msg.COMMAND_PALETTE_ICON]
