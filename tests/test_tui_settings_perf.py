"""Regression tests for the settings screen's per-focus-change cost (bb-p284w.21)."""

from __future__ import annotations

from unittest.mock import patch

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Input, TabbedContent

from lilbee.cli.tui.widgets.stable_footer import StableFooter
from tests._lilbee_app_test_host import LilbeeAppHost


def _make_eager_settings_screen():
    """SettingsScreen subclass that populates every pane on mount.

    Mirrors the fixture in ``test_tui_screens.py`` so this file can drive the
    Generation tab without waiting on lazy tab activation.
    """
    import importlib
    from pathlib import Path

    from lilbee.cli.tui.screens.settings import SettingsScreen

    src_module = importlib.import_module(SettingsScreen.__module__)
    css_path = str(Path(src_module.__file__ or "").parent / "settings.tcss")

    class _EagerSettingsScreen(SettingsScreen):
        CSS_PATH = css_path

        def on_mount(self) -> None:
            super().on_mount()
            self.populate_all_panes()

    return _EagerSettingsScreen


class SettingsPerfTestApp(LilbeeAppHost):
    """Test fixture that pre-populates every Settings pane on mount."""

    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(_make_eager_settings_screen()())


async def test_settings_footer_skips_recompose_between_same_kind_fields():
    """Tabbing between two Input rows must not rebuild the footer each time.

    Textual's stock Footer recomposes (tears down and remounts every
    FooterKey) on every focus change. The Generation tab is mostly Input
    rows, and moving focus between two of them never changes which keys
    the footer shows, so recomposing there is wasted work. Profiled before
    this fix: about 440ms per focus change tabbing through the tab.
    """
    app = SettingsPerfTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.15)
        tabbed = app.screen.query_one("#settings-tabs", TabbedContent)
        tabbed.active = "settings-tab-generation"
        await pilot.pause(0.15)
        pane = app.screen.query_one("#settings-tab-generation")
        inputs = list(pane.query(Input))
        assert len(inputs) >= 4, "Generation tab needs at least 4 Input rows for this test"

        footer = app.screen.query_one(StableFooter)
        # Settle on the first Input so the initial type-change transition
        # (from the tab strip, which is not an Input) does not count against
        # the bound below.
        app.screen.set_focus(inputs[0])
        await pilot.pause()

        with patch.object(footer, "recompose", wraps=footer.recompose) as recompose_spy:
            for widget in inputs[1:4]:
                app.screen.set_focus(widget)
                await pilot.pause()
            await pilot.pause(0.2)

        assert recompose_spy.call_count == 0, (
            "expected no footer recompose moving between Input rows, got "
            f"{recompose_spy.call_count}"
        )


async def test_toggle_fleet_and_sessions_checks_do_not_query_the_screen():
    """check_action for the drawer toggles must not walk the whole screen tree.

    ``Screen.query()`` walks every descendant to build its match list.
    check_action runs on every focus change across the whole app, so a
    full-tree query there scales with however many widgets the current
    screen has mounted, and the Settings screen's largest tab mounts
    dozens. The fleet/session drawer checks use a direct-children lookup
    instead.
    """
    app = SettingsPerfTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(Screen, "query", autospec=True, side_effect=Screen.query) as query_spy:
            app.check_action("toggle_fleet", ())
            app.check_action("toggle_sessions", ())
        assert query_spy.call_count == 0, (
            "expected no Screen.query calls from the drawer no-op checks, got "
            f"{query_spy.call_count}"
        )


async def test_first_direct_child_finds_only_immediate_children():
    """The drawer lookup helper does not recurse past direct children."""
    from textual.app import App
    from textual.containers import Vertical
    from textual.widgets import Static

    from lilbee.cli.tui.app import _first_direct_child

    class _ProbeApp(App[None]):
        def compose(self) -> ComposeResult:
            yield Vertical(Vertical(Static()), id="outer")

    app = _ProbeApp()
    async with app.run_test():
        outer = app.query_one("#outer", Vertical)
        inner = outer.query_one(Vertical)

        assert _first_direct_child(outer, Vertical) is inner
        assert _first_direct_child(outer, Static) is None


def test_stable_footer_skips_when_app_not_focused():
    """bindings_changed marks bindings ready but takes no other action while
    the terminal app itself lacks OS focus, matching stock Footer's guard."""
    from types import SimpleNamespace

    footer = StableFooter()
    fake_screen = SimpleNamespace(app=SimpleNamespace(app_focus=False))

    footer.bindings_changed(fake_screen)

    assert footer._bindings_ready is True
    assert footer._last_bindings_signature is None


def test_stable_footer_skips_when_not_attached_to_the_screen():
    """bindings_changed is a no-op for a footer that is not mounted (or
    briefly belongs to a screen mid-transition), matching stock Footer."""
    from types import SimpleNamespace

    footer = StableFooter()
    assert footer.is_attached is False
    fake_screen = SimpleNamespace(app=SimpleNamespace(app_focus=True))

    footer.bindings_changed(fake_screen)

    assert footer._bindings_ready is True
    assert footer._last_bindings_signature is None


def test_shows_placement_full_screen_true_only_on_fleet_screen():
    """The full-screen check is exact isinstance, not a subtree query."""
    from types import SimpleNamespace

    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.fleet import FleetScreen

    fleet_screen = FleetScreen.__new__(FleetScreen)

    assert LilbeeApp._shows_placement_full_screen(SimpleNamespace(screen=object())) is False
    assert LilbeeApp._shows_placement_full_screen(SimpleNamespace(screen=fleet_screen)) is True
