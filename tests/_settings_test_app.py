"""Settings screen app hosts and wait helpers that tests share."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.widgets import Footer

from tests._async_wait import subtree_mounted, wait_until
from tests._lilbee_app_test_host import LilbeeAppHost


def _make_eager_settings_screen(*, eager: bool = True, keys: tuple[str, ...] = ()):
    """SettingsScreen subclass that populates every pane or only the panes holding *keys*.

    Production lazy-mounts non-active panes on activation; the test
    suite predates that pattern and queries editors across all panes
    by id, so this subclass hooks ``on_mount`` to walk pane groups
    and populate every body before tests query. CSS_PATH is pinned
    to the original module's directory because Textual otherwise
    resolves it next to the test file.

    Building every pane styles hundreds of widgets, which takes seconds alone
    and over a minute under gate load. With ``eager=False``, only the panes
    holding *keys* build on mount and every other pane stays lazy.
    """
    import importlib
    from pathlib import Path

    from lilbee.cli.tui.screens.settings import SettingsScreen

    src_module = importlib.import_module(SettingsScreen.__module__)
    css_path = str(Path(src_module.__file__ or "").parent / "settings.tcss")

    class _EagerSettingsScreen(SettingsScreen):
        CSS_PATH = css_path

        # Textual runs the on_mount of every class in the tree, SettingsScreen's after this one.
        def on_mount(self) -> None:
            if eager:
                self.populate_all_panes()
            else:
                for key in keys:
                    populate_pane_of(self, key)

    return _EagerSettingsScreen


class SettingsTestApp(LilbeeAppHost):
    """Test fixture that pre-populates every Settings pane on mount."""

    CSS = ""
    # A subclass sets this, not its own on_mount: Textual runs every on_mount in the class tree.
    eager_panes: ClassVar[bool] = True
    settings_to_populate: tuple[str, ...] = ()

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(
            _make_eager_settings_screen(eager=self.eager_panes, keys=self.settings_to_populate)()
        )


class LazySettingsTestApp(SettingsTestApp):
    """Test fixture that builds selected Settings panes and leaves the rest lazy."""

    eager_panes = False

    def __init__(self, *keys: str) -> None:
        super().__init__()
        self.settings_to_populate = keys


def populate_pane_of(screen, key: str) -> None:
    """Build only the pane that holds the editor of setting *key*."""
    for pane_id, group in screen._pane_groups.items():
        if any(name == key for name, _defn in group.items):
            screen._populate_pane(pane_id)


async def _wait_for_pane_rows(app, pilot, pane_id):
    """Wait until *pane_id* holds rows and the screen, each dropdown included, has mounted.

    A dropdown that still mounts when the app exits raises for the overlay it misses.
    """
    from textual.widgets import Select

    def _ready() -> bool:
        pane = app.screen.query_one(f"#{pane_id}")
        return (
            bool(pane.query(".setting-row"))
            and subtree_mounted(app.screen)
            and all(select.query("SelectOverlay") for select in app.screen.query(Select))
        )

    assert await wait_until(pilot, _ready, timeout=5.0), f"{pane_id} did not mount all its rows"


async def _blur_after_focus(pilot, app, editor) -> None:
    """Focus *editor*, then move focus to another row so its blur handler runs."""
    from textual.widgets import Input

    editor.focus()
    await wait_until(pilot, lambda: app.screen.focused is editor)
    app.screen.query_one("#ed-max_distance", Input).focus()
    await wait_until(pilot, lambda: not editor.has_focus)
    for _ in range(5):
        await pilot.pause()
