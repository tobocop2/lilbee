"""Main Textual app for lilbee TUI."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.signal import Signal

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.cli.tui.events import ModelChanged
from lilbee.config import cfg
from lilbee.services import reset_services

log = logging.getLogger(__name__)

_DEFAULT_THEME = "gruvbox"  # warm retro CRT aesthetic
_CHAT_SCREEN_NAME = "chat"
DARK_THEMES = (
    "monokai",
    "dracula",
    "tokyo-night",
    "nord",
    "gruvbox",
    "catppuccin-mocha",
    "catppuccin-frappe",
    "atom-one-dark",
    "rose-pine",
    "solarized-dark",
    "textual-dark",
)


def _make_catalog() -> Screen:
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    return CatalogScreen()


def _make_status() -> Screen:
    from lilbee.cli.tui.screens.status import StatusScreen

    return StatusScreen()


def _make_settings() -> Screen:
    from lilbee.cli.tui.screens.settings import SettingsScreen

    return SettingsScreen()


def _make_tasks() -> Screen:
    from lilbee.cli.tui.screens.task_center import TaskCenter

    return TaskCenter()


def _make_wiki() -> Screen:
    from lilbee.cli.tui.screens.wiki import WikiScreen

    return WikiScreen()


_BASE_VIEWS: dict[str, Callable[[], Screen]] = {
    "Catalog": _make_catalog,
    "Status": _make_status,
    "Settings": _make_settings,
    "Tasks": _make_tasks,
}


def get_views() -> dict[str, Callable[[], Screen]]:
    """Return the active view factories, including wiki when enabled."""
    views = dict(_BASE_VIEWS)
    if cfg.wiki:
        views["Wiki"] = _make_wiki
    return views


class LilbeeApp(App[None]):
    """Full-screen TUI for lilbee knowledge base."""

    TITLE = "lilbee"
    CSS_PATH = Path(__file__).parent / "app.tcss"
    ENABLE_COMMAND_PALETTE = True
    COMMANDS = {LilbeeCommandProvider}  # noqa: RUF012

    _NAV_GROUP = Binding.Group("Navigate")

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("question_mark", "push_help", "Help", show=True),
        Binding("f1", "push_help", "Help", show=False),
        Binding("ctrl+h", "push_help", "Help", show=False),
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        Binding("t", "open_tasks", "Tasks", show=False),
        # priority=True is required: even though NavAwareInput lets [ and ]
        # bubble past Input.check_consume_key, Textual's focused Input still
        # handles printable keys in _on_key before a non-priority ancestor
        # binding can fire. Both NavAwareInput and priority=True are needed.
        Binding(
            "left_square_bracket",
            "nav_prev",
            "Prev",
            show=True,
            group=_NAV_GROUP,
            priority=True,
        ),
        Binding(
            "right_square_bracket",
            "nav_next",
            "Next",
            show=True,
            group=_NAV_GROUP,
            priority=True,
        ),
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
    ]

    def __init__(self, *, auto_sync: bool = False, initial_view: str | None = None) -> None:
        super().__init__()
        self._auto_sync = auto_sync
        self._initial_view = initial_view
        self.active_view = msg.DEFAULT_VIEW
        self._switching = False
        self._theme_index = 0
        self.last_quit_time: float = 0.0
        self.settings_changed_signal: Signal[tuple[str, object]] = Signal(self, "settings_changed")
        self.model_changed_signal: Signal[ModelChanged] = Signal(self, "model_changed")
        from lilbee.cli.tui.widgets.task_bar import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()  # screens compose their own ViewTabs + Footer

    def on_mount(self) -> None:
        self.title = f"lilbee — {cfg.chat_model}"
        self.theme = _DEFAULT_THEME

        from lilbee.cli.tui.screens.chat import ChatScreen

        chat = ChatScreen(auto_sync=self._auto_sync)
        self.install_screen(chat, name=_CHAT_SCREEN_NAME)
        self.push_screen(_CHAT_SCREEN_NAME)
        if self._initial_view and self._initial_view != msg.DEFAULT_VIEW:
            self.switch_view(self._initial_view)

    def action_cycle_theme(self) -> None:
        self._theme_index = (self._theme_index + 1) % len(DARK_THEMES)
        name = DARK_THEMES[self._theme_index]
        self.theme = name
        self.notify(msg.THEME_SET.format(name=name))

    def set_theme(self, name: str) -> None:
        """Set theme by name (used by /theme command)."""
        if name in self.available_themes:
            self.theme = name

    async def action_quit(self) -> None:
        """Context-aware Ctrl+C: cancel active task > cancel stream > quit.
        On second Ctrl+C (within 2s), force-exits via os._exit to handle
        cases where the GIL is held by native code.
        """
        import time

        now = time.monotonic()
        if now - self.last_quit_time < 2.0:
            self._force_quit()
            return
        self.last_quit_time = now

        if not self.task_bar.queue.is_empty:
            active = self.task_bar.queue.active_task
            if active:
                self.task_bar.cancel_task(active.task_id)
                self.notify(msg.APP_CANCELLED)
                return
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.screens.setup import SetupWizard

        screen = self.screen
        if isinstance(screen, SetupWizard):
            screen.action_cancel()
            return
        if isinstance(screen, ChatScreen) and screen.streaming:
            screen.action_cancel_stream()
            return
        self.exit()

    def _force_quit(self) -> None:
        """Force-exit when normal quit is blocked (e.g. GIL held by native code)."""
        import os

        with contextlib.suppress(Exception):
            reset_services()
        os._exit(1)

    def switch_view(self, view_name: str) -> None:
        """Switch to a named view via lazy screen factories.

        Guards against concurrent switches: ``switch_screen`` is async
        (processed on the next event-loop tick) but callers read
        ``active_view`` synchronously. Without a guard, rapid keypresses
        queue conflicting switches that corrupt the screen stack.
        ``active_view`` is updated after the switch completes.
        """
        if self._switching:
            return
        self._switching = True

        if view_name == "Chat":
            from lilbee.cli.tui.screens.chat import ChatScreen

            if not isinstance(self.screen, ChatScreen):
                self.switch_screen(_CHAT_SCREEN_NAME)
            # Already on Chat, just update state below.
        else:
            factory = get_views().get(view_name)
            if factory is None:
                self._switching = False
                return
            self.switch_screen(factory())

        def _finish() -> None:
            self.active_view = view_name
            self._switching = False

        self.call_later(_finish)

    def action_push_help(self) -> None:
        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()

    def action_open_tasks(self) -> None:
        """Jump to the Task Center screen (t key)."""
        self.switch_view("Tasks")

    def action_nav_prev(self) -> None:
        """Navigate to previous view ([ key)."""
        view_names = msg.get_nav_views()
        current_idx = view_names.index(self.active_view)
        self.switch_view(view_names[(current_idx - 1) % len(view_names)])

    def action_nav_next(self) -> None:
        """Navigate to next view (] key)."""
        view_names = msg.get_nav_views()
        current_idx = view_names.index(self.active_view)
        self.switch_view(view_names[(current_idx + 1) % len(view_names)])
