"""Main Textual app for lilbee TUI."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.signal import Signal

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.cli.tui.events import ModelChanged
from lilbee.cli.tui.widgets.nav_bar import NavBar
from lilbee.config import cfg

log = logging.getLogger(__name__)

_READY_FILE = "lilbee-splash-ready"

_DEFAULT_THEME = "gruvbox"  # warm retro CRT aesthetic
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


def _get_chat_screen() -> Screen:
    from lilbee.cli.tui.screens.chat import ChatScreen

    return ChatScreen()


def _get_catalog_screen() -> Screen:
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    return CatalogScreen()


def _get_status_screen() -> Screen:
    from lilbee.cli.tui.screens.status import StatusScreen

    return StatusScreen()


def _get_settings_screen() -> Screen:
    from lilbee.cli.tui.screens.settings import SettingsScreen

    return SettingsScreen()


def _get_tasks_screen() -> Screen:
    from lilbee.cli.tui.screens.task_center import TaskCenter

    return TaskCenter()


class LilbeeApp(App[None]):
    """Full-screen TUI for lilbee knowledge base."""

    TITLE = "lilbee"
    CSS_PATH = Path(__file__).parent / "app.tcss"
    ENABLE_COMMAND_PALETTE = True
    COMMANDS = {LilbeeCommandProvider}  # noqa: RUF012

    MODES = {  # noqa: RUF012
        "chat": _get_chat_screen,
        "models": _get_catalog_screen,
        "status": _get_status_screen,
        "settings": _get_settings_screen,
        "tasks": _get_tasks_screen,
    }

    _MODE_FOR_VIEW: ClassVar[dict[str, str]] = {
        "Chat": "chat",
        "Models": "models",
        "Status": "status",
        "Settings": "settings",
        "Tasks": "tasks",
    }

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("question_mark", "push_help", "? help", show=True),
        Binding("f1", "push_help", "Help", show=False),
        Binding("ctrl+h", "push_help", "Help", show=False),
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        Binding("h", "nav_prev", "Prev", show=False),
        Binding("left", "nav_prev", "Prev", show=False),
        Binding("l", "nav_next", "Next", show=False),
        Binding("right", "nav_next", "Next", show=False),
        Binding("ctrl+c", "quit", "^c cancel/quit", show=True, priority=True),
    ]

    def __init__(self, *, auto_sync: bool = False) -> None:
        super().__init__()
        self._auto_sync = auto_sync
        self.active_view = "Chat"
        self._theme_index = 0
        self.last_quit_time: float = 0.0
        self.settings_changed_signal: Signal[tuple[str, object]] = Signal(self, "settings_changed")
        self.model_changed_signal: Signal[ModelChanged] = Signal(self, "model_changed")
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        self.task_bar = TaskBar(id="app-task-bar")

    def compose(self) -> ComposeResult:
        yield NavBar(id="global-nav-bar")

    def on_mount(self) -> None:
        self.title = f"lilbee — {cfg.chat_model}"
        self.theme = _DEFAULT_THEME
        self.mount(self.task_bar)
        self.switch_mode("chat")

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

        screen = self.screen
        if isinstance(screen, ChatScreen) and screen._streaming:
            screen.action_cancel_stream()
            return
        self.exit()

    def _force_quit(self) -> None:
        """Force-exit when normal quit is blocked (e.g. GIL held by native code)."""
        import os

        from lilbee.services import reset_services

        with contextlib.suppress(Exception):
            reset_services()
        os._exit(1)

    def switch_view(self, view_name: str) -> None:
        """Switch to a named view using Textual modes."""
        mode_name = self._MODE_FOR_VIEW.get(view_name)
        if mode_name is None:
            return
        self.switch_mode(mode_name)
        self.active_view = view_name
        self._update_nav(view_name)

    def _update_nav(self, view_name: str) -> None:
        """Update the app-level NavBar after a mode switch."""
        try:
            nav = self.query_one("#global-nav-bar", NavBar)
            nav.active_view = view_name
        except Exception:
            log.debug("NavBar update failed", exc_info=True)

    def action_push_help(self) -> None:
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        self.push_screen(HelpModal())

    def action_nav_prev(self) -> None:
        """Navigate to previous view (h or left arrow)."""
        view_names = msg.NAV_VIEWS
        current_idx = view_names.index(self.active_view)
        self.switch_view(view_names[(current_idx - 1) % len(view_names)])

    def action_nav_next(self) -> None:
        """Navigate to next view (l or right arrow)."""
        view_names = msg.NAV_VIEWS
        current_idx = view_names.index(self.active_view)
        self.switch_view(view_names[(current_idx + 1) % len(view_names)])
