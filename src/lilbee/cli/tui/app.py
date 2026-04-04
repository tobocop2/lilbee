"""Main Textual app for lilbee TUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from textual.app import App
from textual.binding import Binding, BindingType

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.commands import LilbeeCommandProvider
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


class LilbeeApp(App[None]):
    """Full-screen TUI for lilbee knowledge base."""

    TITLE = "lilbee"
    CSS_PATH = Path(__file__).parent / "theme.tcss"
    ENABLE_COMMAND_PALETTE = True
    COMMANDS = {LilbeeCommandProvider}  # noqa: RUF012

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("question_mark", "push_help", "Help", show=False),
        Binding("f1", "push_help", "Help", show=False),
        Binding("ctrl+h", "push_help", "Help", show=False),
        Binding("f2", "push_catalog", "Models", show=False),
        Binding("ctrl+n", "push_catalog", "Models", show=False),
        Binding("f3", "push_status", "Status", show=False),
        Binding("ctrl+s", "push_status", "Status", show=False),
        Binding("f4", "push_settings", "Settings", show=False),
        Binding("ctrl+e", "push_settings", "Settings", show=False),
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        Binding("1", "switch_chat", "Chat", show=False),
        Binding("2", "switch_models", "Models", show=False),
        Binding("3", "switch_status", "Status", show=False),
        Binding("4", "switch_settings", "Settings", show=False),
        Binding("h", "nav_prev", "Prev", show=False),
        Binding("left", "nav_prev", "Prev", show=False),
        Binding("l", "nav_next", "Next", show=False),
        Binding("right", "nav_next", "Next", show=False),
        Binding("ctrl+c", "quit", "Cancel/Quit", show=False, priority=True),
    ]

    def __init__(self, *, auto_sync: bool = False) -> None:
        super().__init__()
        self._auto_sync = auto_sync
        self._active_view = "Chat"
        self._theme_index = 0

    def on_mount(self) -> None:
        self.title = f"lilbee — {cfg.chat_model}"
        self.theme = _DEFAULT_THEME

        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen(auto_sync=self._auto_sync))

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
        """Context-aware Ctrl+C: cancel active task > cancel stream > quit."""
        task_bar = getattr(self, "_task_bar", None)
        if task_bar and not task_bar.queue.is_empty:
            active = task_bar.queue.active_task
            if active:
                task_bar.cancel_task(active.task_id)
                self.notify(msg.APP_CANCELLED)
                return
        screen = self.screen
        if hasattr(screen, "_streaming") and screen._streaming:
            screen.action_cancel_stream()  # type: ignore[attr-defined]
            return
        self.exit()

    def _switch_view(self, view_name: str) -> None:
        """Switch to a named view, popping any overlay screens first."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.screens.settings import SettingsScreen
        from lilbee.cli.tui.screens.status import StatusScreen

        # Pop non-chat screens until we're back at chat
        while len(self.screen_stack) > 1 and not isinstance(self.screen, ChatScreen):
            self.pop_screen()

        if view_name == "Chat":
            from textual.widgets import Input

            self.call_later(lambda: self.screen.query_one("#chat-input", Input).focus())
        elif view_name == "Models":
            self.push_screen(CatalogScreen())
        elif view_name == "Status":
            self.push_screen(StatusScreen())
        elif view_name == "Settings":
            self.push_screen(SettingsScreen())

        # Update NavBar on current screen and persist state for new screens
        self._active_view = view_name
        try:
            nav = self.screen.query_one("#global-nav-bar", NavBar)
            nav.active_view = view_name
        except Exception:
            log.debug("NavBar update failed", exc_info=True)

    def action_push_catalog(self) -> None:
        self._switch_view("Models")

    def action_push_help(self) -> None:
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        self.push_screen(HelpModal())

    def action_push_status(self) -> None:
        self._switch_view("Status")

    def action_push_settings(self) -> None:
        self._switch_view("Settings")

    def action_switch_chat(self) -> None:
        self._switch_view("Chat")

    def action_switch_models(self) -> None:
        self._switch_view("Models")

    def action_switch_status(self) -> None:
        self._switch_view("Status")

    def action_switch_settings(self) -> None:
        self._switch_view("Settings")

    def action_nav_prev(self) -> None:
        """Navigate to previous view (h or left arrow)."""
        view_names = ["Chat", "Models", "Status", "Settings"]
        current_idx = view_names.index(self._active_view)
        self._switch_view(view_names[(current_idx - 1) % 4])

    def action_nav_next(self) -> None:
        """Navigate to next view (l or right arrow)."""
        view_names = ["Chat", "Models", "Status", "Settings"]
        current_idx = view_names.index(self._active_view)
        self._switch_view(view_names[(current_idx + 1) % 4])
