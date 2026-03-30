"""Main Textual app for lilbee TUI."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import App
from textual.binding import Binding, BindingType

from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.config import cfg

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
        Binding("f1", "push_help", "?/F1 Help", show=True),
        Binding("question_mark", "push_help", "Help", show=False),
        Binding("f2", "push_catalog", "F2 Models", show=True),
        Binding("f3", "push_status", "F3 Status", show=True),
        Binding("f4", "push_settings", "F4 Settings", show=True),
        Binding("ctrl+t", "cycle_theme", "Theme", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
    ]

    def __init__(self, *, auto_sync: bool = False) -> None:
        super().__init__()
        self._auto_sync = auto_sync
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
        self.notify(f"Theme: {name}")

    def set_theme(self, name: str) -> None:
        """Set theme by name (used by /theme command)."""
        if name in self.available_themes:
            self.theme = name

    def action_push_catalog(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        self.push_screen(CatalogScreen())

    def action_push_help(self) -> None:
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        self.push_screen(HelpModal())

    def action_push_status(self) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen

        self.push_screen(StatusScreen())

    def action_push_settings(self) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        self.push_screen(SettingsScreen())
