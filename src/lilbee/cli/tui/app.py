"""Main Textual app for lilbee TUI."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.signal import Signal

from lilbee import settings
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.config import cfg
from lilbee.services import get_services, reset_services

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


def _on_settings_changed_evict_cache(payload: tuple[str, object]) -> None:
    """Drop loaded-model state when a load-affecting setting changes."""
    # Lazy: llama_cpp_provider's transitive imports cost ~500ms.
    from lilbee.providers.llama_cpp_provider import LOAD_AFFECTING_KEYS

    key, _value = payload
    if key in LOAD_AFFECTING_KEYS:
        get_services().provider.invalidate_load_cache()


# Terminal-mode reset sequences emitted on a force-quit so the next shell
# command isn't fed escape bytes from a still-armed mode. The targeted
# modes (bracketed paste, mouse tracking, alt-screen) are the same ones
# Textual's driver normally resets at app exit. (bb-6b86)
_RESET_BRACKETED_PASTE = "\x1b[?2004l"
_RESET_MOUSE_TRACKING = "\x1b[?1003l"
_RESET_ALT_SCREEN = "\x1b[?1049l"
_RESET_SHOW_CURSOR = "\x1b[?25h"

_TERMINAL_RESET_SEQUENCE = (
    _RESET_BRACKETED_PASTE + _RESET_MOUSE_TRACKING + _RESET_ALT_SCREEN + _RESET_SHOW_CURSOR
)


def _reset_terminal_modes() -> None:
    """Emit terminal-mode resets so the shell after a force-quit isn't broken.

    Best-effort: write to stdout if it's a TTY, swallow any error so the
    caller can still proceed to ``os._exit``. Skips the writes entirely on
    non-tty stdout (CI, pipes) since there's no terminal state to reset.
    """
    import sys

    try:
        if not sys.stdout.isatty():
            return
        sys.stdout.write(_TERMINAL_RESET_SEQUENCE)
        sys.stdout.flush()
    except (OSError, ValueError):
        # ValueError fires when stdout is closed; OSError covers EBADF and
        # write errors on a still-open but unwritable fd.
        return


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
        Binding("ctrl+t", "cycle_theme", "Theme", show=True),
        Binding("t", "open_tasks", "Tasks", show=True),
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
        from lilbee.cli.tui.widgets.task_bar import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()  # screens compose their own ViewTabs + Footer

    def on_mount(self) -> None:
        self.title = f"lilbee — {cfg.chat_model}"
        # Restore the persisted theme so the TUI opens in whatever the user
        # picked last session, not always the gruvbox default.
        persisted = cfg.theme or _DEFAULT_THEME
        self.theme = persisted if persisted in self.available_themes else _DEFAULT_THEME
        self._sync_theme_index_to_current()

        self.settings_changed_signal.subscribe(self, _on_settings_changed_evict_cache)

        from lilbee.cli.tui.screens.chat import ChatScreen

        chat = ChatScreen(auto_sync=self._auto_sync)
        self.install_screen(chat, name=_CHAT_SCREEN_NAME)
        self.push_screen(_CHAT_SCREEN_NAME)
        if self._initial_view and self._initial_view != msg.DEFAULT_VIEW:
            self.switch_view(self._initial_view)

    def action_cycle_theme(self) -> None:
        self._theme_index = (self._theme_index + 1) % len(DARK_THEMES)
        name = DARK_THEMES[self._theme_index]
        self._apply_and_persist_theme(name)
        self.notify(msg.THEME_SET.format(name=name))

    def set_theme(self, name: str) -> None:
        """Set theme by name (used by /theme command). Persists across sessions."""
        if name in self.available_themes:
            self._apply_and_persist_theme(name)
            self._sync_theme_index_to_current()

    def _apply_and_persist_theme(self, name: str) -> None:
        """Apply *name* live and write it to config.toml."""
        self.theme = name
        cfg.theme = name
        settings.set_value(cfg.data_root, "theme", name)

    def set_active_model(self, key: str, value: str) -> None:
        """Single write boundary for active model refs; persists the
        post-validator value so subscribers see the normalized form."""
        setattr(cfg, key, value)
        normalized = getattr(cfg, key)
        settings.set_value(cfg.data_root, key, normalized)
        self.settings_changed_signal.publish((key, normalized))

    def _sync_theme_index_to_current(self) -> None:
        """Align the cycle index with the active theme so Ctrl+T moves from there."""
        try:
            self._theme_index = DARK_THEMES.index(self.theme)
        except ValueError:
            self._theme_index = 0

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
        """Force-exit when normal quit is blocked (e.g. GIL held by native code).

        ``os._exit`` skips Textual's driver teardown, which is the only path
        that resets terminal modes (alt-screen, mouse tracking, bracketed
        paste). Without an explicit reset, the next shell command receives
        terminal escape bytes as input and rejects them with
        ``No such command '4'`` and friends. (bb-6b86)
        """
        import os

        with contextlib.suppress(Exception):
            reset_services()
        _reset_terminal_modes()
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
            # ViewTabs.on_mount captured active_view before this callback
            # runs, so the highlight would lag by one step without this push.
            with contextlib.suppress(NoMatches):
                self.screen.query_one(ViewTabs).active_view = view_name

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


def apply_active_model(host_app: App[Any], key: str, value: str) -> None:
    """Route model writes through LilbeeApp.set_active_model; bare-App fallback for tests."""
    if isinstance(host_app, LilbeeApp):
        host_app.set_active_model(key, value)
        return
    setattr(cfg, key, value)
    settings.set_value(cfg.data_root, key, getattr(cfg, key))
