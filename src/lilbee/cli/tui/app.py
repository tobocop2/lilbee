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

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.core.services import get_services
from lilbee.providers.llama_cpp.abort_signal import request_abort

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


def _view_screen_name(view_name: str) -> str:
    """Stable install_screen identifier for a top-level view (lower-cased)."""
    return view_name.lower()


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


_MODEL_REF_KEYS = frozenset({"chat_model", "embedding_model", "vision_model", "reranker_model"})


def _on_settings_changed_evict_cache(payload: tuple[str, object]) -> None:
    """Drop loaded-model state when a load-affecting setting changes."""
    from lilbee.providers.llama_cpp.provider import LOAD_AFFECTING_KEYS

    key, _value = payload
    if key in LOAD_AFFECTING_KEYS:
        get_services().provider.invalidate_load_cache()
    if key in _MODEL_REF_KEYS:
        from lilbee.modelhub.model_info import invalidate_cache

        invalidate_cache()


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
        # Non-priority: a focused Input or TextArea consumes the printable
        # before this binding fires, so brackets type literally inside any
        # input. With no input focused, the bindings reach the app and
        # navigate. This mirrors vim-style insert vs. normal modes.
        Binding("left_square_bracket", "nav_prev", "Prev", show=True, group=_NAV_GROUP),
        Binding("right_square_bracket", "nav_next", "Next", show=True, group=_NAV_GROUP),
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
    ]

    def __init__(self, *, auto_sync: bool = False, initial_view: str | None = None) -> None:
        super().__init__()
        self._auto_sync = auto_sync
        self._initial_view = initial_view
        self.active_view = msg.DEFAULT_VIEW
        self._switching = False
        self._theme_index = 0
        # Names of non-Chat screens already installed via install_screen.
        # Subsequent visits switch by name to reuse the same instance,
        # so Footer / signal / worker wiring runs once per session.
        self._installed_screen_names: set[str] = set()
        self.settings_changed_signal: Signal[tuple[str, object]] = Signal(self, "settings_changed")
        self.provider_availability_changed_signal: Signal[tuple[str, object]] = Signal(
            self, "provider_availability_changed"
        )
        from lilbee.cli.tui.widgets.task_bar import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()  # screens compose their own ViewTabs + Footer

    def on_mount(self) -> None:
        self._canonicalize_persisted_models()
        self.title = f"lilbee: {cfg.chat_model}"
        # Restore the persisted theme so the TUI opens in whatever the user
        # picked last session, not always the gruvbox default.
        persisted = cfg.theme or _DEFAULT_THEME
        self.theme = persisted if persisted in self.available_themes else _DEFAULT_THEME
        self._sync_theme_index_to_current()

        self.settings_changed_signal.subscribe(self, _on_settings_changed_evict_cache)
        self.settings_changed_signal.subscribe(self, self._fan_out_provider_availability)

        from lilbee.cli.tui.screens.chat import ChatScreen

        chat = ChatScreen(auto_sync=self._auto_sync)
        self.install_screen(chat, name=_CHAT_SCREEN_NAME)
        self.push_screen(_CHAT_SCREEN_NAME)
        if self._initial_view and self._initial_view != msg.DEFAULT_VIEW:
            self.switch_view(self._initial_view)

    def _canonicalize_persisted_models(self) -> None:
        """Swap stale persisted refs to a working fallback for this session."""
        from lilbee.modelhub.model_manager import (
            ValidationResult,
            canonicalize_chat_model,
            canonicalize_embedding_model,
        )

        for canon, field, label in (
            (canonicalize_chat_model(), "chat_model", "Chat"),
            (canonicalize_embedding_model(), "embedding_model", "Embedding"),
        ):
            if canon.status == ValidationResult.OK or canon.original == canon.effective:
                continue
            setattr(cfg, field, canon.effective)
            self.notify(
                msg.MODEL_FALLBACK_NOTICE.format(
                    label=label, original=canon.original, effective=canon.effective
                ),
                severity="warning",
                timeout=8,
            )

    def _fan_out_provider_availability(self, payload: tuple[str, object]) -> None:
        """Republish on provider_availability_changed_signal when an API key changes."""
        from lilbee.core.config.keys import PROVIDER_API_KEYS

        key, value = payload
        if key in PROVIDER_API_KEYS:
            self.provider_availability_changed_signal.publish((key, value))

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
        """Single write boundary for active model refs."""
        setattr(cfg, key, value)
        normalized = getattr(cfg, key)
        settings.set_value(cfg.data_root, key, normalized)
        self.settings_changed_signal.publish((key, normalized))

    def _sync_theme_index_to_current(self) -> None:
        """Align cycle index with the active theme."""
        try:
            self._theme_index = DARK_THEMES.index(self.theme)
        except ValueError:
            self._theme_index = 0

    async def action_quit(self) -> None:
        """Context-aware Ctrl+C: cancel active task > cancel stream > quit."""
        request_abort()

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

    def switch_view(self, view_name: str) -> None:
        """Switch to a named view, installing each screen at most once.

        Guards against concurrent switches via ``self._switching`` so
        rapid keypresses don't corrupt the screen stack.
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
            screen_name = _view_screen_name(view_name)
            if screen_name not in self._installed_screen_names:
                self.install_screen(factory(), name=screen_name)
                self._installed_screen_names.add(screen_name)
            self.switch_screen(screen_name)

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
    """Route model writes through set_active_model, falling back to direct cfg+settings writes."""
    if isinstance(host_app, LilbeeApp):
        host_app.set_active_model(key, value)
        return
    setattr(cfg, key, value)
    settings.set_value(cfg.data_root, key, getattr(cfg, key))
