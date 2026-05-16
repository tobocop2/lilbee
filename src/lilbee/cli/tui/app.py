"""Main Textual app for lilbee TUI."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, cast

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.signal import Signal
from textual.widgets import Input, TextArea

from lilbee.app.services import get_services
from lilbee.app.themes import DARK_THEMES
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.core.config import cfg
from lilbee.providers.worker.transport import WorkerRole

log = logging.getLogger(__name__)

_DEFAULT_THEME = "rose-pine"  # muted, low-glare; easier on the eyes than the warmer themes
_CHAT_SCREEN_NAME = "chat"


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


class LilbeeApp(App[None]):
    """Full-screen TUI for lilbee knowledge base."""

    TITLE = "lilbee"
    CSS_PATH = Path(__file__).parent / "app.tcss"
    ENABLE_COMMAND_PALETTE = True
    COMMANDS = {LilbeeCommandProvider}  # noqa: RUF012

    _NAV_GROUP = Binding.Group("Navigate")

    BINDINGS: ClassVar[list[BindingType]] = [
        # ``?`` is non-priority so a focused TextArea (chat input in INSERT
        # mode) can swallow it and type the literal character. F1 / Ctrl+H
        # remain priority routes that always open help, even mid-typing.
        Binding("question_mark", "push_help", "Help", show=False),
        Binding("f1", "push_help", "Help", show=True, priority=True),
        Binding("ctrl+h", "push_help", "Help", show=False, priority=True),
        Binding("escape", "dismiss_help_if_open", "Close help", show=False, priority=True),
        Binding("ctrl+t", "cycle_theme", "Theme", show=True),
        Binding("t", "open_tasks", "Tasks", show=True),
        # Non-priority so Chat's "focus_commands" and Catalog's
        # "focus_search" still win on those screens. Fires only on
        # screens that don't bind slash themselves, routing the user
        # to Chat with the slash already typed.
        Binding("slash", "global_slash_to_chat", "Command", show=False),
        # priority=True so a focused TextArea cannot swallow the bracket
        # under stress (multi-key send-keys etc.); type literal brackets
        # via Shift+[ / Shift+] which produce { / } and bypass these.
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
        Binding("S", "run_sync", "Sync", show=False, priority=True),
    ]

    def __init__(self, *, initial_view: str | None = None) -> None:
        super().__init__()
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
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()  # screens compose their own ViewTabs + Footer

    # Test seam: the TUI test fixtures subclass LilbeeApp and set this to True
    # so on_mount short-circuits before the heavyweight setup (model
    # canonicalization, ChatScreen install, signal subscriptions, sync probe).
    # Production never sets it. See tests/_lilbee_app_test_host.py.
    _test_skip_auto_init: ClassVar[bool] = False

    def on_mount(self) -> None:
        if self._test_skip_auto_init:
            return
        self._canonicalize_persisted_models()
        self.title = f"lilbee: {cfg.chat_model}"
        # Restore the persisted theme so the TUI opens in whatever the user
        # picked last session, not always the default.
        persisted = cfg.theme or _DEFAULT_THEME
        self.theme = persisted if persisted in self.available_themes else _DEFAULT_THEME
        self._sync_theme_index_to_current()

        self.settings_changed_signal.subscribe(self, self._fan_out_provider_availability)
        self._wire_worker_pool_notifications()

        from lilbee.cli.tui.screens.chat import ChatScreen

        chat = ChatScreen()
        self.install_screen(chat, name=_CHAT_SCREEN_NAME)
        self.push_screen(_CHAT_SCREEN_NAME)
        if self._initial_view and self._initial_view != msg.DEFAULT_VIEW:
            self.switch_view(self._initial_view)
        # Cheap detection only: filesystem walk + hash compare. The user
        # initiates sync explicitly via S or the command palette.
        self.task_bar.start_detect_pending()

    def _wire_worker_pool_notifications(self) -> None:
        """Surface worker spawn lifecycle in the bottom TaskBar.

        Worker spawns happen on the pool runtime thread, not the TUI's main
        loop, so the listeners marshal back via :meth:`call_from_thread`
        before mutating controller state. A single TaskBar hint covers all
        in-flight roles instead of one toast per role; the chat surface is
        for user content, not implementation detail.
        """

        def _on_spawning(role: WorkerRole) -> None:
            self.call_from_thread(self.task_bar.mark_role_spawning, role.value)

        def _on_spawned(role: WorkerRole) -> None:
            self.call_from_thread(self.task_bar.mark_role_spawned, role.value)

        get_services().add_pool_listener(on_spawning=_on_spawning, on_spawned=_on_spawned)

    def _canonicalize_persisted_models(self) -> None:
        """Swap stale persisted refs to a working fallback, persist, and log once."""
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
            from lilbee.app.settings import apply_settings_update

            apply_settings_update({field: canon.effective})
            log.warning(
                msg.MODEL_FALLBACK_NOTICE.format(
                    label=label, original=canon.original, effective=canon.effective
                )
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
        from lilbee.app.settings import apply_settings_update

        self.theme = name
        apply_settings_update({"theme": name})

    def set_active_model(self, key: str, value: str) -> None:
        """Persist an active model ref through the shared write boundary.

        Refs whose download is still queued or active are refused before the
        boundary runs, so a half-pulled file cannot land in a model slot.
        """
        from lilbee.app.settings import apply_settings_update

        downloading = self.task_bar.downloading_label_for(value)
        if downloading is not None:
            self.notify(
                msg.MODEL_BEING_DOWNLOADED.format(name=downloading),
                severity="warning",
            )
            return
        try:
            apply_settings_update({key: value})
        except ValueError as exc:
            self.notify(msg.MODEL_ASSIGN_REJECTED.format(error=exc), severity="error")
            return
        self.settings_changed_signal.publish((key, getattr(cfg, key)))

    def set_setting(self, key: str, value: object) -> None:
        """Apply a setting through the shared write boundary, then fan out to the UI."""
        from lilbee.app.settings import apply_settings_update

        apply_settings_update({key: value})
        normalized = getattr(cfg, key)
        if key == "theme" and isinstance(normalized, str) and normalized in self.available_themes:
            self.theme = normalized
            self._sync_theme_index_to_current()
        self.settings_changed_signal.publish((key, normalized))

    def _sync_theme_index_to_current(self) -> None:
        """Align cycle index with the active theme."""
        try:
            self._theme_index = DARK_THEMES.index(self.theme)
        except ValueError:
            self._theme_index = 0

    async def action_quit(self) -> None:
        """Context-aware Ctrl+C: cancel active task > cancel stream > quit."""
        get_services().cancel_inference()

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

    def action_command_palette(self) -> None:
        """Ctrl+P: cycle the chat dropdown if visible, else open the palette."""
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        screen = self.screen
        if isinstance(screen, ChatScreen):
            try:
                overlay = screen.query_one("#completion-overlay", CompletionOverlay)
            except NoMatches:
                overlay = None
            if overlay is not None and overlay.is_visible:
                overlay.cycle_prev()
                return
        super().action_command_palette()

    def action_dismiss_help_if_open(self) -> None:
        """Esc dismisses the HelpPanel when it is open; otherwise no-op.

        Without this, focus inside the panel could prevent ``?`` from
        toggling it back off and the user had no key to escape with.
        Bubble the Escape so screens can still receive it when no panel
        is mounted.
        """
        from textual.actions import SkipAction

        if self.screen.query("HelpPanel"):
            self.action_hide_help_panel()
            return
        raise SkipAction()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Hide ``t Tasks`` from the footer while a text input is focused.

        ``t`` is not a priority binding, so a focused ``Input`` / ``TextArea``
        (the chat prompt in INSERT mode, a catalog/settings search box) eats
        it as a literal character. Showing ``t Tasks`` there would lie.
        """
        # isinstance: a focused Input/TextArea consumes printable keys before
        # non-priority screen/app bindings see them, so `t` types a literal there.
        if action == "open_tasks" and isinstance(self.focused, (Input, TextArea)):
            return False
        return super().check_action(action, parameters)

    def action_open_tasks(self) -> None:
        """Jump to the Task Center screen (t key)."""
        self.switch_view("Tasks")

    def action_global_slash_to_chat(self) -> None:
        """Route a slash typed on a non-slash-bound screen back to Chat's prompt.

        Lets the user type ``/setup`` from Settings/Tasks/etc. without
        the next character (``s``, ``t``, ...) hitting a global single-key
        binding before the slash command can compose.
        """
        from lilbee.cli.tui.screens.chat import ChatScreen

        if not isinstance(self.screen, ChatScreen):
            self.switch_view("Chat")
        # Defer the prompt focus until after switch_view's call_later
        # _finish has updated active_view, so the chat input is mounted
        # and ready when we prefill it.
        self.call_later(self._prefill_chat_command)

    def _prefill_chat_command(self) -> None:
        """Focus the chat input and seed it with a leading slash."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        if isinstance(self.screen, ChatScreen):
            self.screen.action_focus_commands()

    def action_run_sync(self) -> None:
        """Trigger an explicit document sync from any screen (S key).

        The TaskBar hint is rendered globally, so the trigger must work
        everywhere. Routes to the registered ChatScreen which owns the
        ``_run_sync`` orchestration; switches to the Chat view first if
        not already there so the user can watch progress.
        """
        from lilbee.cli.tui.screens.chat import ChatScreen

        if isinstance(self.screen, ChatScreen):
            self.screen._run_sync()
            return
        try:
            chat = self.get_screen(_CHAT_SCREEN_NAME, ChatScreen)
        except KeyError:
            return
        self.switch_view("Chat")

        def _start() -> None:
            if isinstance(self.screen, ChatScreen):
                chat._run_sync()

        self.call_later(_start)

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
    """Route model writes through LilbeeApp.set_active_model."""
    cast(LilbeeApp, host_app).set_active_model(key, value)


def apply_setting(host_app: App[Any], key: str, value: object) -> None:
    """Route non-model settings writes through LilbeeApp.set_setting."""
    cast(LilbeeApp, host_app).set_setting(key, value)
