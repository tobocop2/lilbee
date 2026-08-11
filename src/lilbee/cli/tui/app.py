"""Main Textual app for lilbee TUI."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, ClassVar, cast

from rich.console import Console
from textual import work
from textual.app import App, ComposeResult
from textual.await_complete import AwaitComplete
from textual.binding import Binding, BindingType
from textual.command import CommandPalette
from textual.css.query import NoMatches
from textual.filter import LineFilter
from textual.reactive import reactive
from textual.screen import Screen
from textual.signal import Signal
from textual.widgets import Input, TextArea

from lilbee.app.services import get_services, peek_services
from lilbee.app.settings import apply_settings_update
from lilbee.app.setup_state import chat_ready, embedding_ready
from lilbee.app.themes import DARK_THEMES
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.color_compat import (
    EightBitPalette,
    draws_block_glyphs,
    needs_eight_bit,
    resolve_term_program,
)
from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.cli.tui.screens.command_palette import LilbeeCommandPalette
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.config_meta import MODEL_ROLE_FIELDS
from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole

if TYPE_CHECKING:
    from lilbee.app.services import Services
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.screens.startup_gate import StartupGate

log = logging.getLogger(__name__)

_DEFAULT_THEME = "rose-pine"  # muted, low-glare; easier on the eyes than the warmer themes
_CHAT_SCREEN_NAME = "chat"
# Long enough that a model-fallback notice is readable before it fades.
_FALLBACK_TOAST_TIMEOUT_S = 10.0


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


def _make_fleet() -> Screen:
    from lilbee.cli.tui.screens.fleet import FleetScreen

    return FleetScreen()


def _make_sessions() -> Screen:
    from lilbee.cli.tui.screens.sessions import SessionsScreen

    return SessionsScreen()


# Screen factory per managed view name (Chat is special-cased in switch_view and
# has no factory). The active set + order + wiki gate come from msg.get_nav_views,
# so the view universe lives in exactly one place (messages.ALL_NAV_VIEWS).
_VIEW_FACTORIES: dict[str, Callable[[], Screen]] = {
    msg.CATALOG_VIEW: _make_catalog,
    "Status": _make_status,
    "Settings": _make_settings,
    "Tasks": _make_tasks,
    "Wiki": _make_wiki,
    "Fleet": _make_fleet,
    "Sessions": _make_sessions,
}


def _import_chat_stack() -> None:
    """Pull in the chat screen's module graph, the TUI's heaviest import."""
    import lilbee.cli.tui.screens.chat  # noqa: F401 - imported for its side effect


def get_views() -> dict[str, Callable[[], Screen]]:
    """Return the active view factories, derived from the nav view list."""
    return {name: _VIEW_FACTORIES[name] for name in msg.get_nav_views() if name in _VIEW_FACTORIES}


class LilbeeApp(App[None]):
    """Full-screen TUI for lilbee knowledge base."""

    TITLE = "lilbee"
    CSS_PATH = Path(__file__).parent / "app.tcss"
    # Restates Textual's own block-based borders in box-drawing. Loaded only
    # where the terminal needs it; see __init__.
    SAFE_CSS_PATH = Path(__file__).parent / "app_safe.tcss"
    ENABLE_COMMAND_PALETTE = True
    COMMANDS = {LilbeeCommandProvider}  # noqa: RUF012

    # The app row is [ and ] to move between views, plus Help and Quit. Every
    # other app key is help-panel only, which lists every non-system binding whatever
    # its ``show``. A group may hold one action pressed in two directions, where
    # a single label still tells the whole truth, and never keys that do
    # different things: five destinations behind one "Views" label named none of
    # them. Textual groups CONSECUTIVE runs of shown bindings, so a group's
    # members stay adjacent.
    _NAV_GROUP = Binding.Group("Views")

    BINDINGS: ClassVar[list[BindingType]] = [
        # ``?`` is the only key for help. Non-priority on purpose: a focused
        # text field consumes printable keys, which both types the literal
        # character and takes the key out of the footer row, so no guard here is
        # needed to keep the row honest. ChatInput additionally routes ``?`` on
        # an EMPTY prompt to this action, so an untouched prompt still opens
        # help. F1 is gone; one advertised key is enough.
        Binding("question_mark", "push_help", "Help", show=True),
        Binding("escape", "dismiss_help_if_open", "Close help", show=False, priority=True),
        # Guarded like open_tasks in check_action: a focused text input
        # types the literal letter instead.
        # Help-panel only: the view tabs run across the top of every screen and are
        # clickable, so a footer cell per destination is a second copy of
        # something already on screen. [ and ] move between them.
        Binding("c", "open_chat", "Chat", show=False),
        Binding("m", "open_catalog", "Models", show=False),
        Binding("t", "open_tasks", "Tasks", show=False),
        # These two keep their cells: they open a drawer beside the current
        # screen rather than navigating, so unlike the jumps above they do
        # something the tab strip cannot, and nothing else on screen says so.
        Binding("ctrl+g", "toggle_fleet", "Fleet", show=True, priority=True),
        Binding("ctrl+o", "toggle_sessions", "Sessions", show=True, priority=True),
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
        # Off the row, like f4 below: cycling the theme is not something a user
        # needs advertised on every screen, and the row is for getting around.
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        # Hidden: a title-bar display toggle is not worth a permanent footer
        # cell on all twelve screens. show=False only drops it from the footer
        # row -- the help panel lists every non-system binding regardless -- so
        # the key stays discoverable.
        Binding("f4", "toggle_lilbee_path", "Path/Name", show=False),
        # Non-priority so Chat's "focus_commands" and Catalog's
        # "focus_search" still win on those screens. Fires only on
        # screens that don't bind slash themselves, routing the user
        # to Chat with the slash already typed.
        Binding("slash", "global_slash_to_chat", "Command", show=False),
        Binding("S", "run_sync", "Sync", show=False, priority=True),
    ]

    # Per-role readiness, settled by the startup gate before it hands over any
    # screen and re-answered whenever a model role is reassigned. They drive
    # empty states and the landing view; no view is gated on them. Reactive so
    # screens can watch them instead of polling.
    chat_is_ready: reactive[bool] = reactive(True)
    embedding_is_ready: reactive[bool] = reactive(True)

    def __init__(self, *, initial_view: str | None = None) -> None:
        # Both terminal questions are answered once, here: resolve_term_program can
        # shell out to tmux, and get_line_filters runs per widget per repaint. The
        # glyph answer must also land before super() so the stylesheet list is
        # complete when Textual reads it.
        color_system = Console().color_system
        term_program = resolve_term_program(os.environ)
        self._plain_glyphs = not draws_block_glyphs(color_system, term_program)
        # A terminal that cannot tile partial-block glyphs also gets the sheet
        # restating Textual's own block borders.
        self._eight_bit_filter = (
            EightBitPalette() if needs_eight_bit(color_system, term_program) else None
        )
        css: list[str | PurePath] = [self.CSS_PATH]
        if self._plain_glyphs:
            css.append(self.SAFE_CSS_PATH)
        super().__init__(css_path=css)
        self._initial_view = initial_view
        self.active_view = msg.DEFAULT_VIEW
        # The view the user came from; go_back returns here so q/Escape mean
        # "back", not "Chat", on every top-level view.
        self._previous_view: str | None = None
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

    def get_css_variables(self) -> dict[str, str]:
        """Textual's variables, plus the border style the terminal can actually draw.

        `tall` and `thick` are built from partial block glyphs, which segment in
        fonts that do not draw them cell-exact. Carrying the style in a variable
        keeps one switch here instead of a second copy of every rule: a capable
        terminal keeps the block rails lilbee is drawn with, and only a terminal
        that needs it falls back to box-drawing.
        """
        variables = super().get_css_variables()
        variables["rail"] = "solid" if self._plain_glyphs else "tall"
        variables["rail-heavy"] = "heavy" if self._plain_glyphs else "thick"
        return variables

    def get_line_filters(self) -> Sequence[LineFilter]:
        """Textual's filters, plus the 256-color correction where the terminal needs it.

        Added for a terminal that reduces to 256 colors, where Rich's own reduction
        collapses the theme's dark surfaces, and for Terminal.app, which claims
        truecolor it cannot render. See color_compat. A terminal that genuinely has
        truecolor gets no filter and renders byte-identically to before.

        Textual calls this per widget per repaint, so it only reads the decision
        made in __init__.
        """
        filters = list(super().get_line_filters())
        if self._eight_bit_filter is not None:
            filters.append(self._eight_bit_filter)
        return filters

    # Test seam: the TUI test fixtures subclass LilbeeApp and set this to True
    # so on_mount short-circuits before the heavyweight setup (model
    # canonicalization, ChatScreen install, signal subscriptions, sync probe).
    # Production never sets it. See tests/_lilbee_app_test_host.py.
    _test_skip_auto_init: ClassVar[bool] = False

    async def on_mount(self) -> None:
        # The app's own signal graph is part of being a working app, not
        # "heavyweight auto-init": wiring it before the test-skip guard lets a
        # test observe app-level signals without booting the startup gate, whose
        # wait is a timing window that wedges loaded CI runners.
        self.settings_changed_signal.subscribe(self, self._fan_out_provider_availability)
        self.settings_changed_signal.subscribe(self, self._recheck_models_on_model_change)
        if self._test_skip_auto_init:
            return
        # Paint the gate before any other work so the terminal is never blank
        # between the splash handing over and the first screen appearing. Nothing
        # slower than widget mounting may run before the first frame: model
        # canonicalization does disk and network probes, so it lives in the
        # gate's boot worker, off this thread.
        from lilbee.cli.tui.screens.startup_gate import StartupGate

        gate = StartupGate()
        # Awaited: the gate's boot worker treats an unmounted gate as "torn down",
        # so it must be mounted before start_boot can hand over.
        await self.push_screen(gate)
        self.title = msg.app_title(cfg.chat_model)
        # Restore the persisted theme so the TUI opens in whatever the user
        # picked last session, not always the default.
        persisted = cfg.theme or _DEFAULT_THEME
        self.theme = persisted if persisted in self.available_themes else _DEFAULT_THEME
        self._sync_theme_index_to_current()

        # Chat's import graph is the TUI's heaviest; loading it here would hold
        # the first frame back for seconds on a cold disk, leaving the terminal
        # blank exactly where the gate should be. Paint first, then load.
        self.call_after_refresh(self._load_chat_screen, gate)

    def _load_chat_screen(self, gate: StartupGate) -> None:
        """Install chat after the first frame, importing off-thread only when cold.

        The worker exists for the cold-disk case where chat's module graph takes
        seconds to read; once the modules are in sys.modules the import is free,
        and the extra thread hop would only delay the handover.
        """
        if "lilbee.cli.tui.screens.chat" in sys.modules:
            self._install_chat_screen(gate)
            return
        self._chat_import_worker(gate)

    @work(thread=True, name="chat_import", exit_on_error=False)
    def _chat_import_worker(self, gate: StartupGate) -> None:
        try:
            _import_chat_stack()
        except Exception as exc:
            # Without chat the app has no home screen; exit loudly like the old
            # inline import did rather than stranding the user on the gate.
            log.exception("the chat screen failed to import")
            call_from_thread(self, self._exit_on_chat_import_failure, str(exc))
            return
        call_from_thread(self, self._install_chat_screen, gate)

    def _exit_on_chat_import_failure(self, error: str) -> None:
        """Leave the TUI with the import error where the user can read it."""
        self.exit(return_code=1, message=msg.CHAT_STACK_FAILED.format(error=error))

    def _install_chat_screen(self, gate: StartupGate) -> None:
        """Install chat and start the gate's boot; runs once chat's modules exist."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        chat = ChatScreen()
        self.install_screen(chat, name=_CHAT_SCREEN_NAME)
        gate.start_boot()

    def reveal_landing(self) -> None:
        """Swap the startup gate for what the machine can serve.

        A resolvable chat model lands on Chat; anything else lands on the
        Catalog, where models are installed.
        """
        if self.chat_is_ready:
            self.switch_screen(_CHAT_SCREEN_NAME)
            if self._initial_view and self._initial_view != msg.DEFAULT_VIEW:
                self.switch_view(self._initial_view)
        else:
            self.switch_view(msg.CATALOG_VIEW)
        # Cheap detection only: filesystem walk + hash compare. The user
        # initiates sync explicitly via S or the command palette.
        self.task_bar.start_detect_pending()

    def settle_landing(self) -> None:
        """Answer per-role readiness and record it.

        Blocks the calling thread until the answer is recorded on the UI
        thread, so a handover ordered after it cannot read a stale flag.
        """
        call_from_thread(self, self._apply_readiness, chat_ready(), embedding_ready())

    @work(thread=True, name="setup_state", exit_on_error=False)
    def refresh_readiness(self) -> None:
        """Re-answer readiness off the UI thread, leaving the user where they are."""
        chat = chat_ready()
        embedding = embedding_ready()
        if (chat or embedding) and peek_services() is None:
            # The first model landed after the gate stepped aside, so nothing
            # has built the container yet and this thread is the one that should.
            self.adopt_services()
        call_from_thread(self, self._apply_readiness, chat, embedding)

    def adopt_services(self) -> None:
        """Build the services container and subscribe this app to it.

        Never call from the UI thread: building spawns the role servers. Two
        workers can reach here at once during boot; the listeners only add to
        and discard from a set, so a double subscription changes nothing.
        """
        self._wire_worker_pool_notifications(get_services())

    def _apply_readiness(self, chat: bool, embedding: bool) -> None:
        """Record the per-role readiness answers."""
        self.chat_is_ready = chat
        self.embedding_is_ready = embedding

    def _recheck_models_on_model_change(self, payload: tuple[str, object]) -> None:
        """Re-answer readiness whenever a model role is reassigned.

        Every model write lands on the settings boundary, a download included,
        so one subscription covers every surface that assigns one.
        """
        key, _value = payload
        if key in MODEL_ROLE_FIELDS:
            self.refresh_readiness()

    def _wire_worker_pool_notifications(self, services: Services) -> None:
        """Surface worker spawn lifecycle in the bottom TaskBar.

        Worker spawns happen on the pool runtime thread, not the TUI's main
        loop, so the listeners marshal back via :meth:`call_from_thread`
        before mutating controller state. A single TaskBar hint covers all
        in-flight roles instead of one toast per role; the chat surface is
        for user content, not implementation detail.

        Takes the container instead of reaching for one: reaching for it builds
        it, which is ``adopt_services``' job and never the UI thread's.
        """

        def _on_spawning(role: WorkerRole) -> None:
            self.call_from_thread(self.task_bar.mark_role_spawning, role.value)

        def _on_spawned(role: WorkerRole) -> None:
            self.call_from_thread(self.task_bar.mark_role_spawned, role.value)

        services.add_pool_listener(on_spawning=_on_spawning, on_spawned=_on_spawned)

    def canonicalize_persisted_models(self) -> None:
        """Swap stale persisted refs to a working fallback, persist, and log once.

        Canonicalization reads model files and can probe local model servers
        over HTTP/DNS, so the startup gate's boot worker calls this off the
        event loop before the services container builds; anything slower than
        widget mounting on the mount path delays the TUI's first frame. UI
        updates marshal back to the main thread.
        """
        from lilbee.modelhub.model_manager import (
            ValidationResult,
            canonicalize_chat_model,
            canonicalize_embedding_model,
        )

        chat_canon = canonicalize_chat_model()
        embedding_canon = canonicalize_embedding_model()
        for canon, field, label in (
            (chat_canon, "chat_model", "Chat"),
            (embedding_canon, "embedding_model", "Embedding"),
        ):
            if canon.status == ValidationResult.OK:
                continue
            reason = canon.reason or msg.MODEL_REASON_DEFAULT

            if canon.original == canon.effective:
                # Nothing to fall back to: keep the ref and let the catalog
                # landing be the single voice for "pick a model." A toast here
                # would just duplicate it, so log the reason as a breadcrumb
                # but don't surface it. An unconfigured role isn't even a
                # breadcrumb: there is nothing to report about a model nobody
                # chose.
                if canon.original:
                    log.warning(
                        msg.MODEL_UNUSABLE_NO_FALLBACK.format(
                            label=label, original=canon.original, reason=reason
                        )
                    )
                continue

            # A rejected swap (validation or disk error) must not be fatal at startup.
            try:
                apply_settings_update({field: canon.effective})
            except (ValueError, OSError):
                log.warning(
                    msg.MODEL_FALLBACK_FAILED.format(
                        label=label,
                        original=canon.original,
                        effective=canon.effective,
                        reason=reason,
                    ),
                    exc_info=True,
                )
                continue
            if not canon.original:
                # Adopting an installed model into an unconfigured role is the
                # expected path (models pulled before the TUI ever ran), not a
                # fallback worth a warning toast.
                log.info(msg.MODEL_ADOPTED_LOG.format(label=label, effective=canon.effective))
                continue
            notice = msg.MODEL_FALLBACK_NOTICE.format(
                label=label, original=canon.original, effective=canon.effective, reason=reason
            )
            log.warning(notice)
            call_from_thread(
                self, self.notify, notice, severity="warning", timeout=_FALLBACK_TOAST_TIMEOUT_S
            )
        call_from_thread(self, self._refresh_title)

    def _refresh_title(self) -> None:
        """Re-derive the window title after canonicalization may have swapped the ref."""
        self.title = msg.app_title(cfg.chat_model)

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

    def action_toggle_lilbee_path(self) -> None:
        """Flip the status-bar pill between the friendly name and the data-root path."""
        self.set_setting("show_lilbee_path", not cfg.show_lilbee_path)

    def set_theme(self, name: str) -> None:
        """Set theme by name (used by /theme command). Persists across sessions."""
        if name in self.available_themes:
            self._apply_and_persist_theme(name)
            self._sync_theme_index_to_current()

    def _apply_and_persist_theme(self, name: str) -> None:
        """Apply *name* live and write it to config.toml."""

        self.theme = name
        apply_settings_update({"theme": name})

    def _reject_if_downloading(self, value: object) -> bool:
        """Toast and return True if *value* is a model ref still downloading, so a
        half-pulled file can't land in a model slot."""
        if not isinstance(value, str):
            return False
        downloading = self.task_bar.downloading_label_for(value)
        if downloading is None:
            return False
        self.notify(msg.MODEL_BEING_DOWNLOADED.format(name=downloading), severity="warning")
        return True

    def set_active_model(self, key: str, value: str) -> None:
        """Persist an active model ref through the shared write boundary.

        Refs whose download is still queued or active are refused before the
        boundary runs, so a half-pulled file cannot land in a model slot.
        """
        if self._reject_if_downloading(value):
            return
        try:
            apply_settings_update({key: value})
        except ValueError as exc:
            self.notify(msg.MODEL_ASSIGN_REJECTED.format(error=exc), severity="error")
            return
        self.settings_changed_signal.publish((key, getattr(cfg, key)))

    def set_setting(self, key: str, value: object) -> None:
        """Apply a writable / model-role setting through the boundary, then fan out to the UI.

        Raises ``ValueError`` for keys outside ``WRITABLE_CONFIG_FIELDS | MODEL_ROLE_FIELDS``
        or values rejected by pydantic validation. Callers either catch and toast or let it
        propagate.
        """
        # A model-role ref still downloading must not land in a slot (parity with
        # set_active_model); toast and skip rather than half-pull.
        if key in MODEL_ROLE_FIELDS and self._reject_if_downloading(value):
            return
        apply_settings_update({key: value})
        normalized = getattr(cfg, key)
        if key == "theme" and isinstance(normalized, str) and normalized in self.available_themes:
            self.theme = normalized
            self._sync_theme_index_to_current()
        self.settings_changed_signal.publish((key, normalized))
        if key == "wiki" and normalized is False:
            self._offer_wiki_wipe()

    def _offer_wiki_wipe(self) -> None:
        """Ask whether to delete what the wiki generated, now that it is off.

        Lives on the setter rather than on the settings screen so every route
        that turns the wiki off (the settings editor, ``/set``) makes the same
        offer. Disabling stops new pages being written but removes nothing, so
        without this the pages stay on disk and their rows stay in the store.
        """
        from lilbee.cli.tui.messages import WIKI_WIPE_DISABLED_MESSAGE, WIKI_WIPE_DISABLED_TITLE
        from lilbee.cli.tui.screens.wiki import confirm_wiki_wipe

        confirm_wiki_wipe(
            self,
            title=WIKI_WIPE_DISABLED_TITLE,
            message=WIKI_WIPE_DISABLED_MESSAGE,
            notify_when_empty=False,
        )

    def _sync_theme_index_to_current(self) -> None:
        """Align cycle index with the active theme."""
        try:
            self._theme_index = DARK_THEMES.index(self.theme)
        except ValueError:
            self._theme_index = 0

    async def action_quit(self) -> None:
        """Context-aware Ctrl+C: cancel the foreground operation, else quit.

        Only operations the user is actively watching (an in-flight chat
        stream) get the cancel-first treatment; a background task like an
        engine warm or a sync never swallows a quit.
        """
        get_services().cancel_inference()

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = self.screen
        if isinstance(screen, ChatScreen) and screen.streaming:
            screen.action_cancel_stream()
            self.notify(msg.APP_QUIT_AGAIN_HINT)
            return
        self.exit()

    def _view_is_refused(self, view_name: str) -> bool:
        """True when *view_name* cannot be entered now, having handled the refusal."""
        if view_name == msg.SESSIONS_VIEW and not cfg.sessions_enabled:
            # The tab stays visible so the feature is discoverable, but opening it
            # while off shows why rather than an empty list.
            self._notify_sessions_disabled()
            return True
        return view_name != msg.DEFAULT_VIEW and get_views().get(view_name) is None

    def switch_view(self, view_name: str) -> None:
        """Switch to a named view, installing each screen at most once.

        Guards against concurrent switches via ``self._switching`` so rapid
        keypresses can't corrupt the screen stack. ``active_view`` is updated
        after the switch completes.
        """
        if self._switching or self._view_is_refused(view_name):
            return
        self._switching = True
        if view_name != self.active_view:
            self._previous_view = self.active_view

        awaitable: AwaitComplete | None = None
        if view_name == msg.DEFAULT_VIEW:
            from lilbee.cli.tui.screens.chat import ChatScreen

            if not isinstance(self.screen, ChatScreen):
                awaitable = self.switch_screen(_CHAT_SCREEN_NAME)
            # Already on Chat, just update state below.
        else:
            screen_name = _view_screen_name(view_name)
            if screen_name not in self._installed_screen_names:
                self.install_screen(get_views()[view_name](), name=screen_name)
                self._installed_screen_names.add(screen_name)
            awaitable = self.switch_screen(screen_name)

        self.active_view = view_name
        # ViewTabs.on_mount captured active_view before this runs, so the
        # highlight would lag by one step without this push.
        with contextlib.suppress(NoMatches):
            self.screen.query_one(ViewTabs).active_view = view_name

        async def _release() -> None:
            # switch_screen updates the stack synchronously but finishes mounting
            # in a deferred AwaitComplete. Releasing the guard on the next tick
            # (the old call_later) let a rapid second nav re-enter switch_screen
            # mid-transition and pop an empty result-callback stack (a Textual
            # IndexError). Awaiting the transition first keeps the guard up for
            # the whole switch; call_next is flushed by the same event loop, so a
            # single completed switch still releases promptly.
            if awaitable is not None:
                await awaitable
            self._switching = False

        self.call_next(_release)

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
                screen.action_complete_prev()
                return
        # Textual's own action hard-codes its CommandPalette; the subclass carries
        # lilbee's search icon. isinstance rather than CommandPalette.is_open,
        # which is typed for App[object] and so rejects lilbee's own app type.
        if self.use_command_palette and not isinstance(self.screen, CommandPalette):
            self.push_screen(LilbeeCommandPalette(id="--command-palette"))

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
        """Hide the letter view-keys from the footer while a text input is focused.

        ``t`` is not a priority binding, so a focused ``Input`` / ``TextArea``
        (the chat prompt in INSERT mode, a catalog/settings search box) eats
        it as a literal character. Showing ``t Tasks`` there would lie.
        """
        # isinstance: a focused Input/TextArea consumes printable keys before
        # screen/app bindings see them (verified empirically: this holds for
        # priority bindings too), so `t`/`m` type literals there and the guard
        # exists purely to keep the footer honest.
        if action in ("open_tasks", "open_catalog", "open_chat") and isinstance(
            self.focused, (Input, TextArea)
        ):
            return False
        # Nothing to jump to when Chat is already the active view.
        if action == "open_chat" and self.active_view == msg.DEFAULT_VIEW:
            return False
        # False, not None: Textual drops a False binding from the row entirely
        # and renders a None one greyed but present. With sessions off there is
        # nothing to toggle, so the key should not take a footer cell at all.
        if action == "toggle_sessions" and not cfg.sessions_enabled:
            return False
        # Drop each drawer toggle only where pressing it would do nothing, which
        # is the view that already shows the same panel full-screen. An OPEN
        # DRAWER is not that case: the key closes it, and the drawer contains the
        # very widget these predicates look for, so testing the panel alone
        # disabled the key that closes the drawer.
        if action == "toggle_fleet" and self._toggle_fleet_is_noop():
            return False
        if action == "toggle_sessions" and self._toggle_sessions_is_noop():
            return False
        return super().check_action(action, parameters)

    def go_back(self) -> None:
        """Return to the view the user came from (Chat when there is none)."""
        self.switch_view(self._previous_view or msg.DEFAULT_VIEW)

    def action_open_tasks(self) -> None:
        """Jump to the Task Center screen (t key)."""
        self.switch_view("Tasks")

    def action_open_catalog(self) -> None:
        """Jump to the model catalog (m key)."""
        self.switch_view(msg.CATALOG_VIEW)

    def action_open_chat(self) -> None:
        """Jump to Chat (c key), the counterpart to t / m for the busiest view."""
        self.switch_view(msg.DEFAULT_VIEW)

    def _shows_placement_full_screen(self) -> bool:
        """True when this screen shows the placement editor, tab or drawer.

        FleetDrawer composes a FleetBody, so this is also True while the drawer
        is open. Callers that care about "nothing left to do" must rule the
        drawer out first, as :meth:`_toggle_fleet_is_noop` does.
        """
        return bool(self.screen.query("FleetBody"))

    def _shows_sessions_full_screen(self) -> bool:
        """True when this screen shows the session list, tab or drawer.

        SessionsDrawer composes a SessionListPanel, so the same caveat as
        :meth:`_shows_placement_full_screen` applies.
        """
        return bool(self.screen.query("SessionListPanel"))

    def _toggle_fleet_is_noop(self) -> bool:
        """True when ctrl+g would do nothing, mirroring the action's own order.

        The action closes an open drawer first and only then treats a
        full-screen placement editor as a reason to stop, so a drawer that can
        be closed is never a no-op.
        """
        from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

        if self.screen.query(FleetDrawer):
            return False
        return self._shows_placement_full_screen()

    def _toggle_sessions_is_noop(self) -> bool:
        """True when ctrl+o would do nothing. See :meth:`_toggle_fleet_is_noop`."""
        from lilbee.cli.tui.widgets.sessions_drawer import SessionsDrawer

        if self.screen.query(SessionsDrawer):
            return False
        return self._shows_sessions_full_screen()

    def action_toggle_fleet(self) -> None:
        """Toggle the Fleet drawer (ctrl+g): dock placement beside the current
        screen, or close it if already open. No-op on the Fleet tab, which
        already shows the full placement editor."""
        from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer

        drawers = self.screen.query(FleetDrawer)
        if drawers:
            drawers.first().remove()
            return
        if self._shows_placement_full_screen():
            return
        self.screen.mount(FleetDrawer())

    def _notify_sessions_disabled(self) -> None:
        """Show the modal explaining sessions are off. Every session entry point
        (ctrl+o, the Sessions tab, /sessions) routes here when disabled."""
        from lilbee.cli.tui.widgets.notice_dialog import NoticeDialog

        # Guard against stacking a second copy if the entry point is hit twice.
        if isinstance(self.screen, NoticeDialog):
            return
        self.push_screen(NoticeDialog(msg.SESSIONS_DISABLED_TITLE, msg.SESSIONS_DISABLED_MESSAGE))

    def action_toggle_sessions(self) -> None:
        """Toggle the Sessions drawer (ctrl+o), or close it if open. No-op on the
        Sessions tab, which already shows the full list. Shows a notice when
        sessions are turned off."""
        if not cfg.sessions_enabled:
            self._notify_sessions_disabled()
            return
        from lilbee.cli.tui.widgets.sessions_drawer import SessionsDrawer

        drawers = self.screen.query(SessionsDrawer)
        if drawers:
            drawers.first().remove()
            return
        if self._shows_sessions_full_screen():
            return
        self.screen.mount(SessionsDrawer())

    def resume_session(self, session_id: str) -> None:
        """Load a saved session into chat and switch to the chat view."""
        chat = self.chat_screen()
        if chat is None:
            return
        chat.resume_session(session_id)
        self.switch_view(msg.DEFAULT_VIEW)

    def new_chat(self) -> None:
        """Start a fresh conversation and switch to the chat view."""
        chat = self.chat_screen()
        if chat is None:
            return
        chat.start_new_conversation()
        self.switch_view(msg.DEFAULT_VIEW)

    def current_session_id(self) -> str | None:
        """The id of the conversation the chat screen is currently persisting to."""
        chat = self.chat_screen()
        return chat.session_id if chat is not None else None

    def action_global_slash_to_chat(self) -> None:
        """Route a slash typed on a non-slash-bound screen back to Chat's prompt.

        Lets the user type ``/setup`` from Settings/Tasks/etc. without
        the next character (``s``, ``t``, ...) hitting a global single-key
        binding before the slash command can compose.
        """
        from lilbee.cli.tui.screens.chat import ChatScreen

        if not isinstance(self.screen, ChatScreen):
            self.switch_view(msg.DEFAULT_VIEW)
        # Defer the prompt focus until after switch_view's call_later
        # _finish has updated active_view, so the chat input is mounted
        # and ready when we prefill it.
        self.call_later(self._prefill_chat_command)

    def _prefill_chat_command(self) -> None:
        """Focus the chat input and seed it with a leading slash."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        if isinstance(self.screen, ChatScreen):
            self.screen.action_focus_commands()

    def chat_screen(self) -> ChatScreen | None:
        """The installed chat screen, or None before the startup gate installs it."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        try:
            return cast("ChatScreen", self.get_screen(_CHAT_SCREEN_NAME, ChatScreen))
        except KeyError:
            return None

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
        chat = self.chat_screen()
        if chat is None:
            return

        # switch_view drops the request outright while its re-entrancy guard is
        # held (an earlier switch still in flight), so the retry must re-attempt
        # the switch itself, not just wait for one that may never have started.
        def _start(attempts: int = 600) -> None:
            if not self.screen_stack:
                return  # the app is tearing down; nothing left to sync
            if isinstance(self.screen, ChatScreen):
                chat._run_sync()
                return
            if attempts > 0:
                self.switch_view(msg.DEFAULT_VIEW)
                self.set_timer(0.05, lambda: _start(attempts - 1))

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
