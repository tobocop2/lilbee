"""Chat screen: scrollable message log with streaming markdown responses."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import logging
import os
import shlex
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from rich.rule import Rule
from textual import events, getters, on, work
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.dom import DOMNode
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Markdown, Select, Static

# Cancellation check for @work(thread=True) workers. Import at module level
# since it's used in multiple methods.
from textual.worker import NoActiveWorker
from textual.worker import get_current_worker as _get_worker

from lilbee.app.services import get_services, reset_store
from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.app.setup_state import needs_setup
from lilbee.app.themes import DARK_THEMES
from lilbee.app.version import get_version
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp, apply_active_model
from lilbee.cli.tui.screens.chat_helpers import (
    add_indexed_anything,
    build_add_progress_callback,
    build_import_progress_callback,
    build_sync_progress_callback,
    close_stream,
    open_local_file,
    remember_from_input,
    unregister_added_roots,
)
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.arg_hint import ArgHintLine
from lilbee.cli.tui.widgets.autocomplete import (
    PATH_ARG_COMMANDS,
    CompletionOverlay,
    get_completions,
    longest_common_prefix,
    path_completion_prefix,
)
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.cli.tui.widgets.context_chip import ContextChip
from lilbee.cli.tui.widgets.drawer import Drawer
from lilbee.cli.tui.widgets.fleet_body import FleetBody
from lilbee.cli.tui.widgets.fleet_drawer import FleetDrawer
from lilbee.cli.tui.widgets.help_hint import HelpHint
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.cli.tui.widgets.model_bar import ChatModeToggle, ModelBar
from lilbee.cli.tui.widgets.slash_command_catalog import SlashCommandCatalog
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode, CrawlRenderMode
from lilbee.crawler import crawler_available, is_url, require_valid_crawl_url
from lilbee.data.store import (
    ChunkType,
    EmbeddingModelMismatchError,
    SearchScope,
    scope_to_chunk_type,
)
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase, WarmProgress
from lilbee.retrieval.embedder import is_model_available
from lilbee.retrieval.query import SOURCES_BLOCK_MARKER, ChatMessage
from lilbee.retrieval.query.compaction import (
    compaction_due,
    foldable,
    history_budget,
    overflow,
    prompt_history,
    summary_messages,
)
from lilbee.retrieval.query.history_window import estimate_tokens
from lilbee.runtime import asyncio_loop
from lilbee.runtime.progress import (
    EventType,
    ProgressEvent,
)
from lilbee.sessions import (
    MessageRole,
    SessionMessage,
    SessionNotFoundError,
    SessionOrigin,
    SessionStore,
    TitleSource,
    derive_title,
)

if TYPE_CHECKING:
    from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController
log = logging.getLogger(__name__)

# Coalesce per-token UI updates into ~50 ms windows. Tiny reasoning models can
# emit 100+ tokens/sec; one ``call_from_thread`` per token saturates Textual's
# message queue and makes key events visibly lag.
_STREAM_FLUSH_INTERVAL = 0.05


@dataclass
class _StreamTimings:
    """Last-fired monotonic timestamp for the stream flush."""

    last_flush: float


# ``/crawl`` command flags.
_CRAWL_FLAG_DEPTH = "--depth"
_CRAWL_FLAG_MAX_PAGES = "--max-pages"
_CRAWL_FLAG_INCLUDE_SUBDOMAINS = "--include-subdomains"
_CRAWL_FLAG_RENDER = "--render"

# Name for the thread worker that resets and warms the new chat model off the
# event loop.
_MODEL_SWAP_WORKER = "model_swap_reset"


def _engine_status_text(snapshot: WarmProgress) -> str:
    """One status line for an engine-load snapshot: byte progress or the phase."""
    if snapshot.phase is WarmPhase.READING_WEIGHTS and snapshot.bytes_total:
        from lilbee.catalog.formatting import display_label_for_ref

        name = display_label_for_ref(snapshot.model_ref) if snapshot.model_ref else ""
        pct = snapshot.bytes_done * 100 // snapshot.bytes_total
        return f"{msg.ENGINE_READING_WEIGHTS.format(name=name)} {pct}%"
    if snapshot.phase is WarmPhase.LOADING_ENGINE:
        return msg.ENGINE_ALMOST_READY
    return msg.ENGINE_WARMING


_SETTING_TYPE_HINTS: dict[type, str] = {int: "a whole number", float: "a number"}


def _setting_type_hint(kind: type) -> str:
    """Human phrase for what a settings value must be."""
    return _SETTING_TYPE_HINTS.get(kind, f"a valid {kind.__name__} value")


def _closest_source(name: str, known: set[str]) -> str | None:
    """The indexed name most likely meant by *name*, or None when nothing is close."""
    low = name.lower()
    contains = [k for k in known if low in k.lower()]
    if len(contains) == 1:
        return contains[0]
    matches = difflib.get_close_matches(name, sorted(known), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _parse_add_paths(args: str) -> list[Path]:
    """Resolve ``/add`` arguments to filesystem paths.

    A single unquoted path may contain spaces and apostrophes (e.g. macOS
    "Star Wars Collector's Edition.pdf"), which shell parsing would split into
    fragments or reject with "No closing quotation". So when the whole argument
    points at an existing file or directory, take it as one path; otherwise fall
    back to shell-style splitting for multiple, optionally quoted, paths.
    """
    whole = Path(args.strip().strip('"').strip("'")).expanduser()
    if whole.exists():
        return [whole]
    try:
        # posix=False on Windows keeps backslash path separators literal.
        tokens = shlex.split(args, posix=os.name != "nt")
    except ValueError:
        return [whole]  # unbalanced quote in a literal path; treat as one path
    if os.name == "nt":
        tokens = [t.strip('"').strip("'") for t in tokens]
    return [Path(token).expanduser() for token in tokens]


class ChatWelcome(Static):
    """Empty-state welcome posted into the chat log; removed on first message."""

    def __init__(self, *, id: str | None = None) -> None:
        title = Content.styled(msg.CHAT_WELCOME_TITLE, "bold $primary")
        tagline = Content.styled(msg.CHAT_WELCOME_TAGLINE, "$text-muted")
        hint = Content.styled(msg.CHAT_WELCOME_HINT, "$text-muted")
        body = Content.assemble(title, "\n", tagline, "\n\n", hint)
        super().__init__(body, id=id)


class PromptArea(Vertical):
    """Container for chat input that highlights on focus-within."""

    pass


class ChatScreen(Screen[None]):
    """Primary chat interface with streaming LLM responses."""

    # Lilbee always hosts screens on a LilbeeApp (production + LilbeeAppHost
    # in tests), so narrowing the type lets the screen call set_theme /
    # switch_view / task_bar without isinstance dance or # type: ignore.
    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "chat.tcss"
    AUTO_FOCUS = "#chat-input"

    streaming: reactive[bool] = reactive(False)
    # True while a chat-model swap's fleet reload runs in the background. Gates the
    # submit handler and disables the input so the user can't fire a prompt into a
    # half-loaded fleet; cleared when the swap worker finishes (or fails).
    swapping_model: reactive[bool] = reactive(False)
    # True while a placement apply/clear reloads the fleet (from the Fleet drawer);
    # holds chat submissions so they don't race the reload into a 429.
    reloading_placement: reactive[bool] = reactive(False)

    HELP = (
        "# Chat\n\n"
        "Ask questions about your knowledge base.\n\n"
        "Press **Escape** for normal mode (vim keys), "
        "**i**/**a**/**o** to return to insert mode.\n\n"
        "**/** opens the slash-command line and **Tab** completes what you "
        "type there; **F2** lists every command.\n\n"
        "**F6** jumps to the model strip under the prompt, and in normal mode "
        "**h** / **l** or **Left** / **Right** step into it from either end. "
        "**Left** / **Right** walk all six cells (the Chat, Embed, Vision and "
        "Rerank pickers, then the Search and Chat mode pills), **h** / **l** do "
        "the same, **Home** / **End** jump to either end, **Enter** opens or "
        "picks the focused cell, **Escape** goes back."
    )

    _SCROLL_GROUP = Binding.Group("Scroll", compact=True)

    # Hot-path widget refs. ``getters.query_one`` is a typed class-level
    # descriptor that resolves via Textual's indexed DOM lookup on every
    # access. It is O(1) for id selectors, so no cache is needed.
    _chat_input = getters.query_one("#chat-input", ChatInput)
    _chat_log = getters.query_one("#chat-log", VerticalScroll)
    _completion_overlay = getters.query_one("#completion-overlay", CompletionOverlay)
    _arg_hint = getters.query_one("#arg-hint", ArgHintLine)

    BINDINGS: ClassVar[list[BindingType]] = [
        # `/` opens the slash-command line: the one thing this screen is for
        # besides typing, so it keeps a footer cell.
        Binding("slash", "focus_commands", "Commands", show=True),
        # F2 opens the searchable list of every slash command
        # (SlashCommandCatalog) -- not the model catalog, which is `/models`.
        # Help-panel only: `/` already leads there, and the full list is a lookup.
        Binding(
            "f2",
            "show_command_catalog",
            "All commands",
            show=False,
            priority=True,
        ),
        # Hidden: Tab only completes while the slash dropdown is open, and the
        # rest of the time it walks the focus chain, so a permanent
        # "tab Complete" cell overstated it. Named in help beside `/`.
        Binding("tab", "complete", "Complete", show=False, priority=True),
        Binding("ctrl+n", "complete_next", "Next match", show=False, priority=True),
        # Ctrl+P stays bound to the app's command palette by default. The
        # chat screen only intercepts it WHEN the dropdown is visible, via
        # LilbeeApp.action_command_palette overriding to call
        # ChatScreen.action_complete_prev. Action is exposed for direct
        # callers / tests; not bound here so the app-level priority binding
        # for ctrl+p (palette) wins by default.
        Binding("pageup", "scroll_up", "PgUp", show=False, group=_SCROLL_GROUP),
        Binding("pagedown", "scroll_down", "PgDn", show=False, group=_SCROLL_GROUP),
        Binding("ctrl+d", "half_page_down", "^d half PgDn", show=False, group=_SCROLL_GROUP),
        Binding("ctrl+u", "half_page_up", "^u half PgUp", show=False, group=_SCROLL_GROUP),
        Binding("j", "vim_scroll_down", "j down", show=False, group=_SCROLL_GROUP),
        Binding("k", "vim_scroll_up", "k up", show=False, group=_SCROLL_GROUP),
        Binding("g", "vim_scroll_home", "g top", show=False, group=_SCROLL_GROUP),
        Binding("G", "vim_scroll_end", "G bottom", show=False, group=_SCROLL_GROUP),
        # priority=True keeps history navigation fast-path winning over the
        # ChatInput's TextArea cursor_up/_down. Multi-line cursor movement
        # inside the prompt still works via PgUp/PgDn/Home/End.
        Binding("up", "history_prev", "Up", show=False, priority=True),
        Binding("down", "history_next", "Down", show=False, priority=True),
        # Esc always drops back into NORMAL mode so the user can navigate
        # the terminal. Cancel-while-streaming is on Ctrl+C below; the
        # two roles used to share Esc and clobbered each other.
        Binding("escape", "enter_normal_mode", "Normal mode", show=True, priority=True),
        # Ctrl+C cancels the active stream when streaming AND in INSERT
        # mode so the user can interrupt without leaving the input. The
        # screen-level priority binding overrides the App-level Quit;
        # check_action below hides + disables it outside that exact
        # context, so Ctrl+C still quits the app from NORMAL or when
        # nothing is streaming.
        Binding("ctrl+c", "cancel_stream", "Cancel stream", show=True, priority=True),
        Binding("ctrl+r", "toggle_markdown", "Markdown", show=False),
        Binding("s", "cycle_scope", "Scope", show=False),
        Binding("f3", "toggle_chat_mode", "Search/Chat", show=False),
        Binding("f5", "open_setup", "Setup", show=False),
        # A function key, not a letter: the four role pickers are worth
        # reaching mid-sentence, and a focused input consumes printable keys
        # before any binding fires. Tab reaches the bar too, but only from
        # NORMAL mode and only after walking past the log.
        Binding("f6", "focus_model_bar", "Model bar", show=False, priority=True),
        # NORMAL mode walks sideways into the role strip. h / l rather than the
        # whole of hjkl: the transcript owns j / k for scrolling.
        Binding("h", "enter_model_strip(-1)", "Prev role", show=False),
        Binding("l", "enter_model_strip(1)", "Next role", show=False),
        # The arrows reach here too. The focused transcript is a VerticalScroll
        # and binds Left / Right to horizontal scrolling, but Widget's
        # action_scroll_left raises SkipAction when there is nothing to scroll
        # sideways, which resumes the key lookup and lands it here. A transcript
        # wide enough to scroll keeps its own arrows; h / l are unconditional.
        Binding("left", "enter_model_strip(-1)", "Prev role", show=False),
        Binding("right", "enter_model_strip(1)", "Next role", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._history: list[ChatMessage] = []
        # Rolling summary of the turns compaction has folded out of _history.
        # Guarded by _history_lock alongside the turns it stands in for.
        self._summary = ""
        self._history_lock = threading.Lock()
        # The saved session this conversation persists to. None until the first
        # user turn creates one; reset to None on /clear so the next turn opens a
        # fresh session.
        self._session_id: str | None = None
        self._insert_mode: bool = True
        # Count of programmatic input edits whose (async) Changed events should
        # not re-filter the dropdown. The setter posts Changed after our flag
        # window would close, so a counter consumed in the handler is used.
        self._suppress_refresh = 0
        # The user-typed text the open dropdown is filtering against. While
        # navigating, the input holds a previewed candidate; Esc restores this.
        self._completion_origin: str | None = None
        self._sync_active: bool = False
        self._input_history: list[str] = []
        self._history_index: int = -1
        # The warm tip is worth one toast per session, on the first prompt that
        # has to wait out a cold engine load.
        self._warm_tip_shown: bool = False
        # The bubble receiving the in-flight response, so a cancel can leave a
        # visible note in it instead of letting the turn die silently.
        self._active_assistant: AssistantMessage | None = None
        # The live turn's question; a context boundary mounts above it, never
        # after it. Outlives its turn like _active_assistant (next send
        # overwrites, reset clears). Never clear it in _finalize_stream: the
        # input unblocks first, so the clear races the next turn's question.
        self._active_question: UserMessage | None = None
        self._command_handlers: dict[str, Callable[[str], None]] = self._build_command_handlers()

    def _build_command_handlers(self) -> dict[str, Callable[[str], None]]:
        """Bind every COMMANDS entry to its handler method on this instance.

        Run once at construction so /handle_slash dispatches via direct method
        reference (no per-call getattr-by-string-name reflection).
        """
        from lilbee.cli.tui.command_registry import COMMANDS

        handlers: dict[str, Callable[[str], None]] = {}
        for cmd in COMMANDS:
            method = getattr(self, cmd.handler)
            for name in (cmd.name, *cmd.aliases):
                handlers[name] = method
        return handlers

    @property
    def _task_bar(self) -> TaskBarController:
        """The app-level TaskBarController (always set by LilbeeApp)."""
        return self.app.task_bar

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        yield VerticalScroll(
            ChatWelcome(id="chat-welcome"),
            id="chat-log",
        )
        with BottomBars():
            # Sits directly above the prompt area so it never covers the line
            # you're typing (the input stays pinned to the bottom edge).
            yield CompletionOverlay(id="completion-overlay")
            with PromptArea(id="chat-prompt-area"):
                yield ScopeChip(id="scope-chip")
                yield ChatInput(
                    placeholder=msg.CHAT_INPUT_PLACEHOLDER_DEFAULT,
                    id="chat-input",
                )
                yield ArgHintLine(id="arg-hint")
                yield ModelBar(id="model-bar")
            yield TaskBar()
            # The context reading shares the hint band instead of costing the
            # prompt block its own row.
            with Horizontal(id="hint-row"):
                yield HelpHint(id="help-hint")
                yield ContextChip(id="context-chip")
            yield Footer()

    def on_mount(self) -> None:
        self._update_input_style()
        self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)
        self._setup_check_worker()

    @work(thread=True, name="chat_setup_check", exit_on_error=False)
    def _setup_check_worker(self) -> None:
        """Run ``needs_setup`` off the UI thread; push the wizard if needed."""
        if not needs_setup():
            return
        call_from_thread(self, self._push_setup_wizard)

    def _push_setup_wizard(self) -> None:
        """Push the SetupWizard if the screen is still mounted."""
        if not self.is_mounted:
            return
        from lilbee.cli.tui.screens.setup import SetupWizard

        self.app.push_screen(SetupWizard(), self._on_setup_complete)

    def on_show(self) -> None:
        """Called when screen becomes visible."""
        from lilbee.runtime.splash import dismiss

        dismiss()
        self.refresh_model_bar()
        # AUTO_FOCUS only fires once on initial mount. Re-entering the
        # screen via view-nav needs an explicit focus restore. In INSERT
        # mode we send focus to the chat input; in NORMAL mode we send
        # focus to the chat log (the input is intentionally unfocusable
        # so global bindings keep firing).
        with contextlib.suppress(Exception):
            if self._insert_mode:
                self._enter_insert_mode()
            else:
                self._chat_log.focus()

    def _embedding_ready(self) -> bool:
        """Quick check if the embedding model resolves (no network calls)."""
        return is_model_available(cfg.embedding_model, get_services().provider)

    def _on_setup_complete(self, result: str | None) -> None:
        """Called when wizard completes or is skipped."""
        # Re-detect after setup so a freshly-set-up vault gets the hint.
        self.app.task_bar.start_detect_pending()
        self.refresh_model_bar()

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        key, _value = payload
        if key in {"chat_mode", "embedding_model"}:
            self.refresh_model_bar()

    def action_open_setup(self) -> None:
        """Open the setup wizard."""
        self._cmd_setup("")

    def _enter_insert_mode(self) -> None:
        """Switch to insert mode: focus input, update border style."""
        self._insert_mode = True
        self._chat_input.can_focus = True
        self._chat_input.focus()
        self._update_input_style()

    def focus_prompt(self) -> None:
        """Return focus to the chat input in INSERT mode.

        Called when a modal (the model picker) closes: the next act is typing
        a prompt, so focus must not stay parked on the widget that opened it.
        """
        self._enter_insert_mode()

    def action_focus_model_bar(self) -> None:
        """F6: put the cursor on the model strip. Left / Right walk it from there."""
        self.query_one("#model-bar", ModelBar).focus_strip()

    def action_enter_model_strip(self, direction: int) -> None:
        """h / l and the arrows from NORMAL mode: step in from the matching side.

        Only reached while focus is outside the bar. Once a role holds the
        cursor the bar's own keys win, being nearer the focus.
        """
        self.query_one("#model-bar", ModelBar).focus_strip(direction)

    def run_command(self, text: str) -> None:
        """Dispatch *text* as a slash command, as if submitted from the prompt."""
        if self._reject_submit_when_busy():
            return
        self._handle_slash(text)

    def _update_input_style(self) -> None:
        """Toggle input opacity and mode indicator based on current mode."""
        # Lifecycle interleaves (an installed-but-swapped-away screen during
        # app teardown) can invoke this before or after the input exists.
        with contextlib.suppress(NoMatches):
            inp = self._chat_input
            if self._insert_mode:
                inp.remove_class("normal-mode")
            else:
                inp.add_class("normal-mode")
        self._update_mode_indicator()

    def _update_mode_indicator(self) -> None:
        """Update the ViewTabs mode text to reflect the current mode."""
        with contextlib.suppress(NoMatches):
            bar = self.query_one(ViewTabs)
            bar.mode_text = msg.MODE_INSERT if self._insert_mode else msg.MODE_NORMAL

    def on_key(self, event: object) -> None:
        """Handle key events: vim mode and typing from chat log."""
        from textual.events import Key

        if not isinstance(event, Key):
            return
        inp = self._chat_input
        if self._insert_mode:
            if not inp.has_focus and event.is_printable and event.character:
                inp.focus()
                inp.insert(event.character)
                event.prevent_default()
                event.stop()
            return
        if event.key == "enter" or (event.character and event.character in "iao"):
            # Let a focused Select, or anything on the model strip, handle Enter
            # itself; i/a/o mean nothing to those widgets, so they always return
            # to INSERT. Asked of the bar rather than of a list of widget types:
            # the mode pills were missing from that list and Enter on a pill
            # dropped to INSERT instead of switching the mode.
            if event.key == "enter" and (
                isinstance(self.focused, Select) or self._focus_in_model_bar()
            ):
                return
            if self._focus_in_drawer():
                return
            self._enter_insert_mode()
            if event.key == "enter" and inp.value.strip():
                # Enter meant "send": submit the draft the user Esc'd over
                # instead of stranding it invisibly in the dimmed input.
                self._submit_draft(inp, inp.value)
            event.prevent_default()
            event.stop()
            return

    @on(events.DescendantFocus, "#chat-input")
    def _on_chat_input_focused(self, event: events.DescendantFocus) -> None:
        """Mark INSERT mode whenever the chat input takes focus.

        With ``can_focus = False`` while in NORMAL mode, the only way the
        input gains focus is via an explicit user action (click, or the
        :meth:`_enter_insert_mode` helper that sets ``can_focus = True``
        and focuses the input). Either path implies INSERT, so we sync
        the screen mode here.
        """
        if not self._insert_mode:
            self._enter_insert_mode()

    @on(events.Click, "#chat-input")
    def _on_chat_input_clicked(self, event: events.Click) -> None:
        """Click on the chat input bar promotes to INSERT.

        ``can_focus = False`` while in NORMAL mode swallows focus from the
        click, so DescendantFocus never fires. Hook the Click directly so
        a mouse user lands in INSERT just like a keystroke (i / a / o).
        """
        if not self._insert_mode:
            self._enter_insert_mode()
            event.stop()

    def on_click(self, event: events.Click) -> None:
        """Click outside the chat input bar drops back to NORMAL.

        The chat-input click handler above promotes to INSERT; the
        symmetric exit happens here so a mouse user gets the same
        click-to-blur behavior they expect from any other text editor.
        """
        if not self._insert_mode:
            return
        if event.widget is None:
            return
        chat_input = self._chat_input
        node: DOMNode | None = event.widget
        while node is not None:
            if node is chat_input:
                return
            node = node.parent
        self.action_enter_normal_mode()

    @on(ChatInput.Submitted, "#chat-input")
    def _on_chat_submitted(self, event: ChatInput.Submitted) -> None:
        if not self._insert_mode:
            # Vim-style: Enter in normal mode flips back to insert without
            # submitting whatever empty / stale text the input still holds.
            self._enter_insert_mode()
            return
        self._submit_draft(event.chat_input, event.value)

    def _submit_draft(self, chat_input: ChatInput, value: str) -> None:
        """Send *value* as a command or message once the submit gate allows it."""
        text = value.strip()
        if not self._ready_to_submit(text):
            return
        chat_input.value = ""
        self._input_history.append(text)
        self._history_index = -1

        if text.startswith("/"):
            self._handle_slash(text)
            return
        self._send_message(text)

    def _ready_to_submit(self, text: str) -> bool:
        """Gate a submit: busy, consumed, empty, and keep-the-draft cases say no."""
        if self._reject_submit_when_busy() or self._dismiss_overlay_on_submit() or not text:
            return False
        if text.startswith("/"):
            cmd = text.split()[0].lower()
            if cmd not in self._command_handlers:
                # Keep the draft so a typo (or a stale leading slash) can be
                # fixed in place instead of retyped.
                self.notify(msg.CMD_UNKNOWN.format(cmd=cmd), severity="warning")
                return False
            return True
        pending = self._pending_required_model_download()
        if pending is not None:
            # Keep the typed prompt in the input so the user can submit it
            # again once the download finishes, instead of retyping it.
            self.notify(
                msg.CHAT_MODEL_DOWNLOADING.format(name=pending),
                severity="warning",
                timeout=5,
            )
            return False
        return True

    def _reject_submit_when_busy(self) -> bool:
        """Toast and reject a submit while a swap is loading or a stream is in flight.

        Returns True when the prompt was rejected so the caller stops. The swap
        check comes first: a prompt sent mid-swap would race a half-torn-down
        fleet, so the user is asked to wait rather than cancel.
        """
        if self.swapping_model:
            self.notify(msg.CHAT_MODEL_SWITCHING, severity="warning", timeout=3)
            return True
        if self.reloading_placement:
            self.notify(msg.FLEET_RELOADING, severity="warning", timeout=3)
            return True
        if self.streaming:
            # Only one chat message may be in flight at a time; surface a toast
            # so the prompt is visibly rejected, not silently dropped.
            self.notify(msg.CHAT_BUSY, severity="warning", timeout=3)
            return True
        return False

    def _pending_required_model_download(self) -> str | None:
        """Return the in-flight download's name if it's for the configured chat or embedding model.

        Covers the fresh-install case where the default ``cfg.chat_model``
        points at a featured catalog ref whose file isn't on disk yet,
        but a wizard-triggered download for it is queued or active.
        """
        task_bar = self.app.task_bar
        for ref in (cfg.chat_model, cfg.embedding_model):
            label = task_bar.downloading_label_for(ref)
            if label is not None:
                return label
        return None

    def _dismiss_overlay_on_submit(self) -> bool:
        """Close the dropdown on Enter; consume only a bare slash, never a message.

        Enter submits exactly what was typed. Tab and the arrow keys are the
        completion gestures, and a previewed candidate is already in the input,
        so a highlighted-but-unaccepted suggestion must never rewrite or swallow
        a submission.
        """
        overlay = self._completion_overlay
        if overlay.is_visible:
            overlay.hide()
            self._completion_origin = None
        if self._chat_input.value.strip() == "/":
            self._set_input("")
            return True
        return False

    def _handle_slash(self, text: str) -> None:
        """Dispatch slash commands via the per-instance handler registry."""
        cmd = text.split()[0].lower()
        args = text[len(cmd) :].strip()
        handler = self._command_handlers.get(cmd)
        if handler is not None:
            handler(args)
        else:
            self.notify(msg.CMD_UNKNOWN.format(cmd=cmd), severity="warning")

    def _set_streaming(self, value: bool) -> None:
        """Main-thread setter so worker-thread paths can route through ``call_from_thread``."""
        self.streaming = value

    def watch_streaming(self, streaming: bool) -> None:
        if streaming:
            self._enter_streaming_state()
        else:
            self._exit_streaming_state()

    def _enter_streaming_state(self) -> None:
        self.add_class("streaming")
        # Cancel + finalize both write streaming=False; reactive dedupe
        # keeps the watcher a no-op on equal values.
        self.refresh_bindings()

    def _exit_streaming_state(self) -> None:
        self.remove_class("streaming")
        self.refresh_bindings()

    def _cmd_add(self, args: str) -> None:
        from lilbee.app.ingest import source_label_taken

        if not args:
            return
        if self._sync_active:
            self.notify(msg.SYNC_ALREADY_ACTIVE, severity="warning")
            return
        if is_url(args):
            self._cmd_crawl(args)
            return
        paths = _parse_add_paths(args)
        missing = [p for p in paths if not p.exists()]
        if missing:
            self.notify(
                msg.CMD_ADD_NOT_FOUND.format(path=", ".join(str(p) for p in missing)),
                severity="error",
            )
            return
        # A file add registers a root labeled by its basename. Prompt before
        # overwriting only when that label is already taken by a different source
        # (a live root elsewhere, or an owned file of that name); re-adding the
        # same path is idempotent, and a directory is left to register_sources'
        # own skipped notices rather than a duplicate-file prompt.
        duplicates = [p for p in paths if p.is_file() and source_label_taken(p.name, p)]
        if duplicates:
            self._prompt_overwrite(paths, duplicates)
            return
        self._submit_add(paths, force=False)

    def _prompt_overwrite(self, paths: list[Path], duplicates: list[Path]) -> None:
        """Ask to overwrite existing copies before re-syncing."""
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        names = ", ".join(p.name for p in duplicates)

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                self.notify(msg.CMD_ADD_SKIPPED_DUPLICATE.format(name=names))
                return
            self._submit_add(paths, force=True)

        self.app.push_screen(
            ConfirmDialog(
                msg.CMD_ADD_DUPLICATE_TITLE,
                msg.CMD_ADD_DUPLICATE_MESSAGE.format(name=names),
            ),
            _on_confirm,
        )

    def _submit_add(self, paths: list[Path], *, force: bool) -> None:
        """Spawn the add worker. Separated so overwrite confirm can reuse it."""
        from lilbee.cli.tui.task_queue import TaskType

        self._sync_active = True
        label = paths[0].name if len(paths) == 1 else f"{len(paths)} files"

        def _target(reporter: ProgressReporter) -> None:
            try:
                self._do_add(paths, reporter, force=force)
            finally:
                self._sync_active = False

        self._task_bar.start_task(f"Add {label}", TaskType.ADD, _target, indeterminate=True)

    def _do_add(
        self, paths: list[Path], reporter: ProgressReporter, *, force: bool = False
    ) -> None:
        """Register source roots and run sync. Called on worker thread with a reporter."""
        from lilbee.app.ingest import register_sources
        from lilbee.data.ingest import sync

        label = paths[0].name if len(paths) == 1 else f"{len(paths)} files"
        reporter.update(0, f"Adding {label}...", indeterminate=True)
        reg_result = register_sources(paths, force=force)
        registered = reg_result.registered
        for name in reg_result.skipped:
            call_from_thread(self, self.notify, f"{name} already exists (use --force to overwrite)")
        reporter.update(0, f"Added {len(registered)} source(s), syncing...", indeterminate=True)

        try:
            sync_result = asyncio_loop.run(
                sync(quiet=True, on_progress=build_add_progress_callback(reporter))
            )
        except BaseException:
            # On cancel or any failure, un-register the roots this /add created so
            # the next sync doesn't silently re-ingest the source the user just
            # cancelled. Only entries this invocation created are dropped;
            # sources the user put in documents/ themselves are never touched.
            unregister_added_roots(registered)
            raise
        if sync_result.failed:
            unregister_added_roots(registered)
            raise RuntimeError(msg.SYNC_FAILED_FILES.format(files=", ".join(sync_result.failed)))
        if sync_result.skipped:
            # Files yielding no text beside indexed siblings are a partial
            # success; only an add whose own roots contributed nothing failed.
            skipped_msg = msg.sync_skipped_message(", ".join(sync_result.skipped))
            if registered and not add_indexed_anything(registered, sync_result):
                unregister_added_roots(registered)
                raise RuntimeError(skipped_msg)
            call_from_thread(self, self.notify, skipped_msg, severity="warning")
        if sync_result.relocated:
            call_from_thread(
                self,
                self.notify,
                msg.CMD_ADD_RELOCATED.format(count=len(sync_result.relocated)),
            )
        call_from_thread(self, self.notify, msg.CMD_ADD_SUCCESS.format(count=len(registered)))

    def _cmd_cancel(self, _args: str) -> None:
        # _cancel_inflight_stream already cancels every screen worker, so the
        # two branches each cancel everything exactly once.
        if self.streaming:
            self._cancel_inflight_stream(msg.STREAM_CANCELLED)
        else:
            for worker in self.workers:
                worker.cancel()
        self.notify(msg.CMD_CANCEL)

    def _cmd_clear(self, _args: str) -> None:
        self._reset_conversation()
        self.notify(msg.CMD_CLEAR)

    def _reset_conversation(self) -> None:
        """Cancel any stream, empty the log and history, and drop the active session.

        The current session is already persisted, so dropping the id just makes the
        next user turn open a fresh one.
        """
        if self.streaming:
            self._cancel_inflight_stream(msg.STREAM_CANCELLED)
        else:
            for worker in self.workers:
                worker.cancel()
        self.streaming = False
        self._chat_log.remove_children()
        self._active_assistant = None
        self._active_question = None
        with self._history_lock:
            self._history.clear()
            # A new conversation inherits nothing, least of all the last one's
            # summary: carrying it would leak the old chat into the new prompt.
            self._summary = ""
        self._session_id = None

    def _cmd_crawl(self, args: str) -> None:
        if not crawler_available():
            self.notify(msg.CMD_CRAWL_UNAVAILABLE, severity="error")
            return
        if not args:
            self._open_crawl_dialog()
            return
        parts = args.split()
        url = parts[0]
        if not is_url(url):
            url = f"https://{url}"
        try:
            require_valid_crawl_url(url)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        depth, max_pages, include_subdomains, render_mode = self._parse_crawl_flags(parts[1:])
        self._start_crawl(
            url,
            depth,
            max_pages,
            include_subdomains=include_subdomains,
            render_mode=render_mode,
        )

    def _open_crawl_dialog(self) -> None:
        """Push the crawl modal and handle its result."""
        from lilbee.cli.tui.widgets.crawl_dialog import CrawlDialog, CrawlParams

        def _on_result(result: CrawlParams | None) -> None:
            if result is not None:
                self._start_crawl(
                    result.url, result.depth, result.max_pages, render_mode=result.render_mode
                )

        self.app.push_screen(CrawlDialog(), callback=_on_result)

    def _start_crawl(
        self,
        url: str,
        depth: int | None,
        max_pages: int | None,
        *,
        include_subdomains: bool = False,
        render_mode: CrawlRenderMode | None = None,
    ) -> None:
        """Enqueue a crawl task and run it in the background.

        Bootstrap Chromium first via the controller helper, but only for a
        browser-mode crawl. HTTP mode needs no browser, so the SETUP task is
        skipped and the crawl starts immediately. An explicit ``render_mode``
        (from the dialog checkbox or ``--render``) is persisted so the choice
        sticks for the next crawl.
        """
        from lilbee.cli.tui.task_queue import TaskType

        mode = render_mode if render_mode is not None else cfg.crawl_render_mode
        if render_mode is not None and render_mode is not cfg.crawl_render_mode:
            self._persist_crawl_render_mode(render_mode)

        def _kick_off_crawl() -> None:
            self._task_bar.start_task(
                msg.TASK_NAME_CRAWL.format(url=url),
                TaskType.CRAWL,
                lambda reporter: self._do_crawl(
                    url,
                    depth,
                    max_pages,
                    reporter,
                    include_subdomains=include_subdomains,
                    render_mode=mode,
                ),
                on_success=lambda: call_from_thread(self, self._run_sync),
            )

        self.notify(msg.CMD_CRAWL_STARTED.format(url=url))
        if mode is CrawlRenderMode.BROWSER:
            self._task_bar.ensure_chromium(_kick_off_crawl)
        else:
            _kick_off_crawl()

    def _persist_crawl_render_mode(self, render_mode: CrawlRenderMode) -> None:
        """Persist the chosen render mode so the dialog checkbox stays sticky."""
        from lilbee.app.settings import apply_settings_update

        try:
            apply_settings_update({"crawl_render_mode": render_mode.value})
        except (ValueError, OSError) as exc:
            log.warning("Could not persist crawl_render_mode: %s", exc)

    @staticmethod
    def _parse_crawl_flags(
        tokens: list[str],
    ) -> tuple[int | None, int | None, bool, CrawlRenderMode | None]:
        """Extract --depth, --max-pages, --include-subdomains, --render from tokens.

        Numeric flags return None when absent so the caller inherits
        crawl_and_save's unbounded-by-default semantics. The boolean
        ``--include-subdomains`` flag defaults to False (exact-host scope).
        ``--render http|browser`` returns None when absent so the caller
        inherits ``cfg.crawl_render_mode``; an unrecognized value is ignored.
        """
        flag_map = {_CRAWL_FLAG_DEPTH: "depth", _CRAWL_FLAG_MAX_PAGES: "max_pages"}
        parsed: dict[str, int | None] = {"depth": None, "max_pages": None}
        include_subdomains = False
        render_mode: CrawlRenderMode | None = None
        i = 0
        while i < len(tokens):
            if tokens[i] == _CRAWL_FLAG_INCLUDE_SUBDOMAINS:
                include_subdomains = True
                i += 1
                continue
            if tokens[i] == _CRAWL_FLAG_RENDER and i + 1 < len(tokens):
                with contextlib.suppress(ValueError):
                    render_mode = CrawlRenderMode(tokens[i + 1])
                i += 2
                continue
            key = flag_map.get(tokens[i])
            if key and i + 1 < len(tokens):
                with contextlib.suppress(ValueError):
                    parsed[key] = int(tokens[i + 1])
                i += 2
            else:
                i += 1
        return parsed["depth"], parsed["max_pages"], include_subdomains, render_mode

    def _do_crawl(
        self,
        url: str,
        depth: int | None,
        max_pages: int | None,
        reporter: ProgressReporter,
        *,
        include_subdomains: bool = False,
        render_mode: CrawlRenderMode | None = None,
    ) -> None:
        """Crawl body. Runs on worker thread; reporter handles progress + cancel."""
        from lilbee.crawler import crawl_and_save
        from lilbee.runtime.progress import CrawlPageEvent, SetupProgressEvent

        reporter.update(0, msg.CMD_CRAWL_STARTED.format(url=url))

        def on_progress(event_type: EventType, data: ProgressEvent) -> None:
            if event_type == EventType.SETUP_START:
                reporter.update(0, msg.SETUP_CHROMIUM_NAME)
            elif event_type == EventType.SETUP_PROGRESS and isinstance(data, SetupProgressEvent):
                if data.total_bytes:
                    pct = int(data.downloaded_bytes * 100 / data.total_bytes)
                    detail = msg.SETUP_CHROMIUM_DETAIL.format(
                        done=data.downloaded_bytes // (1024 * 1024),
                        total=data.total_bytes // (1024 * 1024),
                    )
                else:
                    pct = 0
                    detail = msg.SETUP_CHROMIUM_DETAIL_UNKNOWN.format(
                        done=data.downloaded_bytes // (1024 * 1024),
                    )
                reporter.update(pct, detail)
            elif event_type == EventType.CRAWL_PAGE and isinstance(data, CrawlPageEvent):
                # Discovery hasn't resolved a sitemap yet (data.total <= 0):
                # show the indeterminate spinner with a count, not a parked
                # 50% bar that looks frozen. Switch to a determinate bar as
                # soon as the total is known.
                if data.total > 0:
                    pct = int(data.current * 100 / data.total)
                    reporter.update(
                        pct,
                        msg.CMD_CRAWL_PAGE.format(
                            current=data.current, total=data.total, url=data.url
                        ),
                        indeterminate=False,
                    )
                else:  # pragma: no cover - live crawl without sitemap
                    reporter.update(
                        0,
                        msg.CMD_CRAWL_PAGE_INDETERMINATE.format(current=data.current, url=data.url),
                        indeterminate=True,
                    )

        paths = asyncio_loop.run(
            crawl_and_save(
                url,
                depth=depth,
                max_pages=max_pages,
                on_progress=on_progress,
                quiet=True,
                include_subdomains=include_subdomains,
                render_mode=render_mode,
            )
        )
        call_from_thread(self, self.notify, msg.CMD_CRAWL_SUCCESS.format(count=len(paths), url=url))

    def _cmd_catalog(self, _args: str) -> None:
        # switch_view already installs and navigates to the managed Catalog view;
        # a push_screen on top would stack a second, orphaned CatalogScreen.
        self.app.switch_view("Catalog")

    def _cmd_delete(self, args: str) -> None:
        """Run /delete in a worker so the chat screen stays interactive."""
        self._cmd_delete_worker(args.strip())

    @work(thread=True, name="chat_cmd_delete", exit_on_error=False)
    def _cmd_delete_worker(self, name: str) -> None:
        """Validate and execute /delete off the UI thread; notify back via dispatch."""
        try:
            sources = get_services().store.get_sources()
        except Exception:
            log.debug("Failed to list documents for /delete", exc_info=True)
            call_from_thread(self, self.notify, msg.CMD_DELETE_READ_FAILED, severity="error")
            return

        known = {s.get("filename", s.get("source", "?")) for s in sources}
        if not known:
            call_from_thread(self, self.notify, msg.CMD_DELETE_NO_DOCS, severity="warning")
            return

        if not name:
            usage = msg.CMD_DELETE_USAGE.format(names=", ".join(sorted(known)))
            call_from_thread(self, self.notify, usage)
            return

        if name not in known:
            message = msg.CMD_DELETE_NOT_FOUND.format(name=name)
            suggestion = _closest_source(name, known)
            if suggestion is not None:
                message = f"{message}. {msg.CMD_DELETE_SUGGESTION.format(name=suggestion)}"
            call_from_thread(self, self.notify, message, severity="error")
            return

        from lilbee.app.ingest import remove_documents_durably
        from lilbee.cli.tui.widgets.autocomplete import invalidate_document_cache

        # Skip-mark so the next sync doesn't re-ingest the kept file (durable,
        # non-destructive delete; the file stays on disk).
        remove_documents_durably([name])
        invalidate_document_cache()
        call_from_thread(self, self.notify, msg.CMD_DELETE_SUCCESS.format(name=name))

    def _cmd_export(self, args: str) -> None:
        """Enqueue /export as a task so progress shows in the task bar."""
        path = args.strip()
        if not path:
            self.notify(msg.CMD_EXPORT_USAGE, severity="warning")
            return
        from lilbee.cli.tui.task_queue import TaskType

        def _target(reporter: ProgressReporter) -> None:
            self._do_export(path, reporter)

        name = msg.TASK_NAME_EXPORT.format(file=Path(path).name)
        self._task_bar.start_task(name, TaskType.EXPORT, _target, indeterminate=True)

    def _do_export(self, raw_path: str, reporter: ProgressReporter) -> None:
        """Export body. Runs on the task worker thread."""
        from lilbee.app.dataset import DatasetError, export_to_path

        output = Path(raw_path).expanduser()
        reporter.update(0, msg.EXPORT_STATUS_RUNNING, indeterminate=True)
        try:
            summary = export_to_path(output, "", None)
        except DatasetError as exc:
            call_from_thread(self, self.notify, str(exc), severity="error")
            raise RuntimeError(str(exc)) from exc
        call_from_thread(
            self,
            self.notify,
            msg.CMD_EXPORT_SUCCESS.format(pages=summary.pages, output=output),
        )

    def _cmd_import(self, args: str) -> None:
        """Enqueue /import as a task so re-embedding progress shows in the task bar."""
        path = args.strip()
        if not path:
            self.notify(msg.CMD_IMPORT_USAGE, severity="warning")
            return
        if self._sync_active:
            self.notify(msg.SYNC_ALREADY_ACTIVE, severity="warning")
            return
        from lilbee.cli.tui.task_queue import TaskType

        self._sync_active = True

        def _target(reporter: ProgressReporter) -> None:
            try:
                self._do_import(path, reporter)
            finally:
                self._sync_active = False
                self._task_bar.start_detect_pending()

        name = msg.TASK_NAME_IMPORT.format(file=Path(path).name)
        self._task_bar.start_task(name, TaskType.IMPORT, _target)

    def _do_import(self, raw_path: str, reporter: ProgressReporter) -> None:
        """Import body. Runs on the task worker thread."""
        from lilbee.app.dataset import DatasetError, import_from_path
        from lilbee.cli.tui.widgets.autocomplete import invalidate_document_cache

        reporter.update(0, msg.IMPORT_STATUS_LOADING, indeterminate=True)
        try:
            summary = asyncio_loop.run(
                import_from_path(
                    Path(raw_path).expanduser(),
                    "",
                    on_progress=build_import_progress_callback(reporter),
                )
            )
        except DatasetError as exc:
            call_from_thread(self, self.notify, str(exc), severity="error")
            raise RuntimeError(str(exc)) from exc
        invalidate_document_cache()
        call_from_thread(
            self,
            self.notify,
            msg.CMD_IMPORT_SUCCESS.format(
                sources=len(summary.sources), pages=summary.pages, chunks=summary.chunks
            ),
        )

    def _cmd_help(self, _args: str) -> None:
        self.action_show_command_catalog()

    def action_show_command_catalog(self) -> None:
        """Push the slash-command catalog modal; selected name is inserted into the input."""
        self.app.push_screen(SlashCommandCatalog(), self._on_catalog_pick)

    def insert_slash_command(self, name: str) -> None:
        """Drop ``name + ' '`` into the chat input and focus it for argument entry."""
        self._enter_insert_mode()
        inp = self._chat_input
        inp.value = f"{name} "
        inp.action_end()

    def _on_catalog_pick(self, name: str | None) -> None:
        if name is None:
            return
        self.insert_slash_command(name)

    def _cmd_login(self, args: str) -> None:
        token = args.strip()
        if not token:
            import webbrowser

            webbrowser.open("https://huggingface.co/settings/tokens")
            self.notify(msg.CHAT_LOGIN_PROMPT)
            return
        self._run_hf_login(token)

    @work(thread=True)
    def _run_hf_login(self, token: str) -> None:
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
            call_from_thread(self, self.notify, msg.CHAT_LOGGED_IN)
        except Exception as exc:
            log.warning("HuggingFace login failed", exc_info=True)
            call_from_thread(
                self, self.notify, msg.CHAT_LOGIN_FAILED.format(error=exc), severity="error"
            )

    def _cmd_model(self, args: str) -> None:
        if args:
            from lilbee.catalog.formatting import display_label_for_ref

            apply_active_model(self.app, "chat_model", args)
            self.app.title = msg.app_title(cfg.chat_model)
            self.notify(msg.CMD_MODEL_SET.format(name=display_label_for_ref(cfg.chat_model)))
            self.apply_model_change()
            self.refresh_model_bar()
        else:
            from lilbee.cli.tui.screens.catalog import CatalogScreen

            self.app.push_screen(CatalogScreen())

    def _cmd_quit(self, _args: str) -> None:
        self.app.exit()

    def _cmd_remove(self, args: str) -> None:
        name = args.strip()
        if not name:
            self.notify(msg.CMD_REMOVE_USAGE, severity="warning")
            return
        self._run_remove_model(name)

    @work(thread=True)
    def _run_remove_model(self, name: str) -> None:
        mgr = get_services().model_manager
        if not mgr.is_installed(name):
            call_from_thread(
                self, self.notify, msg.CMD_REMOVE_NOT_FOUND.format(name=name), severity="error"
            )
            return
        try:
            removed = mgr.remove(name)
            if removed:
                call_from_thread(self, self.notify, msg.CMD_REMOVE_SUCCESS.format(name=name))
            else:
                call_from_thread(
                    self, self.notify, msg.CMD_REMOVE_FAILED.format(name=name), severity="error"
                )
        except Exception:
            log.warning("Remove failed for %s", name, exc_info=True)
            call_from_thread(
                self, self.notify, msg.CMD_REMOVE_FAILED.format(name=name), severity="error"
            )

    def _cmd_rebuild(self, _args: str) -> None:
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self._run_sync(force_rebuild=True)

        self.app.push_screen(
            ConfirmDialog(msg.CMD_REBUILD_CONFIRM_TITLE, msg.CMD_REBUILD_CONFIRM_MESSAGE),
            _on_confirm,
        )

    def _cmd_reset(self, args: str) -> None:
        self.request_reset()

    def request_reset(self) -> None:
        """Public entry for the confirm-then-wipe flow (shared by /reset and the
        command palette), so callers don't reach into a private slash handler."""
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            from lilbee.app.reset import perform_reset

            try:
                result = perform_reset()
            except Exception as exc:
                log.warning("Reset failed", exc_info=True)
                self.notify(msg.CMD_RESET_FAILED.format(error=exc), severity="error")
                return

            # Reopen LanceDB against the now-empty data dir; keep providers loaded.
            reset_store()

            if result.skipped:
                self.notify(
                    msg.CMD_RESET_PARTIAL.format(skipped=len(result.skipped)),
                    severity="warning",
                )
            else:
                self.notify(msg.CMD_RESET_SUCCESS)

        self.app.push_screen(
            ConfirmDialog(msg.CMD_RESET_CONFIRM_TITLE, msg.CMD_RESET_CONFIRM_MESSAGE),
            _on_confirm,
        )

    def _cmd_set(self, args: str) -> None:
        if not args:
            return
        parts = args.split(None, 1)
        key = parts[0]
        value = parts[1] if len(parts) > 1 else ""

        if key not in SETTINGS_MAP:
            self.notify(msg.CMD_SET_UNKNOWN.format(key=key), severity="warning")
            return

        defn = SETTINGS_MAP[key]
        if not defn.writable:
            self.notify(msg.CMD_SET_READONLY.format(key=key), severity="warning")
            return
        try:
            if defn.type is bool:
                parsed = value.lower() in ("true", "1", "yes", "on")
            elif defn.nullable and value.lower() in ("none", "null", ""):
                parsed = None
            else:
                if defn.choices and value not in defn.choices:
                    self.notify(
                        msg.CMD_SET_CHOICES.format(key=key, choices=", ".join(defn.choices)),
                        severity="error",
                    )
                    return
                try:
                    parsed = defn.type(value)
                except (ValueError, TypeError):
                    self.notify(
                        msg.CMD_SET_TYPE_HINT.format(key=key, kind=_setting_type_hint(defn.type)),
                        severity="error",
                    )
                    return
            # Route through set_setting so settings_changed_signal subscribers
            # (model bar, scope chip, status bar) refresh. The boundary's
            # _invalidate_caches now handles llm_provider service reset.
            self.app.set_setting(key, parsed)
            shown = msg.MASKED_VALUE if defn.secret and parsed else parsed
            self.notify(msg.CMD_SET_SUCCESS.format(key=key, value=shown))
        except (ValueError, TypeError) as exc:
            self.notify(msg.CMD_SET_INVALID.format(key=key, error=exc), severity="error")

    def _cmd_settings(self, _args: str) -> None:
        self.app.switch_view("Settings")

    def _cmd_setup(self, _args: str) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        self.app.push_screen(SetupWizard(), self._on_setup_complete)

    def _cmd_remember(self, args: str) -> None:
        """Run /remember in a worker so embedding the text never blocks the UI."""
        self._cmd_remember_worker(args)

    @work(thread=True, name="chat_cmd_remember", exit_on_error=False)
    def _cmd_remember_worker(self, raw: str) -> None:
        """Store the memory off the UI thread; notify the outcome back on it."""
        outcome = remember_from_input(raw)
        call_from_thread(self, self.notify, outcome.message, severity=outcome.severity)

    def _cmd_memories(self, _args: str) -> None:
        from lilbee.cli.tui.screens.memories import MemoriesScreen

        self.app.push_screen(MemoriesScreen())

    def _cmd_status(self, _args: str) -> None:
        self.app.switch_view("Status")

    def _cmd_theme(self, args: str) -> None:
        if not args:
            # Land in the prompt with the dropdown listing every theme.
            self.insert_slash_command("/theme")
            return
        if args not in DARK_THEMES:
            self.notify(
                msg.CMD_THEME_UNKNOWN.format(name=args, names=", ".join(DARK_THEMES)),
                severity="warning",
            )
            return
        self.app.set_theme(args)
        self.notify(msg.THEME_SET.format(name=args))

    def _cmd_version(self, _args: str) -> None:
        self.notify(msg.CHAT_VERSION.format(version=get_version()))

    def _cmd_wiki(self, _args: str) -> None:
        if not cfg.wiki:
            self.notify(msg.CMD_WIKI_DISABLED, severity="warning")
            return
        self.app.switch_view("Wiki")

    def _cmd_sessions(self, _args: str) -> None:
        self.app.action_toggle_sessions()

    def _send_message(self, text: str) -> None:
        """Send a user message and stream the response."""
        from textual.css.query import NoMatches

        log = self._chat_log
        with contextlib.suppress(NoMatches):
            log.query_one("#chat-welcome", ChatWelcome).remove()
        question = UserMessage(text)
        log.mount(question)
        self._active_question = question

        # The assistant bubble owns its own ThinkingHeader animator until
        # the first reasoning or content token swaps it out.
        assistant_msg = AssistantMessage()
        self._active_assistant = assistant_msg
        log.mount(assistant_msg)
        # A fresh turn always follows its own answer, even if the user had
        # scrolled up during the previous response and released the anchor.
        log.anchor()

        with self._history_lock:
            self._history.append({"role": "user", "content": text})
        self._persist_user_turn(text)
        self.streaming = True
        self._stream_response(text, assistant_msg, self._current_chunk_type())

    def _current_scope_value(self) -> str:
        """The ScopeChip's selection, or "both" when the chip isn't mounted."""
        from textual.css.query import NoMatches

        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        try:
            chip = self.query_one("#scope-chip", ScopeChip)
        except NoMatches:
            return SearchScope.BOTH.value
        return chip.scope

    def _current_chunk_type(self) -> ChunkType | None:
        """Translate the ScopeChip selection into a ``chunk_type`` arg.

        Returns ``None`` for "both" (no filter) and the raw/wiki ``ChunkType``
        otherwise.
        """
        return scope_to_chunk_type(self._current_scope_value())

    def _open_session(self, store: SessionStore, first_text: str) -> str:
        """Create the active session, auto-title it, and return its id."""
        session_id = store.create(model_ref=cfg.chat_model, scope=self._current_scope_value())
        store.set_title(session_id, derive_title(first_text), TitleSource.AUTO)
        self._session_id = session_id
        return session_id

    def _persist_user_turn(self, text: str) -> None:
        """Open a session on the first turn (auto-titled), then append the message."""
        if not cfg.sessions_enabled:
            # Sessions turned off: the conversation stays live in memory but is
            # never written to disk. _session_id stays None, so the assistant
            # turn's persist is a no-op too.
            return
        store = get_services().session_store
        session_id = self._session_id or self._open_session(store, text)
        message = SessionMessage(role=MessageRole.USER, content=text)
        try:
            store.add_message(session_id, message, surface=SessionOrigin.TUI)
        except SessionNotFoundError:
            # The active session was deleted mid-chat (e.g. from the drawer);
            # open a fresh one so auto-save keeps working instead of crashing.
            store.add_message(self._open_session(store, text), message, surface=SessionOrigin.TUI)

    def _persist_assistant_turn(self, content: str, sources: list[str]) -> None:
        """Append the assistant turn to the active session. Worker thread."""
        if self._session_id is None or not cfg.sessions_enabled:
            # Sessions switched off mid-conversation: the id outlives the
            # setting, so the toggle has to be re-checked here rather than
            # relying on _persist_user_turn having left the id unset.
            return
        # A concurrent delete of the active session must not crash the worker.
        with contextlib.suppress(SessionNotFoundError):
            get_services().session_store.add_message(
                self._session_id,
                SessionMessage(role=MessageRole.ASSISTANT, content=content, sources=tuple(sources)),
                surface=SessionOrigin.TUI,
            )

    def resume_session(self, session_id: str) -> None:
        """Load a saved session into the chat view and make it the active one."""
        store = get_services().session_store
        session = store.get(session_id)
        self._reset_conversation()
        self._session_id = session_id
        for message in session.messages:
            self._render_restored_message(message)
        # Load the whole transcript and the summary it was compacted with. What
        # does not fit is folded into the summary by _compact_history on the next
        # turn, off the UI thread; windowing it away here would silently lose the
        # turns between the stored summary and the window, which is precisely
        # what a resumed conversation must not do.
        loaded: list[ChatMessage] = [
            {"role": message.role.value, "content": message.content} for message in session.messages
        ]
        with self._history_lock:
            self._history = loaded
            self._summary = session.summary
        self._restore_session_model(session.meta.model_ref)
        self._refresh_context_usage()
        self._chat_log.scroll_end(animate=False)
        self.notify(msg.SESSIONS_RESUMED.format(title=session.meta.title))

    def _restore_session_model(self, model_ref: str) -> None:
        """Switch to the session's chat model if it is still installed.

        A conversation records the model it used, but that model may have been
        deleted since. Restoring a missing ref would be rejected by the model
        boundary with a scary error, so only switch when the model is installed;
        otherwise keep the current model and say the original is gone.
        """
        if not model_ref or model_ref == cfg.chat_model:
            return
        if get_services().registry.is_installed(model_ref):
            apply_active_model(self.app, "chat_model", model_ref)
        else:
            self.notify(
                msg.SESSIONS_MODEL_UNAVAILABLE.format(model=model_ref, current=cfg.chat_model),
                severity="warning",
            )

    @property
    def session_id(self) -> str | None:
        """The saved session this conversation persists to, or None before the first turn."""
        return self._session_id

    def start_new_conversation(self) -> None:
        """Clear the conversation and open a fresh session on the next turn."""
        self._reset_conversation()
        self.notify(msg.SESSIONS_NEW)

    def _render_restored_message(self, message: SessionMessage) -> None:
        """Mount a completed message widget for a resumed turn."""
        log = self._chat_log
        if message.role == MessageRole.USER:
            log.mount(UserMessage(message.content))
            return
        # Constructed complete, not appended-to after mounting: mount() is async,
        # so append_content/finish would both no-op against a content widget that
        # compose has not built yet, and the answer would render empty.
        log.mount(AssistantMessage(content=message.content, sources=list(message.sources)))

    @work(thread=True)
    def _stream_response(
        self, question: str, widget: AssistantMessage, chunk_type: ChunkType | None
    ) -> None:
        """Schedule the response stream on a background thread."""
        self._do_stream_response(question, widget, chunk_type)

    def _do_stream_response(
        self, question: str, widget: AssistantMessage, chunk_type: ChunkType | None
    ) -> None:
        """Stream LLM response, coalescing UI updates. Worker thread."""
        response_parts: list[str] = []
        sources: list[str] = []
        stream: Any = None
        try:
            if not self._await_chat_engine(widget):
                return
            self._compact_history()
            with self._history_lock:
                # [:-1] drops the question, which ask_stream takes separately.
                recent = self._history[:-1]
                summary = self._summary
            history_snapshot = prompt_history(recent, summary, max_tokens=self._history_budget())
            stream = get_services().searcher.ask_stream(
                question, history=history_snapshot, chunk_type=chunk_type
            )
            self._consume_stream(stream, widget, response_parts)
        except EmbeddingModelMismatchError as exc:
            with contextlib.suppress(Exception):
                call_from_thread(self, self._on_embedding_mismatch, exc, question, widget)
        except Exception as exc:
            log.debug("Stream error", exc_info=True)
            # A deliberate cancel severs the transport, which surfaces here as a
            # stream error; the cancel already wrote its note into the bubble.
            if not self._stream_worker_cancelled():
                with contextlib.suppress(Exception):
                    call_from_thread(
                        self, widget.append_content, msg.STREAM_ERROR.format(error=exc)
                    )
        finally:
            close_stream(stream)
            self._finalize_stream(widget, sources, response_parts)
            call_from_thread(self, self._maybe_extract_memories, question, "".join(response_parts))

    @staticmethod
    def _stream_worker_cancelled() -> bool:
        """Whether the calling stream worker was cancelled; False off-worker."""
        try:
            return _get_worker().is_cancelled
        except NoActiveWorker:
            return False

    def _await_chat_engine(self, widget: AssistantMessage) -> bool:
        """Hold the stream until the engine can serve, painting the load into *widget*.

        The default lifecycle loads the engine on demand, so the first prompt of
        a session usually lands here: the answer bubble's thinking row carries the
        live load phase instead of the input locking up. Worker thread. Returns
        False once the wait was cancelled or the load failed, with any failure
        already rendered into the bubble.
        """
        from lilbee.app.placement import (
            chat_engine_ready,
            chat_warm_error,
            request_engine_warm,
            wait_chat_ready,
        )

        # Build the container if nothing holds it (a settings change resets it);
        # readiness is probed via peek_services, which never builds, so without
        # this a prompt sent into the gap would report a dead engine instead of
        # lazily rebuilding the way ask_stream always has.
        get_services()
        if chat_engine_ready():
            return True
        # A failed boot warm leaves nothing in flight; this restarts the engine
        # so the prompt waits out a fresh load instead of bouncing.
        request_engine_warm()
        self._show_warm_tip_once()
        worker = _get_worker()

        def _paint(snapshot: WarmProgress) -> None:
            with contextlib.suppress(Exception):
                call_from_thread(self, widget.set_thinking_status, _engine_status_text(snapshot))

        # Label the wait before the chat warm stamps its first phase: another
        # role loading first (embed on a cold start) leaves the tracker silent
        # for many seconds, and a bare scanner reads as a hang.
        with contextlib.suppress(Exception):
            call_from_thread(self, widget.set_thinking_status, msg.ENGINE_WARMING)
        if wait_chat_ready(on_progress=_paint, should_abort=lambda: worker.is_cancelled):
            with contextlib.suppress(Exception):
                call_from_thread(self, widget.set_thinking_status, "")
            return True
        if worker.is_cancelled:
            return False
        error = chat_warm_error()
        text = (
            f"{msg.ENGINE_LOAD_FAILED.format(error=error)}\n{msg.ENGINE_FAILED_HINT}"
            if error is not None
            else msg.ENGINE_NOT_READY
        )
        with contextlib.suppress(Exception):
            call_from_thread(self, widget.append_content, text)
        return False

    def _show_warm_tip_once(self) -> None:
        """Toast the keep-warm tip on the session's first cold-engine wait. Worker thread."""
        if cfg.keep_engine_warm or self._warm_tip_shown:
            return
        self._warm_tip_shown = True
        with contextlib.suppress(Exception):
            call_from_thread(self, self.notify, msg.ENGINE_WARM_TIP, timeout=8)

    def _maybe_extract_memories(self, question: str, answer: str) -> None:
        """Spawn auto-extraction for the finished turn, when enabled and idle.

        Runs on the main thread (scheduled from the stream worker). Skips while
        indexing so the extraction's embed call never contends with a sync.
        """
        from lilbee.app.memory import auto_extract_enabled

        if not answer or not auto_extract_enabled() or self._indexing_active():
            return
        self._extract_memories_worker(question, answer)

    def _indexing_active(self) -> bool:
        """True while a sync/add/import/wiki task is running (embed worker is busy).

        Wiki counts: a build embeds citations and a draft accept re-chunks and
        re-indexes the page it publishes.
        """
        from lilbee.cli.tui.task_queue import TaskType

        busy = {
            TaskType.SYNC.value,
            TaskType.ADD.value,
            TaskType.IMPORT.value,
            TaskType.WIKI.value,
        }
        return any(task.task_type in busy for task in self._task_bar.queue.active_tasks)

    @work(thread=True, name="chat_memory_extract", exit_on_error=False)
    def _extract_memories_worker(self, question: str, answer: str) -> None:
        """Extract durable memories off the UI thread; notify how many landed."""
        from lilbee.app.memory import auto_extract

        stored = auto_extract(question, answer)
        if stored:
            call_from_thread(self, self.notify, msg.MEMORY_AUTO_EXTRACTED.format(count=len(stored)))

    def _on_embedding_mismatch(
        self, exc: EmbeddingModelMismatchError, question: str, widget: AssistantMessage
    ) -> None:
        """Offer to adopt the index's embedder (same dim) or explain the rebuild path."""
        if not exc.dims_match:
            widget.append_content(msg.EMBED_ADOPT_REBUILD_NOTICE.format(dim=exc.persisted_dim))
            return
        widget.append_content(msg.EMBED_ADOPT_NOTICE.format(model=exc.persisted_model))
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        self.app.push_screen(
            ConfirmDialog(
                msg.EMBED_ADOPT_CONFIRM_TITLE,
                msg.EMBED_ADOPT_CONFIRM_MESSAGE.format(model=exc.persisted_model),
            ),
            lambda ok: self._on_adopt_confirm(ok, exc.persisted_model, question),
        )

    def _on_adopt_confirm(self, confirmed: bool | None, ref: str, question: str) -> None:
        """Run the adopt+retry in a worker thread, or report the cancellation."""
        if not confirmed:
            self.notify(msg.EMBED_ADOPT_CANCELLED)
            return
        self.notify(msg.EMBED_ADOPTING.format(model=ref))
        self._adopt_and_retry(ref, question)

    @work(thread=True)
    def _adopt_and_retry(self, ref: str, question: str) -> None:
        """Schedule the adopt+retry on a worker thread (pull may be slow)."""
        self._do_adopt_and_retry(ref, question)

    def _do_adopt_and_retry(self, ref: str, question: str) -> None:
        """Switch to embedder *ref* (downloading if needed), then re-ask. Worker thread."""
        from lilbee.app.models import adopt_embedder

        try:
            adopt_embedder(ref)
        except Exception as exc:  # surfaced to the user, never silently swallowed
            log.debug("Embedder adopt failed", exc_info=True)
            call_from_thread(
                self, self.notify, msg.EMBED_ADOPT_FAILED.format(error=exc), severity="error"
            )
            return
        call_from_thread(self, self.notify, msg.EMBED_ADOPTED.format(model=ref))
        call_from_thread(self, self._send_message, question)

    def _consume_stream(
        self, stream: Any, widget: AssistantMessage, response_parts: list[str]
    ) -> None:
        """Pull tokens off *stream*, batching UI updates to ~50 ms windows."""
        worker = _get_worker()
        reason_buf: list[str] = []
        content_buf: list[str] = []
        timings = _StreamTimings(last_flush=time.monotonic())

        def flush() -> None:
            if reason_buf:
                call_from_thread(self, widget.append_reasoning, "".join(reason_buf))
                reason_buf.clear()
            if content_buf:
                call_from_thread(self, widget.append_content, "".join(content_buf))
                content_buf.clear()

        for token in stream:
            if worker.is_cancelled:
                break
            try:
                self._buffer_token(token, reason_buf, content_buf, response_parts)
                self._maybe_flush(flush, timings)
            except Exception:
                break  # App shutting down (Ctrl-C) -- stop streaming
        with contextlib.suppress(Exception):
            flush()

    @staticmethod
    def _buffer_token(
        token: Any,
        reason_buf: list[str],
        content_buf: list[str],
        response_parts: list[str],
    ) -> None:
        """Append *token* to the right buffer; record response content for history."""
        if token.is_reasoning:
            reason_buf.append(token.content)
        elif token.content:
            response_parts.append(token.content)
            content_buf.append(token.content)

    def _maybe_flush(self, flush: Callable[[], None], timings: _StreamTimings) -> None:
        """Run *flush* on its interval. The chat log is anchored, so Textual
        keeps the answer's tail in view as it grows without a scroll of ours.
        """
        now = time.monotonic()
        if now - timings.last_flush >= _STREAM_FLUSH_INTERVAL:
            flush()
            timings.last_flush = now

    def _finalize_stream(
        self, widget: AssistantMessage, sources: list[str], response_parts: list[str]
    ) -> None:
        """Persist the assistant turn and update the widget. Always runs."""
        # _stream_response runs in a worker thread; reactive setters mutate
        # widgets, so the streaming flag must flip on the main thread.
        call_from_thread(self, self._set_streaming, False)
        full_response = "".join(response_parts)
        if full_response:
            with self._history_lock:
                self._history.append({"role": "assistant", "content": full_response})
            # No trim here: the next turn compacts before it builds its prompt, so
            # trimming now would drop turns without folding them into the summary.
            self._persist_assistant_turn(full_response, sources)
            call_from_thread(self, self._refresh_context_usage)
        call_from_thread(self, widget.finish, sources)
        if (
            cfg.chat_mode == ChatMode.SEARCH.value
            and self._embedding_ready()
            and full_response
            and SOURCES_BLOCK_MARKER not in full_response
        ):
            call_from_thread(self, self._notify_no_results)

    def _notify_no_results(self) -> None:
        self.notify(msg.CHAT_MODE_SEARCH_NO_RESULTS, severity="warning")

    @staticmethod
    def _history_budget() -> int:
        """Token budget for everything this conversation carries into the prompt."""
        return history_budget(cfg.chat_n_ctx_target)

    def _compact_history(self) -> None:
        """Fold turns that no longer fit into the rolling summary. Worker thread only.

        Runs before a prompt is built rather than after a turn lands, so the
        summary is always current with what is about to be sent, and a resumed
        conversation compacts what it cannot carry instead of dropping it.

        The summarizing model call is slow, so it happens without the lock held;
        only the known prefix is removed afterwards, which stays correct if the
        user sends another turn meanwhile.
        """
        with self._history_lock:
            history = list(self._history)
            summary = self._summary
        budget = self._history_budget()
        if not cfg.chat_compaction:
            # Default path, deliberately free: prune exactly to the limit, no
            # model call. The summary is charged against the same budget so a
            # session compacted on earlier hardware still carries its notes.
            reserved = sum(estimate_tokens(m) for m in summary_messages(summary))
            dropped = overflow(history, max_tokens=max(1, budget - reserved))
            if not dropped:
                return
            with self._history_lock:
                del self._history[: len(dropped)]
            call_from_thread(self, self._on_history_trimmed, len(dropped))
            return
        # Compaction on: fire early, clear deep (see COMPACT_TRIGGER_FRACTION).
        if not compaction_due(history, summary, max_tokens=budget):
            return
        dropped = foldable(history)
        if not dropped:
            # Nothing but the tail, and it alone fills the budget. Folding it
            # would summarize the very turn being answered; prompt_history windows
            # it instead.
            return
        # Condensing blocks this turn on a model call: seconds on a GPU, tens of
        # seconds on a CPU-only host. An unannounced pause that long is
        # indistinguishable from a hang, so say what is happening first.
        call_from_thread(self, self._set_compacting, True)
        try:
            result = get_services().searcher.summarize_history(dropped, summary)
        finally:
            call_from_thread(self, self._set_compacting, False)
        with self._history_lock:
            del self._history[: len(dropped)]
            self._summary = result.summary
        if self._session_id and result.summary and cfg.sessions_enabled:
            # A summary for a session deleted mid-chat is not worth a crash; the
            # next user turn reopens one and re-summarizes from there. The
            # toggle is re-checked because the fold keeps working in memory
            # after sessions go off, but must not reach the disk.
            with contextlib.suppress(SessionNotFoundError):
                get_services().session_store.set_summary(self._session_id, result.summary)
        call_from_thread(self, self._on_history_compacted, result.condensed, result.stranded)

    def _set_compacting(self, compacting: bool) -> None:
        """Flip the chip into (or out of) its condensing state. Main thread only."""
        with contextlib.suppress(NoMatches):
            self.query_one("#context-chip", ContextChip).compacting = compacting

    def _refresh_context_usage(self) -> None:
        """Push current history pressure to the chip. Main thread only.

        Cheap: the same char/4 estimate the windower already uses, over messages
        that are in memory anyway. Recomputed per turn rather than per keystroke.
        """
        with self._history_lock:
            history = list(self._history)
            summary = self._summary
        budget = self._history_budget()
        used = sum(estimate_tokens(m) for m in history)
        used += sum(estimate_tokens(m) for m in summary_messages(summary))
        with contextlib.suppress(NoMatches):
            self.query_one("#context-chip", ContextChip).usage = used / max(1, budget)

    def _mark_context_boundary(self, *titles: str) -> None:
        """Draw rules in the log where the model's view of the chat changed.

        A rich Rule, not a hand-drawn "-- text --": it draws the line out to the
        full width itself, which is what makes it read as a boundary rather than
        as another message. Guarded because the worker can land this after the
        user has navigated off the chat screen.
        """
        # mount() is async: the anchor may not be in the log yet, and mounting
        # before a non-child raises. Appending reads fine in that race.
        anchor = self._active_question
        if anchor is not None and not anchor.is_mounted:
            anchor = None
        with contextlib.suppress(NoMatches):
            for title in titles:
                rule = Static(
                    Rule(title=title, characters="─", style="dim"),
                    classes="compaction-marker",
                )
                if anchor is None:
                    self._chat_log.mount(rule)
                else:
                    self._chat_log.mount(rule, before=anchor)

    def _on_history_trimmed(self, dropped: int) -> None:
        """Mark where turns left the model's view with nothing standing in for them.

        The compaction-off path. Same rule as compaction so the log reads
        consistently, different words because nothing was summarized.
        """
        self._refresh_context_usage()
        self._mark_context_boundary(msg.CHAT_TRIMMED.format(count=dropped))
        with contextlib.suppress(NoMatches):
            self.notify(msg.CHAT_TRIMMED_TOAST, severity="warning")

    def _on_history_compacted(self, condensed: int, stranded: int) -> None:
        """Mark where the model's memory of this conversation turns into a summary.

        Styling lives in chat.tcss under .compaction-marker. Guarded because the
        worker can land this after the user navigated off the chat screen.

        Stranded turns get their own line: they are gone from the model's view
        with nothing standing in for them, and a user whose model has forgotten
        something is owed the reason rather than left to infer it.
        """
        self._refresh_context_usage()
        titles = [msg.CHAT_COMPACTED.format(count=condensed)]
        if stranded:
            titles.append(msg.CHAT_COMPACTION_STRANDED.format(count=stranded))
        self._mark_context_boundary(*titles)
        with contextlib.suppress(NoMatches):
            self.notify(
                msg.CHAT_COMPACTED_STRANDED_TOAST if stranded else msg.CHAT_COMPACTED_TOAST,
                severity="warning",
            )

    def action_scroll_up(self) -> None:
        self._chat_log.scroll_page_up()

    def action_scroll_down(self) -> None:
        self._chat_log.scroll_page_down()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Keep the footer honest about mode-dependent bindings.

        - ``cancel_stream`` (Ctrl+C) only does something while streaming in
          INSERT mode; otherwise the App's Quit binding takes the slot.
        """
        if action == "cancel_stream":
            return self.streaming and self._insert_mode
        if action == "enter_model_strip":
            # NORMAL mode parks the cursor on the transcript, and that is the
            # only place these letters are free. Stated as where they DO apply,
            # so a drawer, a dialog or any later focus target keeps its own
            # letters without having to be named here.
            focused = self.focused
            return focused is not None and focused.id == "chat-log"
        return super().check_action(action, parameters)

    def action_enter_normal_mode(self) -> None:
        """Esc dismisses the overlay if visible; otherwise drops into NORMAL mode."""
        overlay = self._completion_overlay
        if overlay.is_visible:
            # Revert any previewed candidate back to what the user typed.
            if self._completion_origin is not None and self._chat_input.value != (
                self._completion_origin
            ):
                self._set_input(self._completion_origin)
            self._completion_origin = None
            overlay.hide()
            # Backing out of the command list leaves nothing worth keeping in
            # a lone slash, and it would hijack the next message as /word.
            if self._chat_input.value.strip() == "/":
                self._set_input("")
            return
        if isinstance(self.focused, Select) or self._focus_in_model_bar():
            # Leaving the model strip should put us back in INSERT so the
            # user can type their next prompt; routing through the helper
            # makes sure can_focus is re-enabled.
            self._enter_insert_mode()
            return
        self._insert_mode = False
        # Make the chat input unfocusable in NORMAL mode so Tab traversal
        # skips past it AND a programmatic focus restore (modal close,
        # screen pop) cannot land on it. The user re-enters INSERT
        # explicitly via i/a/o/Enter or by clicking the input.
        self._chat_input.can_focus = False
        self._chat_log.focus()
        self._update_input_style()

    def action_cancel_stream(self) -> None:
        """Cancel an in-flight chat stream. Bound to Ctrl+C from INSERT mode."""
        if self.streaming:
            self._cancel_inflight_stream(msg.STREAM_CANCELLED)

    def _cancel_inflight_stream(self, note: str) -> None:
        """Stop the streaming worker, sever its inference call, and say so.

        The worker cancel is cooperative and only observed between tokens, so
        ``cancel_inference`` severs the in-flight stream's transport to unblock
        a reader stuck in a socket read. *note* lands in the answer bubble: a
        cancelled turn must say it was cancelled, not die silently while the
        user waits for an answer that will never arrive.
        """
        for worker in self.workers:
            worker.cancel()
        get_services().cancel_inference()
        bubble = self._active_assistant
        if bubble is not None and bubble.is_mounted:
            bubble.append_content(note)
        self.streaming = False

    def apply_model_change(self) -> None:
        """Swap to the new chat model without freezing the UI.

        Reloading the fleet for the new model is a multi-second restart, so it
        runs in a thread worker instead of on the event loop. The in-flight stream
        is cancelled first, the input is blocked behind a "switching" state with an
        indicator toast, and the worker reloads only the chat role. The provider
        retires any still-busy client across the restart and serializes overlapping
        reloads, so the worker can start at once without waiting for other workers.
        The input re-enables once the fleet has restarted with the new model (which
        loads on the next request).
        """
        if self.swapping_model:
            # A swap is already loading; a second one (rapid /model, or the model
            # bar re-clicked while the input is disabled) would spawn a duplicate
            # worker and a duplicate completion toast. The in-flight reload already
            # coalesces onto the latest cfg, so ignore the re-entry.
            self.notify(msg.CHAT_MODEL_SWITCHING, severity="warning", timeout=3)
            return
        if self.streaming:
            self._cancel_inflight_stream(msg.STREAM_CANCELLED_MODEL_SWITCH)
        self.swapping_model = True
        self.app.notify(msg.MODEL_SWAP_APPLYING)
        self._reload_chat_model_worker()

    def _apply_input_busy_state(self) -> None:
        """Disable the chat input while a swap or placement reload is loading, and
        say why in the placeholder so a person is never left facing a dead input
        with no explanation.

        Restores focus and the default placeholder when the fleet is idle again so
        the user can type without re-clicking the input that was disabled out from
        under them. Guarded because the unblock can fire (via ``call_from_thread``
        or a bubbled message) after the user navigated away and the input is no
        longer mounted.
        """
        busy = self.swapping_model or self.reloading_placement
        with contextlib.suppress(NoMatches):
            inp = self._chat_input
            inp.disabled = busy
            if self.swapping_model:
                from lilbee.catalog.formatting import display_label_for_ref

                inp.placeholder = msg.CHAT_INPUT_SWITCHING.format(
                    name=display_label_for_ref(cfg.chat_model)
                )
            elif self.reloading_placement:
                inp.placeholder = msg.CHAT_INPUT_RELOADING
            else:
                inp.placeholder = msg.CHAT_INPUT_PLACEHOLDER_DEFAULT
            if not busy and self._insert_mode:
                inp.focus()

    def watch_swapping_model(self, swapping: bool) -> None:
        self._apply_input_busy_state()

    def watch_reloading_placement(self, reloading: bool) -> None:
        self._apply_input_busy_state()

    def on_fleet_body_placement_reloading(self, event: FleetBody.PlacementReloading) -> None:
        """Hold chat submissions while the Fleet drawer reloads the fleet."""
        self.reloading_placement = event.active

    @work(thread=True, name=_MODEL_SWAP_WORKER, exit_on_error=False)
    def _reload_chat_model_worker(self) -> None:
        """Reload the chat role and warm the new model before unblocking the input.

        ``reload_role(wait=True)`` re-plans and restarts the fleet for the new chat
        model (retrieval is untouched) and returns once the proxy is back up. The
        model is then warmed here rather than deferred to the user's next prompt:
        ``request_engine_warm`` drives the load and populates the provider warm
        tracker, which the task-bar footer renders (spinner, model, phase), and
        ``wait_chat_ready`` holds the input disabled until the model actually
        serves -- so the switch never hands back a live input in front of a model
        that has not loaded. The provider serializes overlapping reloads, so a
        rapid second swap coalesces onto the latest cfg.
        """
        from lilbee.app.placement import (
            chat_warm_error,
            request_engine_warm,
            wait_chat_ready,
        )

        worker = _get_worker()
        try:
            get_services().reload_role(WorkerRole.CHAT, wait=True)
            request_engine_warm()
            ready = wait_chat_ready(should_abort=lambda: worker.is_cancelled)
        except Exception as exc:  # any reload failure becomes a toast, never a crash
            call_from_thread(self, self._on_model_swap_failed, str(exc))
            return
        if worker.is_cancelled:
            return
        error = None if ready else chat_warm_error()
        if error:
            call_from_thread(self, self._on_model_swap_failed, error)
        else:
            call_from_thread(self, self._on_model_swapped)

    def _on_model_swapped(self) -> None:
        """Main-thread completion: unblock the input and confirm the new model."""
        from lilbee.catalog.formatting import display_label_for_ref

        self.swapping_model = False
        self.app.notify(msg.MODEL_SWAP_DONE.format(name=display_label_for_ref(cfg.chat_model)))

    def _on_model_swap_failed(self, error: str) -> None:
        """Main-thread failure: unblock the input and surface the error."""
        self.swapping_model = False
        self.app.notify(msg.MODEL_SWAP_FAILED.format(error=error), severity="error")

    @on(Markdown.LinkClicked)
    def _open_answer_link(self, event: Markdown.LinkClicked) -> None:
        """Open a link clicked in an answer: ``file:`` citations open in the OS
        default app for the file type; web links open in the browser."""
        event.stop()
        if event.href.startswith("file://"):
            open_local_file(event.href)
        else:
            self.app.open_url(event.href)

    async def action_toggle_markdown(self) -> None:
        """Toggle between Markdown and plain-text rendering for chat responses."""
        cfg.markdown_rendering = not cfg.markdown_rendering
        use_md = cfg.markdown_rendering
        chat_log = self._chat_log
        for widget in chat_log.query(AssistantMessage):
            await widget.rebuild_content_widget(use_md)
        label = "Markdown" if use_md else "Plain text"
        self.notify(msg.CHAT_RENDERING.format(label=label))

    def _run_sync(self, *, force_rebuild: bool = False) -> None:
        """Enqueue a document sync (or full rebuild) in the task bar."""
        if self._sync_active:
            self.notify(msg.SYNC_ALREADY_ACTIVE, severity="warning")
            return
        from lilbee.cli.tui.task_queue import TaskType

        self._sync_active = True
        # Clear the pending hint so the bar shows live sync progress
        # instead of the stale "N docs to sync" line.
        self._task_bar.clear_pending_sync()

        def _target(reporter: ProgressReporter) -> None:
            try:
                self._do_sync(reporter, force_rebuild=force_rebuild)
            finally:
                self._sync_active = False
                # Re-detect after every sync attempt: success drives the
                # count to 0, failure or cancel leaves the still-pending
                # files counted so the hint reappears.
                self._task_bar.start_detect_pending()

        label = msg.TASK_NAME_REBUILD if force_rebuild else msg.TASK_NAME_SYNC
        self._task_bar.start_task(label, TaskType.SYNC, _target, indeterminate=True)

    def _do_sync(self, reporter: ProgressReporter, *, force_rebuild: bool = False) -> None:
        """Sync body. Runs on worker thread."""
        from lilbee.data.ingest import sync

        reporter.update(0, msg.SYNC_STATUS_SYNCING, indeterminate=True)
        on_progress = build_sync_progress_callback(reporter)
        try:
            result = asyncio_loop.run(
                sync(quiet=True, on_progress=on_progress, force_rebuild=force_rebuild)
            )
        except asyncio.CancelledError as exc:
            raise RuntimeError(msg.SYNC_CANCELLED_RESUME) from exc
        if result.failed:
            raise RuntimeError(msg.SYNC_FAILED_FILES.format(files=", ".join(result.failed)))
        if result.skipped:
            call_from_thread(
                self,
                self.notify,
                msg.sync_skipped_message(", ".join(result.skipped)),
                severity="warning",
            )

    def action_focus_commands(self) -> None:
        """Focus chat input and pre-fill with '/' for command entry."""
        # Route through the helper so can_focus is re-enabled when this
        # action fires from NORMAL mode; bare ``inp.focus()`` would
        # silently no-op while the input is intentionally unfocusable.
        self._enter_insert_mode()
        inp = self._chat_input
        if not inp.value.startswith("/"):
            inp.value = "/"
            inp.action_end()

    def action_toggle_chat_mode(self) -> None:
        """F3: flip between Search and Chat mode."""
        try:
            toggle = self.query_one(ChatModeToggle)
        except NoMatches:
            return
        if not toggle.toggle():
            return
        label = (
            msg.CHAT_MODE_SEARCH_LABEL
            if cfg.chat_mode == ChatMode.SEARCH.value
            else msg.CHAT_MODE_CHAT_LABEL
        )
        self.notify(msg.CHAT_MODE_SET.format(label=label))

    def action_cycle_scope(self) -> None:
        """``s``: cycle the scope chip when it is currently visible."""
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        try:
            chip = self.query_one("#scope-chip", ScopeChip)
        except NoMatches:
            return
        if chip.has_class("-hidden"):
            return
        chip.cycle_scope()

    def action_complete(self) -> None:
        """Tab: fill the shared prefix, then cycle matches (readline / vim style).

        - Insert mode + chat input focused + dropdown closed but matches
          exist: open it, fill the longest common prefix, else preview the
          first match.
        - Insert mode + chat input focused + dropdown open: fill any further
          shared prefix, otherwise preview the next match.
        - Insert mode + chat input focused + no matches: insert ``\\t`` so
          users can type tab characters directly.
        - Normal mode or focus elsewhere: advance through the focus
          chain so Tab still walks every focusable widget.
        """
        inp = self._chat_input
        if not self._insert_mode or not inp.has_focus:
            self._tab_into_fleet_or_next()
            return
        overlay = self._completion_overlay
        if not overlay.is_visible and not self._open_completions():
            inp.insert("\t")
            return
        if self._fill_common_prefix():
            return
        self._preview_next()

    def _focus_in_drawer(self) -> bool:
        """True when keyboard focus is inside an open drawer, so Enter / i / a / o
        reach that drawer's own controls instead of entering insert mode.

        Asked of the Drawer base rather than one drawer class: a drawer that had
        to name itself here would otherwise swallow its own Enter until someone
        noticed.
        """
        focused = self.focused
        return bool(focused and any(isinstance(n, Drawer) for n in focused.ancestors_with_self))

    def _focus_in_model_bar(self) -> bool:
        """True when focus is on any model-strip member.

        Asked of the container rather than of each member class so a member
        added later is covered without a second edit here.
        """
        focused = self.focused
        return bool(focused and any(isinstance(n, ModelBar) for n in focused.ancestors_with_self))

    def _tab_into_fleet_or_next(self) -> None:
        """Jump Tab into the open Fleet drawer's first toggle so the placement
        editor is reachable without tabbing past every widget; once focus is
        inside the drawer, Tab cycles within it as usual."""
        drawers = self.screen.query(FleetDrawer)
        if not drawers:
            self.screen.focus_next()
            return
        drawer = drawers.first()
        focused = self.screen.focused
        inside = focused is not None and drawer in focused.ancestors_with_self
        toggles = drawer.query(".dev-toggle")
        if not inside and toggles:
            toggles.first().focus()
            return
        self.screen.focus_next()

    def action_complete_next(self) -> None:
        """Ctrl+N: preview the next match, opening the dropdown if it is closed (vim ``<C-n>``)."""
        if not self._chat_input.has_focus:
            # Not a completion here; skip so an open overlay (e.g. the sessions
            # drawer) can bind Ctrl+N instead of this priority binding eating it.
            raise SkipAction()
        if self._completion_overlay.is_visible or self._open_completions():
            self._preview_next()

    def action_complete_prev(self) -> None:
        """Ctrl+P: preview the previous match, opening the dropdown if it is closed."""
        if not self._chat_input.has_focus:
            return
        if self._completion_overlay.is_visible or self._open_completions():
            self._preview_prev()

    def _preview_next(self) -> None:
        """Preview the highlighted match if none is previewed yet, else step forward."""
        overlay = self._completion_overlay
        if self._chat_input.value == self._completion_origin:
            display = overlay.get_current()
        else:
            display = overlay.cycle_next()
        if display is not None:
            self._preview_completion(display)

    def _preview_prev(self) -> None:
        """Step the highlight backward (wrapping to the last match) and preview it."""
        display = self._completion_overlay.cycle_prev()
        if display is not None:
            self._preview_completion(display)

    def _open_completions(self) -> bool:
        """Show the dropdown for the current input and remember it as the origin."""
        options = get_completions(self._chat_input.value)
        if not options:
            return False
        self._completion_origin = self._chat_input.value
        self._completion_overlay.show_completions(options)
        return True

    def _completion_value(self, display: str) -> str:
        """Full input text produced by accepting ``display``, keeping the typed prefix.

        Path completions are basenames, so the directory the user already
        typed (``~/``, ``./``, absolute) is preserved and only the final
        segment is replaced.
        """
        text = (
            self._completion_origin
            if self._completion_origin is not None
            else (self._chat_input.value)
        )
        if " " not in text:
            return display
        cmd, _, partial = text.partition(" ")
        if cmd.lower() in PATH_ARG_COMMANDS:
            head = path_completion_prefix(partial)
            return f"{cmd} {head}{display}"
        return f"{cmd} {display}"

    def _set_input(self, value: str) -> None:
        """Replace the input value without triggering the live-refresh of the dropdown."""
        inp = self._chat_input
        if inp.value == value:
            return
        # The setter posts Changed asynchronously; flag one event to ignore so
        # the previewed candidate doesn't re-filter (and collapse) the dropdown.
        # (The value setter already moves the cursor to the end.)
        self._suppress_refresh += 1
        inp.value = value

    def _preview_completion(self, display: str) -> None:
        """Write the highlighted candidate into the input, leaving the dropdown open."""
        self._set_input(self._completion_value(display))

    def _fill_common_prefix(self) -> bool:
        """Extend the input to the longest prefix shared by all matches; True if it grew."""
        overlay = self._completion_overlay
        values = [self._completion_value(d) for d in overlay.options]
        shared = longest_common_prefix(values)
        if len(shared) <= len(self._chat_input.value):
            return False
        self._set_input(shared)
        # Re-filter for the newly completed prefix (descends into a directory,
        # narrows the model list, etc.).
        self._refresh_completion_overlay()
        return True

    def action_history_prev(self) -> None:
        """Up arrow: cycle the dropdown if visible, else recall previous history entry."""
        if not self._insert_mode:
            raise SkipAction()
        inp = self._chat_input
        if not inp.has_focus:
            raise SkipAction()
        # When the completion dropdown is up, Up navigates the dropdown
        # (vim/Emacs-style) rather than recalling history.
        overlay = self._completion_overlay
        if overlay.is_visible:
            self._preview_prev()
            return
        if not self._input_history:
            raise SkipAction()
        if self._history_index == -1:
            self._history_index = len(self._input_history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return
        inp.value = self._input_history[self._history_index]
        inp.action_end()

    def action_history_next(self) -> None:
        """Down arrow: cycle the dropdown if visible, else recall next history entry."""
        if not self._insert_mode:
            raise SkipAction()
        inp = self._chat_input
        if not inp.has_focus:
            raise SkipAction()
        # When the completion dropdown is up, Down navigates the dropdown.
        overlay = self._completion_overlay
        if overlay.is_visible:
            self._preview_next()
            return
        if self._history_index == -1:
            raise SkipAction()
        if self._history_index < len(self._input_history) - 1:
            self._history_index += 1
            inp.value = self._input_history[self._history_index]
            inp.action_end()
        else:
            self._history_index = -1
            inp.value = ""

    @on(ChatInput.Changed, "#chat-input")
    def _on_chat_input_changed(self, event: ChatInput.Changed) -> None:
        """Refresh arg-hint and auto-show or hide the completion dropdown."""
        if self._suppress_refresh > 0:
            # A programmatic edit (preview / accept / revert) is managing the
            # overlay itself; consume one Changed and skip the live refresh.
            self._suppress_refresh -= 1
            self._refresh_arg_hint()
            return
        self._refresh_completion_overlay()
        self._refresh_arg_hint()

    def _refresh_completion_overlay(self) -> None:
        """Live-filter the dropdown against the current input, in command and arg modes alike."""
        overlay = self._completion_overlay
        text = self._chat_input.value
        options = get_completions(text)
        if options:
            self._completion_origin = text
            overlay.show_completions(options)
        elif overlay.is_visible:
            overlay.hide()
            self._completion_origin = None

    def _refresh_arg_hint(self) -> None:
        """Push the current input value into the ArgHintLine."""
        self._arg_hint.update_for_input(self._chat_input.value)

    def refresh_model_bar(self) -> None:
        """Re-scan installed models and refresh the model bar."""
        self.query_one("#model-bar", ModelBar).refresh_models()

    def action_vim_scroll_down(self) -> None:
        """Vim j: scroll down in normal mode."""
        if self._insert_mode:
            raise SkipAction()
        self._chat_log.scroll_down()

    def action_vim_scroll_up(self) -> None:
        """Vim k: scroll up in normal mode."""
        if self._insert_mode:
            raise SkipAction()
        self._chat_log.scroll_up()

    def action_vim_scroll_home(self) -> None:
        """Vim g: scroll to top in normal mode."""
        if self._insert_mode:
            raise SkipAction()
        self._chat_log.scroll_home()

    def action_vim_scroll_end(self) -> None:
        """Vim G: scroll to bottom in normal mode."""
        if self._insert_mode:
            raise SkipAction()
        self._chat_log.scroll_end()

    def action_half_page_down(self) -> None:
        """Ctrl-D: half-page down (vim style)."""
        log_widget = self._chat_log
        half = max(1, log_widget.size.height // 2)
        log_widget.scroll_relative(y=half)

    def action_half_page_up(self) -> None:
        """Ctrl-U: half-page up (vim style)."""
        log_widget = self._chat_log
        half = max(1, log_widget.size.height // 2)
        log_widget.scroll_relative(y=-half)
