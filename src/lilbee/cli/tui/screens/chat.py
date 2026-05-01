"""Chat screen: scrollable message log with streaming markdown responses."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual import getters, on, work
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Select, Static

# Cancellation check for @work(thread=True) workers. Import at module level
# since it's used in multiple methods.
from textual.worker import get_current_worker as _get_worker

from lilbee.app.version import get_version
from lilbee.cli.settings_map import SETTINGS_MAP
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import DARK_THEMES, LilbeeApp, apply_active_model
from lilbee.cli.tui.command_registry import build_dispatch_dict
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay, get_completions
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.cli.tui.widgets.chat_stop_button import ChatStopButton
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.cli.tui.widgets.model_bar import ChatModeToggle, ModelBar, ModelPickerButton
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.cli.tui.widgets.task_bar import ProgressReporter, TaskBar
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from lilbee.core.services import get_services, reset_services
from lilbee.crawler import crawler_available, is_url, require_valid_crawl_url
from lilbee.data.store import scope_to_chunk_type
from lilbee.providers.base import ClosableIterator
from lilbee.providers.model_ref import parse_model_ref
from lilbee.retrieval.embedder import is_model_available
from lilbee.retrieval.query import ChatMessage
from lilbee.runtime import asyncio_loop
from lilbee.runtime.progress import EventType, ProgressEvent

if TYPE_CHECKING:
    from lilbee.cli.tui.widgets.task_bar import TaskBarController

log = logging.getLogger(__name__)

_DISPATCH = build_dispatch_dict()

_MAX_HISTORY_MESSAGES = 200

# Treat the user as "still at the bottom" when within this many lines so a tiny
# stray scroll doesn't disable auto-follow during streaming.
_AUTO_SCROLL_TAIL_LINES = 5

# Coalesce per-token UI updates into ~50 ms windows. Tiny reasoning models can
# emit 100+ tokens/sec; one ``call_from_thread`` per token saturates Textual's
# message queue and makes key events visibly lag.
_STREAM_FLUSH_INTERVAL = 0.05

# Auto-scroll throttle. ~6 fps so heavy token streams don't peg the renderer.
_STREAM_SCROLL_INTERVAL = 0.15


def _close_stream(stream: Any) -> None:
    """Close a streaming iterator if it satisfies the ClosableIterator protocol."""
    if isinstance(stream, ClosableIterator):
        with contextlib.suppress(Exception):
            stream.close()


def _remove_copied_files(names: list[str]) -> None:
    """Delete files previously copied into documents/ by a /add invocation.

    Called on cancel or failure of the add task so a cancelled file does not
    re-appear on the next sync. Silently tolerates missing entries;
    the user may have removed them concurrently, and the goal is just to
    prevent accidental indexing.
    """
    for name in names:
        target = cfg.documents_dir / name
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        except OSError:
            log.debug("Could not remove copied file %s", target, exc_info=True)


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

    CSS_PATH = "chat.tcss"
    AUTO_FOCUS = "#chat-input"

    streaming: reactive[bool] = reactive(False)

    HELP = (
        "# Chat\n\n"
        "Ask questions about your knowledge base.\n\n"
        "Press **Escape** for normal mode (vim keys), "
        "**i**/**a**/**o** to return to insert mode."
    )

    _SCROLL_GROUP = Binding.Group("Scroll", compact=True)

    # Hot-path widget refs. ``getters.query_one`` is a typed class-level
    # descriptor that resolves via Textual's indexed DOM lookup on every
    # access. It is O(1) for id selectors, so no cache is needed.
    _chat_input = getters.query_one("#chat-input", ChatInput)
    _chat_log = getters.query_one("#chat-log", VerticalScroll)
    _completion_overlay = getters.query_one("#completion-overlay", CompletionOverlay)

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("slash", "focus_commands", "Commands", show=True),
        Binding("tab", "complete", "Tab models / complete", show=True, priority=True),
        Binding("ctrl+n", "complete_next", "^n next", show=False),
        Binding("ctrl+p", "complete_prev", "^p prev", show=False),
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
        Binding("escape", "enter_normal_mode", "Normal mode", show=True, priority=True),
        Binding("ctrl+r", "toggle_markdown", "Markdown", show=False),
        Binding("m", "focus_model_bar", "Models", show=True),
        Binding("f3", "toggle_chat_mode", "Search/Chat", show=False),
        Binding("f5", "open_setup", "Setup", show=False),
    ]

    def __init__(self, *, auto_sync: bool = False) -> None:
        super().__init__()
        self._auto_sync = auto_sync
        self._history: list[ChatMessage] = []
        self._history_lock = threading.Lock()
        self._insert_mode: bool = True
        self._completing = False
        self._sync_active: bool = False
        self._input_history: list[str] = []
        self._history_index: int = -1

    @property
    def _task_bar(self) -> TaskBarController:
        """The app-level TaskBarController (always set by LilbeeApp)."""
        return self.app.task_bar  # type: ignore[attr-defined,no-any-return]

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
            yield Static(msg.CHAT_ONLY_BANNER, id="chat-only-banner")
        yield VerticalScroll(
            ChatWelcome(id="chat-welcome"),
            id="chat-log",
        )
        yield CompletionOverlay(id="completion-overlay")
        with BottomBars():
            with PromptArea(id="chat-prompt-area"):
                yield ChatInput(
                    placeholder=msg.CHAT_INPUT_PLACEHOLDER_DEFAULT,
                    id="chat-input",
                )
                yield ModelBar(id="model-bar")
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        self._update_input_style()
        self._refresh_mode_banner()
        if self._needs_setup():
            from lilbee.cli.tui.screens.setup import SetupWizard

            self.app.push_screen(SetupWizard(), self._on_setup_complete)
        elif self._auto_sync and self._embedding_ready():
            self._run_sync()
        if isinstance(self.app, LilbeeApp):
            self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)

    def on_show(self) -> None:
        """Called when screen becomes visible."""
        from lilbee.runtime.splash import dismiss

        dismiss()
        self.refresh_model_bar()
        # AUTO_FOCUS only fires once on initial mount. Re-entering the screen
        # via [/] navigation needs an explicit focus restore.
        with contextlib.suppress(Exception):
            self.set_focus(self._chat_input)

    def _needs_setup(self) -> bool:
        """True when the setup wizard should run: fresh data dir or unresolved models.

        Remote-prefixed refs skip the native probe since they resolve
        through the SDK backend at call time.
        """
        if not cfg.lancedb_dir.is_dir():
            log.debug("_needs_setup: lancedb_dir missing (%s)", cfg.lancedb_dir)
            return True
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp.provider import resolve_model_path

        for label, model in (("chat", cfg.chat_model), ("embedding", cfg.embedding_model)):
            if parse_model_ref(model).is_remote:
                continue
            try:
                resolve_model_path(model)
            except (ProviderError, KeyError, ValueError) as exc:
                log.debug("_needs_setup: %s model %r unresolved: %s", label, model, exc)
                return True
        return False

    def _embedding_ready(self) -> bool:
        """Quick check if the embedding model resolves (no network calls)."""
        return is_model_available(cfg.embedding_model, get_services().provider)

    def _on_setup_complete(self, result: str | None) -> None:
        """Called when wizard completes or is skipped."""
        if self._embedding_ready() and self._auto_sync:
            self._run_sync()
        self._refresh_mode_banner()
        self.refresh_model_bar()

    def _refresh_mode_banner(self) -> None:
        """Show the right banner for the current embedding+mode state."""
        try:
            banner = self.query_one("#chat-only-banner", Static)
        except NoMatches:
            return
        if not self._embedding_ready():
            banner.update(msg.CHAT_ONLY_BANNER)
            banner.display = True
        elif cfg.chat_mode == ChatMode.CHAT.value:
            banner.update(msg.CHAT_MODE_BANNER_CHAT)
            banner.display = True
        else:
            banner.display = False

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        key, _value = payload
        if key in {"chat_mode", "embedding_model"}:
            self._refresh_mode_banner()

    def action_open_setup(self) -> None:
        """Open the setup wizard."""
        self._cmd_setup("")

    def _enter_insert_mode(self) -> None:
        """Switch to insert mode: focus input, update border style."""
        self._insert_mode = True
        self._chat_input.focus()
        self._update_input_style()

    def _update_input_style(self) -> None:
        """Toggle input opacity and mode indicator based on current mode."""
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
            # Let a focused Select / picker button handle Enter / i / a / o itself.
            if isinstance(self.focused, (Select, ModelPickerButton)):
                return
            self._enter_insert_mode()
            event.prevent_default()
            event.stop()
            return

    @on(ChatInput.Submitted, "#chat-input")
    def _on_chat_submitted(self, event: ChatInput.Submitted) -> None:
        if not self._insert_mode:
            # Vim-style: Enter in normal mode flips back to insert without
            # submitting whatever empty / stale text the input still holds.
            self._enter_insert_mode()
            return
        text = event.value.strip()
        if not text:
            return
        event.chat_input.value = ""
        self._input_history.append(text)
        self._history_index = -1

        if text.startswith("/"):
            self._handle_slash(text)
            return
        if self.streaming:
            return
        self._send_message(text)

    def _handle_slash(self, text: str) -> None:
        """Dispatch slash commands via the command registry."""
        cmd = text.split()[0].lower()
        args = text[len(cmd) :].strip()
        handler_name = _DISPATCH.get(cmd)
        if handler_name:
            getattr(self, handler_name)(args)
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
        self._chat_input.placeholder = msg.CHAT_INPUT_PLACEHOLDER_STREAMING
        # Both cancel paths and the natural finalize write streaming=False;
        # reactive dedupe makes the watcher a no-op on equal values, so the
        # button mount happens exactly once per streaming session.
        with contextlib.suppress(Exception):
            self.query_one("#chat-prompt-area", PromptArea).mount(
                ChatStopButton(button_id="chat-stop")
            )

    def _exit_streaming_state(self) -> None:
        self.remove_class("streaming")
        self._chat_input.placeholder = msg.CHAT_INPUT_PLACEHOLDER_DEFAULT
        for w in self.query("#chat-stop"):
            w.remove()

    @on(ChatStopButton.Pressed)
    def _on_chat_stop_pressed(self, event: ChatStopButton.Pressed) -> None:
        event.stop()
        self.action_cancel_stream()

    def _cmd_add(self, args: str) -> None:
        if not args:
            return
        if self._sync_active:
            self.notify(msg.SYNC_ALREADY_ACTIVE, severity="warning")
            return
        if is_url(args):
            self._cmd_crawl(args)
            return
        path = Path(args).expanduser()
        if not path.exists():
            self.notify(msg.CMD_ADD_NOT_FOUND.format(path=path), severity="error")
            return
        # Directory adds are whole-tree copies handled by copy_files'
        # recursion; a same-named subdir in documents_dir is not a clean
        # "duplicate file" signal, so skip the prompt there and let
        # copy_files emit its per-file skipped notices.
        dest = cfg.documents_dir / path.name
        if path.is_file() and dest.exists():
            self._prompt_overwrite(path)
            return
        self._submit_add(path, force=False)

    def _prompt_overwrite(self, path: Path) -> None:
        """Ask to overwrite an existing copy before re-syncing."""
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                self.notify(msg.CMD_ADD_SKIPPED_DUPLICATE.format(name=path.name))
                return
            self._submit_add(path, force=True)

        self.app.push_screen(
            ConfirmDialog(
                msg.CMD_ADD_DUPLICATE_TITLE,
                msg.CMD_ADD_DUPLICATE_MESSAGE.format(name=path.name),
            ),
            _on_confirm,
        )

    def _submit_add(self, path: Path, *, force: bool) -> None:
        """Spawn the add worker. Separated so overwrite confirm can reuse it."""
        from lilbee.cli.tui.task_queue import TaskType

        self._sync_active = True

        def _target(reporter: ProgressReporter) -> None:
            try:
                self._do_add(path, reporter, force=force)
            finally:
                self._sync_active = False

        self._task_bar.start_task(f"Add {path.name}", TaskType.ADD, _target, indeterminate=True)

    def _do_add(self, path: Path, reporter: ProgressReporter, *, force: bool = False) -> None:
        """Copy files and run sync. Called on worker thread with a reporter."""
        from lilbee.app.ingest import copy_files
        from lilbee.data.ingest import sync
        from lilbee.runtime.progress import FileStartEvent

        reporter.update(0, f"Copying {path.name}...", indeterminate=True)
        copy_result = copy_files([path], force=force)
        copied = copy_result.copied
        for name in copy_result.skipped:
            call_from_thread(self, self.notify, f"{name} already exists (use --force to overwrite)")
        reporter.update(0, f"Copied {len(copied)} file(s), syncing...", indeterminate=True)

        def on_progress(event_type: EventType, data: ProgressEvent) -> None:
            # Polling point so /c in Task Center can stop a long ingest
            # between file boundaries without having to kill the thread.
            reporter.check_cancelled()
            if event_type == EventType.FILE_START and isinstance(data, FileStartEvent):
                reporter.update(0, f"Syncing {data.file}...", indeterminate=True)

        try:
            sync_result = asyncio_loop.run(sync(quiet=True, on_progress=on_progress))
        except BaseException:
            # On cancel or any failure, remove the files we copied into
            # documents/ so the next sync doesn't silently re-ingest the
            # file the user just cancelled. Only files copied by
            # this /add invocation are removed; pre-existing files the user
            # put in documents/ themselves are never touched.
            _remove_copied_files(copied)
            raise
        if sync_result.failed:
            _remove_copied_files(copied)
            raise RuntimeError(msg.SYNC_FAILED_FILES.format(files=", ".join(sync_result.failed)))
        call_from_thread(self, self.notify, msg.CMD_ADD_SUCCESS.format(count=len(copied)))

    def _cmd_cancel(self, _args: str) -> None:
        for worker in self.workers:
            worker.cancel()
        self.notify(msg.CMD_CANCEL)

    def _cmd_clear(self, _args: str) -> None:
        for worker in self.workers:
            worker.cancel()
        self.streaming = False
        chat_log = self._chat_log
        chat_log.remove_children()
        with self._history_lock:
            self._history.clear()
        self.notify(msg.CMD_CLEAR)

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
        depth, max_pages, include_subdomains = self._parse_crawl_flags(parts[1:])
        self._start_crawl(url, depth, max_pages, include_subdomains=include_subdomains)

    def _open_crawl_dialog(self) -> None:
        """Push the crawl modal and handle its result."""
        from lilbee.cli.tui.widgets.crawl_dialog import CrawlDialog, CrawlParams

        def _on_result(result: CrawlParams | None) -> None:
            if result is not None:
                self._start_crawl(result.url, result.depth, result.max_pages)

        self.app.push_screen(CrawlDialog(), callback=_on_result)

    def _start_crawl(
        self,
        url: str,
        depth: int | None,
        max_pages: int | None,
        *,
        include_subdomains: bool = False,
    ) -> None:
        """Enqueue a crawl task and run it in the background.

        Bootstrap Chromium first via the controller helper. If the
        browser isn't installed yet, a SETUP task renders in the Task
        Center and the crawl kicks off from its on_success hook. On a
        machine where Chromium is already present this is a synchronous
        no-op and the crawl starts immediately (bb-wq8g).
        """
        from lilbee.cli.tui.task_queue import TaskType

        def _kick_off_crawl() -> None:
            self._task_bar.start_task(
                msg.TASK_NAME_CRAWL.format(url=url),
                TaskType.CRAWL,
                lambda reporter: self._do_crawl(
                    url, depth, max_pages, reporter, include_subdomains=include_subdomains
                ),
                on_success=lambda: call_from_thread(self, self._run_sync),
            )

        self.notify(msg.CMD_CRAWL_STARTED.format(url=url))
        self._task_bar.ensure_chromium(_kick_off_crawl)

    @staticmethod
    def _parse_crawl_flags(tokens: list[str]) -> tuple[int | None, int | None, bool]:
        """Extract --depth, --max-pages, and --include-subdomains from tokens.

        Numeric flags return None when absent so the caller inherits
        crawl_and_save's unbounded-by-default semantics. The boolean
        ``--include-subdomains`` flag defaults to False (exact-host scope).
        """
        flag_map = {"--depth": "depth", "--max-pages": "max_pages"}
        parsed: dict[str, int | None] = {"depth": None, "max_pages": None}
        include_subdomains = False
        i = 0
        while i < len(tokens):
            if tokens[i] == "--include-subdomains":
                include_subdomains = True
                i += 1
                continue
            key = flag_map.get(tokens[i])
            if key and i + 1 < len(tokens):
                with contextlib.suppress(ValueError):
                    parsed[key] = int(tokens[i + 1])
                i += 2
            else:
                i += 1
        return parsed["depth"], parsed["max_pages"], include_subdomains

    def _do_crawl(
        self,
        url: str,
        depth: int | None,
        max_pages: int | None,
        reporter: ProgressReporter,
        *,
        include_subdomains: bool = False,
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
            )
        )
        call_from_thread(self, self.notify, msg.CMD_CRAWL_SUCCESS.format(count=len(paths), url=url))

    def _cmd_catalog(self, _args: str) -> None:
        if isinstance(self.app, LilbeeApp):
            self.app.switch_view("Catalog")
            return
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        self.app.push_screen(CatalogScreen())

    def _cmd_delete(self, args: str) -> None:
        try:
            sources = get_services().store.get_sources()
        except Exception:
            log.debug("Failed to list documents for /delete", exc_info=True)
            self.notify(msg.CMD_DELETE_NO_DOCS, severity="warning")
            return

        known = {s.get("filename", s.get("source", "?")) for s in sources}
        if not known:
            self.notify(msg.CMD_DELETE_NO_DOCS, severity="warning")
            return

        name = args.strip()
        if not name:
            self.notify(msg.CMD_DELETE_USAGE.format(names=", ".join(sorted(known))))
            return

        if name not in known:
            self.notify(msg.CMD_DELETE_NOT_FOUND.format(name=name), severity="error")
            return

        store = get_services().store
        store.delete_by_source(name)
        store.delete_source(name)
        self.notify(msg.CMD_DELETE_SUCCESS.format(name=name))

    def _cmd_help(self, _args: str) -> None:
        self.app.action_show_help_panel()

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
            apply_active_model(self.app, "chat_model", args)
            self.app.title = f"lilbee -- {cfg.chat_model}"
            self.notify(msg.CMD_MODEL_SET.format(name=cfg.chat_model))
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

    def _cmd_reset(self, args: str) -> None:
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            from lilbee.app.reset import perform_reset

            try:
                result = perform_reset()
                if result.skipped:
                    self.notify(
                        msg.CMD_RESET_PARTIAL.format(skipped=len(result.skipped)),
                        severity="warning",
                    )
                else:
                    self.notify(msg.CMD_RESET_SUCCESS)
            except Exception as exc:
                log.warning("Reset failed", exc_info=True)
                self.notify(msg.CMD_RESET_FAILED.format(error=exc), severity="error")

        self.app.push_screen(
            ConfirmDialog("Reset Knowledge Base", "This will permanently delete all data."),
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
                parsed = defn.type(value)
            setattr(cfg, key, parsed)
            persisted = str(parsed) if parsed is not None else ""
            settings.set_value(cfg.data_root, key, persisted)
            if key == "llm_provider":  # pragma: no cover
                reset_services()
            self.notify(msg.CMD_SET_SUCCESS.format(key=key, value=parsed))
        except (ValueError, TypeError) as exc:
            self.notify(msg.CMD_SET_INVALID.format(key=key, error=exc), severity="error")

    def _cmd_settings(self, _args: str) -> None:
        if isinstance(self.app, LilbeeApp):
            self.app.switch_view("Settings")
            return
        from lilbee.cli.tui.screens.settings import SettingsScreen

        self.app.push_screen(SettingsScreen())

    def _cmd_setup(self, _args: str) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        self.app.push_screen(SetupWizard(), self._on_setup_complete)

    def _cmd_status(self, _args: str) -> None:
        if isinstance(self.app, LilbeeApp):
            self.app.switch_view("Status")
            return
        from lilbee.cli.tui.screens.status import StatusScreen

        self.app.push_screen(StatusScreen())

    def _cmd_theme(self, args: str) -> None:
        if args and isinstance(self.app, LilbeeApp):
            self.app.set_theme(args)
            self.notify(msg.THEME_SET.format(name=args))
        else:
            theme_list = msg.CMD_THEME_LIST.format(names=", ".join(DARK_THEMES))
            self.notify(theme_list, severity="information")

    def _cmd_version(self, _args: str) -> None:
        self.notify(msg.CHAT_VERSION.format(version=get_version()))

    def _cmd_wiki(self, _args: str) -> None:
        if not cfg.wiki:
            self.notify(msg.CMD_WIKI_DISABLED, severity="warning")
            return
        if isinstance(self.app, LilbeeApp):  # test apps aren't LilbeeApp
            self.app.switch_view("Wiki")

    def _send_message(self, text: str) -> None:
        """Send a user message and stream the response."""
        from textual.css.query import NoMatches

        log = self._chat_log
        with contextlib.suppress(NoMatches):
            log.query_one("#chat-welcome", ChatWelcome).remove()
        log.mount(UserMessage(text))

        assistant_msg = AssistantMessage()
        log.mount(assistant_msg)
        log.scroll_end(animate=False)

        with self._history_lock:
            self._history.append({"role": "user", "content": text})
        self.streaming = True
        self._stream_response(text, assistant_msg, self._current_chunk_type())

    def _current_chunk_type(self) -> str | None:
        """Translate the ModelBar scope selection into a ``chunk_type`` arg.

        Returns ``None`` for "both" (no filter) and the raw/wiki string
        otherwise. Defaults to ``None`` when the ModelBar isn't mounted
        (e.g. test apps).
        """
        from textual.css.query import NoMatches

        try:
            bar = self.query_one("#model-bar", ModelBar)
        except NoMatches:
            return None
        return scope_to_chunk_type(bar.scope)

    @work(thread=True)
    def _stream_response(
        self, question: str, widget: AssistantMessage, chunk_type: str | None
    ) -> None:
        """Stream LLM response in a background thread, coalescing UI updates."""
        response_parts: list[str] = []
        sources: list[str] = []
        stream: Any = None
        try:
            with self._history_lock:
                history_snapshot = self._history[:-1]
            stream = get_services().searcher.ask_stream(
                question, history=history_snapshot, chunk_type=chunk_type
            )
            self._consume_stream(stream, widget, response_parts)
        except Exception as exc:
            log.debug("Stream error", exc_info=True)
            with contextlib.suppress(Exception):
                call_from_thread(self, widget.append_content, msg.STREAM_ERROR.format(error=exc))
        finally:
            _close_stream(stream)
            self._finalize_stream(widget, sources, response_parts)

    def _consume_stream(
        self, stream: Any, widget: AssistantMessage, response_parts: list[str]
    ) -> None:
        """Pull tokens off *stream*, batching UI updates to ~50 ms windows."""
        worker = _get_worker()
        reason_buf: list[str] = []
        content_buf: list[str] = []
        timings = [time.monotonic(), 0.0]  # [last_flush, last_scroll]

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
                self._maybe_flush_and_scroll(flush, timings)
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

    def _maybe_flush_and_scroll(self, flush: Callable[[], None], timings: list[float]) -> None:
        """Run *flush* and the auto-scroll on their respective intervals."""
        now = time.monotonic()
        if now - timings[0] >= _STREAM_FLUSH_INTERVAL:
            flush()
            timings[0] = now
        if now - timings[1] >= _STREAM_SCROLL_INTERVAL:
            call_from_thread(self, self._scroll_to_bottom)
            timings[1] = now

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
                self._trim_history()
        call_from_thread(self, widget.finish, sources)
        call_from_thread(self, self._scroll_to_bottom)
        if (
            cfg.chat_mode == ChatMode.SEARCH.value
            and self._embedding_ready()
            and full_response
            and "\n\nSources:\n" not in full_response
        ):
            call_from_thread(self, self._notify_no_results)

    def _notify_no_results(self) -> None:
        self.notify(msg.CHAT_MODE_BANNER_SEARCH_NO_RESULTS, severity="warning")

    def _trim_history(self) -> None:
        """Trim history to max size, dropping oldest messages. Caller must hold _history_lock."""
        if len(self._history) > _MAX_HISTORY_MESSAGES:
            self._history[:] = self._history[-_MAX_HISTORY_MESSAGES:]

    def _scroll_to_bottom(self) -> None:
        log_widget = self._chat_log
        # Only auto-scroll while the user is still tailing the output.
        # If they scrolled up to read, don't yank them back.
        if log_widget.max_scroll_y - log_widget.scroll_y < _AUTO_SCROLL_TAIL_LINES:
            log_widget.scroll_end(animate=False)

    def action_scroll_up(self) -> None:
        self._chat_log.scroll_page_up()

    def action_scroll_down(self) -> None:
        self._chat_log.scroll_page_down()

    def action_enter_normal_mode(self) -> None:
        """Escape: cancel stream, return from model bar, or enter normal mode."""
        if self.streaming:
            for worker in self.workers:
                worker.cancel()
            self.streaming = False
            return
        if isinstance(self.focused, (Select, ModelPickerButton)):
            self._chat_input.focus()
            return
        self._insert_mode = False
        self._chat_log.focus()
        self._update_input_style()

    def action_cancel_stream(self) -> None:
        """Context-aware Escape: cancel stream -> blur input -> no-op."""
        if self.streaming:
            for worker in self.workers:
                worker.cancel()
            self.streaming = False
            return
        inp = self._chat_input
        if inp.has_focus:
            self._chat_log.focus()

    def apply_model_change(self) -> None:
        """Cancel active stream (if any) and reset services for the new model."""
        if self.streaming:
            self.action_cancel_stream()
            self.call_later(self._deferred_service_reset)
        else:
            reset_services()

    def _deferred_service_reset(self) -> None:
        """Reset services once workers have drained."""
        if self.workers:
            self.call_later(self._deferred_service_reset)
            return
        reset_services()

    async def action_toggle_markdown(self) -> None:
        """Toggle between Markdown and plain-text rendering for chat responses."""
        cfg.markdown_rendering = not cfg.markdown_rendering
        use_md = cfg.markdown_rendering
        chat_log = self._chat_log
        for widget in chat_log.query(AssistantMessage):
            await widget.rebuild_content_widget(use_md)
        label = "Markdown" if use_md else "Plain text"
        self.notify(msg.CHAT_RENDERING.format(label=label))

    def _run_sync(self) -> None:
        """Enqueue a document sync in the task bar."""
        if self._sync_active:
            self.notify(msg.SYNC_ALREADY_ACTIVE, severity="warning")
            return
        from lilbee.cli.tui.task_queue import TaskType

        self._sync_active = True

        def _target(reporter: ProgressReporter) -> None:
            try:
                self._do_sync(reporter)
            finally:
                self._sync_active = False

        self._task_bar.start_task("Sync documents", TaskType.SYNC, _target, indeterminate=True)

    def _do_sync(self, reporter: ProgressReporter) -> None:
        """Sync body. Runs on worker thread."""
        from lilbee.data.ingest import sync
        from lilbee.runtime.progress import EmbedEvent, FileDoneEvent, FileStartEvent, SyncDoneEvent

        reporter.update(0, msg.SYNC_STATUS_SYNCING, indeterminate=True)

        last_embed_update = 0.0
        _throttle_seconds = 0.15

        def on_progress(event_type: EventType, data: ProgressEvent) -> None:
            nonlocal last_embed_update
            if event_type == EventType.FILE_START and isinstance(data, FileStartEvent):
                pct = int((data.current_file - 1) * 100 / data.total_files)
                status = msg.SYNC_FILE_PROGRESS.format(
                    current=data.current_file, total=data.total_files, file=data.file
                )
                reporter.update(pct, status, indeterminate=False)
            elif event_type == EventType.FILE_DONE and isinstance(data, FileDoneEvent):
                reporter.update(0, msg.SYNC_FILE_DONE.format(file=data.file), indeterminate=False)
            elif event_type == EventType.EMBED and isinstance(data, EmbedEvent):
                now = time.monotonic()
                if now - last_embed_update < _throttle_seconds:
                    return
                last_embed_update = now
                pct = int(data.chunk * 100 / data.total_chunks) if data.total_chunks else 0
                reporter.update(pct, msg.SYNC_EMBEDDING.format(file=data.file), indeterminate=False)
            elif event_type == EventType.DONE and isinstance(data, SyncDoneEvent):
                # Without this handler the task never ticks to 100% and the
                # Task Center row never flashes "just-completed" (bb-7enj).
                # "Synced (N docs)" means successfully synced, so failed is excluded.
                total = data.added + data.updated + data.removed
                reporter.update(100, msg.SYNC_STATUS_DONE.format(count=total), indeterminate=False)

        try:
            result = asyncio_loop.run(sync(quiet=True, on_progress=on_progress))
        except asyncio.CancelledError as exc:
            self._auto_sync = False
            raise RuntimeError("Sync cancelled. Use /sync to resume.") from exc
        if result.failed:
            raise RuntimeError(msg.SYNC_FAILED_FILES.format(files=", ".join(result.failed)))

    def action_focus_commands(self) -> None:
        """Focus chat input and pre-fill with '/' for command entry."""
        inp = self._chat_input
        inp.focus()
        if not inp.value.startswith("/"):
            inp.value = "/"
            inp.action_end()

    def action_focus_model_bar(self) -> None:
        """Focus the chat-model picker button in the model bar (normal mode only)."""
        if self._insert_mode:
            raise SkipAction()
        with contextlib.suppress(NoMatches):
            self.query_one("#chat-model-button", ModelPickerButton).focus()

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

    def action_complete(self) -> None:
        """Tab: cycle autocomplete, insert a literal tab, or advance focus.

        - Insert mode + chat input focused + completion overlay open:
          cycle the next completion candidate.
        - Insert mode + chat input focused + no completion: insert
          ``\\t`` so users can type tab characters directly.
        - Normal mode or focus elsewhere: advance through the focus
          chain so Tab still walks every focusable widget.
        """
        inp = self._chat_input
        if not self._insert_mode or not inp.has_focus:
            self.screen.focus_next()
            return
        if self._cycle_completion_forward(inp):
            return
        inp.insert("\t")

    def action_complete_next(self) -> None:
        """Ctrl+N: show completions or cycle forward."""
        inp = self._chat_input
        if not inp.has_focus:
            return
        self._cycle_completion_forward(inp)

    def _cycle_completion_forward(self, inp: ChatInput) -> bool:
        """Show or cycle forward through autocomplete; returns True if it acted."""
        overlay = self._completion_overlay

        if overlay.is_visible:
            selection = overlay.cycle_next()
            if selection:
                cmd_prefix = inp.value.split()[0] + " " if " " in inp.value else ""
                self._completing = True
                inp.value = cmd_prefix + selection
                self._completing = False
                inp.action_end()
            return True

        options = get_completions(inp.value)
        if options:
            overlay.show_completions(options)
            first = overlay.get_current()
            self._completing = True
            if first and " " in inp.value:
                cmd_prefix = inp.value.split()[0] + " "
                inp.value = cmd_prefix + first
                inp.action_end()
            elif first:
                inp.value = first
                inp.action_end()
            self._completing = False
            return True

        return False

    def action_complete_prev(self) -> None:
        """Ctrl+P: cycle backward through completions."""
        overlay = self._completion_overlay
        inp = self._chat_input

        if overlay.is_visible:
            selection = overlay.cycle_prev()
            if selection:
                cmd_prefix = inp.value.split()[0] + " " if " " in inp.value else ""
                self._completing = True
                inp.value = cmd_prefix + selection
                self._completing = False
                inp.action_end()
            return

        options = get_completions(inp.value)
        if options:
            overlay.show_completions(options)
            last = overlay.get_current()
            self._completing = True
            if last and " " in inp.value:
                cmd_prefix = inp.value.split()[0] + " "
                inp.value = cmd_prefix + last
                inp.action_end()
            elif last:
                inp.value = last
                inp.action_end()
            self._completing = False

    def action_history_prev(self) -> None:
        """Up arrow: recall previous input history entry."""
        if not self._insert_mode:
            raise SkipAction()
        inp = self._chat_input
        if not inp.has_focus or not self._input_history:
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
        """Down arrow: recall next input history entry."""
        if not self._insert_mode:
            raise SkipAction()
        inp = self._chat_input
        if not inp.has_focus or self._history_index == -1:
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
        """Hide completion overlay when input changes manually."""
        if self._completing:
            return
        overlay = self._completion_overlay
        if overlay.is_visible:
            overlay.hide()

    def refresh_model_bar(self) -> None:
        """Re-scan installed models and refresh the dropdowns."""
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
