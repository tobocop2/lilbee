"""Chat screen — scrollable message log with streaming markdown responses."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import TYPE_CHECKING, ClassVar

from textual import on, work
from textual.actions import SkipAction
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.reactive import var
from textual.screen import Screen
from textual.widgets import Footer, Input, Label, Select, Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.chat_commands import SlashCommandsMixin
from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay, get_completions
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.config import cfg
from lilbee.progress import EventType, ProgressEvent
from lilbee.query import ChatMessage

if TYPE_CHECKING:
    from lilbee.cli.tui.widgets.task_bar import TaskBar

log = logging.getLogger(__name__)

_MAX_HISTORY_MESSAGES = 200


class ChatStatusLine(Label):
    """One-line status bar showing the current model as a pill badge."""

    model_name: var[str] = var("")

    def watch_model_name(self, name: str) -> None:
        """Re-render when model name changes."""
        if name:
            self.update(pill(name, "$primary", "$text"))
        else:
            self.update("")


class PromptArea(Vertical):
    """Container for chat input that highlights on focus-within."""

    pass


class ChatScreen(SlashCommandsMixin, Screen[None]):
    """Primary chat interface with streaming LLM responses."""

    CSS_PATH = "chat.tcss"
    AUTO_FOCUS = "#chat-input"

    HELP = (
        "# Chat\n\n"
        "Ask questions about your knowledge base.\n\n"
        "Press **Escape** for normal mode (vim keys), "
        "**i**/**a**/**o** to return to insert mode."
    )

    _SCROLL_GROUP = Binding.Group("Scroll", compact=True)

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("slash", "focus_commands", "Commands", show=True),
        Binding("tab", "complete", "Tab", show=False, priority=True),
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
        Binding("up", "history_prev", "Up", show=False),
        Binding("down", "history_next", "Down", show=False),
        Binding("escape", "enter_normal_mode", "Normal mode", show=True, priority=True),
        Binding("ctrl+r", "toggle_markdown", "Markdown", show=False),
        Binding("m", "focus_model_bar", "Models", show=True),
        Binding("f5", "open_setup", "Setup", show=False),
    ]

    def __init__(self, *, auto_sync: bool = False) -> None:
        super().__init__()
        self._auto_sync = auto_sync
        self._history: list[ChatMessage] = []
        self._history_lock = threading.Lock()
        self.streaming = False
        self._insert_mode: bool = True
        self._completing = False
        self._sync_active: bool = False
        self._input_history: list[str] = []
        self._history_index: int = -1

    @property
    def _task_bar(self) -> TaskBar:
        """The app-level TaskBar (created by LilbeeApp)."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar as _TaskBar

        bar = getattr(self.app, "task_bar", None)  # test apps lack task_bar
        if isinstance(bar, _TaskBar):
            return bar
        msg_text = "App does not have a TaskBar"
        raise RuntimeError(msg_text)

    def compose(self) -> ComposeResult:
        yield ModelBar(id="model-bar")
        yield Static(msg.CHAT_ONLY_BANNER, id="chat-only-banner")
        yield VerticalScroll(id="chat-log")
        yield CompletionOverlay(id="completion-overlay")
        yield ChatStatusLine(id="chat-status-line")
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        with PromptArea(id="chat-prompt-area"):
            yield Input(
                placeholder=msg.CHAT_INPUT_PLACEHOLDER,
                id="chat-input",
                suggester=SlashSuggester(use_cache=False),
            )
        yield ViewTabs()
        yield Footer()

    def on_mount(self) -> None:
        self._update_input_style()
        self._refresh_status_line()
        self.query_one("#chat-only-banner", Static).display = False
        if self._needs_setup():
            from lilbee.cli.tui.screens.setup import SetupWizard

            self.app.push_screen(SetupWizard(), self._on_setup_complete)
        elif not self._embedding_ready():
            self._show_chat_only_banner()
        elif self._auto_sync:
            self._run_sync()

    def on_show(self) -> None:
        """Called when screen becomes visible - signal splash to stop."""
        from lilbee.splash import dismiss

        dismiss()

    def _needs_setup(self) -> bool:
        """Check if both chat and embedding models are resolvable."""
        try:
            from lilbee.providers.llama_cpp_provider import resolve_model_path

            resolve_model_path(cfg.chat_model)
            resolve_model_path(cfg.embedding_model)
            return False
        except Exception:
            return True

    def _embedding_ready(self) -> bool:
        """Quick check if embedding model exists (no network calls)."""
        try:
            from lilbee.providers.llama_cpp_provider import resolve_model_path

            resolve_model_path(cfg.embedding_model)
            return True
        except Exception:
            return False

    def _on_setup_complete(self, result: str | None) -> None:
        """Called when wizard completes or is skipped."""
        if result == "skipped":
            self._show_chat_only_banner()
        elif self._auto_sync and self._embedding_ready():
            self._run_sync()
        self._refresh_model_bar()

    def _show_chat_only_banner(self) -> None:
        """Show the persistent chat-only banner."""
        self.query_one("#chat-only-banner", Static).display = True

    def _hide_chat_only_banner(self) -> None:
        """Hide the chat-only banner."""
        self.query_one("#chat-only-banner", Static).display = False

    def action_open_setup(self) -> None:
        """Open the setup wizard."""
        self._cmd_setup("")

    def _enter_insert_mode(self) -> None:
        """Switch to insert mode: focus input, update border style."""
        self._insert_mode = True
        self.query_one("#chat-input", Input).focus()
        self._update_input_style()

    def _update_input_style(self) -> None:
        """Toggle input opacity and mode indicator based on current mode."""
        inp = self.query_one("#chat-input", Input)
        if self._insert_mode:
            inp.remove_class("normal-mode")
        else:
            inp.add_class("normal-mode")
        self._update_mode_indicator()

    def _update_mode_indicator(self) -> None:
        """Update the ViewTabs mode text to reflect the current mode."""
        from textual.css.query import NoMatches

        with contextlib.suppress(NoMatches):
            bar = self.query_one(ViewTabs)
            bar.mode_text = msg.MODE_INSERT if self._insert_mode else msg.MODE_NORMAL

    def on_key(self, event: object) -> None:
        """Handle key events: vim mode and typing from chat log."""
        from textual.events import Key

        if not isinstance(event, Key):
            return
        inp = self.query_one("#chat-input", Input)
        if self._insert_mode:
            if not inp.has_focus and event.is_printable and event.character:
                inp.focus()
                inp.insert_text_at_cursor(event.character)
                event.prevent_default()
                event.stop()
            return
        if event.key == "enter" or (event.character and event.character in "iao"):
            self._enter_insert_mode()
            event.prevent_default()
            event.stop()
            return

    @on(Input.Submitted, "#chat-input")
    def _on_chat_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self._input_history.append(text)
        self._history_index = -1

        if text.startswith("/"):
            self._handle_slash(text)
            return

        self._send_message(text)

    def _send_message(self, text: str) -> None:
        """Send a user message and stream the response."""
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(UserMessage(text))

        assistant_msg = AssistantMessage()
        log.mount(assistant_msg)
        log.scroll_end(animate=False)

        with self._history_lock:
            self._history.append({"role": "user", "content": text})
        self.streaming = True
        self._stream_response(text, assistant_msg)

    @work(thread=True)
    def _stream_response(self, question: str, widget: AssistantMessage) -> None:
        """Stream LLM response in a background thread."""
        from lilbee.services import get_services

        response_parts: list[str] = []
        sources: list[str] = []
        last_scroll = 0.0

        try:
            with self._history_lock:
                history_snapshot = self._history[:-1]
            stream = get_services().searcher.ask_stream(question, history=history_snapshot)
            for token in stream:
                try:
                    if token.is_reasoning:
                        self.app.call_from_thread(widget.append_reasoning, token.content)
                    elif token.content:
                        response_parts.append(token.content)
                        self.app.call_from_thread(widget.append_content, token.content)
                    now = time.monotonic()
                    if now - last_scroll >= 0.15:
                        self.app.call_from_thread(self._scroll_to_bottom)
                        last_scroll = now
                except Exception:
                    break  # App shutting down (Ctrl-C) -- stop streaming
        except Exception as exc:
            log.debug("Stream error", exc_info=True)
            with contextlib.suppress(Exception):
                self.app.call_from_thread(widget.append_content, msg.STREAM_ERROR.format(error=exc))
        finally:
            self.streaming = False
            full_response = "".join(response_parts)
            if full_response:
                with self._history_lock:
                    self._history.append({"role": "assistant", "content": full_response})
                    self._trim_history()
            self.app.call_from_thread(widget.finish, sources)
            self.app.call_from_thread(self._scroll_to_bottom)

    def _trim_history(self) -> None:
        """Trim history to max size, dropping oldest messages. Caller must hold _history_lock."""
        if len(self._history) > _MAX_HISTORY_MESSAGES:
            self._history[:] = self._history[-_MAX_HISTORY_MESSAGES:]

    def _scroll_to_bottom(self) -> None:
        log_widget = self.query_one("#chat-log", VerticalScroll)
        # Only auto-scroll if user is near the bottom (within 5 lines).
        # If they scrolled up to read, don't yank them back.
        if log_widget.max_scroll_y - log_widget.scroll_y < 5:
            log_widget.scroll_end(animate=False)

    def action_scroll_up(self) -> None:
        self.query_one("#chat-log", VerticalScroll).scroll_page_up()

    def action_scroll_down(self) -> None:
        self.query_one("#chat-log", VerticalScroll).scroll_page_down()

    def action_enter_normal_mode(self) -> None:
        """Escape: cancel stream, return from model bar, or enter normal mode."""
        if self.streaming:
            for worker in self.workers:
                worker.cancel()
            self.streaming = False
            return
        if isinstance(self.focused, Select):
            self.query_one("#chat-input", Input).focus()
            return
        self._insert_mode = False
        self.query_one("#chat-log", VerticalScroll).focus()
        self._update_input_style()

    def action_cancel_stream(self) -> None:
        """Context-aware Escape: cancel stream -> blur input -> no-op."""
        if self.streaming:
            for worker in self.workers:
                worker.cancel()
            self.streaming = False
            return
        inp = self.query_one("#chat-input", Input)
        if inp.has_focus:
            self.query_one("#chat-log", VerticalScroll).focus()

    async def action_toggle_markdown(self) -> None:
        """Toggle between Markdown and plain-text rendering for chat responses."""
        cfg.markdown_rendering = not cfg.markdown_rendering
        use_md = cfg.markdown_rendering
        chat_log = self.query_one("#chat-log", VerticalScroll)
        for widget in chat_log.query(AssistantMessage):
            await widget.rebuild_content_widget(use_md)
        label = "Markdown" if use_md else "Plain text"
        self.notify(msg.CHAT_RENDERING.format(label=label))

    def _run_sync(self) -> None:
        """Enqueue a document sync in the task bar."""
        if self._sync_active:
            self.notify(msg.SYNC_ALREADY_ACTIVE, severity="warning")
            return
        task_bar = self._task_bar
        task_id = task_bar.add_task("Sync documents", "sync")
        task_bar.queue.advance("sync")
        self._run_sync_worker(task_id)

    @work(thread=True)
    def _run_sync_worker(self, task_id: str) -> None:
        """Run background document sync in a Textual worker thread.

        Architecture: @work(thread=True) runs this method in a daemon thread,
        keeping the Textual event loop free for UI updates. Progress is reported
        back to the main thread via app.call_from_thread(). The asyncio.run()
        call creates a fresh event loop because Textual workers are plain threads,
        not coroutines on the app's async loop.
        """
        import asyncio

        self._sync_active = True
        task_bar = self._task_bar
        try:
            from lilbee.ingest import sync

            self.app.call_from_thread(task_bar.update_task, task_id, 0, "Syncing...")

            def on_progress(event_type: EventType, data: ProgressEvent) -> None:
                if event_type == EventType.FILE_START:
                    from lilbee.progress import FileStartEvent

                    if not isinstance(data, FileStartEvent):
                        raise TypeError(f"Expected FileStartEvent, got {type(data).__name__}")
                    pct = int(data.current_file * 100 / data.total_files) if data.total_files else 0
                    status = msg.SYNC_FILE_PROGRESS.format(
                        current=data.current_file,
                        total=data.total_files,
                        file=data.file,
                    )
                    self.app.call_from_thread(task_bar.update_task, task_id, pct, status)

            asyncio.run(sync(quiet=True, on_progress=on_progress))
            self.app.call_from_thread(task_bar.complete_task, task_id)
        except asyncio.CancelledError:
            self._auto_sync = False
            self.app.call_from_thread(
                task_bar.fail_task, task_id, "Sync cancelled. Use /sync to resume."
            )
        except Exception:
            log.warning("Background sync failed", exc_info=True)
            self.app.call_from_thread(task_bar.fail_task, task_id, msg.SYNC_STATUS_FAILED)
        finally:
            self._sync_active = False

    def action_focus_commands(self) -> None:
        """Focus chat input and pre-fill with '/' for command entry."""
        inp = self.query_one("#chat-input", Input)
        inp.focus()
        if not inp.value.startswith("/"):
            inp.value = "/"
            inp.action_end()

    def action_focus_model_bar(self) -> None:
        """Focus the first Select in the model bar (normal mode only)."""
        if self._insert_mode:
            raise SkipAction()
        import contextlib

        with contextlib.suppress(Exception):
            self.query_one("#chat-model-select", Select).focus()

    def action_complete(self) -> None:
        """Tab completion: show or cycle autocomplete options."""
        inp = self.query_one("#chat-input", Input)
        if not inp.has_focus:
            raise SkipAction()
        overlay = self.query_one("#completion-overlay", CompletionOverlay)

        if overlay.is_visible:
            selection = overlay.cycle_next()
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

    def action_complete_next(self) -> None:
        """Ctrl+N: show completions or cycle forward."""
        self.action_complete()

    def action_complete_prev(self) -> None:
        """Ctrl+P: cycle backward through completions."""
        overlay = self.query_one("#completion-overlay", CompletionOverlay)
        inp = self.query_one("#chat-input", Input)

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
        inp = self.query_one("#chat-input", Input)
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
        inp = self.query_one("#chat-input", Input)
        if not inp.has_focus or self._history_index == -1:
            raise SkipAction()
        if self._history_index < len(self._input_history) - 1:
            self._history_index += 1
            inp.value = self._input_history[self._history_index]
            inp.action_end()
        else:
            self._history_index = -1
            inp.value = ""

    @on(Input.Changed, "#chat-input")
    def _on_chat_input_changed(self, event: Input.Changed) -> None:
        """Hide completion overlay when input changes manually."""
        if self._completing:
            return
        overlay = self.query_one("#completion-overlay", CompletionOverlay)
        if overlay.is_visible:
            overlay.hide()

    def _refresh_model_bar(self) -> None:
        """Update the model status bar and status line."""
        self.query_one("#model-bar", ModelBar).refresh_models()
        self._refresh_status_line()

    def _refresh_status_line(self) -> None:
        """Update the status line pill with the current chat model."""
        self.query_one("#chat-status-line", ChatStatusLine).model_name = cfg.chat_model

    def action_vim_scroll_down(self) -> None:
        """Vim j: scroll down in normal mode."""
        if self._insert_mode:
            raise SkipAction()
        self.query_one("#chat-log", VerticalScroll).scroll_down()

    def action_vim_scroll_up(self) -> None:
        """Vim k: scroll up in normal mode."""
        if self._insert_mode:
            raise SkipAction()
        self.query_one("#chat-log", VerticalScroll).scroll_up()

    def action_vim_scroll_home(self) -> None:
        """Vim g: scroll to top in normal mode."""
        if self._insert_mode:
            raise SkipAction()
        self.query_one("#chat-log", VerticalScroll).scroll_home()

    def action_vim_scroll_end(self) -> None:
        """Vim G: scroll to bottom in normal mode."""
        if self._insert_mode:
            raise SkipAction()
        self.query_one("#chat-log", VerticalScroll).scroll_end()

    def action_half_page_down(self) -> None:
        """Ctrl-D: half-page down (vim style)."""
        log_widget = self.query_one("#chat-log", VerticalScroll)
        half = max(1, log_widget.size.height // 2)
        log_widget.scroll_relative(y=half)

    def action_half_page_up(self) -> None:
        """Ctrl-U: half-page up (vim style)."""
        log_widget = self.query_one("#chat-log", VerticalScroll)
        half = max(1, log_widget.size.height // 2)
        log_widget.scroll_relative(y=-half)
