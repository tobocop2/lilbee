"""Chat screen — scrollable message log with streaming markdown responses."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Input, Static

from lilbee import settings
from lilbee.cli.helpers import get_version
from lilbee.cli.settings_map import SETTINGS_MAP
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.command_registry import build_dispatch_dict
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay, get_completions
from lilbee.cli.tui.widgets.help_modal import HelpModal
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.cli.tui.widgets.nav_bar import NavBar
from lilbee.config import cfg
from lilbee.crawler import crawler_available, is_url, require_valid_crawl_url
from lilbee.progress import EventType, ProgressEvent
from lilbee.query import ChatMessage

if TYPE_CHECKING:
    from lilbee.cli.tui.widgets.task_bar import TaskBar

log = logging.getLogger(__name__)

_DISPATCH = build_dispatch_dict()

_MAX_HISTORY_MESSAGES = 200

_FOCUSABLE_IDS = ("model-bar", "chat-log", "chat-input")


class ChatScreen(Screen[None]):
    """Primary chat interface with streaming LLM responses."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("slash", "focus_commands", "/ commands", show=True),
        Binding("tab", "complete", "Tab", show=False, priority=True),
        Binding("ctrl+n", "complete_next", "^n next", show=False),
        Binding("ctrl+p", "complete_prev", "^p prev", show=False),
        Binding("pageup", "scroll_up", "PgUp", show=False),
        Binding("pagedown", "scroll_down", "PgDn", show=False),
        Binding("ctrl+d", "half_page_down", "^d half PgDn", show=False),
        Binding("ctrl+u", "half_page_up", "^u half PgUp", show=False),
        Binding("escape", "enter_normal_mode", "Normal", show=False, priority=True),
        Binding("ctrl+r", "toggle_markdown", "Markdown", show=False),
    ]

    def __init__(self, *, auto_sync: bool = False) -> None:
        super().__init__()
        self._auto_sync = auto_sync
        self._history: list[ChatMessage] = []
        self._history_lock = threading.Lock()
        self._streaming = False
        self._insert_mode: bool = True
        self._completing = False
        self._sync_active: bool = False
        self._input_history: list[str] = []
        self._history_index: int = -1

    def _get_task_bar(self) -> TaskBar:
        """Get the app-level TaskBar (created by LilbeeApp)."""
        return self.app._task_bar  # type: ignore[attr-defined, no-any-return]

    def compose(self) -> ComposeResult:
        yield NavBar(id="global-nav-bar")
        yield ModelBar(id="model-bar")
        yield Static(msg.CHAT_ONLY_BANNER, id="chat-only-banner")
        yield VerticalScroll(id="chat-log")
        yield CompletionOverlay(id="completion-overlay")
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        yield Input(
            placeholder=msg.CHAT_INPUT_PLACEHOLDER,
            id="chat-input",
            suggester=SlashSuggester(use_cache=False),
        )

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        self._update_input_style()
        self.query_one("#chat-only-banner", Static).display = False
        if self._needs_setup():
            from lilbee.cli.tui.screens.setup import SetupWizard

            self.app.push_screen(SetupWizard(), self._on_setup_complete)
        elif self._auto_sync and self._embedding_ready():
            self._run_sync()

    def on_show(self) -> None:
        """Called when screen becomes visible - signal splash to stop."""
        import os
        import tempfile

        ready_file = os.path.join(tempfile.gettempdir(), "lilbee-splash-ready")
        try:
            with open(ready_file, "w") as f:
                f.write("ready")
        except OSError:
            pass

    def _needs_setup(self) -> bool:
        """Check if both chat and embedding models are resolvable."""
        try:
            from lilbee.providers.llama_cpp_provider import _resolve_model_path

            _resolve_model_path(cfg.chat_model)
            _resolve_model_path(cfg.embedding_model)
            return False
        except Exception:
            return True

    def _embedding_ready(self) -> bool:
        """Quick check if embedding model exists (no network calls)."""
        try:
            from lilbee.providers.llama_cpp_provider import _resolve_model_path

            _resolve_model_path(cfg.embedding_model)
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

    def key_f5(self) -> None:
        """Open the setup wizard."""
        from lilbee.cli.tui.screens.setup import SetupWizard

        self.app.push_screen(SetupWizard(), self._on_setup_complete)

    def _enter_insert_mode(self) -> None:
        """Switch to insert mode: focus input, update border style."""
        self._insert_mode = True
        self.query_one("#chat-input", Input).focus()
        self._update_input_style()

    def _update_input_style(self) -> None:
        """Toggle input border and mode indicator based on current mode."""
        inp = self.query_one("#chat-input", Input)
        if self._insert_mode:
            inp.remove_class("normal-mode")
            inp.add_class("insert-mode")
        else:
            inp.remove_class("insert-mode")
            inp.add_class("normal-mode")
        self._update_mode_indicator()

    def _update_mode_indicator(self) -> None:
        """Update the NavBar mode text to reflect the current mode."""
        try:
            nav = self.query_one("#global-nav-bar", NavBar)
            nav.mode_text = msg.MODE_INSERT if self._insert_mode else msg.MODE_NORMAL
        except Exception:
            pass

    def on_key(self, event: object) -> None:
        """Handle key events: vim mode and typing from chat log."""
        from textual.events import Key

        if not isinstance(event, Key):
            return
        inp = self.query_one("#chat-input", Input)
        if self._insert_mode and not inp.has_focus and event.is_printable and event.character:
            inp.focus()
            inp.insert_text_at_cursor(event.character)
            event.prevent_default()
            event.stop()
            return
        if self._insert_mode:
            return
        if event.key == "enter":
            self._enter_insert_mode()
            event.prevent_default()
            event.stop()
            return
        if event.character and event.character.isprintable() and len(event.key) == 1:
            if event.character in "jkgG":
                return
            self._enter_insert_mode()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat-input":
            return
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

    def _handle_slash(self, text: str) -> None:
        """Dispatch slash commands via the command registry."""
        cmd = text.split()[0].lower()
        args = text[len(cmd) :].strip()
        handler_name = _DISPATCH.get(cmd)
        if handler_name:
            getattr(self, handler_name)(args)
        else:
            self.notify(msg.CMD_UNKNOWN.format(cmd=cmd), severity="warning")

    # -- Slash command handlers (alphabetical) --------------------------------

    def _cmd_add(self, args: str) -> None:
        if not args:
            return
        if self._sync_active:
            self.notify(msg.SYNC_ALREADY_ACTIVE, severity="warning")
            return
        # Auto-detect URLs and route to crawl logic
        if is_url(args):
            self._cmd_crawl(args)
            return
        path = Path(args).expanduser()
        if not path.exists():
            self.notify(msg.CMD_ADD_NOT_FOUND.format(path=path), severity="error")
            return
        task_bar = self._get_task_bar()
        task_id = task_bar.add_task(f"Add {path.name}", "add")
        task_bar.queue.advance("add")
        self._run_add_background(path, task_id)

    @work(thread=True)
    def _run_add_background(self, path: Path, task_id: str) -> None:
        """Copy files and sync in a background thread."""
        self._sync_active = True
        task_bar = self._get_task_bar()
        self.app.call_from_thread(task_bar.update_task, task_id, 0, f"Copying {path.name}...")
        try:
            from lilbee.cli.helpers import copy_files

            result = copy_files([path])
            copied = result.copied
            for name in result.skipped:
                self.app.call_from_thread(
                    self.notify, f"{name} already exists (use --force to overwrite)"
                )
            self.app.call_from_thread(
                task_bar.update_task, task_id, 50, f"Copied {len(copied)} file(s), syncing..."
            )

            from lilbee.ingest import sync

            def on_progress(event_type: EventType, data: ProgressEvent) -> None:
                if event_type == EventType.FILE_START:
                    from lilbee.progress import FileStartEvent

                    assert isinstance(data, FileStartEvent)
                    if data.total_files:
                        pct = 50 + int(data.current_file * 50 / data.total_files)
                    else:
                        pct = 75
                    self.app.call_from_thread(
                        task_bar.update_task, task_id, pct, f"Syncing {data.file}..."
                    )

            asyncio.run(sync(quiet=True, on_progress=on_progress))
            self.app.call_from_thread(task_bar.complete_task, task_id)
            self.app.call_from_thread(self.notify, msg.CMD_ADD_SUCCESS.format(count=len(copied)))
        except Exception as exc:
            log.warning("Failed to add %s", path, exc_info=True)
            self.app.call_from_thread(task_bar.fail_task, task_id, str(exc))
            self.app.call_from_thread(
                self.notify, msg.CMD_ADD_ERROR.format(error=exc), severity="error"
            )
        finally:
            self._sync_active = False

    def _cmd_cancel(self, _args: str) -> None:
        for worker in self.workers:
            worker.cancel()
        self.notify(msg.CMD_CANCEL)

    def _cmd_crawl(self, args: str) -> None:
        if not crawler_available():
            self.notify(msg.CMD_CRAWL_UNAVAILABLE, severity="error")
            return
        if not args:
            self.notify(msg.CMD_CRAWL_USAGE, severity="warning")
            return
        parts = args.split()
        url = parts[0]
        try:
            require_valid_crawl_url(url)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        depth, max_pages = self._parse_crawl_flags(parts[1:])
        task_bar = self._get_task_bar()
        task_id = task_bar.add_task(f"Crawl {url}", "crawl")
        task_bar.queue.advance("crawl")
        self._run_crawl_background(url, depth, max_pages, task_id)

    @staticmethod
    def _parse_crawl_flags(tokens: list[str]) -> tuple[int, int]:
        """Extract --depth and --max-pages from argument tokens."""
        flag_map = {"--depth": "depth", "--max-pages": "max_pages"}
        parsed: dict[str, int] = {"depth": 0, "max_pages": 0}
        i = 0
        while i < len(tokens):
            key = flag_map.get(tokens[i])
            if key and i + 1 < len(tokens):
                with contextlib.suppress(ValueError):
                    parsed[key] = int(tokens[i + 1])
                i += 2
            else:
                i += 1
        return parsed["depth"], parsed["max_pages"]

    @work(thread=True)
    def _run_crawl_background(self, url: str, depth: int, max_pages: int, task_id: str) -> None:
        """Run a crawl in a background thread, then trigger sync."""
        from lilbee.crawler import crawl_and_save

        task_bar = self._get_task_bar()
        self.app.call_from_thread(task_bar.update_task, task_id, 0, f"Crawling {url}...")

        try:

            def on_progress(event_type: EventType, data: ProgressEvent) -> None:
                if event_type == EventType.CRAWL_PAGE:
                    from lilbee.progress import CrawlPageEvent

                    assert isinstance(data, CrawlPageEvent)
                    pct = int(data.current * 100 / data.total) if data.total > 0 else 50
                    detail = f"[{data.current}/{data.total}]: {data.url}"
                    self.app.call_from_thread(task_bar.update_task, task_id, pct, detail)

            paths = asyncio.run(
                crawl_and_save(url, depth=depth, max_pages=max_pages, on_progress=on_progress)
            )
            self.app.call_from_thread(task_bar.complete_task, task_id)
            self.app.call_from_thread(
                self.notify, msg.CMD_CRAWL_SUCCESS.format(count=len(paths), url=url)
            )
        except Exception as exc:
            self.app.call_from_thread(task_bar.fail_task, task_id, str(exc))
            self.app.call_from_thread(
                self.notify, msg.CMD_CRAWL_FAILED.format(error=exc), severity="error"
            )
            return

        # Trigger sync to ingest the crawled markdown files
        self.app.call_from_thread(self._run_sync)

    def _cmd_catalog(self, _args: str) -> None:
        self.app.push_screen(CatalogScreen())

    def _cmd_delete(self, args: str) -> None:
        from lilbee.services import get_services

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
        self.app.push_screen(HelpModal())

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
            self.app.call_from_thread(self.notify, msg.CHAT_LOGGED_IN)
        except Exception as exc:
            log.warning("HuggingFace login failed", exc_info=True)
            self.app.call_from_thread(
                self.notify, msg.CHAT_LOGIN_FAILED.format(error=exc), severity="error"
            )

    def _cmd_model(self, args: str) -> None:
        if args:
            from lilbee.models import ensure_tag

            tagged = ensure_tag(args)
            cfg.chat_model = tagged
            settings.set_value(cfg.data_root, "chat_model", tagged)
            self.app.title = f"lilbee -- {cfg.chat_model}"
            self.notify(msg.CMD_MODEL_SET.format(name=tagged))
            self._refresh_model_bar()
        else:
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
        from lilbee.model_manager import get_model_manager

        mgr = get_model_manager()
        if not mgr.is_installed(name):
            self.app.call_from_thread(
                self.notify, msg.CMD_REMOVE_NOT_FOUND.format(name=name), severity="error"
            )
            return
        try:
            removed = mgr.remove(name)
            if removed:
                self.app.call_from_thread(self.notify, msg.CMD_REMOVE_SUCCESS.format(name=name))
            else:
                self.app.call_from_thread(
                    self.notify, msg.CMD_REMOVE_FAILED.format(name=name), severity="error"
                )
        except Exception:
            log.warning("Remove failed for %s", name, exc_info=True)
            self.app.call_from_thread(
                self.notify, msg.CMD_REMOVE_FAILED.format(name=name), severity="error"
            )

    def _cmd_reset(self, args: str) -> None:
        if args == "confirm":
            from lilbee.cli.helpers import perform_reset

            try:
                perform_reset()
                self.notify(msg.CMD_RESET_SUCCESS)
            except Exception as exc:
                log.warning("Reset failed", exc_info=True)
                self.notify(msg.CMD_RESET_FAILED.format(error=exc), severity="error")
        else:
            self.notify(msg.CMD_RESET_CONFIRM, severity="warning")

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
        try:
            if defn.type is bool:
                parsed = value.lower() in ("true", "1", "yes", "on")
            elif defn.nullable and value.lower() in ("none", "null", ""):
                parsed = None
            else:
                parsed = defn.type(value)
            setattr(cfg, defn.cfg_attr, parsed)
            persisted = str(parsed) if parsed is not None else ""
            settings.set_value(cfg.data_root, defn.cfg_attr, persisted)
            if defn.cfg_attr == "llm_provider":  # pragma: no cover
                from lilbee.services import reset_services

                reset_services()
            self.notify(msg.CMD_SET_SUCCESS.format(key=key, value=parsed))
        except (ValueError, TypeError) as exc:
            self.notify(msg.CMD_SET_INVALID.format(key=key, error=exc), severity="error")

    def _cmd_settings(self, _args: str) -> None:
        self.app.push_screen(SettingsScreen())

    def _cmd_status(self, _args: str) -> None:
        self.app.push_screen(StatusScreen())

    def _cmd_theme(self, args: str) -> None:
        from lilbee.cli.tui.app import DARK_THEMES, LilbeeApp

        if args and isinstance(self.app, LilbeeApp):
            self.app.set_theme(args)
            self.notify(msg.THEME_SET.format(name=args))
        else:
            theme_list = msg.CMD_THEME_LIST.format(names=", ".join(DARK_THEMES))
            self.notify(theme_list, severity="information")

    def _cmd_version(self, _args: str) -> None:
        self.notify(msg.CHAT_VERSION.format(version=get_version()))

    def _cmd_vision(self, args: str) -> None:
        if args == "off":
            cfg.vision_model = ""
            settings.set_value(cfg.data_root, "vision_model", "")
            self.notify(msg.CMD_VISION_DISABLED)
            self._refresh_model_bar()
            return

        if args:
            cfg.vision_model = args
            settings.set_value(cfg.data_root, "vision_model", args)
            self.notify(msg.CMD_VISION_SET.format(name=args))
            self._refresh_model_bar()
            return

        current = cfg.vision_model or "disabled"
        self.notify(msg.CMD_VISION_STATUS.format(current=current))

    # -- Core chat logic ------------------------------------------------------

    def _send_message(self, text: str) -> None:
        """Send a user message and stream the response."""
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(UserMessage(text))

        assistant_msg = AssistantMessage()
        log.mount(assistant_msg)
        log.scroll_end(animate=False)

        with self._history_lock:
            self._history.append({"role": "user", "content": text})
        self._streaming = True
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
            self._streaming = False
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

    def _cycle_focus(self, direction: int) -> None:
        """Cycle focus between focusable widgets in normal mode."""
        current_id = self.focused.id if self.focused else None
        try:
            idx = _FOCUSABLE_IDS.index(current_id)
        except ValueError:
            idx = 0
        next_idx = (idx + direction) % len(_FOCUSABLE_IDS)
        self.query_one(f"#{_FOCUSABLE_IDS[next_idx]}").focus()

    def action_enter_normal_mode(self) -> None:
        """Escape: cancel stream if active, otherwise enter normal mode."""
        if self._streaming:
            for worker in self.workers:
                worker.cancel()
            self._streaming = False
            return
        self._insert_mode = False
        self.query_one("#chat-log", VerticalScroll).focus()
        self._update_input_style()

    def action_cancel_stream(self) -> None:
        """Context-aware Escape: cancel stream -> blur input -> no-op."""
        if self._streaming:
            for worker in self.workers:
                worker.cancel()
            self._streaming = False
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
        task_bar = self._get_task_bar()
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
        task_bar = self._get_task_bar()
        try:
            from lilbee.ingest import sync

            self.app.call_from_thread(task_bar.update_task, task_id, 0, "Syncing...")

            def on_progress(event_type: EventType, data: ProgressEvent) -> None:
                if event_type == EventType.FILE_START:
                    from lilbee.progress import FileStartEvent

                    assert isinstance(data, FileStartEvent)
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

    def action_complete(self) -> None:
        """Tab completion: show or cycle autocomplete options."""
        overlay = self.query_one("#completion-overlay", CompletionOverlay)
        inp = self.query_one("#chat-input", Input)

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

    def key_up(self) -> None:
        """Up arrow: cycle focus in normal mode, recall input history in insert mode."""
        if not self._insert_mode:
            self._cycle_focus(-1)
            return
        inp = self.query_one("#chat-input", Input)
        if not inp.has_focus or not self._input_history:
            return
        if self._history_index == -1:
            self._history_index = len(self._input_history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return
        inp.value = self._input_history[self._history_index]
        inp.action_end()

    def key_down(self) -> None:
        """Down arrow: cycle focus in normal mode, recall input history in insert mode."""
        if not self._insert_mode:
            self._cycle_focus(1)
            return
        inp = self.query_one("#chat-input", Input)
        if not inp.has_focus or self._history_index == -1:
            return
        if self._history_index < len(self._input_history) - 1:
            self._history_index += 1
            inp.value = self._input_history[self._history_index]
            inp.action_end()
        else:
            self._history_index = -1
            inp.value = ""

    def on_input_changed(self, event: Input.Changed) -> None:
        """Hide completion overlay when input changes manually."""
        if self._completing:
            return
        if event.input.id == "chat-input":
            overlay = self.query_one("#completion-overlay", CompletionOverlay)
            if overlay.is_visible:
                overlay.hide()

    def _refresh_model_bar(self) -> None:
        """Update the model status bar."""
        self.query_one("#model-bar", ModelBar).refresh_models()

    def key_j(self) -> None:
        """Vim j: cycle focus to next widget in normal mode."""
        if not self._insert_mode:
            self._cycle_focus(1)

    def key_k(self) -> None:
        """Vim k: cycle focus to previous widget in normal mode."""
        if not self._insert_mode:
            self._cycle_focus(-1)

    def key_g(self) -> None:
        """Vim: scroll to top."""
        focused = self.focused
        if focused and isinstance(focused, Input):
            return
        self.query_one("#chat-log", VerticalScroll).scroll_home()

    def key_G(self) -> None:
        """Vim: scroll to bottom."""
        focused = self.focused
        if focused and isinstance(focused, Input):
            return
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
