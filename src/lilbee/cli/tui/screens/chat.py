"""Chat screen — scrollable message log with streaming markdown responses."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from pathlib import Path
from typing import Any, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Input

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
from lilbee.cli.tui.widgets.sync_bar import SyncBar
from lilbee.config import cfg
from lilbee.crawler import is_url, require_valid_crawl_url
from lilbee.progress import EventType
from lilbee.query import ChatMessage

log = logging.getLogger(__name__)

_DISPATCH = build_dispatch_dict()

_MAX_HISTORY_MESSAGES = 200


class ChatScreen(Screen[None]):
    """Primary chat interface with streaming LLM responses."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("tab", "complete", "Complete", show=False, priority=True),
        Binding("pageup", "scroll_up", "Scroll Up", show=False),
        Binding("pagedown", "scroll_down", "Scroll Down", show=False),
        Binding("escape", "cancel_stream", "Cancel", show=False),
    ]

    def __init__(self, *, auto_sync: bool = False) -> None:
        super().__init__()
        self._auto_sync = auto_sync
        self._history: list[ChatMessage] = []
        self._history_lock = threading.Lock()
        self._streaming = False

    def compose(self) -> ComposeResult:
        yield ModelBar(id="model-bar")
        yield VerticalScroll(id="chat-log")
        yield SyncBar(id="sync-bar")
        yield CompletionOverlay(id="completion-overlay")
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        yield Input(
            placeholder="Ask a question or type / for commands",
            id="chat-input",
            suggester=SlashSuggester(use_cache=False),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        self._check_embedding_model_async()
        if self._auto_sync:
            self._run_sync()

    @work(thread=True)
    def _check_embedding_model_async(self) -> None:
        """Check for embedding model in a background thread (non-blocking)."""
        from lilbee.model_manager import detect_remote_embedding_models, get_model_manager

        manager = get_model_manager()
        if manager.is_installed(cfg.embedding_model):
            return

        embed_base = cfg.embedding_model.split(":")[0]
        remote_embeds = detect_remote_embedding_models(cfg.litellm_base_url)
        if any(embed_base in name for name in remote_embeds):
            return

        self.app.call_from_thread(self._show_setup_modal, remote_embeds)

    def _show_setup_modal(self, remote_embeds: list[str]) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        def on_setup_complete(name: str | None) -> None:
            if name:
                cfg.embedding_model = name
                self.notify(msg.EMBEDDING_SET.format(name=name))
                self._refresh_model_bar()

        self.app.push_screen(SetupModal(ollama_embeddings=remote_embeds), on_setup_complete)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "chat-input":
            return
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

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
        # Auto-detect URLs and route to crawl logic
        if is_url(args):
            self._cmd_crawl(args)
            return
        path = Path(args).expanduser()
        if not path.exists():
            self.notify(msg.CMD_ADD_NOT_FOUND.format(path=path), severity="error")
            return
        try:
            from lilbee.cli.app import console
            from lilbee.cli.helpers import copy_paths

            copied = copy_paths([path], console)
            self.notify(msg.CMD_ADD_SUCCESS.format(count=len(copied)))
            self._run_sync()
        except Exception as exc:
            log.warning("Failed to add %s", path, exc_info=True)
            self.notify(msg.CMD_ADD_ERROR.format(error=exc), severity="error")

    def _cmd_cancel(self, _args: str) -> None:
        for worker in self.workers:
            worker.cancel()
        self.notify(msg.CMD_CANCEL)

    def _cmd_crawl(self, args: str) -> None:
        if not args:
            self.notify("Usage: /crawl <url> [--depth N] [--max-pages N]", severity="warning")
            return
        parts = args.split()
        url = parts[0]
        try:
            require_valid_crawl_url(url)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        depth, max_pages = self._parse_crawl_flags(parts[1:])
        self.notify(f"Crawling {url}...")
        self._run_crawl_background(url, depth, max_pages)

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
    def _run_crawl_background(self, url: str, depth: int, max_pages: int) -> None:
        """Run a crawl in a background thread, then trigger sync."""
        from lilbee.crawler import crawl_and_save

        sync_bar = self.query_one("#sync-bar", SyncBar)
        self.app.call_from_thread(sync_bar.set_status, f"Crawling {url}...")

        try:

            def on_progress(event_type: EventType, data: dict[str, Any]) -> None:
                if event_type == EventType.CRAWL_PAGE:
                    current = data.get("current", 0)
                    total = data.get("total", 0)
                    page_url = data.get("url", "")
                    msg = f"Crawling [{current}/{total}]: {page_url}"
                    self.app.call_from_thread(sync_bar.set_status, msg)

            paths = asyncio.run(
                crawl_and_save(url, depth=depth, max_pages=max_pages, on_progress=on_progress)
            )
            self.app.call_from_thread(self.notify, f"Crawled {len(paths)} page(s) from {url}")
        except Exception as exc:
            self.app.call_from_thread(self.notify, f"Crawl failed: {exc}", severity="error")
            self.app.call_from_thread(sync_bar.set_status, "Crawl failed")
            return

        # Trigger sync to ingest the crawled markdown files
        self.app.call_from_thread(sync_bar.set_status, "Syncing crawled pages...")
        self._run_sync()

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

    def _cmd_model(self, args: str) -> None:
        if args:
            from lilbee.models import ensure_tag

            tagged = ensure_tag(args)
            cfg.chat_model = tagged
            settings.set_value(cfg.data_root, "chat_model", tagged)
            self.app.title = f"lilbee — {cfg.chat_model}"
            self.notify(msg.CMD_MODEL_SET.format(name=tagged))
            self._refresh_model_bar()
        else:
            self.app.push_screen(CatalogScreen())

    def _cmd_quit(self, _args: str) -> None:
        self.app.exit()

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
            self.notify(f"Theme: {args}")
        else:
            theme_list = msg.CMD_THEME_LIST.format(names=", ".join(DARK_THEMES))
            self.notify(theme_list, severity="information")

    def _cmd_version(self, _args: str) -> None:
        self.notify(f"lilbee {get_version()}")

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

        try:
            with self._history_lock:
                history_snapshot = self._history[:-1]
            stream = get_services().searcher.ask_stream(question, history=history_snapshot)
            for token in stream:
                if token.is_reasoning:
                    self.app.call_from_thread(widget.append_reasoning, token.content)
                elif token.content:
                    response_parts.append(token.content)
                    self.app.call_from_thread(widget.append_content, token.content)
        except Exception as exc:
            log.warning("Stream error", exc_info=True)
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
        self.query_one("#chat-log", VerticalScroll).scroll_end(animate=False)

    def action_scroll_up(self) -> None:
        self.query_one("#chat-log", VerticalScroll).scroll_page_up()

    def action_scroll_down(self) -> None:
        self.query_one("#chat-log", VerticalScroll).scroll_page_down()

    def action_cancel_stream(self) -> None:
        if self._streaming:
            for worker in self.workers:
                worker.cancel()
            self._streaming = False

    @work(thread=True)
    def _run_sync(self) -> None:
        """Run background document sync.

        Uses asyncio.run() because Textual workers run in threads, not the
        async event loop, so we need a fresh loop for the async sync() call.
        """
        import asyncio

        sync_bar = self.query_one("#sync-bar", SyncBar)
        try:
            from lilbee.ingest import sync

            self.app.call_from_thread(sync_bar.set_status, msg.SYNC_STATUS_SYNCING)

            def on_progress(event_type: EventType, data: dict[str, object]) -> None:
                if event_type == EventType.FILE_START:
                    status = msg.SYNC_FILE_PROGRESS.format(
                        current=data.get("current_file", "?"),
                        total=data.get("total_files", "?"),
                        file=data.get("file", ""),
                    )
                    self.app.call_from_thread(sync_bar.set_status, status)

            result = asyncio.run(sync(on_progress=on_progress))
            added = len(result.added)
            self.app.call_from_thread(sync_bar.set_status, msg.SYNC_STATUS_DONE.format(count=added))
        except Exception:
            log.warning("Background sync failed", exc_info=True)
            self.app.call_from_thread(sync_bar.set_status, msg.SYNC_STATUS_FAILED)

    def action_complete(self) -> None:
        """Tab completion: show or cycle autocomplete options."""
        overlay = self.query_one("#completion-overlay", CompletionOverlay)
        inp = self.query_one("#chat-input", Input)

        if overlay.is_visible:
            selection = overlay.cycle_next()
            if selection:
                cmd_prefix = inp.value.split()[0] + " " if " " in inp.value else ""
                inp.value = cmd_prefix + selection
                inp.action_end()
            return

        options = get_completions(inp.value)
        if options:
            overlay.show_completions(options)
            first = overlay.get_current()
            if first and " " in inp.value:
                cmd_prefix = inp.value.split()[0] + " "
                inp.value = cmd_prefix + first
                inp.action_end()
            elif first:
                inp.value = first
                inp.action_end()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Hide completion overlay when input changes manually."""
        if event.input.id == "chat-input":
            overlay = self.query_one("#completion-overlay", CompletionOverlay)
            if overlay.is_visible:
                overlay.hide()

    def _refresh_model_bar(self) -> None:
        """Update the model status bar."""
        self.query_one("#model-bar", ModelBar).refresh_models()

    def key_j(self) -> None:
        """Vim: scroll down."""
        focused = self.focused
        if focused and isinstance(focused, Input):
            return
        self.query_one("#chat-log", VerticalScroll).scroll_down()

    def key_k(self) -> None:
        """Vim: scroll up."""
        focused = self.focused
        if focused and isinstance(focused, Input):
            return
        self.query_one("#chat-log", VerticalScroll).scroll_up()
