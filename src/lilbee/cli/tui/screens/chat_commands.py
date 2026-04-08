"""Slash command handlers for the chat screen.

Extracted from chat.py to reduce monolith size. Mixed into ChatScreen
via ``SlashCommandsMixin``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from textual import work
from textual.screen import Screen

from lilbee import settings
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.command_registry import build_dispatch_dict
from lilbee.config import cfg
from lilbee.crawler import crawler_available, is_url, require_valid_crawl_url
from lilbee.progress import EventType, ProgressEvent

log = logging.getLogger(__name__)

_DISPATCH = build_dispatch_dict()


class SlashCommandsMixin(Screen[None]):  # type: ignore[type-arg]
    """Slash command handlers for the chat screen.

    Inherits Screen for type-checking only — ChatScreen uses this as a
    mixin via multiple inheritance (SlashCommandsMixin, Screen[None]).
    """

    # Stubs for attributes provided by ChatScreen at runtime.
    _sync_active: bool
    _task_bar: Any
    def _refresh_model_bar(self) -> None: ...  # pragma: no cover
    def _run_sync(self) -> None: ...  # pragma: no cover
    def _on_setup_complete(self, result: Any) -> None: ...  # pragma: no cover

    def _handle_slash(self, text: str) -> None:
        """Dispatch slash commands via the command registry."""
        cmd = text.split()[0].lower()
        args = text[len(cmd) :].strip()
        handler_name = _DISPATCH.get(cmd)
        if handler_name:
            getattr(self, handler_name)(args)
        else:
            self.notify(msg.CMD_UNKNOWN.format(cmd=cmd), severity="warning")

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
        task_bar = self._task_bar
        task_id = task_bar.add_task(f"Add {path.name}", "add")
        task_bar.queue.advance("add")
        self._run_add_background(path, task_id)

    @work(thread=True)
    def _run_add_background(self, path: Path, task_id: str) -> None:
        """Copy files and sync in a background thread."""
        self._sync_active = True
        task_bar = self._task_bar
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

                    if not isinstance(data, FileStartEvent):
                        raise TypeError(f"Expected FileStartEvent, got {type(data).__name__}")
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
        task_bar = self._task_bar
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

        task_bar = self._task_bar
        self.app.call_from_thread(task_bar.update_task, task_id, 0, f"Crawling {url}...")

        try:

            def on_progress(event_type: EventType, data: ProgressEvent) -> None:
                if event_type == EventType.CRAWL_PAGE:
                    from lilbee.progress import CrawlPageEvent

                    if not isinstance(data, CrawlPageEvent):
                        raise TypeError(f"Expected CrawlPageEvent, got {type(data).__name__}")
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

        self.app.call_from_thread(self._run_sync)

    def _cmd_catalog(self, _args: str) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

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
        from lilbee.cli.settings_map import SETTINGS_MAP

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
            setattr(cfg, key, parsed)
            persisted = str(parsed) if parsed is not None else ""
            settings.set_value(cfg.data_root, key, persisted)
            if key == "llm_provider":  # pragma: no cover
                from lilbee.services import reset_services

                reset_services()
            self.notify(msg.CMD_SET_SUCCESS.format(key=key, value=parsed))
        except (ValueError, TypeError) as exc:
            self.notify(msg.CMD_SET_INVALID.format(key=key, error=exc), severity="error")

    def _cmd_settings(self, _args: str) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        self.app.push_screen(SettingsScreen())

    def _cmd_setup(self, _args: str) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        self.app.push_screen(SetupWizard(), self._on_setup_complete)

    def _cmd_status(self, _args: str) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen

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
        from lilbee.cli.helpers import get_version

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
