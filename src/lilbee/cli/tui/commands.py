"""Command palette provider for lilbee TUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from textual.command import Hit, Hits, Provider

from lilbee import settings
from lilbee.config import cfg
from lilbee.services import get_services

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp


class LilbeeCommandProvider(Provider):
    """Provides searchable commands for the Textual command palette (Ctrl+P)."""

    @property
    def _app(self) -> LilbeeApp:
        from lilbee.cli.tui.app import LilbeeApp

        if not isinstance(self.screen.app, LilbeeApp):  # test apps aren't LilbeeApp
            raise TypeError(f"Expected LilbeeApp, got {type(self.screen.app).__name__}")
        return self.screen.app

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for cmd_text, help_text, action in self._get_commands():
            score = matcher.match(cmd_text)
            if score > 0:
                yield Hit(score, matcher.highlight(cmd_text), action, help=help_text)

    async def discover(self) -> Hits:
        for cmd_text, help_text, action in self._get_commands():
            yield Hit(1.0, cmd_text, action, help=help_text)

    def _get_commands(self) -> list[tuple[str, str, Any]]:
        app = self._app
        commands: list[tuple[str, str, Any]] = [
            ("Open catalog", "Browse and install models", lambda: app.switch_view("Catalog")),
            ("Run setup wizard", "Configure chat and embedding models", self._action_setup),
            ("Open status", "Knowledge base status", lambda: app.switch_view("Status")),
            ("Open settings", "View and change settings", lambda: app.switch_view("Settings")),
            ("Open task center", "Monitor background tasks", lambda: app.switch_view("Tasks")),
            ("Help", "Show keybinding reference", app.action_push_help),
            ("Cycle theme", "Switch to next color theme", app.action_cycle_theme),
            ("Sync documents", "Sync knowledge base", self._action_sync),
            (
                "Generate wiki pages",
                "Summarize every indexed document into the wiki",
                self._action_wiki_generate,
            ),
            ("Show version", "Display lilbee version", self._action_version),
            (
                "Reset knowledge base",
                "Delete all data (requires /reset confirm)",
                self._action_noop,
            ),
            ("Quit", "Exit lilbee", app.action_quit),
        ]

        commands.extend(self._model_commands())
        commands.extend(self._document_commands())
        return commands

    def _model_commands(self) -> list[tuple[str, str, Any]]:
        """Generate commands for installed models."""
        commands: list[tuple[str, str, Any]] = []
        try:
            from lilbee.models import list_installed_models

            for name in list_installed_models():
                commands.append(
                    (
                        f"Set chat model → {name}",
                        "Switch chat model",
                        lambda n=name: self._set_model("chat_model", n),
                    )
                )
        except Exception:
            log.debug("Failed to list installed models", exc_info=True)

        try:
            from lilbee.models import VISION_CATALOG

            for m in VISION_CATALOG:
                commands.append(
                    (
                        f"Set vision → {m.name}",
                        m.description,
                        lambda n=m.name: self._set_model("vision_model", n),
                    )
                )
            commands.append(
                (
                    "Set vision → off",
                    "Disable vision OCR",
                    lambda: self._set_model("vision_model", ""),
                )
            )
        except Exception:
            log.debug("Failed to load vision catalog", exc_info=True)

        return commands

    def _document_commands(self) -> list[tuple[str, str, Any]]:
        """Generate commands for indexed documents."""
        commands: list[tuple[str, str, Any]] = []
        try:
            for src in get_services().store.get_sources():
                name = src.get("filename", src.get("source", ""))
                if name:
                    commands.append(
                        (
                            f"Delete document → {name}",
                            f"Remove {name} from index",
                            lambda n=name: self._delete_doc(n),
                        )
                    )
        except Exception:
            log.debug("Failed to list documents", exc_info=True)
        return commands

    def _set_model(self, attr: str, value: str) -> None:
        setattr(cfg, attr, value)
        settings.set_value(cfg.data_root, attr, value)
        display = value or "off"
        self.screen.app.notify(f"{attr}: {display}")
        if attr == "chat_model":
            self.screen.app.title = f"lilbee — {value}"

    def _delete_doc(self, name: str) -> None:
        store = get_services().store
        store.delete_by_source(name)
        store.delete_source(name)
        self.screen.app.notify(f"Deleted {name}")

    def _action_sync(self) -> None:
        self.screen.app.notify("Use /add <path> or auto-sync on launch")

    def _action_version(self) -> None:
        from lilbee.cli.helpers import get_version

        self.screen.app.notify(f"lilbee {get_version()}")

    def _action_setup(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        self.screen.app.push_screen(SetupWizard())

    def _action_wiki_generate(self) -> None:
        from lilbee.cli.tui import messages as msg

        self.screen.app.notify(msg.CMD_WIKI_USAGE)

    def _action_noop(self) -> None:
        self.screen.app.notify("Type '/reset confirm' in chat to reset")
