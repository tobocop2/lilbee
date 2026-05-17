"""Command palette provider for lilbee TUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from textual.command import Hit, Hits, Provider

from lilbee.app.services import get_services
from lilbee.app.settings import apply_settings_update
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp


class LilbeeCommandProvider(Provider):
    """Provides searchable commands for the Textual command palette (Ctrl+P)."""

    @property
    def _app(self) -> LilbeeApp:
        return cast("LilbeeApp", self.screen.app)

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
                "Retry skipped documents",
                "Re-attempt files that failed a previous sync",
                self._action_retry_skipped,
            ),
            ("Open wiki", "Browse and generate wiki pages", self._action_open_wiki),
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
            from lilbee.modelhub.models import list_installed_models

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

        apply_settings_update({attr: value})
        display = value or "off"
        self.screen.app.notify(f"{attr}: {display}")
        if attr == "chat_model":
            self.screen.app.title = f"lilbee: {value}"

    def _delete_doc(self, name: str) -> None:
        store = get_services().store
        store.delete_by_source(name)
        store.delete_source(name)
        self.screen.app.notify(f"Deleted {name}")

    def _action_sync(self) -> None:
        self._app.action_run_sync()

    def _action_retry_skipped(self) -> None:
        """Clear the failed-file markers and kick off a sync to retry them.

        Clearing the marker cache and then running a normal sync is
        equivalent to ``lilbee sync --retry-skipped`` / ``POST /api/sync``
        with ``retry_skipped=true``.
        """
        from lilbee.data.ingest.skip_marker import clear_skip_markers, load_skip_markers

        cleared = len(load_skip_markers(cfg.data_root))
        clear_skip_markers(cfg.data_root)
        self.screen.app.notify(msg.retry_skipped_message(cleared))
        self._app.action_run_sync()

    def _action_version(self) -> None:
        from lilbee.app.version import get_version

        self.screen.app.notify(f"lilbee {get_version()}")

    def _action_setup(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        self.screen.app.push_screen(SetupWizard())

    def _action_open_wiki(self) -> None:
        self._app.switch_view("Wiki")

    def _action_noop(self) -> None:
        self.screen.app.notify("Type '/reset confirm' in chat to reset")
