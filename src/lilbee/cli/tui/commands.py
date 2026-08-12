"""Command palette provider for lilbee TUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from textual.command import Hit, Hits, Provider

from lilbee.catalog import display_label_for_ref
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.command_registry import COMMANDS, SlashCommand, get_command
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
            ("Open chat", "Ask questions about your knowledge base", app.action_open_chat),
            ("Open catalog", "Browse and install models", app.action_open_catalog),
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
            (
                "Delete document",
                "Remove a file from the index (Tab completes names)",
                self._action_delete_document,
            ),
            ("Open wiki", "Browse and generate wiki pages", self._action_open_wiki),
            (
                "Wikify",
                "Generate wiki pages from indexed documents (GPU-heavy)",
                self._action_wikify,
            ),
            (
                "Delete wiki",
                "Remove every generated wiki page and its indexed rows",
                self._action_wipe_wiki,
            ),
            ("Show version", "Display lilbee version", self._action_version),
            (
                "Reset knowledge base",
                "Delete all data (asks for confirmation)",
                self._action_reset,
            ),
            ("Quit", "Exit lilbee", app.action_quit),
        ]

        commands.extend(self._slash_commands())
        commands.extend(self._model_commands())
        return commands

    def _slash_commands(self) -> list[tuple[str, str, Any]]:
        """One palette entry per slash command, mirroring the chat surface."""
        return [
            (cmd.name, cmd.help_text, lambda c=cmd: self._run_slash_command(c)) for cmd in COMMANDS
        ]

    def _run_slash_command(self, cmd: SlashCommand) -> None:
        """Run *cmd* through Chat: dispatch it, or prefill it when it needs arguments."""
        app = self._app
        chat = app.chat_screen()
        if chat is None:
            app.notify(f"Open Chat to run {cmd.name}")
            return
        if cmd.args_hint.startswith("<"):
            # Needs an argument: land in the chat prompt for Tab completion.
            app.switch_view(msg.DEFAULT_VIEW)
            chat.insert_slash_command(cmd.name)
        else:
            # Complete as-is: dispatch like a submitted prompt. Handlers that
            # navigate call switch_view themselves, and switch_view no-ops
            # while another switch is in flight, so don't pre-switch to Chat.
            chat.run_command(cmd.name)

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

    def _set_model(self, attr: str, value: str) -> None:
        # Route through LilbeeApp.set_active_model so model-bar / scope chip
        # / status bar subscribers (settings_changed_signal) refresh.
        app = self._app
        app.set_active_model(attr, value)
        display = display_label_for_ref(value) or "off"
        app.notify(f"{attr}: {display}")
        if attr == "chat_model":
            app.title = msg.app_title(value)

    def _action_delete_document(self) -> None:
        """Jump to Chat with /delete prefilled; Tab there completes file names."""
        self._run_slash_command(get_command("/delete"))

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

    def _action_open_wiki(self) -> None:
        self._app.switch_view("Wiki")

    def _action_wikify(self) -> None:
        from lilbee.cli.tui.screens.wiki import start_wikify

        start_wikify(self._app)

    def _action_wipe_wiki(self) -> None:
        """Delete the generated wiki.

        The palette is the only TUI route to this while the wiki is off, since
        the wiki view (and its ``W`` binding) is dropped from the nav in that
        state, which is exactly when the leftover pages need removing.
        """
        from lilbee.cli.tui.screens.wiki import confirm_wiki_wipe

        confirm_wiki_wipe(self._app)

    def _action_reset(self) -> None:
        """Trigger /reset from the palette so the ConfirmDialog flow fires."""
        self._run_slash_command(get_command("/reset"))
