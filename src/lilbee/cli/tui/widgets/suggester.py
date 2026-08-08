"""Tab completion for the chat input via Textual's Suggester API."""

from __future__ import annotations

from textual.suggester import Suggester

from lilbee.cli.tui.command_registry import completion_names
from lilbee.cli.tui.widgets.autocomplete import (
    _fetch_document_names,
    _model_options,
    _setting_options,
    _theme_options,
)

_SLASH_COMMANDS = completion_names()


class SlashSuggester(Suggester):
    """Context-aware suggestions for the chat input.
    Suggests slash command names when input starts with '/'.
    Suggests argument values for commands that take them.
    """

    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None

        if value.startswith("/") and " " not in value:
            return self._suggest_command(value)

        if " " in value:
            return self._suggest_argument(value)

        return None

    def _suggest_command(self, prefix: str) -> str | None:
        for cmd in _SLASH_COMMANDS:
            if cmd.startswith(prefix) and cmd != prefix:
                return cmd
        return None

    def _suggest_argument(self, value: str) -> str | None:
        cmd, _, partial = value.partition(" ")
        cmd = cmd.lower()

        if cmd == "/model":
            return self._suggest_from_list(value, partial, self._get_model_names())
        if cmd == "/set":
            return self._suggest_from_list(value, partial, self._get_setting_names())
        if cmd == "/delete":
            return self._suggest_from_list(value, partial, self._get_document_names())
        if cmd == "/theme":
            return self._suggest_from_list(value, partial, self._get_theme_names())
        return None

    def _suggest_from_list(self, full: str, partial: str, options: list[str]) -> str | None:
        for opt in options:
            if opt.startswith(partial) and opt != partial:
                return full[: len(full) - len(partial)] + opt
        return None

    # Option sources are shared with the completion overlay (autocomplete);
    # these stay as methods so callers and tests can override per-suggester.
    def _get_model_names(self) -> list[str]:
        return _model_options()

    def _get_setting_names(self) -> list[str]:
        return _setting_options()

    def _get_document_names(self) -> list[str]:
        return _fetch_document_names()

    def _get_theme_names(self) -> list[str]:
        return _theme_options()
