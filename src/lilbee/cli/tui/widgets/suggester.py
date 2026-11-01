"""Tab completion for the chat input via Textual's Suggester API."""

from __future__ import annotations

from textual.suggester import Suggester

from lilbee.cli.tui.command_registry import completion_names
from lilbee.cli.tui.widgets.autocomplete import ARG_SOURCES, PATH_ARG_COMMANDS

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
        source = ARG_SOURCES.get(cmd)
        if source is None or cmd in PATH_ARG_COMMANDS:
            return None
        return self._suggest_from_list(value, partial, source())

    def _suggest_from_list(self, full: str, partial: str, options: list[str]) -> str | None:
        for opt in options:
            if opt.startswith(partial) and opt != partial:
                return full[: len(full) - len(partial)] + opt
        return None
