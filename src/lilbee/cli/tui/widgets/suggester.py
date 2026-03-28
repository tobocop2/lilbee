"""Tab completion for the chat input via Textual's Suggester API."""

from __future__ import annotations

from textual.suggester import Suggester

from lilbee.cli.tui.command_registry import command_names

_SLASH_COMMANDS = command_names()


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
        if cmd == "/vision":
            return self._suggest_from_list(value, partial, self._get_vision_names())
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

    def _get_model_names(self) -> list[str]:
        try:
            from lilbee.models import list_installed_models

            return list_installed_models()
        except Exception:
            return []

    def _get_vision_names(self) -> list[str]:
        names = ["off"]
        try:
            from lilbee.models import VISION_CATALOG

            names.extend(m.name for m in VISION_CATALOG)
        except Exception:
            pass
        return names

    def _get_setting_names(self) -> list[str]:
        from lilbee.cli.settings_map import SETTINGS_MAP

        return list(SETTINGS_MAP.keys())

    def _get_document_names(self) -> list[str]:
        try:
            from lilbee.services import get_services

            sources = get_services().store.get_sources()
            return [s.get("filename", s.get("source", "")) for s in sources]
        except Exception:
            return []

    def _get_theme_names(self) -> list[str]:
        from lilbee.cli.tui.app import DARK_THEMES

        return list(DARK_THEMES)
