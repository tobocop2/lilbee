"""Single source of truth for TUI slash commands.

Every slash command is defined once here. All other modules (chat dispatch,
suggester, help modal, autocomplete) read from this registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    """Definition of a single slash command."""

    name: str
    handler: str
    aliases: tuple[str, ...] = ()
    args_hint: str = ""
    help_text: str = ""
    has_arg_completion: bool = False


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand(
        "/model",
        "_cmd_model",
        aliases=(),
        args_hint="[name]",
        help_text="Switch model (no arg = catalog)",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/vision",
        "_cmd_vision",
        aliases=(),
        args_hint="[name]",
        help_text="Set vision model (/vision off)",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/add",
        "_cmd_add",
        aliases=(),
        args_hint="path",
        help_text="Add file to knowledge base",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/delete",
        "_cmd_delete",
        aliases=(),
        args_hint="name",
        help_text="Remove document from index",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/set",
        "_cmd_set",
        aliases=(),
        args_hint="key val",
        help_text="Change a setting",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/theme",
        "_cmd_theme",
        aliases=(),
        args_hint="name",
        help_text="Switch theme",
        has_arg_completion=True,
    ),
    SlashCommand("/reset", "_cmd_reset", args_hint="confirm", help_text="Factory reset"),
    SlashCommand("/status", "_cmd_status", help_text="Knowledge base status"),
    SlashCommand("/settings", "_cmd_settings", help_text="View/change settings"),
    SlashCommand("/models", "_cmd_catalog", aliases=("/m",), help_text="Browse models"),
    SlashCommand("/help", "_cmd_help", aliases=("/h",), help_text="Show help"),
    SlashCommand("/version", "_cmd_version", help_text="Show version"),
    SlashCommand("/cancel", "_cmd_cancel", help_text="Cancel active operations"),
    SlashCommand("/quit", "_cmd_quit", aliases=("/q", "/exit"), help_text="Exit"),
)


def build_dispatch_dict() -> dict[str, str]:
    """Build a mapping from command name (and aliases) to handler method name."""
    dispatch: dict[str, str] = {}
    for cmd in COMMANDS:
        dispatch[cmd.name] = cmd.handler
        for alias in cmd.aliases:
            dispatch[alias] = cmd.handler
    return dispatch


def command_names() -> tuple[str, ...]:
    """All primary command names (excludes aliases), for tab completion."""
    return tuple(cmd.name for cmd in COMMANDS)


def help_text() -> str:
    """Generate help text from the registry for the help modal."""
    lines: list[str] = []
    for cmd in COMMANDS:
        label = cmd.name
        if cmd.args_hint:
            label += f" {cmd.args_hint}"
        if cmd.aliases:
            label += f"  ({', '.join(cmd.aliases)})"
        lines.append(f"  {label:<25s} {cmd.help_text}")
    return "\n".join(lines)
