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
        help_text="Switch chat model (no arg opens the catalog)",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/add",
        "_cmd_add",
        aliases=(),
        args_hint="<path>",
        help_text="Add file or folder to the knowledge base",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/crawl",
        "_cmd_crawl",
        aliases=(),
        args_hint="[url]",
        help_text="Crawl a URL (no arg opens the dialog)",
    ),
    SlashCommand(
        "/delete",
        "_cmd_delete",
        aliases=(),
        args_hint="<name>",
        help_text="Remove a document from the index",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/set",
        "_cmd_set",
        aliases=(),
        args_hint="<key> <value>",
        help_text="Change a setting",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/theme",
        "_cmd_theme",
        aliases=(),
        args_hint="[name]",
        help_text="Switch theme (no arg lists themes)",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/reset",
        "_cmd_reset",
        help_text="Factory reset (asks for confirmation)",
    ),
    SlashCommand(
        "/rebuild",
        "_cmd_rebuild",
        help_text="Re-index the documents directory from scratch",
    ),
    SlashCommand(
        "/export",
        "_cmd_export",
        aliases=(),
        args_hint="<path>",
        help_text="Export a per-page text dataset (parquet or jsonl)",
    ),
    SlashCommand(
        "/import",
        "_cmd_import",
        aliases=(),
        args_hint="<path>",
        help_text="Import a per-page text dataset, re-embedding it",
    ),
    SlashCommand("/status", "_cmd_status", help_text="Show knowledge-base status"),
    SlashCommand("/settings", "_cmd_settings", help_text="Open settings"),
    SlashCommand(
        "/models",
        "_cmd_catalog",
        aliases=("/m", "/catalog"),
        help_text="Browse the model catalog",
    ),
    SlashCommand("/setup", "_cmd_setup", help_text="Run the first-run setup wizard"),
    SlashCommand(
        "/remember",
        "_cmd_remember",
        args_hint="<text>",
        help_text="Save a memory (prefix with 'pref:' for a preference)",
    ),
    SlashCommand(
        "/memories",
        "_cmd_memories",
        help_text="Browse and manage your saved memories",
    ),
    SlashCommand(
        "/wiki",
        "_cmd_wiki",
        help_text="Open the wiki view",
    ),
    SlashCommand(
        "/remove",
        "_cmd_remove",
        aliases=(),
        args_hint="<name>",
        help_text="Uninstall a downloaded model",
        has_arg_completion=True,
    ),
    SlashCommand(
        "/login",
        "_cmd_login",
        args_hint="[token]",
        help_text="Log in to Hugging Face (no arg opens the token page)",
    ),
    SlashCommand("/help", "_cmd_help", aliases=("/h",), help_text="Show the slash-command catalog"),
    SlashCommand("/version", "_cmd_version", help_text="Show the lilbee version"),
    SlashCommand("/cancel", "_cmd_cancel", help_text="Cancel any in-flight operations"),
    SlashCommand("/clear", "_cmd_clear", help_text="Clear the conversation"),
    SlashCommand("/quit", "_cmd_quit", aliases=("/q", "/exit"), help_text="Exit lilbee"),
)


def build_dispatch_dict() -> dict[str, str]:
    """Build a mapping from command name (and aliases) to handler method name."""
    dispatch: dict[str, str] = {}
    for cmd in COMMANDS:
        dispatch[cmd.name] = cmd.handler
        for alias in cmd.aliases:
            dispatch[alias] = cmd.handler
    return dispatch


def completion_names() -> tuple[str, ...]:
    """All command names including aliases, for tab completion."""
    names: list[str] = []
    for cmd in COMMANDS:
        names.append(cmd.name)
        names.extend(cmd.aliases)
    return tuple(names)
