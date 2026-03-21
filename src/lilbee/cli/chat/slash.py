"""Slash command handlers and dispatch for chat mode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lilbee import settings
from lilbee.cli import theme
from lilbee.cli.chat.complete import list_ollama_models
from lilbee.cli.chat.sync import SyncStatus
from lilbee.cli.helpers import add_paths, get_version, perform_reset, render_status
from lilbee.config import cfg


class RenderStyle(StrEnum):
    """How a setting is displayed in /settings."""

    COMPACT = "compact"
    FULL = "full"


@dataclass(frozen=True)
class _SettingDef:
    """Metadata for an interactive setting."""

    cfg_attr: str
    type: type
    nullable: bool
    render: RenderStyle = field(default=RenderStyle.COMPACT)


_SETTINGS_MAP: dict[str, _SettingDef] = {
    "chat_model": _SettingDef("chat_model", str, nullable=False),
    "vision_model": _SettingDef("vision_model", str, nullable=True),
    "embedding_model": _SettingDef("embedding_model", str, nullable=False),
    "top_k": _SettingDef("top_k", int, nullable=False),
    "temperature": _SettingDef("temperature", float, nullable=True),
    "top_p": _SettingDef("top_p", float, nullable=True),
    "top_k_sampling": _SettingDef("top_k_sampling", int, nullable=True),
    "repeat_penalty": _SettingDef("repeat_penalty", float, nullable=True),
    "num_ctx": _SettingDef("num_ctx", int, nullable=True),
    "seed": _SettingDef("seed", int, nullable=True),
    "system_prompt": _SettingDef("system_prompt", str, nullable=False, render=RenderStyle.FULL),
}


class QuitChat(Exception):
    """Raised by /quit to exit the chat loop."""


def _pick_from_catalog(
    catalog: tuple,
    display_fn: Callable,
    con: Console,
    config_attr: str,
    setting_key: str,
    label: str,
) -> None:
    """Shared interactive picker flow for model/vision catalog selection.

    *display_fn* is called with (ram_gb, free_disk_gb) and returns the recommended ModelInfo.
    *config_attr* is the cfg attribute to set (e.g. "chat_model" or "vision_model").
    *setting_key* is the settings.toml key (e.g. "chat_model" or "vision_model").
    *label* is a human-readable label for messages (e.g. "Switched to model").
    """
    from lilbee.models import (
        get_free_disk_gb,
        get_system_ram_gb,
        pull_with_progress,
        validate_disk_and_pull,
    )

    ram_gb = get_system_ram_gb()
    free_disk_gb = get_free_disk_gb(cfg.data_dir)
    recommended = display_fn(ram_gb, free_disk_gb, console=con)
    default_idx = list(catalog).index(recommended) + 1
    installed = set(list_ollama_models())

    try:
        raw = input(f"Choice [{default_idx}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not raw:
        return

    try:
        choice = int(raw)
    except ValueError:
        con.print(f"[{theme.ERROR}]Enter a number 1-{len(catalog)}.[/{theme.ERROR}]")
        return

    if not (1 <= choice <= len(catalog)):
        con.print(f"[{theme.ERROR}]Enter a number 1-{len(catalog)}.[/{theme.ERROR}]")
        return

    model_info = catalog[choice - 1]
    if model_info.name not in installed:
        if config_attr == "chat_model":
            validate_disk_and_pull(model_info, free_disk_gb, console=con)
        else:
            pull_with_progress(model_info.name, console=con)
    setattr(cfg, config_attr, model_info.name)
    settings.set_value(cfg.data_root, setting_key, model_info.name)
    con.print(f"{label} [{theme.LABEL}]{model_info.name}[/{theme.LABEL}] (saved)")


def _set_named_model(
    name: str,
    con: Console,
    config_attr: str,
    setting_key: str,
    label: str,
    *,
    exclude_vision: bool = False,
) -> None:
    """Validate and set a named model directly (no picker)."""
    from lilbee.models import ensure_tag, pull_with_progress

    name = ensure_tag(name)
    available = list_ollama_models(exclude_vision=exclude_vision)
    if available and name not in available:
        try:
            answer = con.input(
                f"[{theme.LABEL}]{name}[/{theme.LABEL}] not installed. Download? (y/n) "
            )
        except (EOFError, KeyboardInterrupt):
            return
        if answer.strip().lower() not in ("y", "yes"):
            return
        pull_with_progress(name, console=con)
    setattr(cfg, config_attr, name)
    settings.set_value(cfg.data_root, setting_key, name)
    con.print(f"{label} [{theme.LABEL}]{name}[/{theme.LABEL}] (saved)")


def handle_slash_status(args: str, con: Console) -> None:
    render_status(con)


def handle_slash_add(args: str, con: Console, *, sync_status: SyncStatus | None = None) -> None:
    raw = args.strip()
    if not raw:
        try:
            from prompt_toolkit import prompt as pt_prompt
            from prompt_toolkit.completion import PathCompleter

            raw = pt_prompt("path: ", completer=PathCompleter(expanduser=True)).strip()
        except (ImportError, EOFError, KeyboardInterrupt):
            return
    if not raw:
        return
    p = Path(raw).expanduser()
    if not p.exists():
        print(f"Path not found: {raw}")
        return
    add_paths([p], con, force=True, background=True, chat_mode=True, sync_status=sync_status)


def handle_slash_quit(args: str, con: Console) -> None:
    raise QuitChat


def handle_slash_model(args: str, con: Console) -> None:
    name = args.strip()
    if name:
        _set_named_model(
            name, con, "chat_model", "chat_model", "Switched to model", exclude_vision=True
        )
        return

    con.print(f"[{theme.LABEL}]Current model:[/{theme.LABEL}] {cfg.chat_model}\n")
    from lilbee.models import MODEL_CATALOG, display_model_picker

    _pick_from_catalog(
        MODEL_CATALOG,
        display_model_picker,
        con,
        config_attr="chat_model",
        setting_key="chat_model",
        label="Switched to model",
    )


def handle_slash_vision(args: str, con: Console) -> None:
    name = args.strip()

    if name == "off":
        cfg.vision_model = ""
        settings.set_value(cfg.data_root, "vision_model", "")
        con.print(f"Vision OCR [{theme.LABEL}]disabled[/{theme.LABEL}] (saved)")
        return

    if name:
        _set_named_model(name, con, "vision_model", "vision_model", "Vision model set to")
        return

    # bare /vision — show status then picker
    if cfg.vision_model:
        con.print(f"[{theme.LABEL}]Vision OCR:[/{theme.LABEL}] {cfg.vision_model}\n")
    else:
        con.print(f"[{theme.LABEL}]Vision OCR:[/{theme.LABEL}] disabled\n")

    from lilbee.models import VISION_CATALOG, display_vision_picker

    _pick_from_catalog(
        VISION_CATALOG,
        display_vision_picker,
        con,
        config_attr="vision_model",
        setting_key="vision_model",
        label="Vision model set to",
    )


def handle_slash_version(args: str, con: Console) -> None:
    con.print(f"lilbee [{theme.LABEL}]{get_version()}[/{theme.LABEL}]")


def handle_slash_reset(args: str, con: Console, *, sync_status: SyncStatus | None = None) -> None:
    con.print(
        f"[{theme.ERROR_BOLD}]This will delete ALL documents and data.[/{theme.ERROR_BOLD}]"
        "\nType 'yes' to confirm:"
    )
    try:
        answer = con.input(f"[{theme.ERROR_BOLD}]> [/{theme.ERROR_BOLD}]")
    except (EOFError, KeyboardInterrupt):
        con.print("Aborted.")
        return
    if answer.strip().lower() != "yes":
        con.print("Aborted.")
        return
    result = perform_reset()
    if sync_status is not None:
        sync_status.clear()
    con.print(
        f"Reset complete: {result.deleted_docs} document(s), "
        f"{result.deleted_data} data item(s) deleted."
    )


def _get_model_defaults() -> dict[str, str]:
    """Fetch generation parameter defaults from Ollama for the current chat model."""
    _OLLAMA_TO_SETTING = {"top_k": "top_k_sampling"}
    try:
        import ollama

        resp = ollama.show(cfg.chat_model)
        defaults: dict[str, str] = {}
        for line in (resp.parameters or "").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                key = _OLLAMA_TO_SETTING.get(parts[0], parts[0])
                if key in _SETTINGS_MAP:
                    defaults[key] = parts[1]
        return defaults
    except (ollama.ResponseError, ConnectionError, OSError):
        return {}


def _format_setting_value(value: object, model_default: str | None = None) -> str:
    """Format a setting value for display."""
    if value is None or value == "":
        if model_default is not None:
            return f"[{theme.MUTED}](model default: {model_default})[/{theme.MUTED}]"
        return f"[{theme.MUTED}](not set)[/{theme.MUTED}]"
    if isinstance(value, str) and len(value) > 60:
        return f"[{theme.MUTED}]({len(value)} chars)[/{theme.MUTED}]"
    return str(value)


def handle_slash_settings(args: str, con: Console) -> None:
    defaults = _get_model_defaults()
    table = Table(show_header=False, box=None, padding=(0, 2))
    for name, defn in _SETTINGS_MAP.items():
        value = getattr(cfg, defn.cfg_attr)
        if defn.render == RenderStyle.FULL:
            con.print(
                Panel(
                    value,
                    title=f"[{theme.LABEL}]{name}[/{theme.LABEL}]",
                    border_style=theme.MUTED,
                    expand=False,
                )
            )
        else:
            table.add_row(
                f"[{theme.LABEL}]{name}[/{theme.LABEL}]",
                _format_setting_value(value, defaults.get(name)),
            )
    con.print(table)


def _validate_setting(cfg_attr: str, raw_value: str, typ: type, con: Console) -> Any:
    """Coerce *raw_value* to *typ* and assign to *cfg_attr* on cfg.

    Returns the parsed value on success, or None on failure (error printed to *con*).
    """
    try:
        parsed = typ(raw_value)
    except (ValueError, TypeError):
        con.print(f"[{theme.ERROR}]Invalid {typ.__name__}:[/{theme.ERROR}] {raw_value}")
        return None

    try:
        setattr(cfg, cfg_attr, parsed)
    except ValidationError as exc:
        msg = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        con.print(f"[{theme.ERROR}]{cfg_attr}: {msg}[/{theme.ERROR}]")
        return None

    return parsed


def handle_slash_set(args: str, con: Console) -> None:
    parts = args.strip().split(None, 1)
    if not parts:
        con.print("Usage: /set <param> [value]  — try /settings to see all params")
        return

    name = parts[0]
    defn = _SETTINGS_MAP.get(name)
    if defn is None:
        con.print(f"[{theme.ERROR}]Unknown setting:[/{theme.ERROR}] {name}")
        con.print(f"Available: {', '.join(_SETTINGS_MAP)}")
        return

    if len(parts) == 1:
        value = getattr(cfg, defn.cfg_attr)
        con.print(f"{name} = {_format_setting_value(value)}")
        return

    raw_value = parts[1]

    if raw_value.lower() in ("off", "none", "default"):
        if not defn.nullable:
            con.print(f"[{theme.ERROR}]{name} cannot be cleared[/{theme.ERROR}]")
            return
        setattr(cfg, defn.cfg_attr, "" if defn.type is str else None)
        settings.delete_value(cfg.data_root, name)
        con.print(f"{name} cleared (saved)")
        return

    parsed = _validate_setting(defn.cfg_attr, raw_value, defn.type, con)
    if parsed is None:
        return

    settings.set_value(cfg.data_root, name, str(parsed))
    con.print(f"{name} = {parsed} (saved)")


def handle_slash_help(args: str, con: Console) -> None:
    con.print(f"[{theme.LABEL}]Slash commands:[/{theme.LABEL}]")
    con.print("  /status  — show indexed documents and config")
    con.print("  /add [path]  — add a file or directory (tab-completes without args)")
    con.print("  /model [name]  — show or switch chat model")
    con.print("  /vision [name|off]  — show or switch vision OCR model")
    con.print("  /settings  — show all generation settings")
    con.print("  /set <param> [value]  — change a setting (tab-completes param names)")
    con.print("  /version — show lilbee version")
    con.print("  /reset   — delete all documents and data")
    con.print("  /help    — show this help")
    con.print("  /quit    — exit chat")


_SLASH_COMMANDS: dict[str, Callable[[str, Console], None]] = {
    "status": handle_slash_status,
    "add": handle_slash_add,
    "model": handle_slash_model,
    "vision": handle_slash_vision,
    "settings": handle_slash_settings,
    "set": handle_slash_set,
    "version": handle_slash_version,
    "reset": handle_slash_reset,
    "help": handle_slash_help,
    "quit": handle_slash_quit,
}


def dispatch_slash(raw_input: str, con: Console, *, sync_status: SyncStatus | None = None) -> bool:
    """Try to dispatch *raw_input* as a ``/command``.  Returns True if handled."""
    stripped = raw_input.strip()
    if not stripped.startswith("/"):
        return False
    parts = stripped[1:].split(None, 1)
    cmd = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    handler = _SLASH_COMMANDS.get(cmd)
    if handler is None:
        con.print(f"[{theme.ERROR}]Unknown command:[/{theme.ERROR}] /{cmd}  — try /help")
        return True
    if cmd == "add":
        handle_slash_add(args, con, sync_status=sync_status)
    elif cmd == "reset":
        handle_slash_reset(args, con, sync_status=sync_status)
    else:
        handler(args, con)
    return True
