"""`lilbee preset` sub-app: list, add, edit and remove server presets."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import typer
from pydantic import BaseModel
from rich.table import Table
from rich.text import Text

from lilbee.app.presets import (
    ServerPreset,
    add_preset,
    edit_preset,
    failure_text,
    list_presets,
    remove_preset,
    split_roles,
)
from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.helpers import json_output
from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole
from lilbee.providers.server_registry import RerankWire
from lilbee.server.models import PresetListResponse, PresetRemovedResponse, PresetResponse

preset_app = typer.Typer(
    name="preset",
    help="List, add, edit and remove server presets.",
    no_args_is_help=True,
)

_T = TypeVar("_T")
_NO_VALUE = "-"
_roles_help = "Comma-separated roles: embed, rerank, chat, vision."
_wire_help = "Rerank request shape: v1 or tei."
_key_stdin_help = "Read the API key from one line of stdin, or ask for it on a terminal."


def _read_key() -> str:
    """The API key from a hidden prompt on a terminal, else one line of stdin."""
    if sys.stdin.isatty():
        return str(typer.prompt("API key", hide_input=True, default="", show_default=False))
    return sys.stdin.readline().rstrip("\r\n")


def _guard(action: Callable[[], _T]) -> _T:
    """Run a service call; a refusal prints the service's text and exits 1."""
    try:
        return action()
    except (ValueError, OSError) as exc:
        text = failure_text(exc)
        if cfg.json_mode:
            json_output({"error": text})
        else:
            console.print(Text(text, style=theme.ERROR), soft_wrap=True)
        raise typer.Exit(code=1) from exc


def _emit(model: BaseModel, human: Callable[[], None]) -> None:
    """Print *model* as JSON in JSON mode, else run *human*."""
    if cfg.json_mode:
        json_output(model.model_dump(mode="json"))
    else:
        human()


def _wire_cell(preset: ServerPreset) -> str:
    if preset.rerank_wire is None or WorkerRole.RERANK not in preset.roles:
        return _NO_VALUE
    return preset.rerank_wire.value


def _render_list(presets: list[ServerPreset]) -> None:
    if not presets:
        console.print("No server presets.", style=theme.MUTED)
        return
    table = Table(title="Server presets")
    for column in ("Name", "URL", "Roles", "Wire", "Key"):
        table.add_column(column)
    for preset in presets:
        table.add_row(
            Text(preset.name),
            Text(preset.base_url or "(invalid)"),
            Text(", ".join(role.value for role in preset.roles) or _NO_VALUE),
            _wire_cell(preset),
            "set" if preset.has_key else "none",
        )
    console.print(table)


@preset_app.command("list")
def list_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List the server presets."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    presets = _guard(list_presets)
    _emit(
        PresetListResponse(presets=[PresetResponse.from_view(p) for p in presets]),
        lambda: _render_list(presets),
    )


@preset_app.command("add")
def add_cmd(
    name: str = typer.Argument(..., help="Preset name."),
    url: str = typer.Argument(..., help="Server base URL."),
    roles: str = typer.Option(..., "--roles", help=_roles_help),
    wire: str = typer.Option(RerankWire.V1.value, "--wire", help=_wire_help),
    key_stdin: bool = typer.Option(False, "--key-stdin", help=_key_stdin_help),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Add a server preset."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    api_key = _read_key() if key_stdin else None
    preset = _guard(
        lambda: add_preset(name, url, roles=split_roles(roles), api_key=api_key, rerank_wire=wire)
    )
    _emit(
        PresetResponse.from_view(preset),
        lambda: console.print(Text(f"Added server preset {name!r}.")),
    )


@preset_app.command("edit")
def edit_cmd(
    name: str = typer.Argument(..., help="Preset name."),
    url: str | None = typer.Option(None, "--url", help="New server base URL."),
    roles: str | None = typer.Option(None, "--roles", help=_roles_help),
    wire: str | None = typer.Option(None, "--wire", help=_wire_help),
    key_stdin: bool = typer.Option(False, "--key-stdin", help=_key_stdin_help),
    clear_key: bool = typer.Option(False, "--clear-key", help="Remove the stored API key."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Change a server preset. Give only what changes."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    api_key = _read_key() if key_stdin else None
    preset = _guard(
        lambda: edit_preset(
            name,
            url=url,
            roles=None if roles is None else split_roles(roles),
            api_key=api_key,
            rerank_wire=wire,
            clear_key=clear_key,
        )
    )
    _emit(
        PresetResponse.from_view(preset),
        lambda: console.print(Text(f"Updated server preset {name!r}.")),
    )


@preset_app.command("remove")
def remove_cmd(
    name: str = typer.Argument(..., help="Preset name."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Remove a server preset."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not (yes or cfg.json_mode or typer.confirm(f"Remove {name}?", default=False)):
        console.print("Aborted.")
        raise typer.Exit(0)
    _guard(lambda: remove_preset(name))
    _emit(
        PresetRemovedResponse(removed=name),
        lambda: console.print(Text(f"Removed server preset {name!r}.")),
    )
