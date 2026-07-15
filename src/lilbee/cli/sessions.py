"""CLI for listing and managing saved chat sessions."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import typer
from rich.table import Table

from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.helpers import json_output
from lilbee.core.config import cfg
from lilbee.sessions import SessionStore, TitleSource

sessions_app = typer.Typer(
    name="sessions",
    help="List and manage saved chat sessions.",
    no_args_is_help=True,
)

_yes_option = typer.Option(False, "--yes", "-y", help="Skip the delete confirmation.")
_id_argument = typer.Argument(..., help="Session id, or a unique prefix of it.")


def _store() -> SessionStore:
    return SessionStore()


def _fail(message: str) -> NoReturn:
    if cfg.json_mode:
        json_output({"error": message})
    else:
        console.print(f"[{theme.ERROR}]{message}[/{theme.ERROR}]")
    raise typer.Exit(1)


def _resolve_id(prefix: str) -> str:
    """Resolve a full id or unique prefix to a session id, or exit 1."""
    matches = [meta.id for meta in _store().list() if meta.id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        _fail(f"No session matching {prefix!r}.")
    _fail(f"Prefix {prefix!r} is ambiguous ({len(matches)} sessions match).")


@sessions_app.command("list")
def list_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List saved conversations, newest first."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    metas = _store().list()
    if cfg.json_mode:
        json_output({"sessions": [asdict(meta) for meta in metas]})
        return
    if not metas:
        console.print("No saved sessions.")
        return
    table = Table(box=None, pad_edge=False)
    for column in ("ID", "Title", "Msgs", "Model", "Updated"):
        table.add_column(column, justify="right" if column == "Msgs" else "left")
    for meta in metas:
        table.add_row(
            meta.id[:8],
            meta.title,
            str(meta.message_count),
            meta.model_ref,
            meta.updated_at[:19],
        )
    console.print(table)


@sessions_app.command("show")
def show_cmd(
    session_id: str = _id_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print a saved conversation's transcript."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    session = _store().get(_resolve_id(session_id))
    if cfg.json_mode:
        json_output(
            {
                "meta": asdict(session.meta),
                "messages": [
                    {"role": m.role.value, "content": m.content, "sources": list(m.sources)}
                    for m in session.messages
                ],
            }
        )
        return
    console.print(f"[{theme.ACCENT}]{session.meta.title}[/{theme.ACCENT}]")
    for message in session.messages:
        console.print(f"[bold]{message.role.value}[/bold]: {message.content}")


@sessions_app.command("rename")
def rename_cmd(
    session_id: str = _id_argument,
    title: str = typer.Argument(..., help="The new title."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Rename a saved conversation."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    resolved = _resolve_id(session_id)
    _store().set_title(resolved, title, TitleSource.CUSTOM)
    if cfg.json_mode:
        json_output({"id": resolved, "title": title})
        return
    console.print(f"Renamed to [{theme.ACCENT}]{title}[/{theme.ACCENT}].")


@sessions_app.command("delete")
def delete_cmd(
    session_id: str = _id_argument,
    yes: bool = _yes_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Delete a saved conversation."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    resolved = _resolve_id(session_id)
    if (
        not yes
        and not cfg.json_mode
        and not typer.confirm(f"Delete {resolved[:8]}?", default=False)
    ):
        raise typer.Abort()
    _store().delete(resolved)
    if cfg.json_mode:
        json_output({"id": resolved, "deleted": True})
        return
    console.print(f"Deleted [{theme.ACCENT}]{resolved[:8]}[/{theme.ACCENT}].")
