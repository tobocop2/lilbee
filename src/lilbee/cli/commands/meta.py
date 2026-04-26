"""Version, status, reset, and init commands."""

from __future__ import annotations

from pathlib import Path

import typer

from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.helpers import (
    gather_status,
    get_version,
    json_output,
    perform_reset,
    render_status,
)
from lilbee.config import cfg

_yes_option = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt.")


def version() -> None:
    """Show the lilbee version."""
    ver = get_version()
    if cfg.json_mode:
        json_output({"command": "version", "version": ver})
        return
    console.print(f"lilbee {ver}")


def status(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show indexed documents, paths, and chunk counts."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if cfg.json_mode:
        json_output(gather_status().model_dump(exclude_none=True))
        return
    render_status(console)


def reset(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    yes: bool = _yes_option,
) -> None:
    """Delete all documents and data (full factory reset)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not yes:
        if cfg.json_mode:
            json_output({"error": "Use --yes to confirm reset in JSON mode"})
            raise SystemExit(1)
        console.print(
            f"[{theme.ERROR_BOLD}]This will delete ALL documents and data.[/{theme.ERROR_BOLD}]\n"
            f"  Documents: {cfg.documents_dir}\n"
            f"  Data:      {cfg.data_dir}"
        )
        confirmed = typer.confirm("Are you sure?", default=False)
        if not confirmed:
            console.print("Aborted.")
            raise SystemExit(0)

    result = perform_reset()

    if cfg.json_mode:
        json_output(result.model_dump())
        return

    console.print(
        f"Reset complete: {result.deleted_docs} document(s), "
        f"{result.deleted_data} data item(s) deleted."
    )
    if result.skipped:
        console.print(
            f"[{theme.WARNING}]{len(result.skipped)} item(s) could not be deleted "
            f"(locked or permission denied).[/{theme.WARNING}]"
        )


def init() -> None:
    """Initialize a local .lilbee/ knowledge base in the current directory."""
    root = Path.cwd() / ".lilbee"
    if root.is_dir():
        if cfg.json_mode:
            json_output({"command": "init", "path": str(root), "created": False})
            return
        console.print(f"Already initialized: {root}")
        return

    docs = root / "documents"
    data = root / "data"
    docs.mkdir(parents=True)
    data.mkdir(parents=True)
    (root / ".gitignore").write_text("data/\n")

    if cfg.json_mode:
        json_output({"command": "init", "path": str(root), "created": True})
        return
    console.print(f"Initialized local knowledge base at {root}")
