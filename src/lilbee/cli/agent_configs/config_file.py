"""Load and atomically write an agent's on-disk config, refusing to clobber a corrupt file."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from lilbee.core.security import write_private_text


def load_config_dict(
    path: Path,
    *,
    parse: Callable[[str], Any],
    parse_error: type[Exception],
    label: str,
) -> dict[str, Any]:
    """Return the parsed mapping at *path*, or ``{}`` when absent or empty.

    Exits non-zero without writing when the file does not parse, so a corrupt
    user config is never overwritten."""
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = parse(raw)
    except parse_error as exc:
        typer.secho(
            f"Your {label} did not parse, so lilbee will not overwrite it. "
            "Fix or remove it, then retry.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1) from exc
    return parsed if isinstance(parsed, dict) else {}


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (temp file + os.replace), creating parents.

    Agent configs carry the lilbee bearer token, so they get the same
    owner-only treatment as the other secret files. This already wrote through
    a temp file, which is created 0600 and keeps that mode across the replace;
    sharing the one implementation just makes that explicit.
    """
    write_private_text(path, text)
