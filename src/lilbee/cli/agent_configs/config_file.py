"""Load and atomically write an agent's on-disk config, refusing to clobber a corrupt file."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer


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
    """Write *text* to *path* atomically (temp file + os.replace), creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise
