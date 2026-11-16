"""First-run consent gate shared by launchers that write outside lilbee's dirs."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Protocol

import typer

from lilbee.core.config import cfg


class PlanPrinter(Protocol):
    """Prints the files a client's setup writes in one MCP mode."""

    def __call__(self, *, include_mcp: bool) -> None: ...


def _marker_path(marker_name: str) -> Path:
    """lilbee's record that a client's setup already ran (so launch doesn't re-prompt)."""
    return cfg.data_dir / "launchers" / marker_name


def _record_setup(marker_name: str) -> None:
    """Persist that the user accepted setup; idempotent (atomic write)."""
    path = _marker_path(marker_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(json.dumps({"accepted": True}).encode("utf-8"))
        os.replace(tmp_name, path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def _is_interactive() -> bool:
    """True when stdin is a TTY, so a confirmation prompt can be answered."""
    return sys.stdin.isatty()


def confirm_first_run_setup(
    *,
    marker_name: str,
    client_name: str,
    print_plan: Callable[[], None],
    assume_yes: bool,
    covered_by: tuple[str, ...] = (),
) -> bool:
    """Prompt before a client's first setup; True means proceed.

    Skipped when this setup or a wider one named in *covered_by* is already
    recorded, when *assume_yes* is set, or when stdin is
    not a TTY (scripts/CI: invoking the launch is the consent there). The
    choice is remembered so later launches don't re-prompt.
    """
    if any(_marker_path(name).exists() for name in (marker_name, *covered_by)):
        return True
    print_plan()
    if assume_yes or not _is_interactive():
        _record_setup(marker_name)
        return True
    if not typer.confirm(f"Proceed with {client_name} setup?", default=True):
        typer.secho(f"Skipped {client_name} setup.", fg=typer.colors.YELLOW)
        return False
    _record_setup(marker_name)
    return True


def confirm_mcp_mode_setup(
    *,
    mcp_marker_name: str,
    no_mcp_marker_name: str,
    client_name: str,
    print_plan: PlanPrinter,
    include_mcp: bool,
    assume_yes: bool,
) -> bool:
    """Ask before the first launch of this MCP mode's plan; MCP consent covers --no-mcp."""
    return confirm_first_run_setup(
        marker_name=mcp_marker_name if include_mcp else no_mcp_marker_name,
        client_name=client_name,
        print_plan=partial(print_plan, include_mcp=include_mcp),
        assume_yes=assume_yes,
        covered_by=() if include_mcp else (mcp_marker_name,),
    )
