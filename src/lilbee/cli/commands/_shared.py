"""Constants and small rendering helpers shared across the CLI command submodules."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import typer
from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from rich.text import Text

from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console
from lilbee.cli.helpers import json_output, print_prefixed
from lilbee.core.config import cfg

CHUNK_PREVIEW_LEN = 80  # characters shown in human-readable search output

REBUILD_HINT = "Run lilbee rebuild so the index uses the new values."

T = TypeVar("T")


def shown_value(value: Any) -> str:
    """A setting value the way a profile or config file writes it: "auto", true, 512."""
    return json.dumps(to_jsonable_python(value))


def emit(model: BaseModel, render: Callable[[], None]) -> None:
    """Print *model* as JSON in --json mode, else run *render*."""
    if cfg.json_mode:
        json_output(model.model_dump(mode="json"))
    else:
        render()


def setup(data_dir: Path | None, use_global: bool) -> None:
    """Apply --data-dir/--global overrides before a profile or settings command runs."""
    apply_overrides(data_dir=data_dir, use_global=use_global)


def fail(message: str) -> NoReturn:
    """Print *message* as the command's error, as JSON or text, and exit 1."""
    if cfg.json_mode:
        json_output({"error": message})
    else:
        print_prefixed(console, "Error: ", message, style=theme.ERROR)
    raise typer.Exit(1)


def line(text: str, style: str | None = None) -> None:
    """Print one line of plain text, soft-wrapped."""
    console.print(Text(text, style=style or ""), soft_wrap=True)


def run_or_fail(operation: Callable[[], T], os_error_message: Callable[[OSError], str]) -> T:
    """Run *operation*; a refusal prints its reason, as JSON or text, and exits 1."""
    try:
        return operation()
    except (ValueError, KeyError) as exc:
        fail(str(exc))
    except OSError as exc:
        fail(os_error_message(exc))
