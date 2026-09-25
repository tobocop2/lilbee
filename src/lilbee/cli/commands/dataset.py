"""Export and import the per-page text dataset."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NoReturn

import typer
from rich.text import Text

from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.helpers import json_output, print_prefixed, sigint_cancel
from lilbee.core.config import cfg
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.runtime.console import styled

_export_output_argument = typer.Argument(
    Path("pages.parquet"),
    help="Output file (suffix sets the format unless --format is given).",
)
_import_dataset_argument = typer.Argument(
    ...,
    help="Dataset file to import (parquet or jsonl).",
)
_format_option = typer.Option(
    "",
    "--format",
    help="Dataset format: parquet or jsonl. Inferred from the file suffix when omitted.",
)
_export_source_option = typer.Option(
    None,
    "--source",
    help="Export only this source (default: every source).",
)


def _fail(message: str) -> NoReturn:
    """Emit *message* as an error in the active output mode and exit non-zero."""
    if cfg.json_mode:
        json_output({"error": message})
    else:
        print_prefixed(console, "Error: ", message, style=theme.ERROR)
    raise SystemExit(1)


def export_cmd(
    output: Path = _export_output_argument,
    fmt: str = _format_option,
    source: str | None = _export_source_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Write a per-page {source, page, text} dataset (drops vectors)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.app.dataset import DatasetError, export_to_path

    try:
        with sigint_cancel() as cancel:
            summary = export_to_path(output, fmt, source, cancel=cancel)
    except TaskCancelledError:
        _fail("Export cancelled; the partial file was removed.")
    except DatasetError as exc:
        _fail(str(exc))

    if cfg.json_mode:
        json_output(summary.model_dump())
        return
    console.print(
        Text.assemble(
            "Wrote ",
            (str(summary.pages), theme.LABEL),
            " pages from ",
            (str(summary.sources), theme.LABEL),
            " source(s) to ",
            (str(output), theme.ACCENT),
        ),
        soft_wrap=True,
    )


def import_cmd(
    dataset: Path = _import_dataset_argument,
    fmt: str = _format_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Import a per-page text dataset, re-embedding it with the current model."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.app.dataset import DatasetError, import_from_path

    try:
        summary = asyncio.run(import_from_path(dataset, fmt))
    except DatasetError as exc:
        _fail(str(exc))

    if cfg.json_mode:
        json_output(summary.model_dump())
        return
    console.print(
        styled(
            "Imported ",
            (str(len(summary.sources)), theme.LABEL),
            " source(s) (",
            (str(summary.pages), theme.LABEL),
            " pages, ",
            (str(summary.chunks), theme.LABEL),
            " chunks)",
        )
    )
