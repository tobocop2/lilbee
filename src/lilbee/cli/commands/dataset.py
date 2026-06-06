"""Export and import the per-page text dataset."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NoReturn

import typer

from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.helpers import json_output
from lilbee.core.config import cfg

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
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {message}")
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
        summary = export_to_path(output, fmt, source)
    except DatasetError as exc:
        _fail(str(exc))

    if cfg.json_mode:
        json_output(summary.model_dump())
        return
    console.print(
        f"Wrote [{theme.LABEL}]{summary.pages}[/{theme.LABEL}] pages from "
        f"[{theme.LABEL}]{summary.sources}[/{theme.LABEL}] source(s) to "
        f"[{theme.ACCENT}]{output}[/{theme.ACCENT}]"
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
        f"Imported [{theme.LABEL}]{len(summary.sources)}[/{theme.LABEL}] source(s) "
        f"([{theme.LABEL}]{summary.pages}[/{theme.LABEL}] pages, "
        f"[{theme.LABEL}]{summary.chunks}[/{theme.LABEL}] chunks)"
    )
