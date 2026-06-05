"""Export and import the per-page text dataset."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NoReturn

import typer

from lilbee.app.services import get_services
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
    from lilbee.data.export import build_page_dataset, resolve_format, write_dataset

    try:
        dataset_format = resolve_format(fmt, output)
    except ValueError as exc:
        _fail(str(exc))

    store = get_services().store
    if source is not None and source not in {s["filename"] for s in store.get_sources()}:
        _fail(f"Source not found: {source}")

    rows = build_page_dataset(store, source)
    if not rows:
        _fail("Nothing to export: the store has no indexed pages.")

    write_dataset(rows, output, dataset_format)
    pages = len(rows)
    sources = len({row["source"] for row in rows})
    if cfg.json_mode:
        json_output(
            {
                "command": "export",
                "format": str(dataset_format),
                "output": str(output),
                "pages": pages,
                "sources": sources,
            }
        )
        return
    console.print(
        f"Wrote [{theme.LABEL}]{pages}[/{theme.LABEL}] pages from "
        f"[{theme.LABEL}]{sources}[/{theme.LABEL}] source(s) to "
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
    from lilbee.data.export import import_dataset, load_page_dataset, resolve_format
    from lilbee.data.store import EmbeddingModelMismatchError

    try:
        dataset_format = resolve_format(fmt, dataset)
        rows = load_page_dataset(dataset, dataset_format)
    except ValueError as exc:
        _fail(str(exc))
    if not rows:
        _fail("Dataset has no pages to import.")

    store = get_services().store
    try:
        result = asyncio.run(import_dataset(store, rows))
    except EmbeddingModelMismatchError as exc:
        _fail(str(exc))

    if cfg.json_mode:
        json_output(
            {
                "command": "import",
                "sources": result.sources,
                "pages": result.pages,
                "chunks": result.chunks,
            }
        )
        return
    console.print(
        f"Imported [{theme.LABEL}]{len(result.sources)}[/{theme.LABEL}] source(s) "
        f"([{theme.LABEL}]{result.pages}[/{theme.LABEL}] pages, "
        f"[{theme.LABEL}]{result.chunks}[/{theme.LABEL}] chunks)"
    )
