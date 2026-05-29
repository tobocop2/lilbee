"""`lilbee model` sub-app: list/show/pull/rm/browse for installed models.

Thin Typer wrapper around the surface-agnostic use-cases in
:mod:`lilbee.app.models`. Bare result models live in ``app.models``;
the Rich renderers below adapt them for human-readable terminal output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from lilbee.app.models import (
    ListModelsResult,
    PullEvent,
    PullProgressEvent,
    PullResult,
    PullStatus,
    ShowModelResult,
    list_models_data,
    pull_model_data,
    remove_model_data,
    show_model_data,
)
from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.helpers import json_output
from lilbee.core.config import cfg

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.catalog import DownloadProgress
    from lilbee.catalog.types import ModelSource


def _render_list(data: ListModelsResult) -> Table:
    table = Table(title="Installed models")
    table.add_column("Name", style=theme.ACCENT)
    table.add_column("Source", style=theme.MUTED)
    table.add_column("Task")
    table.add_column("Size", justify="right")
    for entry in data.models:
        size = f"{entry.size_gb:.2f} GB" if entry.size_gb is not None else ""
        table.add_row(entry.name, entry.source, entry.task or "", size)
    return table


def _render_show(data: ShowModelResult) -> str:
    lines = [f"[{theme.ACCENT}]{data.model}[/{theme.ACCENT}]"]
    if data.catalog is not None:
        lines.extend(
            [
                f"  display_name: {data.catalog.display_name}",
                f"  task:         {data.catalog.task}",
                f"  size_gb:      {data.catalog.size_gb}",
                f"  min_ram_gb:   {data.catalog.min_ram_gb}",
                f"  hf_repo:      {data.catalog.hf_repo}",
                f"  description:  {data.catalog.description}",
            ]
        )
    lines.append(f"  installed:    {data.installed}")
    if data.source:
        lines.append(f"  source:       {data.source}")
    if data.path:
        lines.append(f"  path:         {data.path}")
    if data.manifest is not None:
        lines.append(f"  downloaded:   {data.manifest.downloaded_at}")
    return "\n".join(lines)


model_app = typer.Typer(
    name="model",
    help="Manage installed and available models (pull / list / show / rm / browse).",
    no_args_is_help=True,
)

_source_option = typer.Option(
    None,
    "--source",
    "-s",
    help="Filter by source: native, remote, ollama, lm_studio, or frontier (default: all).",
)
_task_option = typer.Option(
    None,
    "--task",
    "-t",
    help="Filter by task: 'chat', 'embedding', 'vision', or 'rerank'.",
)
_yes_option = typer.Option(
    False,
    "--yes",
    "-y",
    help="Skip confirmation prompt.",
)


def _parse_source_or_bad_param(value: str | None) -> ModelSource | None:
    """Parse a CLI --source value, raising typer.BadParameter on bad input."""
    from lilbee.catalog.types import ModelSource

    try:
        return ModelSource.parse(value)
    except ValueError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        raise typer.BadParameter(str(exc)) from exc


@model_app.command("list")
def list_cmd(
    source: str | None = _source_option,
    task: str | None = _task_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List installed models across all sources."""
    from lilbee.catalog.types import ModelTask

    apply_overrides(data_dir=data_dir, use_global=use_global)
    try:
        parsed_task = ModelTask(task) if task else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    data = list_models_data(source=_parse_source_or_bad_param(source), task=parsed_task)
    if cfg.json_mode:
        json_output(data.model_dump())
        return
    if not data.models:
        console.print("No models installed.")
        return
    console.print(_render_list(data))


@model_app.command("show")
def show_cmd(
    ref: str = typer.Argument(..., help="Model ref (e.g. 'Qwen/Qwen3-0.6B-GGUF')."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show catalog and installed metadata for a model."""
    from lilbee.modelhub.model_manager import ModelNotFoundError

    apply_overrides(data_dir=data_dir, use_global=use_global)
    try:
        data = show_model_data(ref)
    except ModelNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None
    if cfg.json_mode:
        json_output(data.model_dump())
        return
    console.print(_render_show(data))


def _run_pull(
    ref: str,
    src: ModelSource,
    on_update: Callable[[DownloadProgress], None],
    *,
    allow_unsupported: bool = False,
) -> PullResult:
    """Invoke ``pull_model_data`` and translate known errors to typer.Exit."""
    from lilbee.catalog.compat import UnsupportedArchError

    try:
        return pull_model_data(ref, src, on_update=on_update, allow_unsupported=allow_unsupported)
    except UnsupportedArchError as exc:
        msg = (
            f"Architecture {exc.architecture!r} is not supported by this lilbee build.\n"
            "Pass --allow-unsupported to try anyway."
        )
        if cfg.json_mode:
            json_output(
                {
                    "error": "unsupported_arch",
                    "arch": exc.architecture,
                    "ref": exc.ref,
                }
            )
        else:
            console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {msg}")
        raise typer.Exit(1) from None
    except (RuntimeError, PermissionError) as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise typer.Exit(1) from None


def _pull_json_stream(ref: str, src: ModelSource, *, allow_unsupported: bool) -> None:
    """Emit newline-delimited JSON progress events, then the final result."""

    def on_update(p: DownloadProgress) -> None:
        event = PullProgressEvent(
            model=ref, percent=p.percent, detail=p.detail, cache_hit=p.is_cache_hit
        )
        json_output(event.model_dump())

    final = _run_pull(ref, src, on_update, allow_unsupported=allow_unsupported)
    json_output({**final.model_dump(), "event": PullEvent.DONE.value})


def _pull_interactive_progress(ref: str, src: ModelSource, *, allow_unsupported: bool) -> None:
    """Drive Rich's Live progress bar during a native HuggingFace download."""
    err_console = Console(stderr=True, force_terminal=True)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("{task.fields[detail]}"),
        TimeRemainingColumn(),
        console=err_console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(f"Downloading {ref}", total=100, detail="")

        def on_update(p: DownloadProgress) -> None:
            progress.update(task_id, completed=p.percent, detail=p.detail)

        final = _run_pull(ref, src, on_update, allow_unsupported=allow_unsupported)

    if final.status == PullStatus.ALREADY_INSTALLED:
        console.print(f"{ref} is already installed.")
    else:
        console.print(f"Pulled [{theme.ACCENT}]{ref}[/{theme.ACCENT}].")


@model_app.command("pull")
def pull_cmd(
    ref: str = typer.Argument(..., help="Model ref to download (e.g. 'Qwen/Qwen3-0.6B-GGUF')."),
    source: str = typer.Option(
        "native",
        "--source",
        "-s",
        help="Pull from 'native' (HuggingFace GGUF) or 'remote' (SDK-managed).",
    ),
    allow_unsupported: bool = typer.Option(
        False,
        "--allow-unsupported",
        help="Pull even if the architecture isn't in the supported set (load may still fail).",
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Download a model."""
    from lilbee.catalog.types import ModelSource

    apply_overrides(data_dir=data_dir, use_global=use_global)
    src = _parse_source_or_bad_param(source) or ModelSource.NATIVE
    if cfg.json_mode:
        _pull_json_stream(ref, src, allow_unsupported=allow_unsupported)
    else:
        _pull_interactive_progress(ref, src, allow_unsupported=allow_unsupported)


def _confirm_remove_or_exit(ref: str, yes: bool) -> None:
    if yes or cfg.json_mode:
        return
    if not typer.confirm(f"Remove {ref}?", default=False):
        console.print("Aborted.")
        raise typer.Exit(0)


@model_app.command("rm")
def rm_cmd(
    ref: str = typer.Argument(..., help="Model ref to remove."),
    source: str | None = _source_option,
    yes: bool = _yes_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Remove an installed model."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    src = _parse_source_or_bad_param(source)
    _confirm_remove_or_exit(ref, yes)
    data = remove_model_data(ref, source=src)
    if cfg.json_mode:
        json_output(data.model_dump())
        if not data.deleted:
            raise typer.Exit(1)
        return
    if not data.deleted:
        console.print(f"[{theme.WARNING}]Not found: {ref}[/{theme.WARNING}]")
        raise typer.Exit(1)
    suffix = f" ({data.freed_gb:.2f} GB freed)" if data.freed_gb else ""
    console.print(f"Removed [{theme.ACCENT}]{ref}[/{theme.ACCENT}]{suffix}.")


def _is_interactive_terminal() -> bool:
    """Return True when both stdin and stdout are connected to a TTY.

    Extracted as a module-level helper so tests can patch it deterministically;
    CliRunner replaces ``sys.stdin`` during invoke which makes direct
    monkey-patching of ``sys.stdin.isatty`` unreliable.
    """
    import sys

    return sys.stdin.isatty() and sys.stdout.isatty()


@model_app.command("browse")
def browse_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Open the Textual TUI directly on the model catalog screen.

    Exit codes follow the project convention: 2 for invalid flag
    combinations (``--json`` with an interactive-only command), 1 for
    runtime environment failures (no TTY).
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if cfg.json_mode:
        json_output({"error": "model browse is interactive, not available in --json mode"})
        raise typer.Exit(2)
    if not _is_interactive_terminal():
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] model browse requires a terminal.")
        raise typer.Exit(1)

    from lilbee.cli.tui import run_tui

    run_tui(initial_view="Catalog")
