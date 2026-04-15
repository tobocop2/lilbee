"""`lilbee model` sub-app: list/show/pull/rm/browse for installed models.

Thin CLI and shared data helpers over
:class:`lilbee.model_manager.ModelManager`. The ``*_data`` functions
return Pydantic models so MCP tools and CLI commands share a single,
typed implementation.

Heavy imports (:mod:`lilbee.catalog`, :mod:`lilbee.model_manager`,
:mod:`lilbee.registry`, :mod:`lilbee.cli.tui`) are deferred to function
bodies so importing this module at CLI startup stays cheap.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.helpers import json_output
from lilbee.config import cfg

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.catalog import CatalogModel, DownloadProgress
    from lilbee.model_manager import ModelSource, RemoteModel
    from lilbee.registry import ModelManifest


_BYTES_PER_GB = 1024**3  # Model sizes are reported to users in GiB.
_LITELLM_LIST_TIMEOUT_S = 2.0  # Keep `model list` snappy when backend is down.


def _bytes_to_gb(n: int) -> float:
    """Convert bytes to GiB rounded to 2 decimals for user display."""
    return round(n / _BYTES_PER_GB, 2)


class ModelCommand(StrEnum):
    """Command field values for model sub-app JSON output."""

    LIST = "model list"
    SHOW = "model show"
    PULL = "model pull"
    RM = "model rm"


class PullStatus(StrEnum):
    OK = "ok"
    ALREADY_INSTALLED = "already_installed"


class PullEvent(StrEnum):
    PROGRESS = "progress"
    DONE = "done"


class ModelEntry(BaseModel):
    """One row of `lilbee model list` output."""

    name: str
    source: str
    task: str | None = None
    size_gb: float | None = None
    display_name: str = ""

    @classmethod
    def from_native(cls, ref: str, manifest: ModelManifest | None) -> ModelEntry:
        from lilbee.model_manager import ModelSource

        return cls(
            name=ref,
            source=ModelSource.NATIVE.value,
            task=manifest.task if manifest else None,
            size_gb=_bytes_to_gb(manifest.size_bytes) if manifest else None,
            display_name=manifest.display_name if manifest else "",
        )

    @classmethod
    def from_litellm(cls, ref: str, remote: RemoteModel | None) -> ModelEntry:
        from lilbee.model_manager import ModelSource

        return cls(
            name=ref,
            source=ModelSource.LITELLM.value,
            task=remote.task if remote else None,
            size_gb=None,
            display_name=remote.parameter_size if remote else "",
        )


class ListModelsResult(BaseModel):
    command: str = ModelCommand.LIST
    models: list[ModelEntry]
    total: int

    def __rich__(self) -> Table:
        table = Table(title="Installed models")
        table.add_column("Name", style=theme.ACCENT)
        table.add_column("Source", style=theme.MUTED)
        table.add_column("Task")
        table.add_column("Size", justify="right")
        for entry in self.models:
            size = f"{entry.size_gb:.2f} GB" if entry.size_gb is not None else ""
            table.add_row(entry.name, entry.source, entry.task or "", size)
        return table


class CatalogEntryData(BaseModel):
    name: str
    tag: str
    ref: str
    display_name: str
    hf_repo: str
    size_gb: float
    min_ram_gb: float
    description: str
    task: str
    featured: bool
    recommended: bool

    @classmethod
    def from_catalog_model(cls, entry: CatalogModel) -> CatalogEntryData:
        return cls(
            name=entry.name,
            tag=entry.tag,
            ref=entry.ref,
            display_name=entry.display_name,
            hf_repo=entry.hf_repo,
            size_gb=entry.size_gb,
            min_ram_gb=entry.min_ram_gb,
            description=entry.description,
            task=entry.task,
            featured=entry.featured,
            recommended=entry.recommended,
        )


class ManifestData(BaseModel):
    name: str
    display_name: str
    task: str
    size_gb: float
    size_bytes: int
    source_repo: str
    source_filename: str
    downloaded_at: str

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> ManifestData:
        return cls(
            name=f"{manifest.name}:{manifest.tag}",
            display_name=manifest.display_name,
            task=manifest.task,
            size_gb=_bytes_to_gb(manifest.size_bytes),
            size_bytes=manifest.size_bytes,
            source_repo=manifest.source_repo,
            source_filename=manifest.source_filename,
            downloaded_at=manifest.downloaded_at,
        )


class ShowModelResult(BaseModel):
    command: str = ModelCommand.SHOW
    model: str
    catalog: CatalogEntryData | None = None
    installed: bool = False
    source: str | None = None
    path: str | None = None
    manifest: ManifestData | None = None

    def __rich__(self) -> str:
        lines = [f"[{theme.ACCENT}]{self.model}[/{theme.ACCENT}]"]
        if self.catalog is not None:
            lines.extend(
                [
                    f"  display_name: {self.catalog.display_name}",
                    f"  task:         {self.catalog.task}",
                    f"  size_gb:      {self.catalog.size_gb}",
                    f"  min_ram_gb:   {self.catalog.min_ram_gb}",
                    f"  hf_repo:      {self.catalog.hf_repo}",
                    f"  description:  {self.catalog.description}",
                ]
            )
        lines.append(f"  installed:    {self.installed}")
        if self.source:
            lines.append(f"  source:       {self.source}")
        if self.path:
            lines.append(f"  path:         {self.path}")
        if self.manifest is not None:
            lines.append(f"  downloaded:   {self.manifest.downloaded_at}")
        return "\n".join(lines)


class PullResult(BaseModel):
    command: str = ModelCommand.PULL
    model: str
    source: str
    status: str
    path: str | None = None


class PullProgressEvent(BaseModel):
    command: str = ModelCommand.PULL
    event: str = PullEvent.PROGRESS
    model: str
    percent: int
    detail: str
    cache_hit: bool


class RemoveResult(BaseModel):
    command: str = ModelCommand.RM
    model: str
    deleted: bool
    freed_gb: float = Field(default=0.0)


def _native_manifest_index() -> dict[str, ModelManifest]:
    """Map 'name:tag' to manifest for every installed native model."""
    from lilbee.registry import ModelRegistry

    registry = ModelRegistry(cfg.models_dir)
    return {f"{m.name}:{m.tag}": m for m in registry.list_installed()}


def _resolve_native_path(ref: str) -> str | None:
    """Return the on-disk path of an installed native model, if resolvable.

    Swallows ``KeyError`` (manifest present but blob missing) and
    ``ValueError`` (malformed ref) so callers can treat the path as
    optional metadata.
    """
    from lilbee.registry import ModelRegistry

    try:
        return str(ModelRegistry(cfg.models_dir).resolve(ref))
    except (KeyError, ValueError):
        return None


def _collect_native_entries() -> list[ModelEntry]:
    from lilbee.model_manager import ModelSource, get_model_manager

    manifests = _native_manifest_index()
    refs = get_model_manager().list_installed(source=ModelSource.NATIVE)
    return [ModelEntry.from_native(ref, manifests.get(ref)) for ref in refs]


def _collect_litellm_entries() -> list[ModelEntry]:
    from lilbee.model_manager import classify_remote_models

    remote_list = classify_remote_models(cfg.litellm_base_url, timeout=_LITELLM_LIST_TIMEOUT_S)
    remote_by_name = {rm.name: rm for rm in remote_list}
    return [ModelEntry.from_litellm(ref, remote_by_name[ref]) for ref in sorted(remote_by_name)]


def list_models_data(
    source: ModelSource | None = None,
    task: str | None = None,
) -> ListModelsResult:
    """Build the list of installed models with source and task metadata.

    Discovers remote (litellm) models via a single HTTP call with a
    short timeout so the command stays responsive when the backend is
    down.
    """
    from lilbee.model_manager import ModelSource

    entries: list[ModelEntry] = []
    if source is None or source is ModelSource.NATIVE:
        entries.extend(_collect_native_entries())
    if source is None or source is ModelSource.LITELLM:
        entries.extend(_collect_litellm_entries())
    if task:
        entries = [e for e in entries if e.task == task]
    return ListModelsResult(models=entries, total=len(entries))


def show_model_data(ref: str) -> ShowModelResult:
    """Return catalog and install metadata for *ref*.

    Raises :class:`~lilbee.model_manager.ModelNotFoundError` if the ref
    is unknown to both the catalog and the installed set.
    """
    from lilbee.catalog import find_catalog_entry
    from lilbee.model_manager import ModelNotFoundError, get_model_manager

    entry = find_catalog_entry(ref)
    source = get_model_manager().get_source(ref)
    if entry is None and source is None:
        raise ModelNotFoundError(f"model not found: {ref}")
    manifest = _native_manifest_index().get(ref)
    return ShowModelResult(
        model=ref,
        catalog=CatalogEntryData.from_catalog_model(entry) if entry else None,
        installed=source is not None,
        source=source.value if source else None,
        manifest=ManifestData.from_manifest(manifest) if manifest else None,
        path=_resolve_native_path(ref) if manifest is not None else None,
    )


def _litellm_event_to_progress(
    on_update: Callable[[DownloadProgress], None],
    event: dict[str, Any],
) -> None:
    """Adapt an Ollama-style dict event into a DownloadProgress call."""
    from lilbee.catalog import DownloadProgress

    total = event.get("total", 0) or 0
    completed = event.get("completed", 0) or 0
    detail = event.get("status", "") or ""
    pct = int(completed * 100 / total) if total > 0 else 0
    on_update(DownloadProgress(percent=pct, detail=detail, is_cache_hit=False))


def _build_pull_callbacks(
    on_update: Callable[[DownloadProgress], None] | None,
) -> tuple[Callable[[dict[str, Any]], None] | None, Callable[[int, int], None] | None]:
    """Build the (dict_cb, bytes_cb) pair for ModelManager.pull from on_update."""
    import functools

    from lilbee.catalog import make_download_callback

    if on_update is None:
        return None, None
    dict_cb = functools.partial(_litellm_event_to_progress, on_update)
    bytes_cb = make_download_callback(on_update)
    return dict_cb, bytes_cb


def pull_model_data(
    ref: str,
    source: ModelSource,
    *,
    on_update: Callable[[DownloadProgress], None] | None = None,
) -> PullResult:
    """Pull *ref* from *source* and return a typed result.

    Progress updates are throttled by
    :func:`~lilbee.catalog.make_download_callback`, so callers see at
    most roughly 10 Hz of progress events.
    """
    from lilbee.model_manager import get_model_manager

    manager = get_model_manager()

    if manager.is_installed(ref, source):
        return PullResult(model=ref, source=source.value, status=PullStatus.ALREADY_INSTALLED)

    dict_cb, bytes_cb = _build_pull_callbacks(on_update)
    path = manager.pull(ref, source, on_progress=dict_cb, on_bytes=bytes_cb)
    return PullResult(
        model=ref,
        source=source.value,
        status=PullStatus.OK,
        path=str(path) if path is not None else None,
    )


def remove_model_data(
    ref: str,
    source: ModelSource | None = None,
) -> RemoveResult:
    """Remove *ref* and return a typed result with freed size."""
    from lilbee.model_manager import get_model_manager

    manager = get_model_manager()
    manifests = _native_manifest_index()
    size_bytes = manifests[ref].size_bytes if ref in manifests else 0
    removed = manager.remove(ref, source=source)
    return RemoveResult(
        model=ref,
        deleted=removed,
        freed_gb=_bytes_to_gb(size_bytes),
    )


model_app = typer.Typer(
    name="model",
    help="Manage installed and available models (pull / list / show / rm / browse).",
    no_args_is_help=True,
)

_source_option = typer.Option(
    None,
    "--source",
    "-s",
    help="Filter by source: 'native' or 'litellm' (default: all).",
)
_task_option = typer.Option(
    None,
    "--task",
    "-t",
    help="Filter by task: 'chat', 'embedding', or 'vision'.",
)
_yes_option = typer.Option(
    False,
    "--yes",
    "-y",
    help="Skip confirmation prompt.",
)


def _parse_source_or_bad_param(value: str | None) -> ModelSource | None:
    """Parse a CLI --source value, raising typer.BadParameter on bad input."""
    from lilbee.model_manager import ModelSource

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
    apply_overrides(data_dir=data_dir, use_global=use_global)
    data = list_models_data(source=_parse_source_or_bad_param(source), task=task)
    if cfg.json_mode:
        json_output(data.model_dump())
        return
    if not data.models:
        console.print("No models installed.")
        return
    console.print(data)


@model_app.command("show")
def show_cmd(
    ref: str = typer.Argument(..., help="Model ref (e.g. 'qwen3:0.6b')."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show catalog and installed metadata for a model."""
    from lilbee.model_manager import ModelNotFoundError

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
    console.print(data)


def _run_pull(
    ref: str,
    src: ModelSource,
    on_update: Callable[[DownloadProgress], None],
) -> PullResult:
    """Invoke ``pull_model_data`` and translate known errors to typer.Exit."""
    try:
        return pull_model_data(ref, src, on_update=on_update)
    except (RuntimeError, PermissionError) as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise typer.Exit(1) from None


def _pull_json_stream(ref: str, src: ModelSource) -> None:
    """Emit newline-delimited JSON progress events, then the final result."""

    def on_update(p: DownloadProgress) -> None:
        event = PullProgressEvent(
            model=ref, percent=p.percent, detail=p.detail, cache_hit=p.is_cache_hit
        )
        json_output(event.model_dump())

    final = _run_pull(ref, src, on_update)
    json_output({**final.model_dump(), "event": PullEvent.DONE.value})


def _pull_rich_progress(ref: str, src: ModelSource) -> None:
    """Drive a Rich progress bar during a native HuggingFace download."""
    err_console = Console(stderr=True)
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

        final = _run_pull(ref, src, on_update)

    if final.status == PullStatus.ALREADY_INSTALLED:
        console.print(f"{ref} is already installed.")
    else:
        console.print(f"Pulled [{theme.ACCENT}]{ref}[/{theme.ACCENT}].")


@model_app.command("pull")
def pull_cmd(
    ref: str = typer.Argument(..., help="Model ref to download (e.g. 'qwen3:0.6b')."),
    source: str = typer.Option(
        "native",
        "--source",
        "-s",
        help="Pull from 'native' (HuggingFace GGUF) or 'litellm' (remote backend).",
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Download a model."""
    from lilbee.model_manager import ModelSource

    apply_overrides(data_dir=data_dir, use_global=use_global)
    src = _parse_source_or_bad_param(source) or ModelSource.NATIVE
    if cfg.json_mode:
        _pull_json_stream(ref, src)
    else:
        _pull_rich_progress(ref, src)


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

    # Browsing the catalog does not depend on documents, so skip auto-sync.
    from lilbee.cli.tui import run_tui

    run_tui(auto_sync=False, initial_view="Catalog")
