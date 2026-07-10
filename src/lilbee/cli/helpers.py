"""CLI-specific helpers: JSON formatter, Rich rendering, and CLI workflows."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, RenderableType
from rich.table import Table

from lilbee.app.ingest import copy_files
from lilbee.app.status import StatusResult
from lilbee.cli import theme
from lilbee.core.config import cfg

if TYPE_CHECKING:
    from lilbee.cli.sync import SyncStatus


def json_output(data: dict) -> None:
    """Print a JSON object to stdout."""
    print(json.dumps(data))


def announce_cold_start(role: object, model: str) -> Console | None:
    """Print a "Starting <role> engine (loading <model>)..." stderr line if cold.

    Returns a stderr console to print the matching "ready" line through when the
    blocking call returns, or ``None`` when the role's server is already warm (no
    status needed) or output is JSON (machine-readable, no chatter). The role
    parameter is a ``WorkerRole``; typed as ``object`` to keep this CLI helper
    free of a provider-layer import at module top.
    """
    from lilbee.app.services import get_services
    from lilbee.providers.roles import WorkerRole

    if cfg.json_mode or not isinstance(role, WorkerRole):
        return None
    if get_services().provider.role_ready(role):
        return None
    err = Console(stderr=True)
    err.print(f"[{theme.MUTED}]Starting {role.value} engine (loading {model})...[/{theme.MUTED}]")
    return err


def announce_ready(err: Console | None, role: object) -> None:
    """Print the matching "<role> engine ready." stderr line, if cold-start announced.

    A token arriving is not evidence the chat model came up: in RAG mode a grounded
    refusal streams without it. When warm-up recorded a load failure, that reason is
    printed instead of a readiness line.
    """
    from lilbee.providers.roles import WorkerRole

    if err is None or not isinstance(role, WorkerRole):
        return
    failure = _chat_warm_error(role)
    if failure is not None:
        err.print(f"[{theme.ERROR}]{failure}[/{theme.ERROR}]")
        return
    err.print(f"[{theme.MUTED}]{role.value} engine ready.[/{theme.MUTED}]")


def _chat_warm_error(role: object) -> str | None:
    """The chat warm-up's recorded failure, or None when it did not fail.

    Read from the warm tracker rather than re-probing readiness: llama-swap can
    report a freshly loaded model as not-yet-running, which would turn a healthy
    engine into a spurious failure line.
    """
    from lilbee.app.services import get_services
    from lilbee.providers.roles import WorkerRole
    from lilbee.providers.warm_progress import WarmPhase

    if role is not WorkerRole.CHAT:
        return None
    snapshot = get_services().provider.warm_progress()
    if snapshot is None or snapshot.phase is not WarmPhase.ERROR:
        return None
    return snapshot.error or "The chat model did not finish loading."


def render_status_result(status: StatusResult) -> Generator[RenderableType, None, None]:
    """Yield Rich renderables for a :class:`StatusResult`."""
    yield f"[{theme.LABEL}]Documents:[/{theme.LABEL}]  {status.config.documents_dir}"
    yield f"[{theme.LABEL}]Database:[/{theme.LABEL}]   {status.config.data_dir}"
    yield f"[{theme.LABEL}]Chat model:[/{theme.LABEL}] {status.config.chat_model}"
    yield f"[{theme.LABEL}]Embeddings:[/{theme.LABEL}] {status.config.embedding_model}"
    vision = status.config.vision_model or "(disabled)"
    reranker = status.config.reranker_model or "(disabled)"
    yield f"[{theme.LABEL}]Vision:[/{theme.LABEL}]     {vision}"
    yield f"[{theme.LABEL}]Reranker:[/{theme.LABEL}]   {reranker}"
    if status.config.enable_ocr is not None:
        ocr_label = "enabled" if status.config.enable_ocr else "disabled"
        yield f"[{theme.LABEL}]Vision OCR:[/{theme.LABEL}] {ocr_label}"
    yield ""

    if not status.sources:
        yield (
            "No documents indexed. Drop files into the documents directory and run 'lilbee sync'."
        )
        return

    table = Table(title="Indexed Documents")
    table.add_column("File", style=theme.ACCENT)
    table.add_column("Hash", style=theme.MUTED, max_width=12)
    table.add_column("Chunks", justify="right")
    table.add_column("Ingested", style=theme.MUTED)
    for s in status.sources:
        table.add_row(s.filename, s.file_hash, str(s.chunk_count), s.ingested_at)
    yield table
    b = theme.LABEL
    yield f"\n[{b}]{len(status.sources)}[/{b}] documents, [{b}]{status.total_chunks}[/{b}] chunks"


def render_status(con: Console) -> None:
    """Print status info (documents, paths, chunk counts)."""
    from lilbee.app.status import gather_status

    for renderable in render_status_result(gather_status()):
        con.print(renderable)


def copy_paths(paths: list[Path], con: Console, *, force: bool = False) -> list[str]:
    """Copy *paths* into the documents directory. Returns list of copied names."""
    result = copy_files(paths, force=force)
    for name in result.skipped:
        con.print(
            f"[{theme.WARNING}]Warning:[/{theme.WARNING}] {name} already exists in knowledge base "
            f"(use --force to overwrite)"
        )
    return result.copied


def add_paths(
    paths: list[Path],
    con: Console,
    *,
    force: bool = False,
    background: bool = False,
    chat_mode: bool = False,
    sync_status: SyncStatus | None = None,
    run_sync: Callable[[], object] | None = None,
) -> None:
    """Copy *paths* into the knowledge base and sync (human output).
    When *background* is True (chat ``/add``), sync runs in a background thread
    and this function returns immediately after copying files. *run_sync*
    overrides the foreground sync call (the CLI passes a Ctrl+C-cancellable
    runner); it defaults to a plain ``asyncio.run(sync())``.
    """
    copied = copy_paths(paths, con, force=force)
    if chat_mode:
        print(f"Copied {len(copied)} path(s) to {cfg.documents_dir}")
    else:
        con.print(
            f"[{theme.MUTED}]Copied {len(copied)} path(s) to {cfg.documents_dir}[/{theme.MUTED}]"
        )

    if background:
        from lilbee.cli.sync import run_sync_background

        run_sync_background(con, chat_mode=chat_mode, sync_status=sync_status)
        return

    result = run_sync() if run_sync is not None else _run_foreground_sync()
    con.print(result)


def _run_foreground_sync() -> object:
    """Run a blocking sync with no cancellation hook (default for non-CLI callers)."""
    from lilbee.data.ingest import sync

    return asyncio.run(sync())


def sync_result_to_json(result: object) -> dict:
    """Convert a SyncResult to the JSON output envelope."""
    from lilbee.data.ingest import SyncResult

    if not isinstance(result, SyncResult):
        raise TypeError(f"Expected SyncResult, got {type(result).__name__}")
    return {"command": "sync", **result.model_dump()}


def auto_sync(con: Console, *, background: bool = False) -> None:
    """Run document sync before queries.
    When *background* is True, sync runs in a background thread and this
    function returns immediately (for chat/REPL).  When False (default),
    sync blocks until complete (for ``lilbee ask``).
    """
    if background:
        from lilbee.cli.sync import run_sync_background

        run_sync_background(con)
        return

    from lilbee.cli.sync import _format_sync_summary
    from lilbee.data.ingest import sync

    try:
        result = asyncio.run(sync())
    except RuntimeError as exc:
        con.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None
    summary = _format_sync_summary(
        len(result.added),
        len(result.updated),
        len(result.removed),
        len(result.failed),
        len(result.skipped),
    )
    if summary:
        con.print(f"[{theme.MUTED}]Synced: {summary}[/{theme.MUTED}]")
