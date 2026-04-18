"""Shared helper functions for CLI commands and slash commands."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich.console import Console, RenderableType
from rich.table import Table

from lilbee.cli import theme
from lilbee.config import cfg
from lilbee.platform import is_ignored_dir
from lilbee.security import validate_path_within
from lilbee.services import get_services

if TYPE_CHECKING:
    from lilbee.cli.sync import SyncStatus
    from lilbee.store import SearchChunk


class ResetResult(BaseModel):
    """Result of a full knowledge base reset."""

    command: str = "reset"
    deleted_docs: int
    deleted_data: int
    skipped: list[str] = []
    documents_dir: str
    data_dir: str


class StatusConfig(BaseModel):
    """Configuration section of a status response."""

    documents_dir: str
    data_dir: str
    chat_model: str
    embedding_model: str
    enable_ocr: bool | None = None


class SourceInfo(BaseModel):
    """A single indexed source in a status response."""

    filename: str
    file_hash: str
    chunk_count: int
    ingested_at: str


class StatusResult(BaseModel):
    """Full status response for the knowledge base."""

    command: str = "status"
    config: StatusConfig
    sources: list[SourceInfo]
    total_chunks: int

    def __rich_console__(
        self, console: Console, options: object
    ) -> Generator[RenderableType, None, None]:
        yield f"[{theme.LABEL}]Documents:[/{theme.LABEL}]  {self.config.documents_dir}"
        yield f"[{theme.LABEL}]Database:[/{theme.LABEL}]   {self.config.data_dir}"
        yield f"[{theme.LABEL}]Chat model:[/{theme.LABEL}] {self.config.chat_model}"
        yield f"[{theme.LABEL}]Embeddings:[/{theme.LABEL}] {self.config.embedding_model}"
        if self.config.enable_ocr is not None:
            ocr_label = "enabled" if self.config.enable_ocr else "disabled"
            yield f"[{theme.LABEL}]Vision OCR:[/{theme.LABEL}] {ocr_label}"
        yield ""

        if not self.sources:
            yield (
                "No documents indexed. Drop files into the documents directory "
                "and run 'lilbee sync'."
            )
            return

        table = Table(title="Indexed Documents")
        table.add_column("File", style=theme.ACCENT)
        table.add_column("Hash", style=theme.MUTED, max_width=12)
        table.add_column("Chunks", justify="right")
        table.add_column("Ingested", style=theme.MUTED)
        for s in self.sources:
            table.add_row(s.filename, s.file_hash, str(s.chunk_count), s.ingested_at)
        yield table
        b = theme.LABEL
        yield f"\n[{b}]{len(self.sources)}[/{b}] documents, [{b}]{self.total_chunks}[/{b}] chunks"


def _copytree_ignore(directory: str, contents: list[str]) -> set[str]:
    """Ignore callback for shutil.copytree — filters ignored directories."""
    return {
        name
        for name in contents
        if (Path(directory) / name).is_dir() and is_ignored_dir(name, cfg.ignore_dirs)
    }


def get_version() -> str:
    """Return the installed lilbee version."""
    return _pkg_version("lilbee")


def json_output(data: dict) -> None:
    """Print a JSON object to stdout."""
    print(json.dumps(data))


def clean_result(result: SearchChunk) -> dict:
    """Convert SearchChunk to a JSON-friendly dict (no vector, no None scores)."""
    return result.model_dump(exclude={"vector"}, exclude_none=True)


def gather_status() -> StatusResult:
    """Collect status data as a typed model (shared by human + JSON output)."""
    sources = get_services().store.get_sources()
    sorted_sources = sorted(sources, key=lambda x: x["filename"])
    total_chunks = sum(s["chunk_count"] for s in sources)
    return StatusResult(
        config=StatusConfig(
            documents_dir=str(cfg.documents_dir),
            data_dir=str(cfg.data_dir),
            chat_model=cfg.chat_model,
            embedding_model=cfg.embedding_model,
            enable_ocr=cfg.enable_ocr,
        ),
        sources=[
            SourceInfo(
                filename=s["filename"],
                file_hash=s["file_hash"][:12],
                chunk_count=s["chunk_count"],
                ingested_at=s["ingested_at"][:19],
            )
            for s in sorted_sources
        ],
        total_chunks=total_chunks,
    )


def render_status(con: Console) -> None:
    """Print status info (documents, paths, chunk counts)."""
    con.print(gather_status())


@dataclass
class CopyResult:
    """Result of copying files into the documents directory."""

    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def copy_files(paths: list[Path], *, force: bool = False) -> CopyResult:
    """Copy paths into documents dir. Returns structured result (no console output)."""
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    result = CopyResult()
    for p in paths:
        dest = cfg.documents_dir / p.name
        validate_path_within(dest, cfg.documents_dir)
        if dest.exists() and not force:
            result.skipped.append(p.name)
            continue
        if p.is_dir():
            shutil.copytree(p, dest, dirs_exist_ok=True, ignore=_copytree_ignore, symlinks=False)
        else:
            shutil.copy2(p, dest)
        result.copied.append(p.name)
    return result


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
) -> None:
    """Copy *paths* into the knowledge base and sync (human output).
    When *background* is True (chat ``/add``), sync runs in a background thread
    and this function returns immediately after copying files.
    """
    from lilbee.ingest import sync

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

    result = asyncio.run(sync())
    con.print(result)


def _clear_dir(base_dir: Path, skipped: list[str]) -> int:
    """Delete all items in *base_dir*, appending undeletable paths to *skipped*."""
    log = logging.getLogger(__name__)
    deleted = 0
    if not base_dir.exists():
        return deleted
    for item in list(base_dir.iterdir()):
        validate_path_within(item, base_dir)
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except OSError as exc:
            log.warning("Could not delete %s: %s", item, exc)
            skipped.append(str(item))
            continue
        deleted += 1
    return deleted


def perform_reset() -> ResetResult:
    """Delete all documents and data. Returns summary of what was deleted."""
    skipped: list[str] = []
    deleted_docs = _clear_dir(cfg.documents_dir, skipped)
    deleted_data = _clear_dir(cfg.data_dir, skipped)

    return ResetResult(
        deleted_docs=deleted_docs,
        deleted_data=deleted_data,
        skipped=skipped,
        documents_dir=str(cfg.documents_dir),
        data_dir=str(cfg.data_dir),
    )


def sync_result_to_json(result: object) -> dict:
    """Convert a SyncResult to the JSON output envelope."""
    from lilbee.ingest import SyncResult

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

    from lilbee.ingest import sync

    try:
        result = asyncio.run(sync())
    except RuntimeError as exc:
        con.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None
    total = len(result.added) + len(result.updated) + len(result.removed) + len(result.failed)
    if total:
        con.print(
            f"[{theme.MUTED}]Synced: {len(result.added)} added, "
            f"{len(result.updated)} updated, "
            f"{len(result.removed)} removed, "
            f"{len(result.failed)} failed[/{theme.MUTED}]"
        )


@contextmanager
def temporary_ocr_config(
    enable_ocr: bool | None = None,
    ocr_timeout: float | None = None,
) -> Generator[None, None, None]:
    """Temporarily override OCR config for the duration of the block."""
    old_ocr, old_timeout = cfg.enable_ocr, cfg.ocr_timeout
    try:
        if enable_ocr is not None:
            cfg.enable_ocr = enable_ocr
        if ocr_timeout is not None:
            cfg.ocr_timeout = ocr_timeout
        yield
    finally:
        cfg.enable_ocr = old_ocr
        cfg.ocr_timeout = old_timeout
