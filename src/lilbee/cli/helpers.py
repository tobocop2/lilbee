"""Shared helper functions for CLI commands and slash commands."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Generator
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich.console import Console, RenderableType
from rich.table import Table

from lilbee.config import cfg
from lilbee.platform import is_ignored_dir

if TYPE_CHECKING:
    from lilbee.query import ChatMessage
    from lilbee.store import SearchChunk


class ResetResult(BaseModel):
    """Result of a full knowledge base reset."""

    command: str = "reset"
    deleted_docs: int
    deleted_data: int
    documents_dir: str
    data_dir: str


class StatusConfig(BaseModel):
    """Configuration section of a status response."""

    documents_dir: str
    data_dir: str
    chat_model: str
    embedding_model: str
    vision_model: str | None = None


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
        yield f"[bold]Documents:[/bold]  {self.config.documents_dir}"
        yield f"[bold]Database:[/bold]   {self.config.data_dir}"
        yield f"[bold]Chat model:[/bold] {self.config.chat_model}"
        yield f"[bold]Embeddings:[/bold] {self.config.embedding_model}"
        if self.config.vision_model:
            yield f"[bold]Vision OCR:[/bold] {self.config.vision_model}"
        yield ""

        if not self.sources:
            yield (
                "No documents indexed. Drop files into the documents directory "
                "and run 'lilbee sync'."
            )
            return

        table = Table(title="Indexed Documents")
        table.add_column("File", style="cyan")
        table.add_column("Hash", style="dim", max_width=12)
        table.add_column("Chunks", justify="right")
        table.add_column("Ingested", style="dim")
        for s in self.sources:
            table.add_row(s.filename, s.file_hash, str(s.chunk_count), s.ingested_at)
        yield table
        yield (
            f"\n[bold]{len(self.sources)}[/bold] documents, [bold]{self.total_chunks}[/bold] chunks"
        )


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
    """Strip vector field and normalize score fields for JSON output."""
    cleaned = {k: v for k, v in result.items() if k != "vector"}
    if "_relevance_score" in cleaned:
        cleaned["relevance_score"] = cleaned.pop("_relevance_score")
        cleaned.pop("_distance", None)
    elif "_distance" in cleaned:
        cleaned["distance"] = cleaned.pop("_distance")
    return cleaned


def gather_status() -> StatusResult:
    """Collect status data as a typed model (shared by human + JSON output)."""
    from lilbee.store import get_sources

    sources = get_sources()
    sorted_sources = sorted(sources, key=lambda x: x["filename"])
    total_chunks = sum(s["chunk_count"] for s in sources)
    return StatusResult(
        config=StatusConfig(
            documents_dir=str(cfg.documents_dir),
            data_dir=str(cfg.data_dir),
            chat_model=cfg.chat_model,
            embedding_model=cfg.embedding_model,
            vision_model=cfg.vision_model or None,
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
        if dest.exists() and not force:
            result.skipped.append(p.name)
            continue
        if p.is_dir():
            shutil.copytree(p, dest, dirs_exist_ok=True, ignore=_copytree_ignore)
        else:
            shutil.copy2(p, dest)
        result.copied.append(p.name)
    return result


def copy_paths(paths: list[Path], con: Console, *, force: bool = False) -> list[str]:
    """Copy *paths* into the documents directory. Returns list of copied names."""
    result = copy_files(paths, force=force)
    for name in result.skipped:
        con.print(
            f"[yellow]Warning:[/yellow] {name} already exists in knowledge base "
            f"(use --force to overwrite)"
        )
    return result.copied


def add_paths(
    paths: list[Path], con: Console, *, force: bool = False, force_vision: bool = False
) -> None:
    """Copy *paths* into the knowledge base and sync (human output)."""
    from lilbee.ingest import sync

    copied = copy_paths(paths, con, force=force)
    con.print(f"[dim]Copied {len(copied)} path(s) to {cfg.documents_dir}[/dim]")

    result = asyncio.run(sync(force_vision=force_vision))
    con.print(result)


def stream_response(
    question: str,
    history: list[ChatMessage],
    con: Console,
) -> None:
    """Stream an LLM answer and append the exchange to *history*."""
    from lilbee.query import ask_stream

    stream = ask_stream(question, history=history)
    response_parts: list[str] = []
    cancelled = False

    try:
        # Show a spinner while waiting for the first token from the LLM.
        with con.status("Thinking..."):
            first_token = next(stream, None)

        if first_token is not None:
            con.print(first_token, end="")
            response_parts.append(first_token)

        for token in stream:
            con.print(token, end="")
            response_parts.append(token)
    except KeyboardInterrupt:
        cancelled = True
        stream.close()
        con.print("\n[dim](stopped)[/dim]")
    except RuntimeError as exc:
        con.print(f"\n[red]Error:[/red] {exc}")
        return

    if not cancelled:
        con.print("\n")
    full = "".join(response_parts)
    if full:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full})


def perform_reset() -> ResetResult:
    """Delete all documents and data. Returns summary of what was deleted."""
    deleted_docs = 0
    deleted_data = 0

    if cfg.documents_dir.exists():
        for item in list(cfg.documents_dir.iterdir()):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            deleted_docs += 1

    if cfg.data_dir.exists():
        for item in list(cfg.data_dir.iterdir()):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            deleted_data += 1

    return ResetResult(
        deleted_docs=deleted_docs,
        deleted_data=deleted_data,
        documents_dir=str(cfg.documents_dir),
        data_dir=str(cfg.data_dir),
    )


def sync_result_to_json(result: object) -> dict:
    """Convert a SyncResult to the JSON output envelope."""
    from lilbee.ingest import SyncResult

    assert isinstance(result, SyncResult)
    return {"command": "sync", **result.model_dump()}


def auto_sync(con: Console) -> None:
    """Run document sync before queries."""
    from lilbee.ingest import sync

    try:
        result = asyncio.run(sync())
    except RuntimeError as exc:
        con.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None
    total = len(result.added) + len(result.updated) + len(result.removed) + len(result.failed)
    if total:
        con.print(
            f"[dim]Synced: {len(result.added)} added, "
            f"{len(result.updated)} updated, "
            f"{len(result.removed)} removed, "
            f"{len(result.failed)} failed[/dim]"
        )
