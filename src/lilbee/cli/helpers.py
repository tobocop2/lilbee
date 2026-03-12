"""Shared helper functions for CLI commands and slash commands."""

import asyncio
import json
import shutil
from dataclasses import asdict, dataclass, field
from importlib.metadata import version as _pkg_version
from pathlib import Path

from rich.console import Console
from rich.table import Table

from lilbee.config import cfg
from lilbee.platform import is_ignored_dir


def get_version() -> str:
    """Return the installed lilbee version."""
    return _pkg_version("lilbee")


def json_output(data: dict) -> None:
    """Print a JSON object to stdout."""
    print(json.dumps(data))


def clean_result(result: dict) -> dict:
    """Strip vector field and rename _distance for JSON output."""
    cleaned = {k: v for k, v in result.items() if k != "vector"}
    if "_distance" in cleaned:
        cleaned["distance"] = cleaned.pop("_distance")
    return cleaned


def gather_status() -> dict:
    """Collect status data as a plain dict (shared by human + JSON output)."""
    from lilbee.store import get_sources

    sources = get_sources()
    sorted_sources = sorted(sources, key=lambda x: x["filename"])
    total_chunks = sum(s["chunk_count"] for s in sources)
    return {
        "command": "status",
        "config": {
            "documents_dir": str(cfg.documents_dir),
            "data_dir": str(cfg.data_dir),
            "chat_model": cfg.chat_model,
            "embedding_model": cfg.embedding_model,
        },
        "sources": [
            {
                "filename": s["filename"],
                "file_hash": s["file_hash"][:12],
                "chunk_count": s["chunk_count"],
                "ingested_at": s["ingested_at"][:19],
            }
            for s in sorted_sources
        ],
        "total_chunks": total_chunks,
    }


def render_status(con: Console) -> None:
    """Print status info (documents, paths, chunk counts)."""
    data = gather_status()
    conf = data["config"]
    con.print(f"[bold]Documents:[/bold]  {conf['documents_dir']}")
    con.print(f"[bold]Database:[/bold]   {conf['data_dir']}")
    con.print(f"[bold]Chat model:[/bold] {conf['chat_model']}")
    con.print(f"[bold]Embeddings:[/bold] {conf['embedding_model']}")
    con.print()

    if not data["sources"]:
        con.print(
            "No documents indexed. Drop files into the documents directory and run 'lilbee sync'."
        )
        return

    table = Table(title="Indexed Documents")
    table.add_column("File", style="cyan")
    table.add_column("Hash", style="dim", max_width=12)
    table.add_column("Chunks", justify="right")
    table.add_column("Ingested", style="dim")

    for s in data["sources"]:
        table.add_row(
            s["filename"],
            s["file_hash"],
            str(s["chunk_count"]),
            s["ingested_at"],
        )

    con.print(table)
    con.print(
        f"\n[bold]{len(data['sources'])}[/bold] documents, "
        f"[bold]{data['total_chunks']}[/bold] chunks"
    )


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

            def ignore_dirs(directory: str, contents: list[str]) -> set[str]:
                return {
                    name
                    for name in contents
                    if (Path(directory) / name).is_dir() and is_ignored_dir(name, cfg.ignore_dirs)
                }

            shutil.copytree(p, dest, dirs_exist_ok=True, ignore=ignore_dirs)
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


def add_paths(paths: list[Path], con: Console, *, force: bool = False) -> None:
    """Copy *paths* into the knowledge base and sync (human output)."""
    from lilbee.ingest import sync

    copied = copy_paths(paths, con, force=force)
    con.print(f"[dim]Copied {len(copied)} path(s) to {cfg.documents_dir}[/dim]")

    result = asyncio.run(sync())
    con.print(result)


def stream_response(
    question: str,
    history: list[dict],
    con: Console,
) -> None:
    """Stream an LLM answer and append the exchange to *history*."""
    from lilbee.query import ask_stream

    stream = ask_stream(question, history=history)
    response_parts: list[str] = []

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
    except RuntimeError as exc:
        con.print(f"\n[red]Error:[/red] {exc}")
        return

    con.print("\n")
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "".join(response_parts)})


def perform_reset() -> dict:
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

    return {
        "command": "reset",
        "deleted_docs": deleted_docs,
        "deleted_data": deleted_data,
        "documents_dir": str(cfg.documents_dir),
        "data_dir": str(cfg.data_dir),
    }


def sync_result_to_json(result: object) -> dict:
    """Convert a SyncResult to the JSON output envelope."""
    return {"command": "sync", **asdict(result)}  # type: ignore[call-overload]


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
