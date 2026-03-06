"""Shared helper functions for CLI commands and slash commands."""

import json
import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _get_version() -> str:
    """Return the installed lilbee version."""
    from importlib.metadata import version

    return version("lilbee")


def _json_output(data: dict) -> None:
    """Print a JSON object to stdout."""
    print(json.dumps(data))


def _clean_result(result: dict) -> dict:
    """Strip vector field and rename _distance for JSON output."""
    cleaned = {k: v for k, v in result.items() if k != "vector"}
    if "_distance" in cleaned:
        cleaned["distance"] = cleaned.pop("_distance")
    return cleaned


def _gather_status() -> dict:
    """Collect status data as a plain dict (shared by human + JSON output)."""
    from lilbee.config import CHAT_MODEL, DATA_DIR, DOCUMENTS_DIR, EMBEDDING_MODEL
    from lilbee.store import get_sources

    sources = get_sources()
    sorted_sources = sorted(sources, key=lambda x: x["filename"])
    total_chunks = sum(s["chunk_count"] for s in sources)
    return {
        "command": "status",
        "config": {
            "documents_dir": str(DOCUMENTS_DIR),
            "data_dir": str(DATA_DIR),
            "chat_model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
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


def _render_status(con: Console) -> None:
    """Print status info (documents, paths, chunk counts)."""
    data = _gather_status()
    cfg = data["config"]
    con.print(f"[bold]Documents:[/bold]  {cfg['documents_dir']}")
    con.print(f"[bold]Database:[/bold]   {cfg['data_dir']}")
    con.print(f"[bold]Chat model:[/bold] {cfg['chat_model']}")
    con.print(f"[bold]Embeddings:[/bold] {cfg['embedding_model']}")
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


def _copy_paths(paths: list[Path], con: Console, *, force: bool = False) -> list[str]:
    """Copy *paths* into the documents directory. Returns list of copied names."""
    import lilbee.config as cfg

    cfg.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for p in paths:
        dest = cfg.DOCUMENTS_DIR / p.name
        if dest.exists() and not force:
            con.print(
                f"[yellow]Warning:[/yellow] {p.name} already exists in knowledge base "
                f"(use --force to overwrite)"
            )
            continue
        if p.is_dir():
            from lilbee.config import is_ignored_dir

            def _ignore_dirs(directory: str, contents: list[str]) -> set[str]:
                return {
                    name
                    for name in contents
                    if (Path(directory) / name).is_dir() and is_ignored_dir(name)
                }

            shutil.copytree(p, dest, dirs_exist_ok=True, ignore=_ignore_dirs)
        else:
            shutil.copy2(p, dest)
        copied.append(p.name)
    return copied


def _add_paths(paths: list[Path], con: Console, *, force: bool = False) -> None:
    """Copy *paths* into the knowledge base and sync (human output)."""
    import lilbee.config as cfg
    from lilbee.ingest import sync

    copied = _copy_paths(paths, con, force=force)
    con.print(f"[dim]Copied {len(copied)} path(s) to {cfg.DOCUMENTS_DIR}[/dim]")

    result = sync()
    con.print(f"Added: {len(result['added'])}")
    con.print(f"Updated: {len(result['updated'])}")
    con.print(f"Unchanged: {result['unchanged']}")


def _stream_response(
    question: str,
    history: list[dict],
    con: Console,
) -> None:
    """Stream an LLM answer and append the exchange to *history*."""
    from lilbee.query import ask_stream

    stream = ask_stream(question, history=history)
    response_parts: list[str] = []

    # Show a spinner while waiting for the first token from the LLM.
    with con.status("Thinking..."):
        first_token = next(stream, None)

    if first_token is not None:
        con.print(first_token, end="")
        response_parts.append(first_token)

    for token in stream:
        con.print(token, end="")
        response_parts.append(token)
    con.print("\n")
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "".join(response_parts)})


def _perform_reset() -> dict:
    """Delete all documents and data. Returns summary of what was deleted."""
    import lilbee.config as cfg

    deleted_docs = 0
    deleted_data = 0

    if cfg.DOCUMENTS_DIR.exists():
        for item in list(cfg.DOCUMENTS_DIR.iterdir()):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            deleted_docs += 1

    if cfg.DATA_DIR.exists():
        for item in list(cfg.DATA_DIR.iterdir()):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            deleted_data += 1

    return {
        "command": "reset",
        "deleted_docs": deleted_docs,
        "deleted_data": deleted_data,
        "documents_dir": str(cfg.DOCUMENTS_DIR),
        "data_dir": str(cfg.DATA_DIR),
    }


def _sync_result_to_json(result: dict) -> dict:
    """Convert a sync result dict to the JSON output envelope."""
    return {
        "command": "sync",
        "added": result["added"],
        "updated": result["updated"],
        "removed": result["removed"],
        "unchanged": result["unchanged"],
        "failed": result["failed"],
    }


def _auto_sync(con: Console) -> None:
    """Run document sync before queries."""
    from lilbee.ingest import sync

    result = sync()
    total = (
        len(result["added"])
        + len(result["updated"])
        + len(result["removed"])
        + len(result.get("failed", []))
    )
    if total:
        con.print(
            f"[dim]Synced: {len(result['added'])} added, "
            f"{len(result['updated'])} updated, "
            f"{len(result['removed'])} removed, "
            f"{len(result.get('failed', []))} failed[/dim]"
        )
