"""CLI command definitions registered on the app."""

import asyncio
from pathlib import Path

import typer
from rich.table import Table

from lilbee.cli._app import _apply_overrides, _data_dir_option, _model_option, app, console
from lilbee.cli._helpers import (
    _add_paths,
    _auto_sync,
    _clean_result,
    _copy_paths,
    _gather_status,
    _get_version,
    _json_output,
    _perform_reset,
    _render_status,
    _sync_result_to_json,
)

_CHUNK_PREVIEW_LEN = 80  # characters shown in human-readable search output

_paths_argument = typer.Argument(
    ...,
    exists=True,
    help="Files or directories to add to the knowledge base.",
)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of results"),
    data_dir: Path | None = _data_dir_option,
) -> None:
    """Search the knowledge base for relevant chunks."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.cli import _state
    from lilbee.config import TOP_K
    from lilbee.query import search_context

    results = search_context(query, top_k=top_k or TOP_K)
    cleaned = [_clean_result(r) for r in results]

    if _state["json_mode"]:
        _json_output({"command": "search", "query": query, "results": cleaned})
        return

    if not cleaned:
        console.print("No results found.")
        return

    table = Table(title="Search Results")
    table.add_column("Source", style="cyan")
    table.add_column("Chunk", max_width=80)
    table.add_column("Distance", justify="right", style="dim")

    for r in cleaned:
        preview = r.get("chunk", "")[:_CHUNK_PREVIEW_LEN]
        if len(r.get("chunk", "")) > _CHUNK_PREVIEW_LEN:
            preview += "..."
        table.add_row(
            r.get("source", ""),
            preview,
            f"{r.get('distance', 0):.4f}",
        )
    console.print(table)


@app.command(name="sync")
def sync_cmd(data_dir: Path | None = _data_dir_option) -> None:
    """Manually trigger document sync."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.cli import _state
    from lilbee.ingest import sync

    try:
        result = asyncio.run(sync(quiet=_state["json_mode"]))
    except RuntimeError as exc:
        if _state["json_mode"]:
            _json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None
    if _state["json_mode"]:
        _json_output(_sync_result_to_json(result))
        return
    console.print(result)


@app.command()
def rebuild(data_dir: Path | None = _data_dir_option) -> None:
    """Nuke the DB and re-ingest everything from documents/."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.cli import _state
    from lilbee.ingest import sync

    try:
        result = asyncio.run(sync(force_rebuild=True, quiet=_state["json_mode"]))
    except RuntimeError as exc:
        if _state["json_mode"]:
            _json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None
    if _state["json_mode"]:
        _json_output({"command": "rebuild", "ingested": len(result.added)})
        return
    console.print(f"Rebuilt: {len(result.added)} documents ingested")


_force_option = typer.Option(False, "--force", "-f", help="Overwrite existing files.")


@app.command()
def add(
    paths: list[Path] = _paths_argument,
    data_dir: Path | None = _data_dir_option,
    force: bool = _force_option,
) -> None:
    """Copy files into the knowledge base and ingest them."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.cli import _state

    try:
        if _state["json_mode"]:
            from lilbee.ingest import sync

            copied = _copy_paths(paths, console, force=force)
            result = asyncio.run(sync(quiet=True))
            _json_output({"command": "add", "copied": copied, "sync": _sync_result_to_json(result)})
            return
        _add_paths(paths, console, force=force)
    except RuntimeError as exc:
        if _state["json_mode"]:
            _json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None


_chunks_source_argument = typer.Argument(..., help="Source name to inspect chunks for.")


@app.command()
def chunks(
    source: str = _chunks_source_argument,
    data_dir: Path | None = _data_dir_option,
) -> None:
    """Show chunks a document was split into (useful for debugging retrieval)."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.cli import _state
    from lilbee.store import get_chunks_by_source, get_sources

    known = {s["filename"] for s in get_sources()}
    if source not in known:
        if _state["json_mode"]:
            _json_output({"error": f"Source not found: {source}"})
            raise SystemExit(1)
        console.print(f"[red]Source not found:[/red] {source}")
        raise SystemExit(1)

    raw_chunks = get_chunks_by_source(source)
    cleaned = sorted(
        [_clean_result(c) for c in raw_chunks],
        key=lambda c: c.get("chunk_index", 0),
    )

    if _state["json_mode"]:
        _json_output({"command": "chunks", "source": source, "chunks": cleaned})
        return

    console.print(f"[bold]{len(cleaned)}[/bold] chunks from [cyan]{source}[/cyan]\n")
    for c in cleaned:
        idx = c.get("chunk_index", "?")
        preview = c.get("chunk", "")[:_CHUNK_PREVIEW_LEN]
        if len(c.get("chunk", "")) > _CHUNK_PREVIEW_LEN:
            preview += "..."
        console.print(f"  [{idx}] {preview}")


_remove_names_argument = typer.Argument(
    ..., help="Source name(s) to remove from the knowledge base."
)

_delete_file_option = typer.Option(
    False, "--delete", help="Also delete the file from the documents directory."
)


@app.command()
def remove(
    names: list[str] = _remove_names_argument,
    data_dir: Path | None = _data_dir_option,
    delete_file: bool = _delete_file_option,
) -> None:
    """Remove documents from the knowledge base by source name."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.cli import _state
    from lilbee.store import delete_by_source, delete_source, get_sources

    known = {s["filename"] for s in get_sources()}
    removed: list[str] = []
    not_found: list[str] = []

    for name in names:
        if name not in known:
            not_found.append(name)
            continue
        delete_by_source(name)
        delete_source(name)
        removed.append(name)
        if delete_file:
            import lilbee.config as cfg

            path = cfg.DOCUMENTS_DIR / name
            if path.exists():
                path.unlink()

    if _state["json_mode"]:
        payload: dict = {"command": "remove", "removed": removed}
        if not_found:
            payload["not_found"] = not_found
        _json_output(payload)
        return

    for name in removed:
        console.print(f"Removed [cyan]{name}[/cyan]")
    for name in not_found:
        console.print(f"[red]Not found:[/red] {name}")
    if not removed and not_found:
        raise SystemExit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    data_dir: Path | None = _data_dir_option,
    model: str | None = _model_option,
) -> None:
    """Ask a one-shot question (auto-syncs first)."""
    _apply_overrides(data_dir=data_dir, model=model)
    _auto_sync(console)
    from lilbee.cli import _state

    try:
        if _state["json_mode"]:
            from lilbee.query import ask_raw

            result = ask_raw(question)
            _json_output(
                {
                    "command": "ask",
                    "question": question,
                    "answer": result.answer,
                    "sources": [_clean_result(s) for s in result.sources],
                }
            )
            return

        from lilbee.query import ask_stream

        for token in ask_stream(question):
            console.print(token, end="")
        console.print()
    except RuntimeError as exc:
        if _state["json_mode"]:
            _json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None


@app.command()
def chat(
    data_dir: Path | None = _data_dir_option,
    model: str | None = _model_option,
) -> None:
    """Interactive chat loop (auto-syncs first)."""
    _apply_overrides(data_dir=data_dir, model=model)
    _auto_sync(console)
    from lilbee.cli._chat import _chat_loop

    _chat_loop(console)


@app.command()
def version() -> None:
    """Show the lilbee version."""
    from lilbee.cli import _state

    ver = _get_version()
    if _state["json_mode"]:
        _json_output({"command": "version", "version": ver})
        return
    console.print(f"lilbee {ver}")


@app.command()
def status(data_dir: Path | None = _data_dir_option) -> None:
    """Show indexed documents, paths, and chunk counts."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.cli import _state

    if _state["json_mode"]:
        _json_output(_gather_status())
        return
    _render_status(console)


_yes_option = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt.")


@app.command()
def reset(
    data_dir: Path | None = _data_dir_option,
    yes: bool = _yes_option,
) -> None:
    """Delete all documents and data (full factory reset)."""
    _apply_overrides(data_dir=data_dir)
    import lilbee.config as cfg
    from lilbee.cli import _state

    if not yes:
        if _state["json_mode"]:
            _json_output({"error": "Use --yes to confirm reset in JSON mode"})
            raise SystemExit(1)
        console.print(
            f"[bold red]This will delete ALL documents and data.[/bold red]\n"
            f"  Documents: {cfg.DOCUMENTS_DIR}\n"
            f"  Data:      {cfg.DATA_DIR}"
        )
        confirmed = typer.confirm("Are you sure?", default=False)
        if not confirmed:
            console.print("Aborted.")
            raise SystemExit(0)

    result = _perform_reset()

    if _state["json_mode"]:
        _json_output(result)
        return

    console.print(
        f"Reset complete: {result['deleted_docs']} document(s), "
        f"{result['deleted_data']} data item(s) deleted."
    )


@app.command(name="mcp")
def mcp_cmd() -> None:
    """Start the MCP server (stdio transport) for agent integration."""
    from lilbee.mcp import main

    main()
