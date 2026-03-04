"""CLI entry point for lilbee."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="lilbee — Local RAG knowledge base")
console = Console()


def _apply_overrides(
    data_dir: Path | None = None,
    model: str | None = None,
) -> None:
    """Apply CLI overrides to config before any work begins."""
    import lilbee.config as cfg
    import lilbee.store as store_mod

    if data_dir is not None:
        cfg.DOCUMENTS_DIR = data_dir / "documents"
        cfg.DATA_DIR = data_dir / "data"
        cfg.LANCEDB_DIR = data_dir / "data" / "lancedb"
        store_mod.LANCEDB_DIR = cfg.LANCEDB_DIR

    if model is not None:
        cfg.CHAT_MODEL = model


_data_dir_option = typer.Option(
    None,
    "--data-dir",
    "-d",
    help="Override data directory (default: platform-specific, see 'lilbee status')",
)

_model_option = typer.Option(
    None,
    "--model",
    "-m",
    help="Override chat model (default: $LILBEE_CHAT_MODEL or 'mistral')",
)


def _auto_sync() -> None:
    """Run document sync before queries."""
    from lilbee.ingest import sync

    result = sync()
    total = len(result["added"]) + len(result["updated"]) + len(result["removed"])
    if total:
        console.print(
            f"[dim]Synced: {len(result['added'])} added, "
            f"{len(result['updated'])} updated, "
            f"{len(result['removed'])} removed[/dim]"
        )


@app.command(name="sync")
def sync_cmd(data_dir: Path | None = _data_dir_option) -> None:
    """Manually trigger document sync."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.ingest import sync

    result = sync()
    console.print(f"Added: {len(result['added'])}")
    console.print(f"Updated: {len(result['updated'])}")
    console.print(f"Removed: {len(result['removed'])}")
    console.print(f"Unchanged: {result['unchanged']}")


@app.command()
def rebuild(data_dir: Path | None = _data_dir_option) -> None:
    """Nuke the DB and re-ingest everything from documents/."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.ingest import sync

    result = sync(force_rebuild=True)
    console.print(f"Rebuilt: {len(result['added'])} documents ingested")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    data_dir: Path | None = _data_dir_option,
    model: str | None = _model_option,
) -> None:
    """Ask a one-shot question (auto-syncs first)."""
    _apply_overrides(data_dir=data_dir, model=model)
    _auto_sync()
    from lilbee.query import ask_stream

    for token in ask_stream(question):
        console.print(token, end="")
    console.print()


@app.command()
def chat(
    data_dir: Path | None = _data_dir_option,
    model: str | None = _model_option,
) -> None:
    """Interactive chat loop (auto-syncs first)."""
    _apply_overrides(data_dir=data_dir, model=model)
    _auto_sync()
    from lilbee.query import ask_stream

    console.print("[bold]lilbee chat[/bold] — type 'quit' to exit\n")
    while True:
        try:
            question = console.input("[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in ("quit", "exit", "q"):
            break
        if not question.strip():
            continue
        for token in ask_stream(question):
            console.print(token, end="")
        console.print("\n")


@app.command()
def status(data_dir: Path | None = _data_dir_option) -> None:
    """Show indexed documents, paths, and chunk counts."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.config import CHAT_MODEL, DATA_DIR, DOCUMENTS_DIR, EMBEDDING_MODEL
    from lilbee.store import get_sources

    console.print(f"[bold]Documents:[/bold]  {DOCUMENTS_DIR}")
    console.print(f"[bold]Database:[/bold]   {DATA_DIR}")
    console.print(f"[bold]Chat model:[/bold] {CHAT_MODEL}")
    console.print(f"[bold]Embeddings:[/bold] {EMBEDDING_MODEL}")
    console.print()

    sources = get_sources()
    if not sources:
        console.print(
            "No documents indexed. Drop files into the documents directory and run 'lilbee sync'."
        )
        return

    table = Table(title="Indexed Documents")
    table.add_column("File", style="cyan")
    table.add_column("Hash", style="dim", max_width=12)
    table.add_column("Chunks", justify="right")
    table.add_column("Ingested", style="dim")

    total_chunks = 0
    for s in sorted(sources, key=lambda x: x["filename"]):
        table.add_row(
            s["filename"],
            s["file_hash"][:12],
            str(s["chunk_count"]),
            s["ingested_at"][:19],
        )
        total_chunks += s["chunk_count"]

    console.print(table)
    console.print(f"\n[bold]{len(sources)}[/bold] documents, [bold]{total_chunks}[/bold] chunks")
