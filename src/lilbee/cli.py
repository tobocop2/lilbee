"""CLI entry point for lilbee."""

import shutil
from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="lilbee — Local RAG knowledge base", invoke_without_command=True)
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

_paths_argument = typer.Argument(
    ...,
    exists=True,
    help="Files or directories to add to the knowledge base.",
)


# ---------------------------------------------------------------------------
# Extracted helpers (shared by commands and slash commands)
# ---------------------------------------------------------------------------


def _render_status(con: Console) -> None:
    """Print status info (documents, paths, chunk counts)."""
    from lilbee.config import CHAT_MODEL, DATA_DIR, DOCUMENTS_DIR, EMBEDDING_MODEL
    from lilbee.store import get_sources

    con.print(f"[bold]Documents:[/bold]  {DOCUMENTS_DIR}")
    con.print(f"[bold]Database:[/bold]   {DATA_DIR}")
    con.print(f"[bold]Chat model:[/bold] {CHAT_MODEL}")
    con.print(f"[bold]Embeddings:[/bold] {EMBEDDING_MODEL}")
    con.print()

    sources = get_sources()
    if not sources:
        con.print(
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

    con.print(table)
    con.print(f"\n[bold]{len(sources)}[/bold] documents, [bold]{total_chunks}[/bold] chunks")


def _add_paths(paths: list[Path], con: Console) -> None:
    """Copy *paths* into the knowledge base and sync."""
    import lilbee.config as cfg
    from lilbee.ingest import sync

    cfg.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for p in paths:
        dest = cfg.DOCUMENTS_DIR / p.name
        if p.is_dir():
            shutil.copytree(p, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(p, dest)
        copied.append(p.name)

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


# ---------------------------------------------------------------------------
# Slash-command dispatch
# ---------------------------------------------------------------------------


def _handle_slash_status(args: str, con: Console) -> None:
    _render_status(con)


def _handle_slash_add(args: str, con: Console) -> None:
    raw = args.strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.exists():
            con.print(f"[red]Path not found:[/red] {raw}")
            return
        _add_paths([p], con)
    else:
        try:
            from prompt_toolkit import prompt as pt_prompt
            from prompt_toolkit.completion import PathCompleter

            raw = pt_prompt("path: ", completer=PathCompleter(expanduser=True))
        except (ImportError, EOFError, KeyboardInterrupt):
            return
        raw = raw.strip()
        if not raw:
            return
        p = Path(raw).expanduser()
        if not p.exists():
            con.print(f"[red]Path not found:[/red] {raw}")
            return
        _add_paths([p], con)


class _QuitChat(Exception):
    """Raised by /quit to exit the chat loop."""


def _handle_slash_quit(args: str, con: Console) -> None:
    raise _QuitChat


def _handle_slash_help(args: str, con: Console) -> None:
    con.print("[bold]Slash commands:[/bold]")
    con.print("  /status  — show indexed documents and config")
    con.print("  /add [path]  — add a file or directory (tab-completes without args)")
    con.print("  /help    — show this help")
    con.print("  /quit    — exit chat")


_SLASH_COMMANDS: dict[str, Callable[[str, Console], None]] = {
    "status": _handle_slash_status,
    "add": _handle_slash_add,
    "help": _handle_slash_help,
    "quit": _handle_slash_quit,
}


def _dispatch_slash(raw_input: str, con: Console) -> bool:
    """Try to dispatch *raw_input* as a ``/command``.  Returns True if handled."""
    stripped = raw_input.strip()
    if not stripped.startswith("/"):
        return False
    parts = stripped[1:].split(None, 1)
    cmd = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    handler = _SLASH_COMMANDS.get(cmd)
    if handler is None:
        con.print(f"[red]Unknown command:[/red] /{cmd}  — try /help")
        return True
    handler(args, con)
    return True


# ---------------------------------------------------------------------------
# Chat loop (shared by _default callback and chat command)
# ---------------------------------------------------------------------------


_ADD_PREFIX = "/add "


def _make_completer():  # type: ignore[no-untyped-def]
    """Build a completer class that inherits from prompt_toolkit.completion.Completer."""
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document

    class LilbeeCompleter(Completer):
        def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def,override]
            text = document.text_before_cursor
            if text.startswith(_ADD_PREFIX):
                sub_text = text[len(_ADD_PREFIX) :]
                sub_doc = Document(sub_text, len(sub_text))
                yield from PathCompleter(expanduser=True).get_completions(sub_doc, complete_event)
            elif text.startswith("/"):
                prefix = text[1:]
                for cmd in _SLASH_COMMANDS:
                    if cmd.startswith(prefix):
                        yield Completion(f"/{cmd}", start_position=-len(text))

    return LilbeeCompleter()


def _chat_loop(con: Console) -> None:
    """Interactive REPL with slash-command support."""
    import sys

    con.print("[bold]lilbee chat[/bold] — type /help for commands\n")
    history: list[dict] = []

    _prompt_fn: Callable[[], str] | None = None
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession

            _session: PromptSession[str] = PromptSession(completer=_make_completer())
            _prompt_fn = lambda: _session.prompt("> ")  # noqa: E731
        except ImportError:
            pass

    while True:
        try:
            if _prompt_fn is not None:
                question = _prompt_fn()
            else:
                question = con.input("[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            break
        if not question.strip():
            continue
        try:
            if _dispatch_slash(question, con):
                continue
        except _QuitChat:
            break
        _stream_response(question, history, con)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.callback()
def _default(
    ctx: typer.Context,
    data_dir: Path | None = _data_dir_option,
    model: str | None = _model_option,
) -> None:
    """Start interactive chat when no command is given."""
    if ctx.invoked_subcommand is None:
        _apply_overrides(data_dir=data_dir, model=model)
        _auto_sync()
        _chat_loop(console)


def _auto_sync() -> None:
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
        console.print(
            f"[dim]Synced: {len(result['added'])} added, "
            f"{len(result['updated'])} updated, "
            f"{len(result['removed'])} removed, "
            f"{len(result.get('failed', []))} failed[/dim]"
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
    console.print(f"Failed: {len(result['failed'])}")
    for f in result.get("failed", []):
        console.print(f"  [red]{f}[/red]")


@app.command()
def rebuild(data_dir: Path | None = _data_dir_option) -> None:
    """Nuke the DB and re-ingest everything from documents/."""
    _apply_overrides(data_dir=data_dir)
    from lilbee.ingest import sync

    result = sync(force_rebuild=True)
    console.print(f"Rebuilt: {len(result['added'])} documents ingested")


@app.command()
def add(
    paths: list[Path] = _paths_argument,
    data_dir: Path | None = _data_dir_option,
) -> None:
    """Copy files into the knowledge base and ingest them."""
    _apply_overrides(data_dir=data_dir)
    _add_paths(paths, console)


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
    _chat_loop(console)


@app.command()
def status(data_dir: Path | None = _data_dir_option) -> None:
    """Show indexed documents, paths, and chunk counts."""
    _apply_overrides(data_dir=data_dir)
    _render_status(console)
