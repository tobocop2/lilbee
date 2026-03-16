"""CLI command definitions registered on the app."""

import asyncio
from pathlib import Path

import typer
from rich.table import Table

from lilbee import settings
from lilbee.cli.app import (
    _global_option,
    app,
    apply_overrides,
    console,
    data_dir_option,
    model_option,
    num_ctx_option,
    repeat_penalty_option,
    seed_option,
    temperature_option,
    top_k_sampling_option,
    top_p_option,
)
from lilbee.cli.helpers import (
    add_paths,
    auto_sync,
    clean_result,
    copy_paths,
    gather_status,
    get_version,
    json_output,
    perform_reset,
    render_status,
    sync_result_to_json,
)
from lilbee.config import cfg

CHUNK_PREVIEW_LEN = 80  # characters shown in human-readable search output

_vision_option = typer.Option(False, "--vision", help="Enable vision OCR for scanned PDFs.")
_vision_timeout_option = typer.Option(
    None,
    "--vision-timeout",
    help="Per-page timeout in seconds for vision OCR (default: 120, 0 = no limit).",
)


def _ensure_vision_model() -> None:
    """Ensure a vision model is configured and available for this run."""
    if cfg.vision_model:
        _validate_configured_vision()
        return

    # Restore persisted model from TOML (--vision is explicit even if model was cleared)
    saved = settings.get(cfg.data_root, "vision_model") or ""
    if saved:
        cfg.vision_model = saved
        _validate_configured_vision()
        return

    import sys

    from lilbee.cli.chat import list_ollama_models

    try:
        installed = set(list_ollama_models())
    except Exception:
        console.print("[yellow]Warning: Cannot connect to Ollama. Vision OCR disabled.[/yellow]")
        return

    if sys.stdin.isatty():
        _pick_vision_interactive(installed)
    else:
        _pick_vision_auto(installed)


def _validate_configured_vision() -> None:
    """Check that a pre-configured vision model is available; pull if needed."""
    from lilbee.cli.chat import list_ollama_models
    from lilbee.models import ensure_tag

    tagged = ensure_tag(cfg.vision_model)
    cfg.vision_model = tagged

    try:
        installed = set(list_ollama_models())
    except Exception:
        # Can't reach Ollama — keep the config and let downstream handle errors
        return

    if tagged in installed:
        return

    console.print(f"Vision model '{tagged}' not installed. Pulling...")
    if not _try_pull(tagged):
        cfg.vision_model = ""


def _pick_vision_interactive(installed: set[str]) -> None:
    """Interactive vision model picker for TTY sessions."""
    from lilbee.models import (
        VISION_CATALOG,
        display_vision_picker,
        get_free_disk_gb,
        get_system_ram_gb,
    )

    ram_gb = get_system_ram_gb()
    free_gb = get_free_disk_gb(cfg.data_dir)
    recommended = display_vision_picker(ram_gb, free_gb)
    default_idx = list(VISION_CATALOG).index(recommended) + 1

    try:
        raw = input(f"Choice [{default_idx}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not raw:
        model_info = recommended
    else:
        try:
            choice = int(raw)
        except ValueError:
            console.print(f"[red]Enter a number 1-{len(VISION_CATALOG)}.[/red]")
            return
        if not (1 <= choice <= len(VISION_CATALOG)):
            console.print(f"[red]Enter a number 1-{len(VISION_CATALOG)}.[/red]")
            return
        model_info = VISION_CATALOG[choice - 1]

    _pull_and_save_vision(model_info.name, installed)


def _pick_vision_auto(installed: set[str]) -> None:
    """Non-interactive vision model auto-selection."""
    import sys

    from lilbee.models import pick_default_vision_model

    model_info = pick_default_vision_model()
    sys.stderr.write(f"No vision model configured. Auto-selecting '{model_info.name}'...\n")
    _pull_and_save_vision(model_info.name, installed)


def _try_pull(model_name: str) -> bool:
    """Attempt to pull a model. Returns True on success, False on failure."""
    from lilbee.models import pull_with_progress

    try:
        pull_with_progress(model_name)
    except Exception as exc:
        console.print(f"[yellow]Warning: Failed to pull '{model_name}': {exc}[/yellow]")
        console.print("[yellow]Continuing without vision OCR.[/yellow]")
        return False
    return True


def _pull_and_save_vision(model_name: str, installed: set[str]) -> None:
    """Pull if needed and persist vision model choice."""
    if model_name not in installed and not _try_pull(model_name):
        return

    cfg.vision_model = model_name
    settings.set_value(cfg.data_root, "vision_model", model_name)


_paths_argument = typer.Argument(
    ...,
    exists=True,
    help="Files or directories to add to the knowledge base.",
)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of results"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
    """Search the knowledge base for relevant chunks."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.query import search_context

    results = search_context(query, top_k=top_k or cfg.top_k)
    cleaned = [clean_result(r) for r in results]

    if cfg.json_mode:
        json_output({"command": "search", "query": query, "results": cleaned})
        return

    if not cleaned:
        console.print("No results found.")
        return

    has_relevance = any("relevance_score" in r for r in cleaned)
    table = Table(title="Search Results")
    table.add_column("Source", style="cyan")
    table.add_column("Chunk", max_width=80)
    score_label = "Score" if has_relevance else "Distance"
    table.add_column(score_label, justify="right", style="dim")

    for r in cleaned:
        chunk_text = r["chunk"]
        preview = chunk_text[:CHUNK_PREVIEW_LEN]
        if len(chunk_text) > CHUNK_PREVIEW_LEN:
            preview += "..."
        score = r.get("relevance_score") or r.get("distance") or 0
        table.add_row(r["source"], preview, f"{score:.4f}")
    console.print(table)


@app.command(name="sync")
def sync_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
    vision: bool = _vision_option,
    vision_timeout: float | None = _vision_timeout_option,
) -> None:
    """Manually trigger document sync."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if vision_timeout is not None:
        cfg.vision_timeout = vision_timeout
    if vision:
        _ensure_vision_model()
    from lilbee.ingest import sync

    try:
        result = asyncio.run(sync(quiet=cfg.json_mode, force_vision=vision))
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None
    if cfg.json_mode:
        json_output(sync_result_to_json(result))
        return
    console.print(result)


@app.command()
def rebuild(
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
    vision: bool = _vision_option,
    vision_timeout: float | None = _vision_timeout_option,
) -> None:
    """Nuke the DB and re-ingest everything from documents/."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if vision_timeout is not None:
        cfg.vision_timeout = vision_timeout
    if vision:
        _ensure_vision_model()
    from lilbee.ingest import sync

    try:
        result = asyncio.run(sync(force_rebuild=True, quiet=cfg.json_mode, force_vision=vision))
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None
    if cfg.json_mode:
        json_output({"command": "rebuild", "ingested": len(result.added)})
        return
    console.print(f"Rebuilt: {len(result.added)} documents ingested")


_force_option = typer.Option(False, "--force", "-f", help="Overwrite existing files.")


@app.command()
def add(
    paths: list[Path] = _paths_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
    force: bool = _force_option,
    vision: bool = _vision_option,
    vision_timeout: float | None = _vision_timeout_option,
) -> None:
    """Copy files into the knowledge base and ingest them."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if vision_timeout is not None:
        cfg.vision_timeout = vision_timeout
    if vision:
        _ensure_vision_model()
    try:
        if cfg.json_mode:
            from lilbee.ingest import sync

            copied = copy_paths(paths, console, force=force)
            result = asyncio.run(sync(quiet=True, force_vision=vision))
            json_output({"command": "add", "copied": copied, "sync": sync_result_to_json(result)})
            return
        add_paths(paths, console, force=force, force_vision=vision)
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None


_chunks_source_argument = typer.Argument(..., help="Source name to inspect chunks for.")


@app.command()
def chunks(
    source: str = _chunks_source_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
    """Show chunks a document was split into (useful for debugging retrieval)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    from lilbee.store import get_chunks_by_source, get_sources

    known = {s["filename"] for s in get_sources()}
    if source not in known:
        if cfg.json_mode:
            json_output({"error": f"Source not found: {source}"})
            raise SystemExit(1)
        console.print(f"[red]Source not found:[/red] {source}")
        raise SystemExit(1)

    raw_chunks = get_chunks_by_source(source)
    cleaned = sorted(
        [clean_result(c) for c in raw_chunks],
        key=lambda c: c.get("chunk_index", 0),
    )

    if cfg.json_mode:
        json_output({"command": "chunks", "source": source, "chunks": cleaned})
        return

    console.print(f"[bold]{len(cleaned)}[/bold] chunks from [cyan]{source}[/cyan]\n")
    for c in cleaned:
        idx = c.get("chunk_index", "?")
        preview = c.get("chunk", "")[:CHUNK_PREVIEW_LEN]
        if len(c.get("chunk", "")) > CHUNK_PREVIEW_LEN:
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
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
    delete_file: bool = _delete_file_option,
) -> None:
    """Remove documents from the knowledge base by source name."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

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
            path = cfg.documents_dir / name
            if path.exists():
                path.unlink()

    if cfg.json_mode:
        payload: dict = {"command": "remove", "removed": removed}
        if not_found:
            payload["not_found"] = not_found
        json_output(payload)
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
    data_dir: Path | None = data_dir_option,
    model: str | None = model_option,
    use_global: bool = _global_option,
    temperature: float | None = temperature_option,
    top_p: float | None = top_p_option,
    top_k_sampling: int | None = top_k_sampling_option,
    repeat_penalty: float | None = repeat_penalty_option,
    num_ctx: int | None = num_ctx_option,
    seed: int | None = seed_option,
) -> None:
    """Ask a one-shot question (auto-syncs first)."""
    apply_overrides(
        data_dir=data_dir,
        model=model,
        use_global=use_global,
        temperature=temperature,
        top_p=top_p,
        top_k_sampling=top_k_sampling,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        seed=seed,
    )

    from lilbee.embedder import validate_model
    from lilbee.models import ensure_chat_model

    ensure_chat_model()
    validate_model()
    auto_sync(console)

    try:
        if cfg.json_mode:
            from lilbee.query import ask_raw

            result = ask_raw(question)
            json_output(
                {
                    "command": "ask",
                    "question": question,
                    "answer": result.answer,
                    "sources": [clean_result(s) for s in result.sources],
                }
            )
            return

        from lilbee.query import ask_stream

        for token in ask_stream(question):
            console.print(token, end="")
        console.print()
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from None


@app.command()
def chat(
    data_dir: Path | None = data_dir_option,
    model: str | None = model_option,
    use_global: bool = _global_option,
    temperature: float | None = temperature_option,
    top_p: float | None = top_p_option,
    top_k_sampling: int | None = top_k_sampling_option,
    repeat_penalty: float | None = repeat_penalty_option,
    num_ctx: int | None = num_ctx_option,
    seed: int | None = seed_option,
) -> None:
    """Interactive chat loop (auto-syncs first)."""
    apply_overrides(
        data_dir=data_dir,
        model=model,
        use_global=use_global,
        temperature=temperature,
        top_p=top_p,
        top_k_sampling=top_k_sampling,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        seed=seed,
    )
    from lilbee.embedder import validate_model
    from lilbee.models import ensure_chat_model

    ensure_chat_model()
    validate_model()
    auto_sync(console)
    from lilbee.cli.chat import chat_loop

    chat_loop(console)


@app.command()
def version() -> None:
    """Show the lilbee version."""
    ver = get_version()
    if cfg.json_mode:
        json_output({"command": "version", "version": ver})
        return
    console.print(f"lilbee {ver}")


@app.command()
def status(
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
    """Show indexed documents, paths, and chunk counts."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if cfg.json_mode:
        json_output(gather_status().model_dump(exclude_none=True))
        return
    render_status(console)


_yes_option = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt.")


@app.command()
def reset(
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
    yes: bool = _yes_option,
) -> None:
    """Delete all documents and data (full factory reset)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not yes:
        if cfg.json_mode:
            json_output({"error": "Use --yes to confirm reset in JSON mode"})
            raise SystemExit(1)
        console.print(
            f"[bold red]This will delete ALL documents and data.[/bold red]\n"
            f"  Documents: {cfg.documents_dir}\n"
            f"  Data:      {cfg.data_dir}"
        )
        confirmed = typer.confirm("Are you sure?", default=False)
        if not confirmed:
            console.print("Aborted.")
            raise SystemExit(0)

    result = perform_reset()

    if cfg.json_mode:
        json_output(result.model_dump())
        return

    console.print(
        f"Reset complete: {result.deleted_docs} document(s), "
        f"{result.deleted_data} data item(s) deleted."
    )


@app.command()
def init() -> None:
    """Initialize a local .lilbee/ knowledge base in the current directory."""
    from pathlib import Path

    root = Path.cwd() / ".lilbee"
    if root.is_dir():
        if cfg.json_mode:
            json_output({"command": "init", "path": str(root), "created": False})
            return
        console.print(f"Already initialized: {root}")
        return

    docs = root / "documents"
    data = root / "data"
    docs.mkdir(parents=True)
    data.mkdir(parents=True)
    (root / ".gitignore").write_text("data/\n")

    if cfg.json_mode:
        json_output({"command": "init", "path": str(root), "created": True})
        return
    console.print(f"Initialized local knowledge base at {root}")


@app.command()
def serve(
    host: str = typer.Option(None, "--host", "-H", help="Bind address (default: 127.0.0.1)"),
    port: int = typer.Option(None, "--port", "-p", help="Port (default: 7433)"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
    """Start the HTTP API server for Obsidian and other clients."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if host is not None:
        cfg.server_host = host
    if port is not None:
        cfg.server_port = port

    import logging

    import uvicorn

    from lilbee.server import create_app

    logging.getLogger("asyncio").setLevel(logging.ERROR)

    uvicorn.run(
        create_app(),
        host=cfg.server_host,
        port=cfg.server_port,
    )


@app.command(name="mcp")
def mcp_cmd() -> None:
    """Start the MCP server (stdio transport) for agent integration."""
    from lilbee.mcp import main

    main()
