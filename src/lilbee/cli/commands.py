"""CLI command definitions registered on the app."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    import uvicorn
from rich.table import Table

from lilbee.cli import theme
from lilbee.cli.app import (
    app,
    apply_overrides,
    console,
    data_dir_option,
    global_option,
    model_option,
    num_ctx_option,
    repeat_penalty_option,
    seed_option,
    temperature_option,
    top_k_sampling_option,
    top_p_option,
)
from lilbee.cli.helpers import (
    CopyResult,
    add_paths,
    auto_sync,
    clean_result,
    copy_files,
    gather_status,
    get_version,
    json_output,
    perform_reset,
    render_status,
    sync_result_to_json,
)
from lilbee.config import cfg
from lilbee.crawler import is_url
from lilbee.services import get_services

CHUNK_PREVIEW_LEN = 80  # characters shown in human-readable search output

_ocr_option = typer.Option(None, "--ocr/--no-ocr", help="Force vision OCR on/off for scanned PDFs.")
_ocr_timeout_option = typer.Option(
    None,
    "--ocr-timeout",
    help="Per-page timeout in seconds for vision OCR (default: 120, 0 = no limit).",
)


def _apply_ocr_overrides(ocr: bool | None, ocr_timeout: float | None) -> None:
    """Apply --ocr/--no-ocr and --ocr-timeout CLI overrides to config."""
    if ocr is not None:
        cfg.enable_ocr = ocr
    if ocr_timeout is not None:
        cfg.ocr_timeout = ocr_timeout


_paths_argument = typer.Argument(
    ...,
    help="Files, directories, or URLs to add to the knowledge base.",
)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of results"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Search the knowledge base for relevant chunks."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    results = get_services().searcher.search(query, top_k=top_k or cfg.top_k)
    cleaned = [clean_result(r) for r in results]

    if cfg.json_mode:
        json_output({"command": "search", "query": query, "results": cleaned})
        return

    if not cleaned:
        console.print("No results found.")
        return

    has_relevance = any("relevance_score" in r for r in cleaned)
    table = Table(title="Search Results")
    table.add_column("Source", style=theme.ACCENT)
    table.add_column("Chunk", max_width=80)
    score_label = "Score" if has_relevance else "Distance"
    table.add_column(score_label, justify="right", style=theme.MUTED)

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
    use_global: bool = global_option,
    ocr: bool | None = _ocr_option,
    ocr_timeout: float | None = _ocr_timeout_option,
) -> None:
    """Manually trigger document sync."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _apply_ocr_overrides(ocr, ocr_timeout)
    from lilbee.ingest import sync

    try:
        result = asyncio.run(sync(quiet=cfg.json_mode))
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None
    if cfg.json_mode:
        json_output(sync_result_to_json(result))
        return
    console.print(result)


@app.command()
def rebuild(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    ocr: bool | None = _ocr_option,
    ocr_timeout: float | None = _ocr_timeout_option,
) -> None:
    """Nuke the DB and re-ingest everything from documents/."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _apply_ocr_overrides(ocr, ocr_timeout)
    from lilbee.ingest import sync

    try:
        result = asyncio.run(sync(force_rebuild=True, quiet=cfg.json_mode))
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None
    if cfg.json_mode:
        json_output({"command": "rebuild", "ingested": len(result.added)})
        return
    console.print(f"Rebuilt: {len(result.added)} documents ingested")


_force_option = typer.Option(False, "--force", "-f", help="Overwrite existing files.")
_crawl_option = typer.Option(False, "--crawl", help="Recursively crawl URLs (follow links).")
_depth_option = typer.Option(None, "--depth", help="Maximum crawl depth (default: from config).")
_max_pages_option = typer.Option(
    None, "--max-pages", help="Maximum pages to crawl (default: from config)."
)


def _partition_inputs(inputs: list[str]) -> tuple[list[Path], list[str]]:
    """Split inputs into file paths and URLs."""
    paths: list[Path] = []
    urls: list[str] = []
    for inp in inputs:
        if is_url(inp):
            urls.append(inp)
        else:
            paths.append(Path(inp))
    return paths, urls


def _crawl_urls_blocking(
    urls: list[str], *, crawl: bool, depth: int | None, max_pages: int | None
) -> list[Path]:
    """Crawl URLs synchronously (for CLI), returning paths written."""
    from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

    from lilbee.crawler import crawl_and_save
    from lilbee.progress import CrawlPageEvent, DetailedProgressCallback, EventType, ProgressEvent

    effective_depth = depth if depth is not None else (cfg.crawl_max_depth if crawl else 0)
    effective_pages = max_pages if max_pages is not None else cfg.crawl_max_pages

    all_paths: list[Path] = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        for url in urls:
            ptask = progress.add_task(f"Crawling {url}...", total=None)

            def _make_callback(_t: TaskID = ptask) -> DetailedProgressCallback:
                def on_progress(event_type: EventType, data: ProgressEvent) -> None:
                    if event_type == EventType.CRAWL_PAGE:
                        if not isinstance(data, CrawlPageEvent):
                            raise TypeError(f"Expected CrawlPageEvent, got {type(data).__name__}")
                        progress.update(
                            _t, description=f"Crawled {data.current}/{data.total}: {data.url}"
                        )

                return on_progress

            paths = asyncio.run(
                crawl_and_save(
                    url,
                    depth=effective_depth,
                    max_pages=effective_pages,
                    on_progress=_make_callback(),
                )
            )
            all_paths.extend(paths)
            progress.update(ptask, description=f"Done: {url} ({len(paths)} pages)")
    return all_paths


@app.command()
def add(
    paths: list[str] = _paths_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    force: bool = _force_option,
    ocr: bool | None = _ocr_option,
    ocr_timeout: float | None = _ocr_timeout_option,
    crawl: bool = _crawl_option,
    depth: int | None = _depth_option,
    max_pages: int | None = _max_pages_option,
) -> None:
    """Copy files or crawl URLs into the knowledge base and ingest them."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _apply_ocr_overrides(ocr, ocr_timeout)

    file_paths, urls = _partition_inputs(paths)
    # Validate file paths exist
    for fp in file_paths:
        if not fp.exists():
            if cfg.json_mode:
                json_output({"error": f"Path not found: {fp}"})
                raise SystemExit(1)
            console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] Path not found: {fp}")
            raise SystemExit(1)

    try:
        # Crawl URLs first (saves .md files into documents/_web/)
        crawled_paths: list[Path] = []
        if urls:
            from lilbee.crawler import crawler_available

            if not crawler_available():
                console.print(
                    f"[{theme.ERROR}]Web crawling requires: "
                    f"pip install 'lilbee[crawler]'[/{theme.ERROR}]"
                )
                raise SystemExit(1)
            crawled_paths = _crawl_urls_blocking(
                urls, crawl=crawl, depth=depth, max_pages=max_pages
            )
            if not cfg.json_mode:
                console.print(
                    f"[{theme.MUTED}]Crawled {len(crawled_paths)} page(s)"
                    f" from {len(urls)} URL(s)[/{theme.MUTED}]"
                )

        if cfg.json_mode:
            from lilbee.ingest import sync

            copy_result = CopyResult()
            if file_paths:
                copy_result = copy_files(file_paths, force=force)
            result = asyncio.run(sync(quiet=True))
            json_output(
                {
                    "command": "add",
                    "copied": copy_result.copied,
                    "skipped": copy_result.skipped,
                    "crawled": len(crawled_paths),
                    "sync": sync_result_to_json(result),
                }
            )
            return

        if file_paths:
            add_paths(file_paths, console, force=force)
        elif urls:
            # URLs already saved; just trigger sync
            from lilbee.ingest import sync

            result = asyncio.run(sync())
            console.print(result)
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None


_chunks_source_argument = typer.Argument(..., help="Source name to inspect chunks for.")


@app.command()
def chunks(
    source: str = _chunks_source_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show chunks a document was split into (useful for debugging retrieval)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    store = get_services().store
    known = {s["filename"] for s in store.get_sources()}
    if source not in known:
        if cfg.json_mode:
            json_output({"error": f"Source not found: {source}"})
            raise SystemExit(1)
        console.print(f"[{theme.ERROR}]Source not found:[/{theme.ERROR}] {source}")
        raise SystemExit(1)

    raw_chunks = store.get_chunks_by_source(source)
    cleaned = sorted(
        [clean_result(c) for c in raw_chunks],
        key=lambda c: c.get("chunk_index", 0),
    )

    if cfg.json_mode:
        json_output({"command": "chunks", "source": source, "chunks": cleaned})
        return

    console.print(
        f"[{theme.LABEL}]{len(cleaned)}[/{theme.LABEL}]"
        f" chunks from [{theme.ACCENT}]{source}[/{theme.ACCENT}]\n"
    )
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
    use_global: bool = global_option,
    delete_file: bool = _delete_file_option,
) -> None:
    """Remove documents from the knowledge base by source name."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    result = get_services().store.remove_documents(
        names, delete_files=delete_file, documents_dir=cfg.documents_dir
    )

    if cfg.json_mode:
        payload: dict = {"command": "remove", "removed": result.removed}
        if result.not_found:
            payload["not_found"] = result.not_found
        json_output(payload)
        if not result.removed and result.not_found:
            raise SystemExit(1)
        return

    for name in result.removed:
        console.print(f"Removed [{theme.ACCENT}]{name}[/{theme.ACCENT}]")
    for name in result.not_found:
        console.print(f"[{theme.ERROR}]Not found:[/{theme.ERROR}] {name}")
    if not result.removed and result.not_found:
        raise SystemExit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    data_dir: Path | None = data_dir_option,
    model: str | None = model_option,
    use_global: bool = global_option,
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

    from lilbee.models import ensure_chat_model

    ensure_chat_model()
    get_services().embedder.validate_model()
    auto_sync(console)

    try:
        if cfg.json_mode:
            result = get_services().searcher.ask_raw(question)
            json_output(
                {
                    "command": "ask",
                    "question": question,
                    "answer": result.answer,
                    "sources": [clean_result(s) for s in result.sources],
                }
            )
            return

        for token in get_services().searcher.ask_stream(question):
            console.print(token.content, end="")
        console.print()
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None


@app.command()
def chat(
    data_dir: Path | None = data_dir_option,
    model: str | None = model_option,
    use_global: bool = global_option,
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

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] Chat requires a terminal.")
        raise SystemExit(1)
    from lilbee.cli.tui import run_tui

    run_tui(auto_sync=True)


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
    use_global: bool = global_option,
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
    use_global: bool = global_option,
    yes: bool = _yes_option,
) -> None:
    """Delete all documents and data (full factory reset)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not yes:
        if cfg.json_mode:
            json_output({"error": "Use --yes to confirm reset in JSON mode"})
            raise SystemExit(1)
        console.print(
            f"[{theme.ERROR_BOLD}]This will delete ALL documents and data.[/{theme.ERROR_BOLD}]\n"
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


def _port_file() -> Path:
    return cfg.data_dir / "server.port"


async def _run_server(server: uvicorn.Server, config: uvicorn.Config, host: str) -> None:
    """Start uvicorn, write port file, and clean up on shutdown."""
    port_path = _port_file()
    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)
    await server.startup()
    try:
        if server.servers:
            sock = server.servers[0].sockets[0]
            actual_port = sock.getsockname()[1]
            port_path.parent.mkdir(parents=True, exist_ok=True)
            port_path.write_text(str(actual_port))
            console.print(f"Listening on http://{host}:{actual_port}")
        await server.main_loop()
    finally:
        port_path.unlink(missing_ok=True)
        await server.shutdown()


@app.command()
def serve(
    host: str = typer.Option(None, "--host", "-H", help="Bind address (default: 127.0.0.1)"),
    port: int = typer.Option(None, "--port", "-p", help="Port (default: 0/random)"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Start the HTTP API server."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if host is not None:
        cfg.server_host = host
    if port is not None:
        cfg.server_port = port

    import logging

    import uvicorn

    from lilbee.server import create_app

    logging.getLogger("asyncio").setLevel(logging.ERROR)

    config = uvicorn.Config(create_app(), host=cfg.server_host, port=cfg.server_port)
    server = uvicorn.Server(config)
    asyncio.run(_run_server(server, config, cfg.server_host))


@app.command()
def topics(
    query: str = typer.Argument(None, help="Optional query to find related concepts."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show top concept communities or concepts related to a query."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    from lilbee.concepts import concepts_available

    if not concepts_available():
        msg = "Concept graph requires: pip install 'lilbee[graph]'"
        if cfg.json_mode:
            json_output({"error": msg})
            raise SystemExit(1)
        console.print(f"[{theme.ERROR}]{msg}[/{theme.ERROR}]")
        raise SystemExit(1)

    if not cfg.concept_graph:
        if cfg.json_mode:
            json_output({"error": "Concept graph is disabled (LILBEE_CONCEPT_GRAPH=false)"})
            raise SystemExit(1)
        console.print(
            f"[{theme.ERROR}]Concept graph is disabled.[/{theme.ERROR}] "
            "Enable with LILBEE_CONCEPT_GRAPH=true"
        )
        raise SystemExit(1)

    if not get_services().concepts.get_graph():
        if cfg.json_mode:
            json_output({"error": "Concept graph not available"})
            raise SystemExit(1)
        console.print(f"[{theme.ERROR}]Concept graph not available.[/{theme.ERROR}]")
        raise SystemExit(1)

    if query:
        _topics_for_query(query)
    else:
        _topics_overview(top_k)


def _topics_for_query(query: str) -> None:
    """Show concepts related to a query."""
    cg = get_services().concepts
    concepts = cg.extract_concepts(query)
    related = cg.expand_query(query)
    all_concepts = concepts + [r for r in related if r not in concepts]

    if cfg.json_mode:
        json_output({"command": "topics", "query": query, "concepts": all_concepts})
        return
    if not all_concepts:
        console.print("No concepts found for this query.")
        return
    console.print(f"Concepts related to [{theme.ACCENT}]{query}[/{theme.ACCENT}]:")
    for c in all_concepts:
        console.print(f"  {c}")


def _topics_overview(top_k: int) -> None:
    """Show top concept communities."""
    from dataclasses import asdict

    communities = get_services().concepts.top_communities(k=top_k)
    if cfg.json_mode:
        json_output({"command": "topics", "communities": [asdict(c) for c in communities]})
        return
    if not communities:
        console.print("No concept communities found. Try syncing some documents first.")
        return
    table = Table(title="Concept Communities")
    table.add_column("Cluster", justify="right", style=theme.MUTED)
    table.add_column("Size", justify="right")
    table.add_column("Top Concepts", style=theme.ACCENT)
    for comm in communities:
        preview = ", ".join(comm.concepts[:5])
        if len(comm.concepts) > 5:
            preview += f" (+{len(comm.concepts) - 5} more)"
        table.add_row(str(comm.cluster_id), str(comm.size), preview)
    console.print(table)


@app.command()
def login() -> None:
    """Log in to HuggingFace for access to gated models (Mistral, Llama, etc.)."""
    import webbrowser

    from huggingface_hub import get_token
    from huggingface_hub import login as hf_login

    if get_token():
        typer.echo("Already logged in to HuggingFace.")
        if not typer.confirm("Log in again?", default=False):
            return

    typer.echo("Opening HuggingFace token page in your browser...")
    typer.echo("Create a token with 'Read' access, then paste it below.\n")
    webbrowser.open("https://huggingface.co/settings/tokens")

    token = typer.prompt("Paste your HuggingFace token", hide_input=True)
    if not token.strip():
        typer.echo("No token provided.", err=True)
        raise typer.Exit(1)

    hf_login(token=token.strip(), add_to_git_credential=False)
    typer.echo("Logged in! Gated models (Mistral, Llama, etc.) are now accessible.")


@app.command(name="mcp")
def mcp_cmd() -> None:
    """Start the MCP server (stdio transport) for agent integration."""
    from lilbee.mcp import main

    main()


wiki_app = typer.Typer(help="Wiki layer commands: lint, citations, status.")
app.add_typer(wiki_app, name="wiki")


@wiki_app.command(name="lint")
def wiki_lint(
    wiki_source: str = typer.Argument("", help="Wiki page path (empty = lint all)."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Lint wiki pages for stale citations, missing sources, and unmarked claims."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.lint import lint_all as _lint_all
    from lilbee.wiki.lint import lint_wiki_page

    store = get_services().store
    if wiki_source:
        issues = lint_wiki_page(wiki_source, store)
    else:
        report = _lint_all(store)
        issues = report.issues

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_lint",
                "issues": [i.to_dict() for i in issues],
                "total": len(issues),
            }
        )
        return

    if not issues:
        console.print("No issues found.")
        return

    table = Table(title="Wiki Lint Issues")
    table.add_column("Page", style=theme.ACCENT)
    table.add_column("Severity")
    table.add_column("Message")
    for issue in issues:
        sev_style = theme.ERROR if issue.severity.value == "error" else theme.WARNING
        sev_text = f"[{sev_style}]{issue.severity.value}[/{sev_style}]"
        table.add_row(issue.wiki_source, sev_text, issue.message)
    console.print(table)


@wiki_app.command(name="citations")
def wiki_citations(
    wiki_source: str = typer.Argument(..., help="Wiki page path, e.g. wiki/summaries/doc.md."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show citations for a wiki page."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    records = get_services().store.get_citations_for_wiki(wiki_source)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_citations",
                "wiki_source": wiki_source,
                "citations": [dict(r) for r in records],
                "total": len(records),
            }
        )
        return

    if not records:
        console.print(f"No citations found for [{theme.ACCENT}]{wiki_source}[/{theme.ACCENT}]")
        return

    table = Table(title=f"Citations: {wiki_source}")
    table.add_column("Key", style=theme.ACCENT)
    table.add_column("Source")
    table.add_column("Type", style=theme.MUTED)
    table.add_column("Excerpt", max_width=60)
    for rec in records:
        excerpt = rec["excerpt"][:57] + "..." if len(rec["excerpt"]) > 60 else rec["excerpt"]
        table.add_row(rec["citation_key"], rec["source_filename"], rec["claim_type"], excerpt)
    console.print(table)


@wiki_app.command(name="status")
def wiki_status(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show wiki layer status: page counts and lint summary."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    wiki_root = cfg.data_root / cfg.wiki_dir
    if not wiki_root.exists():
        if cfg.json_mode:
            json_output({"wiki_enabled": cfg.wiki, "pages": 0, "issues": 0})
            return
        console.print("Wiki directory does not exist yet. Run sync with wiki enabled.")
        return

    from lilbee.wiki.shared import DRAFTS_SUBDIR, SUMMARIES_SUBDIR

    summaries = _count_md_files(wiki_root / SUMMARIES_SUBDIR)
    drafts = _count_md_files(wiki_root / DRAFTS_SUBDIR)

    from lilbee.wiki.lint import lint_all as _lint_all

    report = _lint_all(get_services().store)

    if cfg.json_mode:
        json_output(
            {
                "wiki_enabled": cfg.wiki,
                SUMMARIES_SUBDIR: summaries,
                DRAFTS_SUBDIR: drafts,
                "pages": summaries + drafts,
                "lint_errors": report.error_count,
                "lint_warnings": report.warning_count,
            }
        )
        return

    color = "green" if cfg.wiki else "red"
    label = "enabled" if cfg.wiki else "disabled"
    console.print(f"Wiki: [{color}]{label}[/{color}]")
    console.print(f"  Summaries: [{theme.LABEL}]{summaries}[/{theme.LABEL}]")
    console.print(f"  Drafts:    [{theme.LABEL}]{drafts}[/{theme.LABEL}]")
    if report.error_count or report.warning_count:
        console.print(
            f"  Lint: [{theme.ERROR}]{report.error_count} error(s)[/{theme.ERROR}], "
            f"[{theme.WARNING}]{report.warning_count} warning(s)[/{theme.WARNING}]"
        )
    else:
        console.print("  Lint: all clean")


@wiki_app.command(name="prune")
def wiki_prune(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Prune stale and orphaned wiki pages."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.prune import prune_wiki

    report = prune_wiki(get_services().store)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_prune",
                "records": [r.to_dict() for r in report.records],
                "archived": report.archived_count,
                "flagged": report.flagged_count,
            }
        )
        return

    if not report.records:
        console.print("No pages pruned.")
        return

    table = Table(title="Wiki Prune Results")
    table.add_column("Page", style=theme.ACCENT)
    table.add_column("Action")
    table.add_column("Reason")
    for rec in report.records:
        action_style = theme.ERROR if rec.action.value == "archived" else theme.WARNING
        action_text = f"[{action_style}]{rec.action.value}[/{action_style}]"
        table.add_row(rec.wiki_source, action_text, rec.reason)
    console.print(table)


def _count_md_files(directory: Path) -> int:
    """Count markdown files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.md")))
