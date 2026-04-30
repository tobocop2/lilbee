"""CLI command definitions registered on the app."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    import uvicorn

    from lilbee.wiki.entity_extractor import ExtractedEntity
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
from lilbee.cli.tui import messages as msg
from lilbee.config import cfg
from lilbee.crawler import CrawlerBrowserMissing, bootstrap_chromium, chromium_installed, is_url
from lilbee.progress import EventType, SetupProgressEvent
from lilbee.providers.base import ProviderError
from lilbee.services import get_services
from lilbee.store import SearchScope, scope_to_chunk_type
from lilbee.wiki.shared import (
    DRAFTS_SUBDIR,
    SUMMARIES_SUBDIR,
)

CHUNK_PREVIEW_LEN = 80  # characters shown in human-readable search output

_ocr_option = typer.Option(None, "--ocr/--no-ocr", help="Force vision OCR on/off for scanned PDFs.")
_ocr_timeout_option = typer.Option(
    None,
    "--ocr-timeout",
    help="Per-page timeout in seconds for vision OCR (default: 120, 0 = no limit).",
)
_scope_option = typer.Option(
    SearchScope.BOTH,
    "--scope",
    "-s",
    help="Restrict the pool to raw chunks, wiki pages, or both (default).",
    case_sensitive=False,
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
    scope: SearchScope = _scope_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Search the knowledge base for relevant chunks."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    if not query or not query.strip():
        if cfg.json_mode:
            json_output({"error": "query must not be empty"})
            raise SystemExit(1)
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] query must not be empty")
        raise SystemExit(1)

    try:
        results = get_services().searcher.search(
            query,
            top_k=top_k or cfg.top_k,
            chunk_type=scope_to_chunk_type(scope),
        )
    except Exception as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None
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
_crawl_option = typer.Option(
    False,
    "--crawl",
    help="Recursively crawl URLs (whole site by default; see --depth and --max-pages).",
)
_depth_option = typer.Option(
    None,
    "--depth",
    help="Cap link-follow depth for --crawl. Unset = unbounded; 0 = single URL only.",
)
_max_pages_option = typer.Option(
    None,
    "--max-pages",
    help="Cap total pages for --crawl. Unset = no limit; positive int = hard cap.",
)
_include_subdomains_option = typer.Option(
    False,
    "--include-subdomains",
    help=(
        "Allow --crawl to follow links into sibling subdomains of the start "
        "host (e.g. en.wikipedia.org plus af.wikipedia.org). Default scopes "
        "the crawl to the exact start host only."
    ),
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
    urls: list[str],
    *,
    crawl: bool,
    depth: int | None,
    max_pages: int | None,
    include_subdomains: bool = False,
) -> list[Path]:
    """Crawl URLs synchronously (for CLI), returning paths written.

    Without --crawl, each URL is fetched as a single page (depth=0).
    With --crawl, the default is whole-site unbounded (depth=None, pages=None).
    Explicit --depth / --max-pages override both.

    Ctrl-C is handled by running the crawl through _run_crawl_with_signal_cancel,
    which installs a signal.signal handler that sets a threading.Event passed
    into crawl_and_save. crawl_recursive polls the event between pages so the
    signal flows through as a clean cancel instead of asyncio.run's default
    KeyboardInterrupt-raising (which left browser contexts mid-teardown).
    """
    import threading

    from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

    from lilbee.crawler import crawl_and_save
    from lilbee.progress import CrawlPageEvent, DetailedProgressCallback, EventType, ProgressEvent

    if crawl:
        effective_depth = depth
        effective_pages = max_pages
    else:
        effective_depth = 0
        effective_pages = None

    cancel_event = threading.Event()

    from rich.console import Console as RichConsole

    err_console = RichConsole(stderr=True)
    all_paths: list[Path] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
        console=err_console,
        disable=cfg.json_mode,
    ) as progress:
        for url in urls:
            if cancel_event.is_set():
                break
            ptask = progress.add_task(f"Crawling {url}...", total=None)

            def _make_callback(_t: TaskID = ptask) -> DetailedProgressCallback:
                def on_progress(event_type: EventType, data: ProgressEvent) -> None:
                    if event_type == EventType.CRAWL_PAGE:
                        if not isinstance(data, CrawlPageEvent):
                            raise TypeError(f"Expected CrawlPageEvent, got {type(data).__name__}")
                        total_str = str(data.total) if data.total > 0 else "?"
                        progress.update(
                            _t,
                            description=f"Crawled {data.current}/{total_str}: {data.url}",
                        )

                return on_progress

            paths = _run_crawl_with_signal_cancel(
                url,
                depth=effective_depth,
                max_pages=effective_pages,
                on_progress=_make_callback(),
                cancel_event=cancel_event,
                crawl_and_save=crawl_and_save,
                include_subdomains=include_subdomains,
            )
            all_paths.extend(paths)
            progress.update(ptask, description=f"Done: {url} ({len(paths)} pages)")
    return all_paths


def _run_crawl_with_signal_cancel(
    url: str,
    *,
    depth: int | None,
    max_pages: int | None,
    on_progress: object,
    cancel_event: object,
    crawl_and_save: object,
    include_subdomains: bool = False,
) -> list[Path]:
    """Run crawl_and_save on a dedicated event loop with a SIGINT->cancel hook.

    asyncio.run() installs its own SIGINT handler that raises
    KeyboardInterrupt, which tears the crawl down ungracefully. Registering a
    plain signal.signal handler on the main thread AND running the crawl on a
    loop we own (instead of asyncio.run) lets Ctrl-C set our threading.Event,
    which crawl_recursive polls between pages so it can close the stream and
    stop dispatch cleanly.
    """
    import signal

    previous_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(_signum: int, _frame: object) -> None:
        # Set the cancel event that crawl_recursive polls between pages, so
        # a Ctrl-C flows through as a clean cancel instead of asyncio.run's
        # default KeyboardInterrupt-raising dance.
        cancel_event.set()  # type: ignore[attr-defined]

    signal.signal(signal.SIGINT, _on_sigint)
    # Manage the event loop explicitly. In the CLI this runs once per process,
    # but under pytest-xdist the same worker thread runs many tests; leaving a
    # closed loop set as the "current" loop for the thread poisons every later
    # asyncio.get_event_loop() call and hangs macOS 3.12/3.13 unit-test CI.
    # Always clear the thread-current loop in finally.
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        coro = crawl_and_save(  # type: ignore[operator]
            url,
            depth=depth,
            max_pages=max_pages,
            on_progress=on_progress,
            cancel=cancel_event,
            quiet=cfg.json_mode,
            include_subdomains=include_subdomains,
        )
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
        signal.signal(signal.SIGINT, previous_handler)


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
    include_subdomains: bool = _include_subdomains_option,
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
                urls,
                crawl=crawl,
                depth=depth,
                max_pages=max_pages,
                include_subdomains=include_subdomains,
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
    scope: SearchScope = _scope_option,
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

    try:
        from lilbee.models import ensure_chat_model

        ensure_chat_model()
        get_services().embedder.validate_model()
        if cfg.json_mode:
            from rich.console import Console as _QuietConsole

            auto_sync(_QuietConsole(quiet=True))
        else:
            auto_sync(console)

        chunk_type = scope_to_chunk_type(scope)

        if cfg.json_mode:
            result = get_services().searcher.ask_raw(question, chunk_type=chunk_type)
            json_output(
                {
                    "command": "ask",
                    "question": question,
                    "answer": result.answer,
                    "sources": [clean_result(s) for s in result.sources],
                }
            )
            return

        for token in get_services().searcher.ask_stream(question, chunk_type=chunk_type):
            console.print(token.content, end="")
        console.print()
    except (RuntimeError, ProviderError) as exc:
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

    if cfg.json_mode:
        json_output({"error": "Chat requires a terminal, not --json"})
        raise SystemExit(1)
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


_SELF_CHECK_CHAT_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_SELF_CHECK_CHAT_FILE = "SmolLM2-135M-Instruct-Q3_K_S.gguf"
_SELF_CHECK_EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
_SELF_CHECK_EMBED_FILE = "nomic-embed-text-v1.5.Q4_K_M.gguf"


def _download_self_check_model(repo: str, filename: str) -> Path:
    """Fetch a GGUF from the HuggingFace CDN via urllib (stdlib only).

    Avoids huggingface_hub / httpx entirely. Inside the PyInstaller --onefile
    bundle, huggingface_hub's retry path has re-entered a closed httpx client
    after transient DNS failures on macOS runners. urllib is synchronous,
    lives in the stdlib, and has no long-lived client to close.
    """
    import tempfile
    import urllib.request

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest_dir = Path(tempfile.mkdtemp(prefix="lilbee-self-check-"))
    dest = dest_dir / filename
    console.print(f"Downloading {url}")
    last_exc: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 — literal https url
                dest.write_bytes(response.read())
            return dest
        except (OSError, urllib.error.URLError) as exc:
            last_exc = exc
            console.print(f"download attempt {attempt + 1} failed: {exc!r}")
    raise RuntimeError(f"GGUF download failed after 3 attempts: {last_exc!r}")


_self_check_chat_path_option = typer.Option(
    None,
    "--chat-model-path",
    help="Path to a chat GGUF file. Skips the HuggingFace download.",
)
_self_check_embed_path_option = typer.Option(
    None,
    "--embed-model-path",
    help="Path to an embedding GGUF file. Skips the HuggingFace download.",
)
_self_check_max_tokens_option = typer.Option(5, "--max-tokens", help="Tokens to generate.")
_self_check_skip_embedding_option = typer.Option(
    False,
    "--skip-embedding",
    help="Skip the embedding-model leg of the self-check.",
)


def _self_check_emit_failure(error: str) -> None:
    if cfg.json_mode:
        json_output({"ok": False, "error": error})
    else:
        console.print(f"[{theme.ERROR}]SELF-CHECK FAILED:[/{theme.ERROR}] {error}")


@app.command("self-check")
def self_check_cmd(
    chat_model_path: Path | None = _self_check_chat_path_option,
    embed_model_path: Path | None = _self_check_embed_path_option,
    max_tokens: int = _self_check_max_tokens_option,
    skip_embedding: bool = _self_check_skip_embedding_option,
) -> None:
    """Verify the installation can load llama.cpp and run real inference.

    Two legs:

    1. **Chat**: downloads ``SmolLM2-135M-Instruct-Q3_K_S.gguf`` (~90MB) and
       runs a tiny ``create_completion`` so we know decoder-style models work
       end-to-end and the vendored shared libraries load.
    2. **Embedding**: downloads ``nomic-embed-text-v1.5.Q4_K_M.gguf`` (~84MB)
       and runs ``create_embedding``. This is the leg that catches the
       "Memory is not initialized" assert from llama-cpp-python <0.3.19, where
       BERT-style encoders trip ``kv_cache_clear`` on a context that never
       allocated memory.

    Exits 0 on success, 1 on any failure. Intended for post-install
    verification and as the end-to-end gate in release CI.
    """
    from typing import Any, cast

    try:
        chat_path = chat_model_path or _download_self_check_model(
            _SELF_CHECK_CHAT_REPO, _SELF_CHECK_CHAT_FILE
        )
        console.print(f"Loading chat model {chat_path}")

        from lilbee.providers.llama_cpp_provider import (
            import_llama_cpp,
            install_llama_log_handler,
        )

        llama_cpp = import_llama_cpp()
        install_llama_log_handler()
        llm = llama_cpp.Llama(model_path=str(chat_path), n_ctx=256, verbose=False)
        # stream=False (default) returns a dict, not an iterator, but
        # create_completion's return type is a union; cast to Any so the
        # indexing below type-checks without forcing llama_cpp to be a
        # typecheck-time dep of lilbee.
        out = cast(Any, llm.create_completion("2+2=", max_tokens=max_tokens))
        text: str = out["choices"][0]["text"]
    except Exception as exc:
        _self_check_emit_failure(repr(exc))
        raise typer.Exit(1) from exc

    if not text.strip():
        _self_check_emit_failure("empty inference response")
        raise typer.Exit(1)

    embedding_dims: int | None = None
    if not skip_embedding:
        try:
            embed_path = embed_model_path or _download_self_check_model(
                _SELF_CHECK_EMBED_REPO, _SELF_CHECK_EMBED_FILE
            )
            console.print(f"Loading embedding model {embed_path}")
            enc = llama_cpp.Llama(
                model_path=str(embed_path),
                embedding=True,
                n_ctx=512,
                verbose=False,
            )
            emb = cast(Any, enc.create_embedding(input=["test"]))
            vec = emb["data"][0]["embedding"]
        except Exception as exc:
            _self_check_emit_failure(repr(exc))
            raise typer.Exit(1) from exc

        if not vec:
            _self_check_emit_failure("empty embedding vector")
            raise typer.Exit(1)
        embedding_dims = len(vec)

    if cfg.json_mode:
        payload: dict[str, Any] = {
            "ok": True,
            "chat_response": text,
            "chat_model": str(chat_path),
        }
        if embedding_dims is not None:
            payload["embedding_dims"] = embedding_dims
        json_output(payload)
    else:
        console.print(f"Chat response: {text!r}")
        if embedding_dims is not None:
            console.print(f"Embedding dims: {embedding_dims}")
        console.print(f"[{theme.ACCENT}]SELF-CHECK PASSED[/{theme.ACCENT}]")


_SELF_CHECK_EXTRAS = ("litellm", "crawl4ai", "spacy", "graspologic_native")


@app.command("self-check-extras")
def self_check_extras_cmd() -> None:
    """Verify optional extras (crawler, litellm, graph) are bundled and importable.

    Frozen-binary smoke gate. Replaces the byte-grep over the binary that
    PyInstaller-era smoke used: under Nuitka's C-compiled output the literal
    package-name strings may not appear in the binary even when the modules
    are correctly bundled, so we exercise the import paths instead.
    """
    results: dict[str, Any] = {}
    failed: list[str] = []
    for name in _SELF_CHECK_EXTRAS:
        try:
            importlib.import_module(name)
            results[name] = True
        except ImportError as exc:
            results[name] = False
            results[f"{name}_error"] = str(exc)
            failed.append(name)

    if cfg.json_mode:
        json_output({"ok": not failed, **results})
    else:
        for name in _SELF_CHECK_EXTRAS:
            ok = results.get(name) is True
            tag = (
                f"[{theme.ACCENT}]ok[/{theme.ACCENT}]"
                if ok
                else f"[{theme.ERROR}]MISSING[/{theme.ERROR}]"
            )
            console.print(f"  {name}: {tag}")
            if not ok:
                console.print(f"    {results.get(f'{name}_error', '')}")

    if failed:
        raise typer.Exit(1)


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
    if result.skipped:
        console.print(
            f"[{theme.WARNING}]{len(result.skipped)} item(s) could not be deleted "
            f"(locked or permission denied).[/{theme.WARNING}]"
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
    import atexit

    from lilbee.parent_monitor import parse_parent_pid, watch_parent_async

    port_path = _port_file()

    def _cleanup_port_file() -> None:
        port_path.unlink(missing_ok=True)

    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)
    await server.startup()

    parent_pid = parse_parent_pid()
    parent_watcher: asyncio.Task[None] | None = None
    if parent_pid is not None:

        def _on_parent_death() -> None:
            server.should_exit = True

        parent_watcher = asyncio.create_task(watch_parent_async(parent_pid, _on_parent_death))

    try:
        if server.servers:
            sock = server.servers[0].sockets[0]
            actual_port = sock.getsockname()[1]
            port_path.parent.mkdir(parents=True, exist_ok=True)
            port_path.write_text(str(actual_port))
            atexit.register(_cleanup_port_file)
            console.print(f"Listening on http://{host}:{actual_port}")
        await server.main_loop()
    finally:
        if parent_watcher is not None and not parent_watcher.done():
            parent_watcher.cancel()
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
def token(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print the auth token for a running server."""
    from lilbee.server.auth import server_json_path

    apply_overrides(data_dir=data_dir, use_global=use_global)
    path = server_json_path()
    if not path.exists():
        if cfg.json_mode:
            json_output({"error": "No running server found"})
        else:
            console.print("No running server found (server.json missing).")
        raise SystemExit(1)
    try:
        data = json.loads(path.read_text())
        tok = data.get("token", "")
    except (json.JSONDecodeError, OSError) as exc:
        if cfg.json_mode:
            json_output({"error": f"Could not read server.json: {exc}"})
        else:
            console.print(
                f"[{theme.ERROR}]Error:[/{theme.ERROR}] Could not read server.json: {exc}"
            )
        raise SystemExit(1) from None
    if cfg.json_mode:
        json_output({"token": tok})
        return
    console.print(tok)


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
    from lilbee.mcp_server import main

    main()


setup_app = typer.Typer(help="One-time setup for optional runtime components.")
app.add_typer(setup_app, name="setup")


@setup_app.command(name="crawler")
def setup_crawler_cmd() -> None:
    """Install Playwright's Chromium browser, needed for /crawl.

    No-op when Chromium is already present. Emits a simple progress
    readout; use '--json' mode on the top-level 'lilbee' command to get
    a single JSON blob with the final install state instead.
    """
    if chromium_installed():
        if cfg.json_mode:
            typer.echo(json.dumps({"component": "chromium", "already_installed": True}))
        else:
            typer.echo("Chromium already installed.")
        return

    last_pct: list[int] = [-1]

    def _on_progress(event_type: object, data: object) -> None:
        if event_type != EventType.SETUP_PROGRESS or not isinstance(data, SetupProgressEvent):
            return
        total = data.total_bytes or 0
        pct = int(data.downloaded_bytes * 100 / total) if total > 0 else 0
        if pct != last_pct[0] and not cfg.json_mode:
            last_pct[0] = pct
            typer.echo(msg.SETUP_CHROMIUM_CLI_PROGRESS.format(pct=pct), err=True)

    try:
        asyncio.run(bootstrap_chromium(on_progress=_on_progress))
    except CrawlerBrowserMissing as exc:
        if cfg.json_mode:
            typer.echo(json.dumps({"component": "chromium", "error": str(exc)}))
        else:
            typer.secho(f"Install failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if cfg.json_mode:
        typer.echo(json.dumps({"component": "chromium", "installed": True}))
    else:
        typer.echo("Chromium installed.")


wiki_app = typer.Typer(help="Wiki layer commands: generate, lint, citations, status, prune.")
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


@wiki_app.command(name="synthesize")
def wiki_synthesize(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Generate synthesis pages for concept clusters spanning 3+ sources."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
        return
    from lilbee.wiki.gen import generate_synthesis_pages

    svc = get_services()
    paths = generate_synthesis_pages(svc.provider, svc.store, svc.clusterer)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_synthesize",
                "paths": [str(p) for p in paths],
                "count": len(paths),
            }
        )
        return

    if not paths:
        console.print("No synthesis pages generated (need 3+ sources per cluster).")
        return

    console.print(f"Generated [{theme.LABEL}]{len(paths)}[/{theme.LABEL}] synthesis pages:")
    for path in paths:
        console.print(f"  {path}")


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


def _fail_wiki_disabled() -> None:
    """Emit the standard wiki-disabled message in the caller's output mode."""
    if cfg.json_mode:
        json_output({"error": msg.CMD_WIKI_DISABLED})
        return
    console.print(msg.CMD_WIKI_DISABLED)


@wiki_app.command(name="build")
def wiki_build(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Run extraction only; skip every LLM call. Prints the NER entity candidates. "
            "LLM-curated concept pages require a build call and are not shown in dry-run."
        ),
    ),
) -> None:
    """Build the concept and entity wiki across all ingested sources."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
        return

    if dry_run:
        from lilbee.store import SearchChunk
        from lilbee.wiki.entity_extractor import get_entity_extractor

        svc = get_services()
        chunks: list[SearchChunk] = []
        for record in svc.store.get_sources():
            chunks.extend(svc.store.get_chunks_by_source(record["filename"]))
        extractor = get_entity_extractor(cfg.wiki_entity_mode, svc.provider, cfg)
        entities = extractor.extract(chunks)
        _wiki_build_dry_run_output(entities)
        return

    from lilbee.wiki import run_full_build

    result = run_full_build(cfg)

    if cfg.json_mode:
        json_output({"command": "wiki_build", **result})
        return

    pages = result["paths"]
    if not pages:
        console.print("No concept or entity pages generated.")
        return

    console.print(
        f"Generated [{theme.LABEL}]{result['count']}[/{theme.LABEL}] "
        f"wiki pages from {result['entities']} extracted records:"
    )
    for path in pages:
        console.print(f"  {path}")


_DRY_RUN_CONCEPT_NOTE = (
    "Note: LLM-curated concepts are not shown in --dry-run. "
    "Run `lilbee wiki build` to see which concepts the LLM proposes."
)


def _wiki_build_dry_run_output(entities: list[ExtractedEntity]) -> None:
    """Render the extraction result as JSON or table without calling any LLM.

    Phase D: concepts come from the per-source batched LLM call, so
    listing them would require the call we are trying to avoid. The
    dry-run surface is NER-entity only, with a trailing note so a
    user who expected concepts in the output knows why they are
    missing.
    """
    rows: list[dict[str, Any]] = [
        {
            "slug": e.slug,
            "label": e.label,
            "kind": e.kind.value,
            "type_hint": e.type_hint,
            "mentions": len(e.chunk_refs),
            "sources": sorted({r.source for r in e.chunk_refs}),
        }
        for e in entities
    ]

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_build",
                "dry_run": True,
                "entities": rows,
                "count": len(rows),
                "note": _DRY_RUN_CONCEPT_NOTE,
            }
        )
        return

    if not rows:
        console.print("No candidate entities extracted. Run sync first.")
        console.print(f"[{theme.MUTED}]{_DRY_RUN_CONCEPT_NOTE}[/{theme.MUTED}]")
        return

    table = Table(title=f"Wiki build dry-run ({len(rows)} NER entity candidates)")
    table.add_column("Slug", style=theme.ACCENT)
    table.add_column("Kind", style=theme.MUTED)
    table.add_column("Type")
    table.add_column("Mentions")
    table.add_column("Sources")
    for row in rows:
        sources_list: list[str] = row["sources"]
        table.add_row(
            str(row["slug"]),
            str(row["kind"]),
            str(row["type_hint"]),
            str(row["mentions"]),
            ", ".join(sources_list[:3]) + (", ..." if len(sources_list) > 3 else ""),
        )
    console.print(table)
    console.print(
        f"Dry run: [{theme.LABEL}]{len(rows)}[/{theme.LABEL}] candidate entities. "
        "No LLM calls were made."
    )
    console.print(f"[{theme.MUTED}]{_DRY_RUN_CONCEPT_NOTE}[/{theme.MUTED}]")


@wiki_app.command(name="update")
def wiki_update(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Refresh the concept and entity wiki after an ingest.

    Currently a full rebuild. The incremental touched-slug regeneration
    lands in the ingest-hook task and will re-route this command then.
    """
    wiki_build(data_dir=data_dir, use_global=use_global, dry_run=False)


drafts_app = typer.Typer(help="Review wiki drafts: list, diff, accept, reject.")
wiki_app.add_typer(drafts_app, name="drafts")


@drafts_app.command(name="list")
def wiki_drafts_list(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List pending wiki drafts with drift, faithfulness, and pairing info."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import PENDING_KIND_DRIFT, list_drafts

    wiki_root = cfg.data_root / cfg.wiki_dir
    drafts = list_drafts(wiki_root)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_drafts_list",
                "drafts": [d.to_dict() for d in drafts],
                "total": len(drafts),
            }
        )
        return

    if not drafts:
        console.print("No drafts pending review.")
        return

    table = Table(title="Wiki Drafts")
    table.add_column("Slug", style=theme.ACCENT)
    table.add_column("Kind", style=theme.MUTED)
    table.add_column("Drift")
    table.add_column("Faithfulness")
    table.add_column("Published?", style=theme.MUTED)
    for d in drafts:
        kind = d.pending_kind or PENDING_KIND_DRIFT
        drift = f"{d.drift_ratio:.0%}" if d.drift_ratio is not None else "-"
        faith = f"{d.faithfulness_score:.2f}" if d.faithfulness_score is not None else "-"
        published = "yes" if d.published_exists else "no"
        table.add_row(d.slug, kind, drift, faith, published)
    console.print(table)


@drafts_app.command(name="diff")
def wiki_drafts_diff(
    slug: str = typer.Argument(..., help="Draft slug (e.g. chevrolet)."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show a unified diff of the draft against its published counterpart."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import diff_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        diff = diff_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_diff", "slug": slug, "diff": diff})
        return
    console.print(diff or "(no differences)")


@drafts_app.command(name="accept")
def wiki_drafts_accept(
    slug: str = typer.Argument(..., help="Draft slug to accept."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Overwrite the published page with the draft and re-index its chunks."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import accept_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        result = accept_draft(slug, wiki_root, get_services().store)
    except FileNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_accept", **result.to_dict()})
        return
    console.print(
        f"Accepted [{theme.ACCENT}]{slug}[/{theme.ACCENT}] -> "
        f"{result.moved_to} ({result.reindexed_chunks} chunks re-indexed)"
    )


@drafts_app.command(name="reject")
def wiki_drafts_reject(
    slug: str = typer.Argument(..., help="Draft slug to reject."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Delete the draft file. Does not touch the published page or index."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import reject_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        reject_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_reject", "slug": slug})
        return
    console.print(f"Rejected [{theme.ACCENT}]{slug}[/{theme.ACCENT}]")
