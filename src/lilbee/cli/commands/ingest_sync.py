"""Sync, rebuild, add, chunks, and remove commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.commands._shared import CHUNK_PREVIEW_LEN
from lilbee.cli.helpers import (
    CopyResult,
    add_paths,
    clean_result,
    copy_files,
    json_output,
    sync_result_to_json,
)
from lilbee.core.config import cfg
from lilbee.core.services import get_services
from lilbee.crawler import is_url

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


def sync_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    ocr: bool | None = _ocr_option,
    ocr_timeout: float | None = _ocr_timeout_option,
) -> None:
    """Manually trigger document sync."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _apply_ocr_overrides(ocr, ocr_timeout)
    from lilbee.data.ingest import sync

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


def rebuild(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    ocr: bool | None = _ocr_option,
    ocr_timeout: float | None = _ocr_timeout_option,
) -> None:
    """Nuke the DB and re-ingest everything from documents/."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _apply_ocr_overrides(ocr, ocr_timeout)
    from lilbee.data.ingest import sync

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
            from lilbee.data.ingest import sync

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
            from lilbee.data.ingest import sync

            result = asyncio.run(sync())
            console.print(result)
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        console.print(f"[{theme.ERROR}]Error:[/{theme.ERROR}] {exc}")
        raise SystemExit(1) from None


_chunks_source_argument = typer.Argument(..., help="Source name to inspect chunks for.")


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
