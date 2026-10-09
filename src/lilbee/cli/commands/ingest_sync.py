"""Sync, rebuild, add, chunks, and remove commands."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import typer
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from lilbee.runtime.progress import DetailedProgressCallback

from lilbee.app.ingest import (
    AddRollback,
    RegisterResult,
    expand_remove_targets,
    leave_as_cancel,
    register_sources,
    removable_names,
    remove_documents_durably,
)
from lilbee.app.search import clean_result
from lilbee.app.services import get_services
from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.commands._shared import CHUNK_PREVIEW_LEN
from lilbee.cli.helpers import (
    add_paths,
    json_output,
    print_prefixed,
    sync_result_to_json,
)
from lilbee.core.config import cfg
from lilbee.core.config.enums import OcrMode
from lilbee.crawler import is_url
from lilbee.data.ingest.skip_marker import SkipRecordsLockError

_SYNC_CANCELLED_MESSAGE = "Sync cancelled."
# The shell's exit status for a command stopped by Ctrl+C (128 + SIGINT).
_EXIT_INTERRUPTED = 130

_ocr_option = typer.Option(
    None,
    "--ocr",
    help=(
        "Scanned pages for this run: auto reads each page without usable text and keeps "
        "the text of every other page; all reads every page of every file; off skips them. "
        "Leave it out to use the ocr setting."
    ),
)
_retry_skipped_option = typer.Option(
    False,
    "--retry-skipped",
    help="Retry files that were skipped on a previous sync (clears the failed-file markers).",
)
_prune_ignored_option = typer.Option(
    False,
    "--prune-ignored",
    help="Also drop indexed documents a .lilbeeignore now excludes. Source files are kept.",
)
_ocr_timeout_option = typer.Option(
    None,
    "--ocr-timeout",
    help="Per-page timeout in seconds for vision OCR (default: 300, 0 = no limit).",
)


def _apply_ocr_overrides(ocr: OcrMode | None, ocr_timeout: float | None) -> None:
    """Apply --ocr and --ocr-timeout CLI overrides to config.

    The CLI is a single-shot, single-process invocation, so mutating the global
    cfg here is safe (it mirrors ``apply_overrides`` for the data dir). The
    daemon-shared per-request OCR override uses a ContextVar instead; see
    ``temporary_ocr_config``.
    """
    if ocr is not None:
        cfg.ocr = ocr
    if ocr_timeout is not None:
        cfg.ocr_timeout = ocr_timeout


_paths_argument = typer.Argument(
    ...,
    help="Files, directories, or URLs to add to the knowledge base.",
)

_force_option = typer.Option(False, "--force", "-f", help="Overwrite existing files.")
_max_cpus_option = typer.Option(
    None,
    "--max-cpus",
    min=1,
    help="Cap the workers used to discover and hash files. Unset = auto (all available cores).",
)
_processes_option = typer.Option(
    None,
    "--processes",
    min=0,
    help=(
        "Ingest worker processes, one GPU each: N explicit, 0 = auto (one per card),"
        " 1 = this process only."
    ),
)
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
    help="Cap pages for --crawl. Unset = protective default; 0 = unlimited; N = hard cap.",
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
    cancel_event: threading.Event,
    include_subdomains: bool = False,
) -> list[Path]:
    """Crawl each URL in turn and return the pages saved; a set *cancel_event* raises a cancel."""
    from rich.progress import Progress, SpinnerColumn, TaskID

    from lilbee.crawler import crawl_and_save
    from lilbee.runtime.progress import (
        CrawlDoneEvent,
        CrawlPageEvent,
        EventType,
        ProgressEvent,
    )
    from lilbee.runtime.progress.columns import literal_text_column

    if crawl:
        effective_depth = depth
        effective_pages = max_pages
    else:
        effective_depth = 0
        effective_pages = None

    from lilbee.runtime.console import PlainConsole

    err_console = PlainConsole(stderr=True)
    all_paths: list[Path] = []
    with Progress(
        SpinnerColumn(),
        literal_text_column("{task.description}"),
        transient=True,
        console=err_console,
        disable=cfg.json_mode,
    ) as progress:
        for url in urls:
            ptask = progress.add_task(f"Crawling {url}...", total=None)
            crawled: dict[str, int] = {}

            def _make_callback(
                _t: TaskID = ptask, _crawled: dict[str, int] = crawled
            ) -> DetailedProgressCallback:
                def on_progress(event_type: EventType, data: ProgressEvent) -> None:
                    if event_type == EventType.CRAWL_PAGE:
                        if not isinstance(data, CrawlPageEvent):
                            raise TypeError(f"Expected CrawlPageEvent, got {type(data).__name__}")
                        total_str = str(data.total) if data.total > 0 else "?"
                        progress.update(
                            _t,
                            description=f"Crawled {data.current}/{total_str}: {data.url}",
                        )
                    elif event_type == EventType.CRAWL_DONE and isinstance(data, CrawlDoneEvent):
                        _crawled["n"] = data.pages_crawled

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
            # No explicit cap given and the crawl filled the protective default:
            # tell the user how to go unlimited without editing settings.
            default_cap = cfg.crawl_max_pages or cfg.crawl_safety_max_pages
            if crawl and max_pages is None and crawled.get("n", 0) >= default_cap:
                err_console.print(
                    f"Stopped at the default {default_cap}-page limit; "
                    f"pass --max-pages 0 to crawl unlimited (or --max-pages N for a higher cap).",
                )
    return all_paths


def _run_crawl_with_signal_cancel(
    url: str,
    *,
    depth: int | None,
    max_pages: int | None,
    on_progress: DetailedProgressCallback,
    cancel_event: threading.Event,
    crawl_and_save: Callable[..., Awaitable[list[Path]]],
    include_subdomains: bool = False,
) -> list[Path]:
    """Crawl one URL on its own event loop; a crawl that ends under a set cancel raises it."""
    # Manage the event loop explicitly. In the CLI this runs once per process,
    # but under pytest-xdist the same worker thread runs many tests; leaving a
    # closed loop set as the "current" loop for the thread poisons every later
    # asyncio.get_event_loop() call and hangs macOS 3.12/3.13 unit-test CI.
    # Always clear the thread-current loop in finally.
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        coro = crawl_and_save(
            url,
            depth=depth,
            max_pages=max_pages,
            on_progress=on_progress,
            cancel=cancel_event,
            quiet=cfg.json_mode,
            include_subdomains=include_subdomains,
        )
        result: list[Path] = loop.run_until_complete(coro)
        # A cancelled crawl returns the pages it saved; the command still stops here.
        if cancel_event.is_set():
            raise asyncio.CancelledError
        return result
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _cancellable_progress(
    cancel_event: threading.Event, chain: DetailedProgressCallback
) -> DetailedProgressCallback:
    """Wrap *chain* so a set *cancel_event* aborts the in-flight file cooperatively.

    The ingest pipeline and the per-page vision OCR loop both call the progress
    callback between units of work; raising :class:`TaskCancelledError` there is
    the established cooperative-cancel signal, so a Ctrl+C stops a long OCR
    between pages instead of after the whole document.
    """
    from lilbee.runtime.cancellation import TaskCancelledError

    def _callback(event_type: object, data: object) -> None:
        if cancel_event.is_set():
            raise TaskCancelledError
        chain(event_type, data)  # type: ignore[arg-type]

    return _callback


def _exit_cancelled(rollback: AddRollback) -> NoReturn:
    """Report a cancelled command and exit with the Ctrl+C status."""
    message = rollback.message(_SYNC_CANCELLED_MESSAGE)
    not_added = rollback.not_added
    if cfg.json_mode:
        json_output({"error": message, "not_added": not_added} if not_added else {"error": message})
    else:
        console.print(Text(message, style=theme.WARNING))
    raise SystemExit(_EXIT_INTERRUPTED) from None


@contextmanager
def _ctrl_c_sets(cancel_event: threading.Event) -> Iterator[None]:
    """Make Ctrl+C set *cancel_event* instead of raising KeyboardInterrupt, for the block."""
    import signal

    # signal.signal raises ValueError off the main thread (e.g. under pytest-xdist
    # workers); the cancel_event can still be driven externally there.
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _on_sigint(_signum: int, _frame: object) -> None:
        cancel_event.set()

    previous_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)


@contextmanager
def _ctrl_c_stops(cancel_event: threading.Event, rollback: AddRollback) -> Iterator[None]:
    """Run a command under Ctrl+C; one it stops exits 130 after *rollback*."""
    try:
        # Only the user's Ctrl+C sets this cancel.
        with _ctrl_c_sets(cancel_event), leave_as_cancel(rollback, cancel_event, lambda: True):
            yield
    except asyncio.CancelledError:
        _exit_cancelled(rollback)


def run_sync_with_signal_cancel(
    *,
    force_rebuild: bool = False,
    retry_skipped: bool = False,
    prune_ignored: bool = False,
    on_progress: DetailedProgressCallback | None = None,
) -> object:
    """Run ``sync`` under Ctrl+C; a sync it stops ends the command with exit status 130."""
    cancel_event = threading.Event()
    with _ctrl_c_stops(cancel_event, AddRollback()):
        return _run_sync(
            cancel_event,
            force_rebuild=force_rebuild,
            retry_skipped=retry_skipped,
            prune_ignored=prune_ignored,
            on_progress=on_progress,
        )


def _run_sync(
    cancel_event: threading.Event,
    *,
    force_rebuild: bool = False,
    retry_skipped: bool = False,
    prune_ignored: bool = False,
    on_progress: DetailedProgressCallback | None = None,
    before_sync: Callable[[], None] | None = None,
) -> object:
    """Run ``sync`` on its own event loop; it polls *cancel_event* between files and pages.

    *before_sync* runs once the eager warm is off, so a store read in it starts no model.
    """
    from lilbee.data.ingest import sync
    from lilbee.runtime.progress import noop_callback

    # Batch ingest is a headless one-shot: skip the eager warm so services init
    # doesn't spawn every role. With lazy per-role spawn, the sync brings up only
    # the embed server (plus vision/chat if those steps actually run), instead of
    # holding an idle chat server's VRAM for the whole build.
    cfg.worker_pool_eager_start = False
    if before_sync is not None:
        before_sync()

    callback = _cancellable_progress(cancel_event, on_progress or noop_callback)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            sync(
                force_rebuild=force_rebuild,
                quiet=cfg.json_mode,
                on_progress=callback,
                cancel=cancel_event,
                retry_skipped=retry_skipped,
                prune_ignored=prune_ignored,
            )
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def sync_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    ocr: OcrMode | None = _ocr_option,
    ocr_timeout: float | None = _ocr_timeout_option,
    retry_skipped: bool = _retry_skipped_option,
    prune_ignored: bool = _prune_ignored_option,
    max_cpus: int | None = _max_cpus_option,
    processes: int | None = _processes_option,
) -> None:
    """Manually trigger document sync."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _apply_ocr_overrides(ocr, ocr_timeout)
    if max_cpus is not None:
        cfg.ingest_workers = max_cpus
    if processes is not None:
        cfg.ingest_processes = processes

    try:
        result = run_sync_with_signal_cancel(
            retry_skipped=retry_skipped, prune_ignored=prune_ignored
        )
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        print_prefixed(console, "Error: ", exc, style=theme.ERROR)
        raise SystemExit(1) from None
    if cfg.json_mode:
        json_output(sync_result_to_json(result))
        return
    console.print(result)


def rebuild(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    ocr: OcrMode | None = _ocr_option,
    ocr_timeout: float | None = _ocr_timeout_option,
    max_cpus: int | None = _max_cpus_option,
    processes: int | None = _processes_option,
) -> None:
    """Nuke the DB and re-ingest everything from documents/."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _apply_ocr_overrides(ocr, ocr_timeout)
    if max_cpus is not None:
        cfg.ingest_workers = max_cpus
    if processes is not None:
        cfg.ingest_processes = processes
    from lilbee.data.ingest import SyncResult

    try:
        result = run_sync_with_signal_cancel(force_rebuild=True)
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        print_prefixed(console, "Error: ", exc, style=theme.ERROR)
        raise SystemExit(1) from None
    if not isinstance(result, SyncResult):
        raise TypeError(f"Expected SyncResult, got {type(result).__name__}")
    if cfg.json_mode:
        json_output(
            {
                "command": "rebuild",
                "ingested": len(result.added),
                "skip_records_error": result.skip_records_error,
            }
        )
        return
    console.print(f"Rebuilt: {len(result.added)} documents ingested")
    if result.skip_records_error is not None:
        print_prefixed(console, "Error: ", result.skip_records_error, style=theme.ERROR)


def index(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Build the search indexes now (vector ANN + full-text).

    Useful before publishing a large index so downloaders get fast search
    without waiting for it to build on first query. Forces the vector index
    even below the auto-build threshold.
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    store = get_services().store
    store.ensure_fts_index()
    store.ensure_scalar_indexes()
    built = store.ensure_vector_index(force=True)
    if cfg.json_mode:
        json_output({"command": "index", "vector_index": built})
        return
    if built:
        console.print("Search indexes built (vector ANN + full-text).")
    else:
        console.print("Full-text index built; vector index needs more chunks.")


def _validate_file_paths(file_paths: list[Path]) -> None:
    """Exit on the first missing path; respects ``cfg.json_mode``."""
    for fp in file_paths:
        if fp.exists():
            continue
        if cfg.json_mode:
            json_output({"error": f"Path not found: {fp}"})
            raise SystemExit(1)
        print_prefixed(console, "Error: ", f"Path not found: {fp}", style=theme.ERROR)
        raise SystemExit(1)


def _crawl_urls_step(
    urls: list[str],
    *,
    crawl: bool,
    depth: int | None,
    max_pages: int | None,
    include_subdomains: bool,
    cancel_event: threading.Event,
) -> list[Path]:
    """Crawl URLs (or fail fast when crawler extra is missing). Returns saved paths."""
    if not urls:
        return []
    from lilbee.crawler import crawler_available

    if not crawler_available():
        console.print(
            "Web crawling requires: pip install 'lilbee[crawler]'",
            style=theme.ERROR,
            soft_wrap=True,
        )
        raise SystemExit(1)
    crawled_paths = _crawl_urls_blocking(
        urls,
        crawl=crawl,
        depth=depth,
        max_pages=max_pages,
        cancel_event=cancel_event,
        include_subdomains=include_subdomains,
    )
    if not cfg.json_mode:
        console.print(
            f"Crawled {len(crawled_paths)} page(s) from {len(urls)} URL(s)",
            style=theme.MUTED,
        )
    return crawled_paths


def _add_json_mode(
    file_paths: list[Path],
    crawled_paths: list[Path],
    *,
    force: bool,
    run_sync: Callable[[RegisterResult], object],
) -> dict:
    """Run the JSON-mode finish: register roots, sync, return the one structured result."""
    reg_result = RegisterResult()
    if file_paths:
        reg_result = register_sources(file_paths, force=force)
    # A sync is a whole-vault pass; run it only when something named reached the corpus.
    result = run_sync(reg_result) if reg_result.reached_corpus or crawled_paths else None
    return {
        "command": "add",
        "copied": reg_result.registered,
        "name_taken": reg_result.name_taken,
        "overlapping": reg_result.overlapping,
        "absorbed": reg_result.absorbed,
        "tracked": reg_result.tracked,
        "refused": reg_result.refused,
        "crawled": len(crawled_paths),
        "sync": None if result is None else sync_result_to_json(result),
    }


def _register_and_sync(
    file_paths: list[Path],
    crawled_paths: list[Path],
    *,
    sync_urls: bool,
    force: bool,
    cancel_event: threading.Event,
    rollback: AddRollback,
) -> dict | None:
    """Register the files and sync; returns the JSON result, or None after human output."""

    def _sync(registration: RegisterResult) -> object:
        def _sync_starts() -> None:
            rollback.registered(registration.revocable, cancel_event)

        return _run_sync(cancel_event, before_sync=_sync_starts)

    if cfg.json_mode:
        return _add_json_mode(file_paths, crawled_paths, force=force, run_sync=_sync)
    if file_paths:
        # Crawled pages are new content even when no file reached the corpus.
        add_paths(file_paths, console, force=force, run_sync=_sync, sync_anyway=bool(crawled_paths))
    elif sync_urls:
        # URLs already saved; just trigger sync
        console.print(_sync(RegisterResult()))
    return None


def add(
    paths: list[str] = _paths_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    force: bool = _force_option,
    ocr: OcrMode | None = _ocr_option,
    ocr_timeout: float | None = _ocr_timeout_option,
    crawl: bool = _crawl_option,
    depth: int | None = _depth_option,
    max_pages: int | None = _max_pages_option,
    include_subdomains: bool = _include_subdomains_option,
    max_cpus: int | None = _max_cpus_option,
    processes: int | None = _processes_option,
) -> None:
    """Link files or crawl URLs into the knowledge base and ingest them."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _apply_ocr_overrides(ocr, ocr_timeout)
    if max_cpus is not None:
        cfg.ingest_workers = max_cpus
    if processes is not None:
        cfg.ingest_processes = processes

    file_paths, urls = _partition_inputs(paths)
    _validate_file_paths(file_paths)

    cancel_event = threading.Event()
    rollback = AddRollback(paths=file_paths, at_sync=False)
    try:
        with _ctrl_c_stops(cancel_event, rollback):
            crawled_paths = _crawl_urls_step(
                urls,
                crawl=crawl,
                depth=depth,
                max_pages=max_pages,
                include_subdomains=include_subdomains,
                cancel_event=cancel_event,
            )
            payload = _register_and_sync(
                file_paths,
                crawled_paths,
                sync_urls=bool(urls),
                force=force,
                cancel_event=cancel_event,
                rollback=rollback,
            )
            # An add that ends before its sync under a Ctrl+C still stops; a sync
            # that returned has finished, and its result stands.
            if cancel_event.is_set() and not rollback.at_sync:
                raise asyncio.CancelledError
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
            raise SystemExit(1) from None
        print_prefixed(console, "Error: ", exc, style=theme.ERROR)
        raise SystemExit(1) from None
    if payload is not None:
        json_output(payload)


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
        print_prefixed(console, "Source not found: ", source, style=theme.ERROR)
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
        Text.assemble(
            (str(len(cleaned)), theme.LABEL), " chunks from ", (source, theme.ACCENT), "\n"
        ),
        soft_wrap=True,
    )
    for c in cleaned:
        idx = c.get("chunk_index", "?")
        preview = c.get("chunk", "")[:CHUNK_PREVIEW_LEN]
        if len(c.get("chunk", "")) > CHUNK_PREVIEW_LEN:
            preview += "..."
        console.print(Text.assemble(f"  [{idx}] ", preview), soft_wrap=True)


_remove_names_argument = typer.Argument(
    ..., help="Source name(s), folder(s), or glob pattern(s) to remove from the knowledge base."
)

_remove_yes_option = typer.Option(
    False, "--yes", "-y", help="Skip the confirmation prompt when a name expands to many documents."
)


def remove(
    names: list[str] = _remove_names_argument,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    yes: bool = _remove_yes_option,
) -> None:
    """Remove documents from the knowledge base by source name, folder, or glob pattern.

    A folder name removes every document indexed beneath it; a glob pattern
    (containing ``*``, ``?``, or ``[]``) removes every source it matches. A file
    held out because its ingestion failed can be named too; it stays out of later
    syncs. Source files on disk are never deleted.
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    # Remove only touches the store, never the engine; skip the eager fleet warm.
    cfg.worker_pool_eager_start = False

    known = removable_names()
    targets = expand_remove_targets(names, known=known)
    expanded = sorted(set(targets)) != sorted(set(names))
    if expanded and not yes and not cfg.json_mode:
        # Count only what actually exists; not-found names are kept in targets.
        removable = sum(1 for t in targets if t in set(known))
        typer.confirm(f"Remove {removable} document(s)? Source files on disk are kept.", abort=True)

    try:
        result = remove_documents_durably(names, targets=targets)
    except SkipRecordsLockError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            print_prefixed(console, "Error: ", exc, style=theme.ERROR)
        raise SystemExit(1) from None

    if cfg.json_mode:
        payload: dict = {"command": "remove", "removed": result.removed}
        if result.not_found:
            payload["not_found"] = result.not_found
        json_output(payload)
        if not result.removed and result.not_found:
            raise SystemExit(1)
        return

    for name in result.removed:
        console.print(Text.assemble("Removed ", (name, theme.ACCENT)), soft_wrap=True)
    for name in result.not_found:
        print_prefixed(console, "Not found: ", name, style=theme.ERROR)
    if not result.removed and result.not_found:
        raise SystemExit(1)
