"""Web crawling — fetch pages as markdown and save to the documents directory."""

import asyncio
import contextlib
import hashlib
import io
import ipaddress
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lilbee.config import cfg
from lilbee.progress import (
    CrawlDoneEvent,
    CrawlPageEvent,
    CrawlStartEvent,
    DetailedProgressCallback,
    EventType,
    SetupDoneEvent,
    SetupProgressEvent,
    SetupStartEvent,
)
from lilbee.security import validate_path_within

log = logging.getLogger(__name__)


def crawler_available() -> bool:
    """Check if crawl4ai is installed."""
    try:
        import crawl4ai  # noqa: F401

        return True
    except ImportError:
        return False


class CrawlerBrowserMissing(RuntimeError):
    """Playwright is installed but its Chromium browser binary is not.

    Raised early by ``_open_crawler`` so task workers route to FAILED with
    an actionable message instead of letting Playwright print its raw
    ASCII install banner into the TUI.
    """


def _browsers_cache_path() -> Path:
    """Return the root path where Playwright stores browser binaries."""
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "ms-playwright"
    return Path.home() / ".cache" / "ms-playwright"


def chromium_installed() -> bool:
    """Return True if at least one chromium-* install directory exists."""
    root = _browsers_cache_path()
    if not root.exists():
        return False
    return any(p.is_dir() and p.name.startswith("chromium-") for p in root.iterdir())


def crawler_browsers_path() -> Path:
    """Public accessor for the crawler browser cache root.

    Used by the HTTP status endpoint to tell plugins where Chromium
    lives. The underlying resolver stays private because callers should
    not depend on the Playwright-specific directory layout.
    """
    return _browsers_cache_path()


_CHROMIUM_COMPONENT = "chromium"
# Rough size estimate for the Chromium download; Playwright bundles vary
# slightly per platform but this gives the UI a decent denominator before
# 'Total bytes' parses out of stdout.
_CHROMIUM_ESTIMATE_MB = 180
_CHROMIUM_SIZE_ESTIMATE_BYTES = _CHROMIUM_ESTIMATE_MB * 1024 * 1024

# Unit -> bytes scale for Playwright stdout progress lines.
_BYTE_UNIT_SCALE: dict[str, int] = {
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024 * 1024,
    "mib": 1024 * 1024,
}

# Playwright 1.58 prints lines like
# ``|■■■■■■■■                                  |  10% of 162.3 MiB`` during
# the chromium download. The percent comes first, then "of <total> <unit>".
_PROGRESS_LINE_RE = re.compile(
    r"(\d+)\s*%\s*of\s*(\d+(?:\.\d+)?)\s*(MiB|Mb|MB|KiB|KB|B)",
    re.IGNORECASE,
)


def _bytes_from_stdout(line: str) -> tuple[int, int] | None:
    """Extract (downloaded_bytes, total_bytes) from a Playwright stdout line.

    Matches the ``NN% of N.N MiB`` shape Playwright 1.58+ emits for the
    Chromium download. Returns None when the line doesn't match. The
    percent and total both parse out of the same line so callers never
    have to handle a missing total.
    """
    match = _PROGRESS_LINE_RE.search(line)
    if match is None:
        return None
    pct = int(match.group(1))
    raw_total = float(match.group(2))
    unit = match.group(3).lower()
    scale = _BYTE_UNIT_SCALE.get(unit, 1)
    total = int(raw_total * scale)
    downloaded = int(total * pct / 100)
    return downloaded, total


def _emit_setup_start(on_progress: DetailedProgressCallback | None) -> None:
    if on_progress is None:
        return
    on_progress(
        EventType.SETUP_START,
        SetupStartEvent(
            component=_CHROMIUM_COMPONENT,
            size_estimate_bytes=_CHROMIUM_SIZE_ESTIMATE_BYTES,
        ),
    )


def _emit_setup_done(
    on_progress: DetailedProgressCallback | None,
    *,
    success: bool,
    error: str | None,
) -> None:
    if on_progress is None:
        return
    on_progress(
        EventType.SETUP_DONE,
        SetupDoneEvent(component=_CHROMIUM_COMPONENT, success=success, error=error),
    )


async def _drain_stdout_to_progress(
    stream: asyncio.StreamReader,
    on_progress: DetailedProgressCallback | None,
) -> None:
    while True:
        line_bytes = await stream.readline()
        if not line_bytes:
            return
        line = line_bytes.decode(errors="replace").rstrip()
        parsed = _bytes_from_stdout(line)
        if parsed is None or on_progress is None:
            continue
        downloaded, total = parsed
        on_progress(
            EventType.SETUP_PROGRESS,
            SetupProgressEvent(
                component=_CHROMIUM_COMPONENT,
                downloaded_bytes=downloaded,
                total_bytes=total,
                detail=line,
            ),
        )


async def _drain_stderr(stream: asyncio.StreamReader, tail: list[str]) -> None:
    while True:
        line_bytes = await stream.readline()
        if not line_bytes:
            return
        tail.append(line_bytes.decode(errors="replace").rstrip())


async def bootstrap_chromium(
    on_progress: DetailedProgressCallback | None = None,
) -> None:
    """Run ``playwright install chromium`` as a subprocess, emitting events.

    Short-circuits when ``chromium_installed()`` is already True. Emits
    ``setup_start`` before spawning, ``setup_progress`` for each
    recognizable progress line on stdout, and ``setup_done`` on exit
    (``success=False`` + the subprocess stderr tail on failure). Raises
    :class:`CrawlerBrowserMissing` with the tail so task workers route
    to FAILED cleanly.

    Uses the current Python interpreter's ``playwright`` module so this
    works under ``uv tool install`` and bundled installs alike without
    relying on a globally-installed ``playwright`` CLI.
    """
    if chromium_installed():
        _emit_setup_done(on_progress, success=True, error=None)
        return

    _emit_setup_start(on_progress)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "playwright",
        "install",
        "chromium",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr_tail: list[str] = []
    await asyncio.gather(
        _drain_stdout_to_progress(proc.stdout, on_progress),
        _drain_stderr(proc.stderr, stderr_tail),
    )
    returncode = await proc.wait()

    if returncode != 0:
        tail = "\n".join(stderr_tail[-10:]) or f"exit code {returncode}"
        _emit_setup_done(on_progress, success=False, error=tail)
        raise CrawlerBrowserMissing(f"Chromium bootstrap failed (exit {returncode}): {tail}")

    _emit_setup_done(on_progress, success=True, error=None)


class CrawlerState:
    """Per-process mutable state for the crawler (semaphore, periodic sync tracking).
    Encapsulates state that would otherwise live as bare module-level globals.
    A single module-level instance (_state) is used because this state is inherently
    per-process (threading primitives, asyncio tasks tied to the running loop).
    """

    def __init__(self) -> None:
        self.semaphore: asyncio.Semaphore | None = None
        self.semaphore_limit: int = 0
        self.last_sync_time: float = 0.0
        self.sync_running: threading.Lock = threading.Lock()
        self.background_tasks: set[asyncio.Task[None]] = set()

    def reset(self) -> None:
        """Reset all state (useful for testing)."""
        self.semaphore = None
        self.semaphore_limit = 0
        self.last_sync_time = 0.0
        self.sync_running = threading.Lock()
        self.background_tasks = set()


_state = CrawlerState()


def _get_crawl_semaphore() -> asyncio.Semaphore | None:
    """Return an asyncio semaphore for crawl concurrency, or None if unlimited (0)."""
    limit = cfg.crawl_max_concurrent
    if limit <= 0:
        return None
    if _state.semaphore is None or _state.semaphore_limit != limit:
        _state.semaphore = asyncio.Semaphore(limit)
        _state.semaphore_limit = limit
    return _state.semaphore


# Maximum filename length before truncation (most filesystems cap at 255 bytes)
_MAX_FILENAME_LEN = 200

# Sentinel for index pages (trailing slash or empty path)
_INDEX_FILENAME = "index.md"


_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
)


def get_blocked_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return blocked network list. Override in tests via monkeypatch."""
    return _BLOCKED_NETWORKS


def is_url(value: str) -> bool:
    """Check if a string is an HTTP/HTTPS URL."""
    return value.startswith(("http://", "https://"))


def validate_crawl_url(url: str) -> None:
    """Validate a URL for crawling. Raises ValueError for unsafe URLs.
    Rejects private IPs, loopback, link-local, and non-HTTP schemes.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Only http:// and https:// URLs are allowed, got {scheme}://")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from exc

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in get_blocked_networks():
            if ip in network:
                raise ValueError(f"Crawling private/reserved IP {ip} is not allowed")


def require_valid_crawl_url(url: str) -> None:
    """Validate URL for crawling. Raises ValueError if invalid."""
    if not is_url(url):
        raise ValueError("URL must start with http:// or https://")
    validate_crawl_url(url)


@dataclass
class CrawlResult:
    """Outcome of crawling a single URL."""

    url: str
    markdown: str = ""
    success: bool = True
    error: str | None = None


def url_to_filename(url: str) -> str:
    """Convert a URL to a safe filesystem path ending in .md.
    Examples:
        https://docs.python.org/3/tutorial/ → docs.python.org/3/tutorial/index.md
        https://example.com/page?q=1#frag   → example.com/page.md
        https://example.com/                → example.com/index.md
    """
    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    path = parsed.path.rstrip("/")

    if not path or path == "/":
        return f"{host}/{_INDEX_FILENAME}"

    # Strip leading slash
    path = path.lstrip("/")

    # Neutralize path traversal segments
    path = re.sub(r"\.\.+", "_", path)

    # Replace unsafe filesystem characters
    path = re.sub(r'[<>:"|?*]', "_", path)

    # If the last segment has no extension, treat as directory
    last_segment = path.rsplit("/", 1)[-1]
    if "." not in last_segment:
        path = f"{path}/{_INDEX_FILENAME}"
    else:
        # Replace existing extension with .md
        path = re.sub(r"\.[^./]+$", ".md", path)

    full = f"{host}/{path}"

    # Truncate if too long, preserving .md extension
    if len(full) > _MAX_FILENAME_LEN:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        full = full[: _MAX_FILENAME_LEN - 16] + f"_{url_hash}.md"

    return full


def _web_dir() -> Path:
    """Return the _web/ subdirectory under documents."""
    return cfg.documents_dir / "_web"


def save_crawl_results(results: list[CrawlResult]) -> list[Path]:
    """Write successful crawl results as .md files under documents/_web/.
    Returns list of paths written.
    """
    written: list[Path] = []
    web_dir = _web_dir()
    resolved_web_dir = web_dir.resolve()
    for result in results:
        if not result.success or not result.markdown.strip():
            continue
        rel = url_to_filename(result.url)
        dest = web_dir / rel
        try:
            validate_path_within(dest, resolved_web_dir)
        except ValueError:
            log.warning("Path traversal blocked: %s → %s", result.url, dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.markdown, encoding="utf-8")
        written.append(dest)
    return written


def _crawl_meta_path() -> Path:
    """Path to the crawl metadata sidecar JSON."""
    return cfg.data_dir / "crawl_meta.json"


@dataclass
class CrawlMeta:
    """Metadata for a single crawled URL."""

    file: str
    content_hash: str
    crawled_at: str


def load_crawl_metadata() -> dict[str, CrawlMeta]:
    """Load URL→metadata mapping from the JSON sidecar."""
    path = _crawl_meta_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    result: dict[str, CrawlMeta] = {}
    for url, data in raw.items():
        try:
            result[url] = CrawlMeta(**data)
        except (TypeError, KeyError):
            log.warning("Skipping malformed crawl metadata entry: %s", url)
    return result


def save_crawl_metadata(meta: dict[str, CrawlMeta]) -> None:
    """Persist URL→metadata mapping to the JSON sidecar (atomic write)."""
    path = _crawl_meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    serializable = {
        url: {"file": m.file, "content_hash": m.content_hash, "crawled_at": m.crawled_at}
        for url, m in meta.items()
    }
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(json.dumps(serializable, indent=2).encode("utf-8"))
        Path(tmp_name).replace(path)
    except BaseException:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def content_hash(text: str) -> str:
    """SHA-256 hex digest of text content."""
    return hashlib.sha256(text.encode()).hexdigest()


def update_metadata(results: list[CrawlResult]) -> None:
    """Update crawl metadata with successful results."""
    meta = load_crawl_metadata()
    now = datetime.now(UTC).isoformat()
    for r in results:
        if r.success and r.markdown.strip():
            meta[r.url] = CrawlMeta(
                file=url_to_filename(r.url),
                content_hash=content_hash(r.markdown),
                crawled_at=now,
            )
    save_crawl_metadata(meta)


@contextlib.asynccontextmanager
async def _open_crawler(*, quiet: bool = False) -> AsyncIterator[Any]:
    """Open an AsyncWebCrawler, suppressing stdout when quiet.

    Raises :class:`CrawlerBrowserMissing` early if the Chromium binary
    hasn't been downloaded. Without this guard Playwright prints a full
    ASCII install banner that leaks into the TUI.
    """
    if not chromium_installed():
        raise CrawlerBrowserMissing(
            "Playwright Chromium browser not installed. "
            "Run 'uv run playwright install chromium' to enable /crawl."
        )

    from crawl4ai import AsyncWebCrawler

    stdout_ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    stderr_ctx = contextlib.redirect_stderr(io.StringIO()) if quiet else contextlib.nullcontext()
    with stdout_ctx, stderr_ctx:
        async with AsyncWebCrawler(verbose=not quiet) as crawler:
            yield crawler


async def crawl_single(url: str, *, quiet: bool = False) -> CrawlResult:
    """Fetch a single URL and return its markdown content."""
    validate_crawl_url(url)
    from crawl4ai import CrawlerRunConfig

    config = CrawlerRunConfig(
        page_timeout=cfg.crawl_timeout * 1000,  # ms
    )
    try:
        async with _open_crawler(quiet=quiet) as crawler:
            result = await crawler.arun(url=url, config=config)
        markdown = (result.markdown or "").strip()
        if markdown:
            return CrawlResult(url=url, markdown=markdown, success=True)
        return CrawlResult(
            url=url,
            success=False,
            error=result.error_message or "No content extracted",
        )
    except CrawlerBrowserMissing:
        raise
    except Exception as exc:
        log.warning("Failed to crawl %s: %s", url, exc)
        return CrawlResult(url=url, success=False, error=str(exc))


async def crawl_recursive(
    url: str,
    max_depth: int = 0,
    max_pages: int = 0,
    on_progress: DetailedProgressCallback | None = None,
    *,
    quiet: bool = False,
) -> list[CrawlResult]:
    """Crawl a URL recursively using BFS, returning results for all pages.
    Uses crawl4ai's deep crawl strategy for link discovery.
    Falls back to cfg defaults when max_depth/max_pages are 0.
    """
    validate_crawl_url(url)
    from crawl4ai import CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

    depth = max_depth if max_depth > 0 else cfg.crawl_max_depth
    pages = min(max_pages if max_pages > 0 else cfg.crawl_max_pages, cfg.crawl_max_pages)

    strategy = BFSDeepCrawlStrategy(
        max_depth=depth,
        max_pages=pages,
    )
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        page_timeout=cfg.crawl_timeout * 1000,
    )

    results: list[CrawlResult] = []
    try:
        async with _open_crawler(quiet=quiet) as crawler:
            crawl_results = await crawler.arun(url=url, config=config)
        if not isinstance(crawl_results, list):
            crawl_results = [crawl_results]
        for i, cr in enumerate(crawl_results):
            if on_progress:
                on_progress(
                    EventType.CRAWL_PAGE,
                    CrawlPageEvent(url=cr.url, current=i + 1, total=len(crawl_results)),
                )
            if cr.success:
                results.append(CrawlResult(url=cr.url, markdown=cr.markdown or ""))
            else:
                results.append(
                    CrawlResult(
                        url=cr.url,
                        success=False,
                        error=cr.error_message or "Unknown error",
                    )
                )
    except CrawlerBrowserMissing:
        raise
    except Exception as exc:
        log.warning("Recursive crawl of %s failed: %s", url, exc)
        if not results:
            results.append(CrawlResult(url=url, success=False, error=str(exc)))

    return results


async def _maybe_periodic_sync() -> None:
    """Fire off a background sync if the crawl_sync_interval has elapsed.
    Skips if a sync is already running or periodic sync is disabled (interval=0).
    Uses a threading.Lock to avoid asyncio event-loop binding issues when called
    from different loops.
    """
    interval = cfg.crawl_sync_interval
    if interval <= 0 or not _state.sync_running.acquire(blocking=False):
        return

    now = time.monotonic()
    if now - _state.last_sync_time < interval:
        _state.sync_running.release()
        return

    _state.last_sync_time = now

    async def _run_sync() -> None:
        try:
            from lilbee.ingest import sync

            await sync(quiet=True)
        except Exception as exc:
            log.warning("Periodic sync during crawl failed: %s", exc)
        finally:
            _state.sync_running.release()

    task = asyncio.create_task(_run_sync())
    _state.background_tasks.add(task)
    task.add_done_callback(_state.background_tasks.discard)


async def crawl_and_save(
    url: str,
    *,
    depth: int = 0,
    max_pages: int = 0,
    on_progress: DetailedProgressCallback | None = None,
    cancel: threading.Event | None = None,
    quiet: bool = False,
) -> list[Path]:
    """Crawl URL(s), save as markdown, update metadata. Returns paths written.
    Uses hash-based change detection: always fetches, but only saves files
    whose content has changed (or is new).
    When *cancel* is set, returns early with an empty list.
    """
    max_pages = min(max_pages if max_pages > 0 else cfg.crawl_max_pages, cfg.crawl_max_pages)

    # Auto-bootstrap Chromium on first use so every crawl entry point works
    # on a fresh install without a separate setup step. bootstrap_chromium
    # short-circuits when Chromium is already installed. Any progress is
    # forwarded through the same on_progress callback so downstream UIs
    # surface a 'setup' stage before the crawl events.
    if not chromium_installed():
        await bootstrap_chromium(on_progress=on_progress)

    sem = _get_crawl_semaphore()
    if sem is not None:
        await sem.acquire()
    try:
        if on_progress:
            on_progress(EventType.CRAWL_START, CrawlStartEvent(url=url, depth=depth))

        if depth > 0:
            results = await crawl_recursive(
                url, max_depth=depth, max_pages=max_pages, on_progress=on_progress, quiet=quiet
            )
        else:
            result = await crawl_single(url, quiet=quiet)
            results = [result]
            if on_progress:
                on_progress(EventType.CRAWL_PAGE, CrawlPageEvent(url=url, current=1, total=1))

        if cancel and cancel.is_set():
            return []

        changed = _filter_changed(results)
        paths = save_crawl_results(changed)
        update_metadata(changed)
        await _maybe_periodic_sync()

        if on_progress:
            on_progress(
                EventType.CRAWL_DONE,
                CrawlDoneEvent(pages_crawled=len(results), files_written=len(paths)),
            )

        return paths
    finally:
        if sem is not None:
            sem.release()


def _filter_changed(results: list[CrawlResult]) -> list[CrawlResult]:
    """Return only results whose content differs from the last crawl."""
    meta = load_crawl_metadata()
    web_dir = _web_dir()
    changed: list[CrawlResult] = []
    for r in results:
        if not r.success or not r.markdown.strip():
            continue
        prev = meta.get(r.url)
        file_path = web_dir / url_to_filename(r.url)
        new_hash = content_hash(r.markdown)
        if prev is not None and prev.content_hash == new_hash and file_path.exists():
            log.info("Content unchanged, skipping save: %s", r.url)
            continue
        changed.append(r)
    return changed
