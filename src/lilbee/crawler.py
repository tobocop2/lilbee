"""Web crawling — fetch pages as markdown and save to the documents directory."""

import asyncio
import contextlib
import hashlib
import io
import ipaddress
import json
import logging
import math
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
    CRAWL_TOTAL_UNKNOWN,
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


def _build_rate_limited_dispatcher() -> Any:
    """Build a SemaphoreDispatcher + RateLimiter from cfg, or None when disabled.

    BFSDeepCrawlStrategy calls crawler.arun_many() without a dispatcher kwarg,
    so per-domain rate limiting is only reachable by threading a dispatcher
    through AsyncWebCrawler itself. This helper centralizes the cfg read so the
    TUI / CLI / server all get identical behavior.
    """
    if not cfg.crawl_retry_on_rate_limit:
        return None
    from crawl4ai.async_dispatcher import RateLimiter, SemaphoreDispatcher

    rate_limiter = RateLimiter(
        base_delay=(cfg.crawl_retry_base_delay_min, cfg.crawl_retry_base_delay_max),
        max_delay=cfg.crawl_retry_max_backoff,
        max_retries=cfg.crawl_retry_max_attempts,
    )
    return SemaphoreDispatcher(
        semaphore_count=cfg.crawl_concurrent_requests,
        rate_limiter=rate_limiter,
    )


class _LilbeeAsyncCrawler:
    """AsyncWebCrawler wrapper that injects a default dispatcher on arun_many.

    crawl4ai's BFSDeepCrawlStrategy hard-codes crawler.arun_many(urls, config)
    without a dispatcher kwarg, so per-domain rate limiting and 429/503 retries
    can't be wired via CrawlerRunConfig. By giving the crawler a default
    dispatcher, every strategy-originated arun_many picks it up. An explicit
    dispatcher= on the call still wins.
    """

    def __init__(self, *, verbose: bool, dispatcher: Any) -> None:
        from crawl4ai import AsyncWebCrawler

        self._inner = AsyncWebCrawler(verbose=verbose)
        self._dispatcher = dispatcher

    async def __aenter__(self) -> "_LilbeeAsyncCrawler":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return await self._inner.__aexit__(exc_type, exc, tb)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.arun(*args, **kwargs)

    async def arun_many(
        self, urls: Any, config: Any = None, dispatcher: Any = None, **kwargs: Any
    ) -> Any:
        return await self._inner.arun_many(
            urls,
            config=config,
            dispatcher=dispatcher if dispatcher is not None else self._dispatcher,
            **kwargs,
        )


@contextlib.asynccontextmanager
async def _open_crawler(*, quiet: bool = False, dispatcher: Any = None) -> AsyncIterator[Any]:
    """Open a crawler.

    Raises :class:`CrawlerBrowserMissing` early if the Chromium binary
    hasn't been downloaded. Without this guard Playwright prints a full
    ASCII install banner that leaks into the TUI.

    When *dispatcher* is provided, wrap AsyncWebCrawler in _LilbeeAsyncCrawler
    so every strategy-originated arun_many call picks it up. The single-URL
    path (crawl_single) doesn't need a dispatcher because arun() doesn't accept
    one, so it passes None and gets a bare AsyncWebCrawler.
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
        if dispatcher is not None:
            async with _LilbeeAsyncCrawler(verbose=not quiet, dispatcher=dispatcher) as crawler:
                yield crawler
        else:
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


def _resolve_limit(value: int | None, cfg_ceiling: int | None) -> float:
    """Resolve a caller-provided crawl limit to the number crawl4ai consumes.

    None    -> cfg_ceiling (itself may be None, which collapses to math.inf)
    n > 0   -> n (explicit caller intent; cfg is not a ceiling here)
    n <= 0  -> ValueError (use None for unbounded, not 0)
    """
    effective = value if value is not None else cfg_ceiling
    if effective is None:
        return math.inf
    if effective <= 0:
        raise ValueError("crawl limit must be a positive int or None")
    return effective


# Sitemap lookups are best-effort progress hints; never block the actual crawl.
_SITEMAP_FETCH_TIMEOUT_SECONDS = 5.0
_SITEMAP_MAX_URLS = 10_000
_SITEMAP_URL_TAG_RE = re.compile(r"<loc>\s*([^<]+?)\s*</loc>", re.IGNORECASE)


def _count_sitemap_urls(start_url: str, *, include_subdomains: bool) -> int:
    """Best-effort count of URLs in the host's /sitemap.xml that match the crawl scope.

    Returns CRAWL_TOTAL_UNKNOWN on any failure (missing sitemap, timeout,
    parse error, redirect away from the starting host). This is purely a
    progress-hint denominator, so correctness is not load-bearing.

    Only fetches sitemap.xml directly at the root of the starting host; does
    not follow robots.txt references or nested sitemap indexes.
    """
    import httpx

    parsed = urlparse(start_url)
    host = (parsed.hostname or "").lower()
    if not host:
        return CRAWL_TOTAL_UNKNOWN
    sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
    try:
        resp = httpx.get(sitemap_url, timeout=_SITEMAP_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
    except (httpx.HTTPError, OSError):
        return CRAWL_TOTAL_UNKNOWN
    if resp.status_code >= 400:
        return CRAWL_TOTAL_UNKNOWN

    count = 0
    for match in _SITEMAP_URL_TAG_RE.finditer(resp.text):
        candidate = match.group(1).strip()
        link_host = (urlparse(candidate).hostname or "").lower()
        if not link_host:
            continue
        if include_subdomains:
            matches = link_host == host or link_host.endswith(f".{host}")
        else:
            matches = link_host == host
        if matches:
            count += 1
        if count >= _SITEMAP_MAX_URLS:
            break
    return count if count > 0 else CRAWL_TOTAL_UNKNOWN


def _host_scope_filter(start_url: str, *, include_subdomains: bool) -> Any:
    """Build a URLFilter that scopes a crawl to the starting URL's host.

    Default behavior (``include_subdomains=False``) restricts link-following to
    the exact host of *start_url*. For ``https://en.wikipedia.org/...`` this
    excludes ``af.wikipedia.org`` and every other language subdomain.

    When ``include_subdomains=True``, crawl4ai's DomainFilter matches the host
    plus any of its subdomains (``foo.example.com`` matches ``example.com``),
    which is the loose "whole registrable domain" behavior users may want.
    """
    from crawl4ai.deep_crawling.filters import DomainFilter, URLFilter

    host = (urlparse(start_url).hostname or "").lower()
    if include_subdomains:
        return DomainFilter(allowed_domains=host) if host else None

    class _ExactHostFilter(URLFilter):  # type: ignore[misc]
        def __init__(self, allowed_host: str) -> None:
            super().__init__()
            self._host = allowed_host

        def apply(self, url: str) -> bool:
            link_host = (urlparse(url).hostname or "").lower()
            ok = link_host == self._host
            self._update_stats(ok)
            return ok

    return _ExactHostFilter(host) if host else None


async def crawl_recursive(
    url: str,
    max_depth: int | None = None,
    max_pages: int | None = None,
    on_progress: DetailedProgressCallback | None = None,
    cancel: threading.Event | None = None,
    *,
    quiet: bool = False,
    include_subdomains: bool = False,
) -> list[CrawlResult]:
    """Crawl a URL recursively using BFS, streaming per-page progress.

    None values for max_depth / max_pages mean unbounded (constrained only by
    whatever ceiling the user has set in cfg.crawl_max_{depth,pages}, if any).
    Positive ints are explicit caps. CRAWL_PAGE events fire as each page
    completes; total is CRAWL_TOTAL_UNKNOWN since BFS doesn't know the final
    page count up front.

    By default the crawl is scoped to the exact starting host so a Wikipedia
    article doesn't wander into other language editions. Pass
    ``include_subdomains=True`` to broaden scope to the starting host plus any
    subdomains (e.g. ``en.wikipedia.org`` plus ``af.wikipedia.org``).
    """
    validate_crawl_url(url)
    depth = _resolve_limit(max_depth, cfg.crawl_max_depth)
    pages = _resolve_limit(max_pages, cfg.crawl_max_pages)

    # Fail fast before pulling in crawl4ai submodules so callers get a clear
    # CrawlerBrowserMissing instead of a Playwright install banner or a
    # dispatcher import path.
    if not chromium_installed():
        raise CrawlerBrowserMissing(
            "Playwright Chromium browser not installed. "
            "Run 'uv run playwright install chromium' to enable /crawl."
        )

    from crawl4ai import CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter

    def _should_cancel() -> bool:
        return cancel is not None and cancel.is_set()

    # Reject noise URLs at link-discovery time instead of fetching and
    # filtering later. Patterns are treated as regex (use_glob=False);
    # reverse=True flips the filter's sense from include to exclude.
    exclude_patterns = list(cfg.crawl_exclude_patterns)
    filters: list[Any] = []
    host_filter = _host_scope_filter(url, include_subdomains=include_subdomains)
    if host_filter is not None:
        filters.append(host_filter)
    if exclude_patterns:
        filters.append(URLPatternFilter(exclude_patterns, use_glob=False, reverse=True))
    filter_chain = FilterChain(filters) if filters else FilterChain()

    strategy = BFSDeepCrawlStrategy(
        max_depth=depth,
        max_pages=pages,
        should_cancel=_should_cancel,
        filter_chain=filter_chain,
    )
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        page_timeout=cfg.crawl_timeout * 1000,
        mean_delay=cfg.crawl_mean_delay,
        max_range=cfg.crawl_max_delay_range,
        semaphore_count=cfg.crawl_concurrent_requests,
        stream=True,
    )

    # Best-effort sitemap lookup so the TUI / CLI can render a real page-count
    # denominator instead of [n/-1]. Falls back to CRAWL_TOTAL_UNKNOWN on any
    # failure; off the hot path so a slow/missing sitemap never blocks the crawl.
    sitemap_total = await asyncio.to_thread(
        _count_sitemap_urls, url, include_subdomains=include_subdomains
    )

    results: list[CrawlResult] = []
    counter = 0
    dispatcher = _build_rate_limited_dispatcher()
    stream: Any = None
    try:
        async with _open_crawler(quiet=quiet, dispatcher=dispatcher) as crawler:
            stream = await crawler.arun(url=url, config=config)
            try:
                async for cr in _iter_crawl_stream(stream):
                    if _should_cancel():
                        _safe_strategy_cancel(strategy)
                        break
                    counter += 1
                    if on_progress:
                        on_progress(
                            EventType.CRAWL_PAGE,
                            CrawlPageEvent(url=cr.url, current=counter, total=sitemap_total),
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
                    # Hard cap on visible progress. crawl4ai's BFS uses
                    # max_pages to count successful pages, so failed /
                    # redirected pages can push our per-result counter past
                    # the cap even after crawl4ai has stopped dispatching.
                    # Break explicitly so the user-visible count never
                    # exceeds the number they asked for.
                    if counter >= pages:
                        _safe_strategy_cancel(strategy)
                        break
            finally:
                # Close the async generator (if it is one) before the crawler
                # context exits, so Playwright tears down in-flight URLs in
                # order. Skipping this is what produced the "BrowserContext.
                # new_page: Connection closed" spam on cancel.
                await _safe_aclose(stream)
    except CrawlerBrowserMissing:
        raise
    except Exception as exc:
        # After cancel, crawl4ai may raise BrowserContext teardown errors as
        # in-flight URLs bail. That's expected noise, not a failure worth
        # surfacing. Log at debug and drop the synthetic error result.
        if _should_cancel():
            log.debug("Recursive crawl of %s ended during cancel teardown: %s", url, exc)
        else:
            log.warning("Recursive crawl of %s failed: %s", url, exc)
            if not results:
                results.append(CrawlResult(url=url, success=False, error=str(exc)))

    return results


def _safe_strategy_cancel(strategy: Any) -> None:
    """Call strategy.cancel() if available, swallowing if the method is missing.

    BFSDeepCrawlStrategy has .cancel() in crawl4ai 0.8.6. Older versions or
    third-party strategies may not. Belt-and-suspenders: should_cancel already
    gates between BFS levels, but cancel() also short-circuits arun_many.
    """
    cancel_method = getattr(strategy, "cancel", None)
    if callable(cancel_method):
        try:
            cancel_method()
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("strategy.cancel() raised: %s", exc)


async def _safe_aclose(stream: Any) -> None:
    """Close an async generator stream if that is what it is.

    _iter_crawl_stream normalizes over async-generator / list / single-result
    shapes; only the generator shape has an aclose() to call. A list or single
    object is a no-op.
    """
    import inspect

    if stream is None:
        return
    if inspect.isasyncgen(stream):
        with contextlib.suppress(Exception):
            await stream.aclose()


async def _iter_crawl_stream(stream: Any) -> AsyncIterator[Any]:
    """Normalize crawl4ai's arun() return to an async iterator.

    With stream=True on CrawlerRunConfig, crawl4ai 0.8 returns an async
    generator. Older call sites and some crawl4ai code paths return a list
    (batch mode) or a single CrawlResult. Accept all three shapes so tests
    that mock arun() with a plain list keep working.
    """
    import inspect

    if inspect.isasyncgen(stream):
        async for item in stream:
            yield item
        return
    if isinstance(stream, list):
        for item in stream:
            yield item
        return
    yield stream


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
    depth: int | None = None,
    max_pages: int | None = None,
    on_progress: DetailedProgressCallback | None = None,
    cancel: threading.Event | None = None,
    quiet: bool = False,
    include_subdomains: bool = False,
) -> list[Path]:
    """Crawl URL(s), save as markdown, update metadata. Returns paths written.

    depth: None = whole-site unbounded recursion (default). 0 = single URL, no
    recursion. N > 0 = max link-follow depth. max_pages: None = no limit.
    Positive int = cap. cfg.crawl_max_{depth,pages} act as user-opted-in
    ceilings applied only when depth/max_pages are None.

    When recursing, the crawl is scoped to the exact starting host by default.
    Set ``include_subdomains=True`` to also follow links into sibling
    subdomains of the starting host.

    Uses hash-based change detection: always fetches, but only saves files
    whose content has changed (or is new). When *cancel* is set, returns
    early with an empty list.
    """
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
            start_depth = depth if depth is not None else 0
            on_progress(EventType.CRAWL_START, CrawlStartEvent(url=url, depth=start_depth))

        if depth == 0:
            result = await crawl_single(url, quiet=quiet)
            results = [result]
            if on_progress:
                on_progress(EventType.CRAWL_PAGE, CrawlPageEvent(url=url, current=1, total=1))
        else:
            results = await crawl_recursive(
                url,
                max_depth=depth,
                max_pages=max_pages,
                on_progress=on_progress,
                cancel=cancel,
                quiet=quiet,
                include_subdomains=include_subdomains,
            )

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
