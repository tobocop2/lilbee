"""Web crawling — fetch pages as markdown and save to the documents directory."""

import asyncio
import contextlib
import hashlib
import io
import ipaddress
import json
import logging
import math
import re
import socket
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

    When *dispatcher* is provided, wrap AsyncWebCrawler in _LilbeeAsyncCrawler
    so every strategy-originated arun_many call picks it up. The single-URL
    path (crawl_single) doesn't need a dispatcher because arun() doesn't accept
    one, so it passes None and gets a bare AsyncWebCrawler.
    """
    from crawl4ai import AsyncWebCrawler

    stdout_ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with stdout_ctx:
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


async def crawl_recursive(
    url: str,
    max_depth: int | None = None,
    max_pages: int | None = None,
    on_progress: DetailedProgressCallback | None = None,
    cancel: threading.Event | None = None,
    *,
    quiet: bool = False,
) -> list[CrawlResult]:
    """Crawl a URL recursively using BFS, streaming per-page progress.

    None values for max_depth / max_pages mean unbounded (constrained only by
    whatever ceiling the user has set in cfg.crawl_max_{depth,pages}, if any).
    Positive ints are explicit caps. CRAWL_PAGE events fire as each page
    completes; total is CRAWL_TOTAL_UNKNOWN since BFS doesn't know the final
    page count up front.
    """
    validate_crawl_url(url)
    depth = _resolve_limit(max_depth, cfg.crawl_max_depth)
    pages = _resolve_limit(max_pages, cfg.crawl_max_pages)

    from crawl4ai import CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

    def _should_cancel() -> bool:
        return cancel is not None and cancel.is_set()

    strategy = BFSDeepCrawlStrategy(
        max_depth=depth,
        max_pages=pages,
        should_cancel=_should_cancel,
    )
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        page_timeout=cfg.crawl_timeout * 1000,
        mean_delay=cfg.crawl_mean_delay,
        max_range=cfg.crawl_max_delay_range,
        semaphore_count=cfg.crawl_concurrent_requests,
        stream=True,
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
                            CrawlPageEvent(url=cr.url, current=counter, total=CRAWL_TOTAL_UNKNOWN),
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
            finally:
                # Close the async generator (if it is one) before the crawler
                # context exits, so Playwright tears down in-flight URLs in
                # order. Skipping this is what produced the "BrowserContext.
                # new_page: Connection closed" spam on cancel.
                await _safe_aclose(stream)
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
) -> list[Path]:
    """Crawl URL(s), save as markdown, update metadata. Returns paths written.

    depth: None = whole-site unbounded recursion (default). 0 = single URL, no
    recursion. N > 0 = max link-follow depth. max_pages: None = no limit.
    Positive int = cap. cfg.crawl_max_{depth,pages} act as user-opted-in
    ceilings applied only when depth/max_pages are None.

    Uses hash-based change detection: always fetches, but only saves files
    whose content has changed (or is new). When *cancel* is set, returns
    early with an empty list.
    """
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
