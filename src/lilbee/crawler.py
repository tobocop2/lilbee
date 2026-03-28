"""Web crawling — fetch pages as markdown and save to the documents directory."""

import asyncio
import hashlib
import ipaddress
import json
import logging
import re
import socket
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from lilbee.config import cfg
from lilbee.progress import DetailedProgressCallback, EventType
from lilbee.security import validate_path_within

log = logging.getLogger(__name__)


def crawler_available() -> bool:
    """Check if crawl4ai is installed."""
    try:
        import crawl4ai  # noqa: F401

        return True
    except ImportError:
        return False


_crawl_semaphore: threading.Semaphore | None = None
_crawl_semaphore_limit: int = 0


def _get_crawl_semaphore() -> threading.Semaphore | None:
    """Return a module-level threading semaphore for crawl concurrency, or None if unlimited (0).

    Uses threading.Semaphore instead of asyncio.Semaphore to avoid event-loop binding issues
    when crawl_and_save is called from different loops (e.g. CLI via asyncio.run).
    """
    global _crawl_semaphore, _crawl_semaphore_limit
    limit = cfg.crawl_max_concurrent
    if limit <= 0:
        return None
    if _crawl_semaphore is None or _crawl_semaphore_limit != limit:
        _crawl_semaphore = threading.Semaphore(limit)
        _crawl_semaphore_limit = limit
    return _crawl_semaphore


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
    serializable = {
        url: {"file": m.file, "content_hash": m.content_hash, "crawled_at": m.crawled_at}
        for url, m in meta.items()
    }
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        tmp.rename(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
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


async def crawl_single(url: str) -> CrawlResult:
    """Fetch a single URL and return its markdown content."""
    validate_crawl_url(url)
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    config = CrawlerRunConfig(
        page_timeout=cfg.crawl_timeout * 1000,  # ms
    )
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
            # crawl4ai may set success=False for sub-resource failures (e.g. favicon 404)
            # even when the main page has valid markdown. Trust the content, not the flag.
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


async def crawl_recursive(
    url: str,
    max_depth: int = 0,
    max_pages: int = 0,
    on_progress: DetailedProgressCallback | None = None,
) -> list[CrawlResult]:
    """Crawl a URL recursively using BFS, returning results for all pages.

    Uses crawl4ai's deep crawl strategy for link discovery.
    Falls back to cfg defaults when max_depth/max_pages are 0.
    """
    validate_crawl_url(url)
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
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
        async with AsyncWebCrawler() as crawler:
            crawl_results = await crawler.arun(url=url, config=config)
            # arun with deep crawl returns a list
            if not isinstance(crawl_results, list):
                crawl_results = [crawl_results]
            for i, cr in enumerate(crawl_results):
                if on_progress:
                    on_progress(
                        EventType.CRAWL_PAGE,
                        {"url": cr.url, "current": i + 1, "total": len(crawl_results)},
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
    except Exception as exc:
        log.warning("Recursive crawl of %s failed: %s", url, exc)
        if not results:
            results.append(CrawlResult(url=url, success=False, error=str(exc)))

    return results


_last_sync_time: float = 0.0
_sync_running = threading.Lock()
_background_tasks: set[asyncio.Task[None]] = set()


async def _maybe_periodic_sync() -> None:
    """Fire off a background sync if the crawl_sync_interval has elapsed.

    Skips if a sync is already running or periodic sync is disabled (interval=0).
    Uses a threading.Lock to avoid asyncio event-loop binding issues when called
    from different loops.
    """
    global _last_sync_time
    interval = cfg.crawl_sync_interval
    if interval <= 0 or not _sync_running.acquire(blocking=False):
        return

    now = time.monotonic()
    if now - _last_sync_time < interval:
        _sync_running.release()
        return

    _last_sync_time = now

    async def _run_sync() -> None:
        try:
            from lilbee.ingest import sync

            await sync(quiet=True)
        except Exception as exc:
            log.warning("Periodic sync during crawl failed: %s", exc)
        finally:
            _sync_running.release()

    task = asyncio.create_task(_run_sync())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def crawl_and_save(
    url: str,
    *,
    depth: int = 0,
    max_pages: int = 0,
    on_progress: DetailedProgressCallback | None = None,
) -> list[Path]:
    """Crawl URL(s), save as markdown, update metadata. Returns paths written.

    Uses hash-based change detection: always fetches, but only saves files
    whose content has changed (or is new).
    """
    max_pages = min(max_pages if max_pages > 0 else cfg.crawl_max_pages, cfg.crawl_max_pages)

    sem = _get_crawl_semaphore()
    if sem is not None:
        sem.acquire()
    try:
        if on_progress:
            on_progress(EventType.CRAWL_START, {"url": url, "depth": depth})

        if depth > 0:
            results = await crawl_recursive(
                url, max_depth=depth, max_pages=max_pages, on_progress=on_progress
            )
        else:
            result = await crawl_single(url)
            results = [result]
            if on_progress:
                on_progress(EventType.CRAWL_PAGE, {"url": url, "current": 1, "total": 1})

        changed = _filter_changed(results)
        paths = save_crawl_results(changed)
        update_metadata(changed)
        await _maybe_periodic_sync()

        if on_progress:
            on_progress(
                EventType.CRAWL_DONE,
                {"pages_crawled": len(results), "files_written": len(paths)},
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
