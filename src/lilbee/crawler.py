"""Web crawling — fetch pages as markdown and save to the documents directory."""

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from lilbee.config import cfg

log = logging.getLogger(__name__)

# Maximum filename length before truncation (most filesystems cap at 255 bytes)
_MAX_FILENAME_LEN = 200

# Sentinel for index pages (trailing slash or empty path)
_INDEX_FILENAME = "index.md"


@dataclass
class CrawlResult:
    """Outcome of crawling a single URL."""

    url: str
    markdown: str = ""
    success: bool = True
    error: str | None = None


CrawlProgressCallback = Callable[[int, int, str], None]
"""(pages_crawled, pages_total, current_url) → None"""


def _noop_progress(_crawled: int, _total: int, _url: str) -> None:
    """Default no-op progress callback."""


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
    for result in results:
        if not result.success or not result.markdown.strip():
            continue
        rel = url_to_filename(result.url)
        dest = web_dir / rel
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
    return {url: CrawlMeta(**data) for url, data in raw.items()}


def save_crawl_metadata(meta: dict[str, CrawlMeta]) -> None:
    """Persist URL→metadata mapping to the JSON sidecar."""
    path = _crawl_meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        url: {"file": m.file, "content_hash": m.content_hash, "crawled_at": m.crawled_at}
        for url, m in meta.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _content_hash(text: str) -> str:
    """SHA-256 hex digest of text content."""
    return hashlib.sha256(text.encode()).hexdigest()


def _update_metadata(results: list[CrawlResult]) -> None:
    """Update crawl metadata with successful results."""
    meta = load_crawl_metadata()
    now = datetime.now(UTC).isoformat()
    for r in results:
        if r.success and r.markdown.strip():
            meta[r.url] = CrawlMeta(
                file=url_to_filename(r.url),
                content_hash=_content_hash(r.markdown),
                crawled_at=now,
            )
    save_crawl_metadata(meta)


async def crawl_single(url: str) -> CrawlResult:
    """Fetch a single URL and return its markdown content."""
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    config = CrawlerRunConfig(
        page_timeout=cfg.crawl_timeout * 1000,  # ms
    )
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
            if not result.success:
                return CrawlResult(
                    url=url,
                    success=False,
                    error=result.error_message or "Unknown crawl error",
                )
            return CrawlResult(
                url=url,
                markdown=result.markdown or "",
                success=True,
            )
    except Exception as exc:
        log.warning("Failed to crawl %s: %s", url, exc)
        return CrawlResult(url=url, success=False, error=str(exc))


async def crawl_recursive(
    url: str,
    max_depth: int = 0,
    max_pages: int = 0,
    on_progress: CrawlProgressCallback = _noop_progress,
) -> list[CrawlResult]:
    """Crawl a URL recursively using BFS, returning results for all pages.

    Uses crawl4ai's deep crawl strategy for link discovery.
    Falls back to cfg defaults when max_depth/max_pages are 0.
    """
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

    depth = max_depth if max_depth > 0 else cfg.crawl_max_depth
    pages = max_pages if max_pages > 0 else cfg.crawl_max_pages

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
                on_progress(i + 1, len(crawl_results), cr.url)
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


async def crawl_and_save(
    url: str,
    *,
    depth: int = 0,
    max_pages: int = 0,
    on_progress: CrawlProgressCallback = _noop_progress,
) -> list[Path]:
    """Crawl URL(s), save as markdown, update metadata. Returns paths written."""
    if depth > 0:
        results = await crawl_recursive(
            url, max_depth=depth, max_pages=max_pages, on_progress=on_progress
        )
    else:
        result = await crawl_single(url)
        results = [result]

    paths = save_crawl_results(results)
    _update_metadata(results)
    return paths
