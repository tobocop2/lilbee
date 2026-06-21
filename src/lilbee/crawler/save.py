"""URL-to-filename mapping, metadata I/O, and per-page save-to-disk."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.crawler.models import CrawlResult

log = logging.getLogger(__name__)

# Maximum filename length before truncation (most filesystems cap at 255 bytes)
_MAX_FILENAME_LEN = 200

# Sentinel for index pages (trailing slash or empty path)
_INDEX_FILENAME = "index.md"

# Length of the query-string hash folded into a filename to keep distinct
# queries on the same path from mapping to (and overwriting) the same file.
_QUERY_HASH_LEN = 8

# How often the crawl metadata JSON is rewritten during a streaming crawl.
# Markdown files are durable per-page; metadata batches to keep write volume
# bounded. Worst-case loss on crash is N-1 entries, recoverable from the files.
METADATA_FLUSH_INTERVAL = 10


def url_to_filename(url: str) -> str:
    """Convert a URL to a safe filesystem path ending in .md.

    Examples:
        https://docs.python.org/3/tutorial/ → docs.python.org/3/tutorial/index.md
        https://example.com/page?q=1        → example.com/page/index_q<hash>.md
        https://example.com/                → example.com/index.md

    A distinct query string folds a short hash of the query into the filename so
    two URLs differing only in their query do not map to the same file and
    overwrite each other. The fragment is client-side and intentionally ignored.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    path = parsed.path.strip("/")

    if not path:
        rel = _INDEX_FILENAME
    else:
        # Neutralize path traversal segments and unsafe filesystem characters.
        path = re.sub(r"\.\.+", "_", path)
        path = re.sub(r'[<>:"|?*]', "_", path)
        last_segment = path.rsplit("/", 1)[-1]
        if re.search(r"\.[^./]+$", last_segment):
            rel = re.sub(r"\.[^./]+$", ".md", path)  # real extension: swap it for .md
        else:
            # No real extension (incl. a bare "." segment) -> treat as a directory.
            # This guarantees rel ends in .md so the query suffix below always applies.
            rel = f"{path}/{_INDEX_FILENAME}"

    if parsed.query:
        query_hash = hashlib.sha256(parsed.query.encode()).hexdigest()[:_QUERY_HASH_LEN]
        rel = re.sub(r"\.md$", f"_q{query_hash}.md", rel)

    full = f"{host}/{rel}"

    # Truncate if too long, preserving .md extension
    if len(full) > _MAX_FILENAME_LEN:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        full = full[: _MAX_FILENAME_LEN - 16] + f"_{url_hash}.md"

    return full


def _web_dir() -> Path:
    """Return the _web/ subdirectory under documents."""
    return cfg.documents_dir / "_web"


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


# Reference-style nested-bracket links, e.g. Wikipedia footnote markers like
# ``[[1]](https://en.wikipedia.org/wiki/Foo#cite_note-1)``. The inner brackets
# make this a normal Markdown link with the text ``[1]``, but readers that
# treat ``[[...]]`` as a wikilink (Obsidian) mis-parse it as a broken wikilink
# followed by the literal URL.
_REFERENCE_LINK_RE = re.compile(r"\[\[([^\]]*)\]\]\(([^)]*)\)")


def normalize_crawled_markdown(markdown: str) -> str:
    """Collapse reference-style ``[[N]](url)`` links to plain ``[N](url)``.

    This fixes the double-bracket/wikilink collision without dropping the
    link text or URL. Ordinary single-bracket links are left untouched.
    """
    return _REFERENCE_LINK_RE.sub(r"[\1](\2)", markdown)


@dataclass(frozen=True)
class SaveOutcome:
    """Return value of ``_save_single_result``: written path and the hash/filename used."""

    path: Path
    filename: str
    content_hash: str


def _save_single_result(result: CrawlResult, meta: dict[str, CrawlMeta]) -> SaveOutcome | None:
    """Write one crawl result to disk if it's new or changed.

    Returns the outcome (written path plus reusable filename/hash), or
    None if skipped (failure, empty markdown, unchanged hash with file
    on disk, or blocked by path traversal).
    """
    if not result.success or not result.markdown.strip():
        return None
    markdown = normalize_crawled_markdown(result.markdown)
    filename = url_to_filename(result.url)
    web_dir = _web_dir()
    file_path = web_dir / filename
    resolved_web_dir = web_dir.resolve()
    try:
        validate_path_within(file_path, resolved_web_dir)
    except ValueError:
        log.warning("Path traversal blocked: %s -> %s", result.url, file_path)
        return None
    new_hash = content_hash(markdown)
    prev = meta.get(result.url)
    if prev is not None and prev.content_hash == new_hash and file_path.exists():
        log.info("Content unchanged, skipping save: %s", result.url)
        return None
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(markdown, encoding="utf-8")
    return SaveOutcome(path=file_path, filename=filename, content_hash=new_hash)


def _update_single_metadata(
    meta: dict[str, CrawlMeta],
    url: str,
    outcome: SaveOutcome,
    now: str,
) -> None:
    """Update the metadata dict in place with a previously-computed outcome."""
    meta[url] = CrawlMeta(
        file=outcome.filename,
        content_hash=outcome.content_hash,
        crawled_at=now,
    )
