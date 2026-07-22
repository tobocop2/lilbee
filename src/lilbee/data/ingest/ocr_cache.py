"""Content-addressed cache for per-page OCR text.

OCR is the most expensive ingest step (a 595-page scanned manual runs ~18 min).
Its per-page output is cached keyed on the file's content hash plus the OCR
backend and model, so a failure in a downstream step (chunk, embed, or the
LanceDB write) no longer discards the whole OCR pass: the retry reads the cached
pages and goes straight to chunk + embed.

Storage is ``diskcache``, for its size cap and LRU eviction. The key includes the
file's content hash, so an edited file, a new vision model, or a changed timeout
all mint a *new* key and leave the old one behind; across tens of thousands of
scanned PDFs re-ingested under different vision settings, an unbounded store of
per-page text is a silent multi-GB disk sink. The cache only exists to make a
retry cheap, so evicting the least recently used entry at the cap costs at most
one re-OCR of something nothing has asked for in a while.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from diskcache import Cache

from lilbee.core.config import active_config

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# A cached page is stored as a (page_number, text) pair.
_PAGE_ENTRY_LEN = 2

_CACHE_DIRNAME = "ocr_cache"
# Bump when the entry format or what counts as "the same OCR" changes, so old
# entries miss instead of being misread.
_CACHE_VERSION = "2"
_KEY_SEP = "\0"
# Ceiling on the whole cache. Page text is small per document, so this holds a
# large working set while keeping the store bounded.
_SIZE_LIMIT_BYTES = 2 * 1024**3


def _cache_dir() -> Path:
    return active_config().data_dir / _CACHE_DIRNAME


def _open_cache() -> Cache:
    return Cache(
        directory=str(_cache_dir()),
        size_limit=_SIZE_LIMIT_BYTES,
        eviction_policy="least-recently-used",
    )


def ocr_cache_key(file_hash: str, *, backend: str, model: str, extra: str = "") -> str:
    """Stable cache key for one file's OCR output under a backend/model/config tuple.

    ``file_hash`` ties the entry to exact file content (an edit changes the hash
    and so misses); ``backend`` / ``model`` / ``extra`` capture what would change
    the OCR text for the same bytes.
    """
    payload = _KEY_SEP.join((_CACHE_VERSION, file_hash, backend, model, extra))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parsed_pages(raw: Any) -> list[tuple[int, str]] | None:
    """*raw* as ``(page, text)`` pairs, or ``None`` if it is not that shape."""
    # An empty list is corruption too: store_ocr_pages never writes one.
    if not isinstance(raw, list) or not raw:
        return None
    pages: list[tuple[int, str]] = []
    for entry in raw:
        if not (
            isinstance(entry, (list, tuple))
            and len(entry) == _PAGE_ENTRY_LEN
            and isinstance(entry[0], int)
            and isinstance(entry[1], str)
        ):
            # Dropping the bad entry would hand the caller a page-short
            # document it cannot tell from a complete one.
            return None
        pages.append((entry[0], entry[1]))
    return pages


def load_ocr_pages(key: str) -> list[tuple[int, str]] | None:
    """Return cached ``(page, text)`` pairs for *key*, or ``None`` on miss.

    A missing, unreadable, or malformed entry returns ``None`` so the caller
    re-runs OCR rather than trusting partial data.
    """
    try:
        with _open_cache() as cache:
            raw = cache.get(key)
    except Exception:
        log.debug("OCR cache read failed for %s", key, exc_info=True)
        return None
    if raw is None:
        return None
    pages = _parsed_pages(raw)
    if pages is None:
        log.debug("OCR cache entry for %s is malformed; treating as a miss", key)
    return pages


def store_ocr_pages(key: str, pages: list[tuple[int, str]]) -> None:
    """Persist ``(page, text)`` pairs for *key*, evicting least-recent entries at the cap.

    Best-effort: a write failure is logged and swallowed (a cache miss next time
    is harmless). Empty results are not cached so a failed OCR run is retried.
    """
    if not pages:
        return
    try:
        with _open_cache() as cache:
            cache.set(key, list(pages))
    except Exception:
        log.debug("OCR cache write failed for %s", key, exc_info=True)
