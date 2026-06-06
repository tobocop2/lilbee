"""Content-addressed cache for per-page OCR text.

OCR is the most expensive ingest step (a 595-page scanned manual runs ~18 min).
Its per-page output is cached keyed on the file's content hash plus the OCR
backend and model, so a failure in a downstream step (chunk, embed, or the
LanceDB write) no longer discards the whole OCR pass: the retry reads the cached
pages and goes straight to chunk + embed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from lilbee.core.config import cfg

log = logging.getLogger(__name__)

# A cached page is serialized as a [page_number, text] pair.
_PAGE_ENTRY_LEN = 2

_CACHE_DIRNAME = "ocr_cache"
# Bump when the entry format or what counts as "the same OCR" changes, so old
# entries miss instead of being misread.
_CACHE_VERSION = "1"
_KEY_SEP = "\0"


def _cache_dir() -> Path:
    return cfg.data_dir / _CACHE_DIRNAME


def ocr_cache_key(file_hash: str, *, backend: str, model: str, extra: str = "") -> str:
    """Stable cache key for one file's OCR output under a backend/model/config tuple.

    ``file_hash`` ties the entry to exact file content (an edit changes the hash
    and so misses); ``backend`` / ``model`` / ``extra`` capture what would change
    the OCR text for the same bytes.
    """
    payload = _KEY_SEP.join((_CACHE_VERSION, file_hash, backend, model, extra))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_ocr_pages(key: str) -> list[tuple[int, str]] | None:
    """Return cached ``(page, text)`` pairs for *key*, or ``None`` on miss.

    A missing, unreadable, or malformed entry returns ``None`` so the caller
    re-runs OCR rather than trusting partial data.
    """
    path = _cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.debug("OCR cache read failed for %s", key, exc_info=True)
        return None
    if not isinstance(raw, list):
        return None
    pages: list[tuple[int, str]] = []
    for entry in raw:
        if (
            isinstance(entry, list)
            and len(entry) == _PAGE_ENTRY_LEN
            and isinstance(entry[0], int)
            and isinstance(entry[1], str)
        ):
            pages.append((entry[0], entry[1]))
    return pages


def store_ocr_pages(key: str, pages: list[tuple[int, str]]) -> None:
    """Persist ``(page, text)`` pairs for *key*.

    Best-effort: a write failure is logged and swallowed (a cache miss next time
    is harmless). Empty results are not cached so a failed OCR run is retried.
    """
    if not pages:
        return
    cache_dir = _cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([[page, text] for page, text in pages]), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        log.debug("OCR cache write failed for %s", key, exc_info=True)
