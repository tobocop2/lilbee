"""The page record of crawled pages: one JSON line per save, the last line of an address counts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.core.web_names import WEB_PREFIX

log = logging.getLogger(__name__)

PAGES_RECORD_FILE = "crawl_pages.jsonl"
# The field of a page record line that holds the page's canonical address.
RECORD_ADDRESS_FIELD = "url"


@dataclass
class CrawlMeta:
    """Metadata for a single crawled URL."""

    file: str
    content_hash: str
    crawled_at: str
    # The hash of the text the file held when this entry was recorded; empty for a new file.
    replaces: str = ""
    # The page's own title; empty when the page has none or an older lilbee saved it.
    title: str = ""


def pages_path() -> Path:
    """The page record file under the data directory."""
    return cfg.data_dir / PAGES_RECORD_FILE


def record_line(address: str, entry: CrawlMeta) -> str:
    """The page record line of *entry* for the page at *address*."""
    return json.dumps({RECORD_ADDRESS_FIELD: address, **asdict(entry)}, ensure_ascii=False)


def read_pages(lines: list[str]) -> dict[str, CrawlMeta]:
    """The last entry of each address in the page record *lines*; a malformed line is skipped."""
    pages: dict[str, CrawlMeta] = {}
    for line in filter(None, lines):
        try:
            data = json.loads(line)
            address = data.pop(RECORD_ADDRESS_FIELD)
            pages[address] = CrawlMeta(**data)
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
            log.warning("Skipping malformed crawl record line in %s", pages_path())
    return pages


@lru_cache(maxsize=1)
def _titles_at(path: Path, mtime_ns: int, size: int) -> dict[str, str]:
    """The non-empty title of each page in the record at *path*, by source name."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        f"{WEB_PREFIX}{entry.file}": entry.title
        for entry in read_pages(lines).values()
        if entry.title
    }


def page_titles() -> dict[str, str]:
    """The title the page record holds for each crawled page, by source name."""
    path = pages_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    return _titles_at(path, stat.st_mtime_ns, stat.st_size)
