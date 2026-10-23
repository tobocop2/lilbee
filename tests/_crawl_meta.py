"""Seeds the crawl page table as a crawl, or the import of an older lilbee's files, leaves it."""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.core.config import cfg
from lilbee.crawler.save import CrawlRecord, load_crawl_record
from lilbee.data.store import CrawlPage, Store


@dataclass(frozen=True)
class CrawlMeta:
    """One page's file as a test gives it: the fields of a row less its address."""

    file: str
    content_hash: str
    crawled_at: str
    replaces: str = ""
    title: str = ""


def crawl_store() -> Store:
    """A store on the configured library."""
    return Store(cfg)


def _seed(entries: dict[str, CrawlMeta], *, current: bool) -> None:
    pages = crawl_store().crawl_pages
    with pages.held():
        for address, entry in entries.items():
            pages.claim(CrawlPage(address, **vars(entry), current=current))


def write_crawl_metadata(meta: dict[str, CrawlMeta]) -> None:
    """Give each address of *meta* its entry as an earlier copy, keyed as written."""
    _seed(meta, current=False)


def claim_pages(pages: dict[str, CrawlMeta]) -> None:
    """Make each entry of *pages* the current file of its address."""
    _seed(pages, current=True)


def crawl_record() -> CrawlRecord:
    """The crawl record a crawl of the configured library starts with."""
    return load_crawl_record(crawl_store())


def recorded() -> dict[str, CrawlPage]:
    """The current row of each address, as the last commit holds it."""
    return {row.address: row for row in crawl_store().crawl_pages.rows() if row.current}


def earlier_rows() -> list[CrawlPage]:
    """Every earlier copy, as the last commit holds it."""
    return [row for row in crawl_store().crawl_pages.rows() if not row.current]
