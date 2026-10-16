"""The move, once per crawl, of each saved page to the name its canonical address gives it."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.core.web_names import WEB_DIR
from lilbee.crawler.save import CrawlMeta, canonical_address, content_hash, page_name, web_dir
from lilbee.data.store import Store
from lilbee.data.store.types import SourceMove, SourceStat
from lilbee.runtime.lock import LOCK_TIMEOUT, library_held

log = logging.getLogger(__name__)

CRAWL_MOVE_HELD = (
    "A sync is running on this library. Start the crawl again when the sync ends."
)


@dataclass(frozen=True)
class _SavedPage:
    """A saved page: its canonical address, its entry, its file's stat, its file's earlier names."""

    address: str
    entry: CrawlMeta
    stat: os.stat_result
    earlier_names: tuple[str, ...]

    @property
    def name(self) -> str:
        """The name the page's address gives it."""
        return page_name(self.address)


def _latest_by_address(meta: dict[str, CrawlMeta]) -> dict[str, CrawlMeta]:
    """One entry per canonical address of *meta*: the one crawled last.

    A key whose file holds its own hash is a canonical address already; it is not
    normalized again, because the escape step can make a new escape (``%%341``).
    """
    pages: dict[str, CrawlMeta] = {}
    for url, entry in meta.items():
        address = url if entry.file == page_name(url) else canonical_address(url)
        held = pages.get(address)
        if held is None or entry.crawled_at > held.crawled_at:
            pages[address] = entry
    return pages


def _file_stat(name: str) -> os.stat_result | None:
    """The stat of the saved file *name* under ``_web``, or None when there is no such file."""
    try:
        return validate_path_within(web_dir() / name, web_dir().resolve()).stat()
    except (ValueError, OSError):
        return None


def _owner(group: list[_SavedPage]) -> _SavedPage:
    """The page that owns the file *group* shares: the one whose text it holds, else the latest."""
    latest = max(group, key=lambda page: page.entry.crawled_at)
    if len(group) == 1:
        return latest
    held = content_hash((web_dir() / latest.entry.file).read_text(encoding="utf-8"))
    owner = next((page for page in group if page.entry.content_hash == held), latest)
    names = tuple(dict.fromkeys(name for page in group for name in page.earlier_names))
    return replace(owner, entry=replace(owner.entry, content_hash=held), earlier_names=names)


def _file_owners(pages: dict[str, CrawlMeta]) -> list[_SavedPage]:
    """The page that owns each file, as the disk resolves it; a page with no file is left out."""
    sharing: dict[tuple[int, int], list[_SavedPage]] = {}
    for address, entry in pages.items():
        stat = _file_stat(entry.file)
        if stat is not None:
            page = _SavedPage(address, entry, stat, (entry.file,))
            sharing.setdefault((stat.st_dev, stat.st_ino), []).append(page)
    return [_owner(group) for group in sharing.values()]


def _source(name: str) -> str:
    """The index source of the saved file *name*, as the sync names it."""
    return PurePosixPath(WEB_DIR, name).as_posix()


def _remove_empty_folders(folder: Path) -> None:
    """Remove *folder* and each parent a move left empty, up to ``_web``."""
    root = web_dir()
    while folder != root and folder.is_relative_to(root) and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def _move_file(page: _SavedPage) -> CrawlMeta:
    """Move *page*'s file to its name and remove the folders that leaves empty."""
    earlier = web_dir() / page.entry.file
    target = web_dir() / page.name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(earlier, target)
    _remove_empty_folders(earlier.parent)
    return replace(page.entry, file=page.name)


def _index_move(page: _SavedPage, now: int) -> SourceMove:
    """The re-key of *page*'s index source, under any earlier name of its file, to its name."""
    sources = tuple(dict.fromkeys(_source(name) for name in page.earlier_names))
    return SourceMove(
        sources, _source(page.name), SourceStat(page.stat.st_size, page.stat.st_mtime_ns, now)
    )


def _drop_earlier_copies(pages: list[_SavedPage], store: Store) -> None:
    """Remove the earlier file of each of *pages*, and its index source; each has a newer one."""
    for page in pages:
        earlier = web_dir() / page.entry.file
        earlier.unlink()
        _remove_empty_folders(earlier.parent)
    if pages:
        sources = (_source(name) for page in pages for name in page.earlier_names)
        store.remove_documents(list(dict.fromkeys(sources)))


def _move_files(pages: list[_SavedPage], store: Store) -> list[tuple[str, CrawlMeta]]:
    """Move each of *pages* to its name, its index source first; the entries that stay."""
    # heavy: lilbee.data.ingest costs 263 ms (python -X importtime) past lilbee.crawler
    from lilbee.data.ingest.pipeline import relocate_in_holds

    stays = [page for page in pages if page.entry.file == page.name]
    moving = [page for page in pages if page.entry.file != page.name]
    replaced = [page for page in moving if (web_dir() / page.name).exists()]
    _drop_earlier_copies(replaced, store)
    moving = [page for page in moving if page not in replaced]
    now = time.time_ns()
    relocate_in_holds(store, [_index_move(page, now) for page in moving])
    kept = [(page.address, page.entry) for page in stays]
    return kept + [(page.address, _move_file(page)) for page in moving]


def move_saved_pages(meta: dict[str, CrawlMeta]) -> dict[str, CrawlMeta]:
    """*meta* with one entry per canonical address, each saved page at the name it gives.

    A page whose file is gone, or whose file another page owns, loses its entry and is
    saved afresh by the crawl. Text never changes here, so a second run moves nothing.
    """
    pages = _file_owners(_latest_by_address(meta))
    if all(page.entry.file == page.name for page in pages):
        return {page.address: page.entry for page in pages}
    with library_held(cfg, CRAWL_MOVE_HELD, LOCK_TIMEOUT):
        return dict(_move_files(pages, get_services().store))
