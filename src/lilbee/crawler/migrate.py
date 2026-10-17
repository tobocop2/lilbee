"""The move, once per crawl, of each saved page to the name its canonical address gives it."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.core.web_names import WEB_DIR
from lilbee.crawler import save
from lilbee.crawler.save import CrawlMeta, canonical_address, content_hash, page_name, web_dir
from lilbee.data.store import Store
from lilbee.data.store.types import SourceMove
from lilbee.runtime.lock import LOCK_TIMEOUT, library_held

CRAWL_MOVE_HELD = "A sync is running on this library. Start the crawl again when the sync ends."


@dataclass(frozen=True)
class _SavedPage:
    """A saved page: its address, its entry, where its file is, its stat and its earlier names."""

    address: str
    entry: CrawlMeta
    found: str
    stat: os.stat_result
    earlier_names: tuple[str, ...]

    @property
    def name(self) -> str:
        """The name the page's address gives it."""
        return page_name(self.address)


def _latest_by_address(meta: dict[str, CrawlMeta]) -> dict[str, CrawlMeta]:
    """One entry per canonical address of *meta*, the one crawled last; a key at its name stays."""
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


def _found(address: str, entry: CrawlMeta) -> tuple[str, os.stat_result] | None:
    """Where the file of *address* is: at its recorded name, else at its own after a cut move."""
    for name in dict.fromkeys((entry.file, page_name(address))):
        stat = _file_stat(name)
        if stat is not None:
            return name, stat
    return None


def _owner(group: list[_SavedPage]) -> _SavedPage:
    """The page that owns the file *group* shares: the one whose text it holds, else the latest."""
    latest = max(group, key=lambda page: page.entry.crawled_at)
    if len(group) == 1:
        return latest
    text = (web_dir() / latest.found).read_text(encoding="utf-8", errors="replace")
    held = content_hash(text)
    owner = next((page for page in group if page.entry.content_hash == held), latest)
    names = tuple(dict.fromkeys(name for page in group for name in page.earlier_names))
    return replace(owner, entry=replace(owner.entry, content_hash=held), earlier_names=names)


def _file_key(name: str, stat: os.stat_result) -> tuple[int, int | str]:
    """What two names of one file share: its inode, or its resolved path where the disk has none."""
    if stat.st_ino:
        return stat.st_dev, stat.st_ino
    return stat.st_dev, os.path.normcase((web_dir() / name).resolve())


def _file_owners(pages: dict[str, CrawlMeta]) -> list[_SavedPage]:
    """The page that owns each file, as the disk resolves it; a page with no file is left out."""
    sharing: dict[tuple[int, int | str], list[_SavedPage]] = {}
    for address, entry in pages.items():
        found = _found(address, entry)
        if found is not None:
            name, stat = found
            page = _SavedPage(address, entry, name, stat, (entry.file,))
            sharing.setdefault(_file_key(name, stat), []).append(page)
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


def _move_file(page: _SavedPage) -> None:
    """Move *page*'s file to its name, over any older copy there, and remove emptied folders."""
    earlier = web_dir() / page.found
    target = web_dir() / page.name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(earlier, target)
    _remove_empty_folders(earlier.parent)


def _index_move(page: _SavedPage) -> SourceMove:
    """The re-key of *page*'s index source, under any earlier name, to its name; its stat stays."""
    sources = tuple(dict.fromkeys(_source(name) for name in page.earlier_names))
    return SourceMove(sources, _source(page.name), None)


def _move_index(rekeyed: list[_SavedPage], dropped: list[_SavedPage], store: Store) -> None:
    """Re-key the sources of *rekeyed* to their names; drop the earlier sources of *dropped*."""
    # heavy: lilbee.data.ingest costs 263 ms (python -X importtime) past lilbee.crawler
    from lilbee.data.ingest.pipeline import relocate_in_holds

    if dropped:
        sources = (_source(name) for page in dropped for name in page.earlier_names)
        store.remove_documents(list(dict.fromkeys(sources)))
    relocate_in_holds(store, [_index_move(page) for page in rekeyed])


def _move_pages(pages: list[_SavedPage], store: Store) -> dict[str, CrawlMeta]:
    """Move every file of *pages* to its name, then the index; the entries to record."""
    moving = [page for page in pages if page.found != page.name]
    taken = {page.address for page in moving if (web_dir() / page.name).exists()}
    for page in moving:
        _move_file(page)
    rekeyed = [page for page in moving if page.address not in taken]
    keys = {page.address for page in rekeyed}
    dropped = [page for page in pages if page.entry.file != page.name and page.address not in keys]
    _move_index(rekeyed, dropped, store)
    return {page.address: replace(page.entry, file=page.name) for page in pages}


def _plan() -> tuple[dict[str, CrawlMeta], list[_SavedPage]]:
    """The stored crawl metadata and the page that owns each saved file in it."""
    stored = save.load_crawl_metadata()
    return stored, _file_owners(_latest_by_address(stored))


def _written(stored: dict[str, CrawlMeta], meta: dict[str, CrawlMeta]) -> dict[str, CrawlMeta]:
    """*meta*, written over *stored* when the two differ."""
    if meta != stored:
        save.save_crawl_metadata(meta)
    return meta


def move_saved_pages(store: Store) -> dict[str, CrawlMeta]:
    """The crawl metadata with one entry per canonical address, each page at its name."""
    stored, pages = _plan()
    if all(page.entry.file == page.name for page in pages):
        return _written(stored, {page.address: page.entry for page in pages})
    with library_held(cfg, CRAWL_MOVE_HELD, LOCK_TIMEOUT):
        stored, pages = _plan()
        return _written(stored, _move_pages(pages, store))
