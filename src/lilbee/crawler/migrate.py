"""The move, once per crawl, of each saved page to the name its canonical address gives it."""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from lilbee.core.config import cfg
from lilbee.core.security import validate_path_within
from lilbee.core.web_names import WEB_DIR
from lilbee.crawler import save
from lilbee.crawler.save import CrawlMeta, canonical_address, content_hash, page_name, web_dir
from lilbee.data.store import Store
from lilbee.data.store.types import SourceMove
from lilbee.runtime.lock import LOCK_TIMEOUT, library_held

log = logging.getLogger(__name__)

CRAWL_MOVE_HELD = "A sync is running on this library. Start the crawl again when the sync ends."

# What two names of one saved file share: its device and inode, or its resolved path.
_FileKey = tuple[int, int | str]


class _Stored(NamedTuple):
    """One page of the stored crawl metadata: its latest entry, that entry's key, its address."""

    key: str
    address: str
    entry: CrawlMeta


@dataclass(frozen=True)
class _SavedPage:
    """A saved page: its stored key, address, entry naming the file it is in, and earlier names."""

    key: str
    address: str
    entry: CrawlMeta
    stat: os.stat_result
    earlier_names: tuple[str, ...]

    @property
    def name(self) -> str:
        """The name the page's address gives it."""
        return page_name(self.address)

    @property
    def record_key(self) -> str:
        """The key to record the page under: its address, unless that key reads as another page."""
        if self.entry.file == self.name or canonical_address(self.address) == self.address:
            return self.address
        return self.key


def _address(url: str, entry: CrawlMeta) -> str:
    """The canonical address of the page stored under *url*; a key at its name stays."""
    return url if entry.file == page_name(url) else canonical_address(url)


def _file_stat(name: str) -> os.stat_result | None:
    """The stat of *name* under ``_web``, or None when nothing is there."""
    try:
        return validate_path_within(web_dir() / name, web_dir().resolve()).stat()
    except (ValueError, OSError):
        return None


def _file_key(name: str, file_stat: os.stat_result) -> _FileKey:
    """What two names of one file share: its inode, or its resolved path where the disk has none."""
    if file_stat.st_ino:
        return file_stat.st_dev, file_stat.st_ino
    return file_stat.st_dev, os.path.normcase((web_dir() / name).resolve())


class _Record:
    """The stored crawl metadata, and the pages it ties each saved file to."""

    def __init__(self, stored: dict[str, CrawlMeta]) -> None:
        self.stored = stored
        self._entries = [_Stored(url, _address(url, entry), entry) for url, entry in stored.items()]

    @cached_property
    def _addresses_by_file(self) -> dict[_FileKey, set[str]]:
        """The addresses whose entries name each saved file, as the disk resolves it."""
        by_file: dict[_FileKey, set[str]] = {}
        for _, address, entry in self._entries:
            file_stat = _file_stat(entry.file)
            if file_stat is not None:
                by_file.setdefault(_file_key(entry.file, file_stat), set()).add(address)
        return by_file

    def latest(self) -> list[_Stored]:
        """One entry per canonical address, the one crawled last."""
        pages: dict[str, _Stored] = {}
        for stored in self._entries:
            held = pages.get(stored.address)
            if held is None or stored.entry.crawled_at > held.entry.crawled_at:
                pages[stored.address] = stored
        return list(pages.values())

    def holds_another(self, address: str, name: str) -> bool:
        """Whether *name* is a folder, or a file the record ties to a page other than *address*."""
        file_stat = _file_stat(name)
        if file_stat is None:
            return False
        if not stat.S_ISREG(file_stat.st_mode):
            return True
        return bool(self._addresses_by_file.get(_file_key(name, file_stat), set()) - {address})


def _text_hash(name: str) -> str:
    """The content hash of the text of the saved file *name*."""
    return content_hash((web_dir() / name).read_text(encoding="utf-8", errors="replace"))


def _own_copy(page: _Stored, record: _Record, earlier: tuple[str, ...]) -> _SavedPage | None:
    """*page* in a file at its own name that the record ties to no other page, if any."""
    name = page_name(page.address)
    file_stat = _file_stat(name)
    if file_stat is None or record.holds_another(page.address, name):
        return None
    kept = replace(page.entry, file=name, content_hash=_text_hash(name))
    return _SavedPage(page.key, page.address, kept, file_stat, earlier)


def _found(page: _Stored, record: _Record) -> _SavedPage | None:
    """*page* in its recorded file, else in its own copy after a cut move."""
    file_stat = _file_stat(page.entry.file)
    if file_stat is not None and stat.S_ISREG(file_stat.st_mode):
        return _SavedPage(*page, file_stat, (page.entry.file,))
    return _own_copy(page, record, (page.entry.file,))


def _owner(group: list[_SavedPage]) -> _SavedPage:
    """The page that owns the file *group* shares: the one whose text it holds, else the latest."""
    latest = max(group, key=lambda page: page.entry.crawled_at)
    if len(group) == 1:
        return latest
    held = _text_hash(latest.entry.file)
    owner = next((page for page in group if page.entry.content_hash == held), latest)
    names = tuple(dict.fromkeys(name for page in group for name in page.earlier_names))
    return replace(owner, entry=replace(owner.entry, content_hash=held), earlier_names=names)


def _kept(group: list[_SavedPage], record: _Record) -> list[_SavedPage]:
    """The owner of the file *group* shares, and each other page that has a copy of its own."""
    owner = _owner(group)
    others = (page for page in group if page.address != owner.address)
    stored = (_Stored(page.key, page.address, page.entry) for page in others)
    copies = (_own_copy(page, record, ()) for page in stored)
    return [owner, *(page for page in copies if page is not None)]


def _file_owners(record: _Record) -> list[_SavedPage]:
    """The pages that keep an entry: each file's owner, as the disk resolves it, and own copies."""
    sharing: dict[_FileKey, list[_SavedPage]] = {}
    for stored in record.latest():
        page = _found(stored, record)
        if page is not None:
            sharing.setdefault(_file_key(page.entry.file, page.stat), []).append(page)
    return [page for group in sharing.values() for page in _kept(group, record)]


def _source(name: str) -> str:
    """The index source of the saved file *name*, as the sync names it."""
    return PurePosixPath(WEB_DIR, name).as_posix()


def _remove_empty_folders(folder: Path) -> None:
    """Remove *folder* and each parent a move left empty, up to ``_web``."""
    root = web_dir()
    while folder != root and folder.is_relative_to(root) and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def _remove_earlier_folders(pages: list[_SavedPage]) -> None:
    """Remove the empty folders of the earlier names of *pages*, which are at their names."""
    for page in pages:
        for name in page.earlier_names:
            folder = (web_dir() / name).parent
            if name != page.name and folder.is_dir():
                _remove_empty_folders(folder)


def _move_file(page: _SavedPage) -> None:
    """Move *page*'s file to its name, over its own older copy there, and remove emptied folders."""
    earlier = web_dir() / page.entry.file
    target = web_dir() / page.name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(earlier, target)
    _remove_empty_folders(earlier.parent)


def _moved(page: _SavedPage) -> bool:
    """Whether *page*'s file moved to its name; a page that cannot move stays where it is."""
    try:
        _move_file(page)
    except OSError as exc:
        log.warning(
            "Saved page %s stays at %s, not %s: %s", page.address, page.entry.file, page.name, exc
        )
        return False
    return True


def _index_move(page: _SavedPage) -> SourceMove:
    """The re-key of *page*'s index source, under any earlier name, to its name; its stat stays."""
    sources = tuple(dict.fromkeys(_source(name) for name in page.earlier_names))
    return SourceMove(sources, _source(page.name), None)


def _move_index(rekeyed: list[_SavedPage], dropped: list[_SavedPage], store: Store) -> None:
    """Re-key the sources of *rekeyed* to their names; drop the earlier sources of *dropped*."""
    # heavy: lilbee.data.ingest costs 263 ms (python -X importtime) past lilbee.crawler
    from lilbee.data.ingest.pipeline import relocate_in_holds

    names = (name for page in dropped for name in page.earlier_names if name != page.name)
    sources = list(dict.fromkeys(_source(name) for name in names))
    if sources:
        store.remove_documents(sources)
    relocate_in_holds(store, [_index_move(page) for page in rekeyed])


def _move_round(ready: list[_SavedPage], store: Store) -> list[_SavedPage]:
    """Move the files of *ready*, then their index sources; the pages that moved, at their names."""
    taken = {page.address for page in ready if _file_stat(page.name) is not None}
    moved = [page for page in ready if _moved(page)]
    rekeyed = [page for page in moved if page.address not in taken]
    _move_index(rekeyed, [page for page in moved if page.address in taken], store)
    return [replace(page, entry=replace(page.entry, file=page.name)) for page in moved]


def _skip(waiting: list[_SavedPage]) -> None:
    """Log each page of *waiting*, whose name holds a folder or another page's file."""
    for page in waiting:
        log.warning(
            "Saved page %s stays at %s: %s holds a folder or another page's file",
            page.address,
            page.entry.file,
            page.name,
        )


def _move_pages(record: _Record, pages: list[_SavedPage], store: Store) -> dict[str, CrawlMeta]:
    """Move each page whose name is free or holds its own copy, round by round; the entries."""
    placed = {page.address: page for page in pages}
    at_name = [page for page in pages if page.entry.file == page.name]
    _move_index([], at_name, store)
    _remove_earlier_folders(at_name)
    written = _written(record.stored, _entries(pages))
    waiting = [page for page in pages if page.entry.file != page.name]
    while ready := [page for page in waiting if not record.holds_another(page.address, page.name)]:
        placed.update({page.address: page for page in _move_round(ready, store)})
        written = _written(written, _entries(placed.values()))
        tried = {page.address for page in ready}
        waiting = [page for page in waiting if page.address not in tried]
    _skip(waiting)
    return written


def _entries(pages: Iterable[_SavedPage]) -> dict[str, CrawlMeta]:
    """The crawl metadata of *pages*, each entry under its record key."""
    return {page.record_key: page.entry for page in pages}


def _settled(page: _SavedPage) -> bool:
    """Whether *page* is at its name with no earlier index source to drop."""
    return page.entry.file == page.name and set(page.earlier_names) <= {page.name}


def _plan() -> tuple[_Record, list[_SavedPage]]:
    """The stored crawl metadata and the page that owns each saved file in it."""
    record = _Record(save.load_crawl_metadata())
    return record, _file_owners(record)


def _written(stored: dict[str, CrawlMeta], meta: dict[str, CrawlMeta]) -> dict[str, CrawlMeta]:
    """*meta*, written over *stored* when the two differ."""
    if meta != stored:
        save.save_crawl_metadata(meta)
    return meta


def move_saved_pages(store: Store) -> dict[str, CrawlMeta]:
    """The crawl metadata with one entry per page, each page at its name where it can be."""
    record, pages = _plan()
    if all(_settled(page) for page in pages):
        return _written(record.stored, _entries(pages))
    with library_held(cfg, CRAWL_MOVE_HELD, LOCK_TIMEOUT):
        record, pages = _plan()
        return _move_pages(record, pages, store)
