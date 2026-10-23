"""The crawl page table: which file each crawled page owns, and the earlier copies to retire."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from lilbee.core.config import CRAWL_PAGES_TABLE
from lilbee.core.web_names import WEB_PREFIX
from lilbee.runtime.lock import LOCK_TIMEOUT, crawl_pages_lock

from .lance_helpers import ensure_table, escape_sql_string, open_committed

if TYPE_CHECKING:
    from lancedb.db import LanceDBConnection
    from lancedb.table import LanceTable

log = logging.getLogger(__name__)

# The file an older lilbee kept its crawled pages in, under the data directory.
CRAWL_META_FILE = "crawl_meta.json"
# The column that orders the rows of one [address, file] key: the highest one counts.
_SEQ = "seq"
# The schema metadata key of an id each new table gets, so a table made again reads as new.
_TABLE_ID = b"lilbee.table_id"
# The commits a store makes between two folds of the table, and the keys one delete names.
_FOLD_EVERY = 100
_KEYS_PER_DELETE = 200
# How long a version a fold replaces stays readable: far longer than one read of the
# table, which takes milliseconds. Kept short, as each kept version slows the check for
# the latest one (about 1 ms per 1,000 versions on a laptop).
_KEEP_VERSIONS = timedelta(seconds=10)


class CrawlPagesError(OSError):
    """The crawl page table refused a read or a write; the crawl skips that one page."""


@dataclass(frozen=True)
class CrawlPage:
    """One crawled page's file: the current one the page owns, or an earlier copy of it."""

    address: str
    file: str
    content_hash: str
    crawled_at: str
    # The hash of the text the file held when the page claimed it; empty for a new file.
    replaces: str = ""
    # The page's own title; empty when the page has none or an older lilbee saved it.
    title: str = ""
    # False for an earlier copy: a file an earlier claim or name rule gave the page.
    current: bool = True


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("address", pa.utf8()),
            pa.field("file", pa.utf8()),
            pa.field("content_hash", pa.utf8()),
            pa.field("crawled_at", pa.utf8()),
            pa.field("replaces", pa.utf8()),
            pa.field("title", pa.utf8()),
            pa.field("current", pa.bool_()),
            pa.field(_SEQ, pa.int64()),
        ],
        metadata={_TABLE_ID: uuid.uuid4().hex.encode()},
    )


_COLUMNS = [f.name for f in fields(CrawlPage)]
_Key = tuple[str, str]
# The id of a table and the version of it a view holds.
_Stamp = tuple[bytes, int]


def _stamp(table: LanceTable) -> _Stamp:
    """Which table *table* is, and at which version: a reset that reaches the version differs."""
    return (table.schema.metadata or {}).get(_TABLE_ID, b""), table.version


def _where(**columns: str) -> str:
    return " AND ".join(f"{name} = '{escape_sql_string(value)}'" for name, value in columns.items())


def _index(index: dict[str, dict[_Key, CrawlPage]], value: str, page: CrawlPage) -> None:
    index.setdefault(value, {})[(page.address, page.file)] = page


def _unindex(index: dict[str, dict[_Key, CrawlPage]], value: str, key: _Key) -> None:
    entries = index.get(value, {})
    entries.pop(key, None)
    if not entries:
        index.pop(value, None)


class _View:
    """The rows of the table at one version, found by address, by owned file or by text.

    A save adds a row and never rewrites one, so a key can have several rows: the one
    with the highest ``seq`` counts, and the others wait for :meth:`CrawlPages.compact`.
    """

    def __init__(self, stamp: _Stamp | None, rows: list[dict[str, Any]]) -> None:
        self.stamp = stamp
        # The next seq to write: above every seq the table holds.
        self.next_seq = 0
        self._rows: dict[_Key, tuple[int, CrawlPage]] = {}
        # Keys with rows a later row of the key replaced.
        self.replaced: set[_Key] = set()
        self._by_address: dict[str, dict[_Key, CrawlPage]] = {}
        # Current rows by their file, and earlier copies by the hash of their text.
        self._owners: dict[str, dict[_Key, CrawlPage]] = {}
        self._copies: dict[str, dict[_Key, CrawlPage]] = {}
        for row in rows:
            self.put(CrawlPage(**{name: row[name] for name in _COLUMNS}), row[_SEQ])

    def put(self, page: CrawlPage, seq: int) -> None:
        """Note *page*, written at *seq*; a lower seq of its key loses to the row held."""
        key = (page.address, page.file)
        self.next_seq = max(self.next_seq, seq + 1)
        held = self._rows.get(key)
        if held is not None:
            self.replaced.add(key)
            if held[0] > seq:
                return
        self.drop(page)
        self._rows[key] = (seq, page)
        _index(self._by_address, page.address, page)
        if page.current:
            _index(self._owners, page.file, page)
        else:
            _index(self._copies, page.content_hash, page)

    def drop(self, page: CrawlPage) -> None:
        key = (page.address, page.file)
        held = self._rows.pop(key, None)
        if held is None:
            return
        old = held[1]
        _unindex(self._by_address, old.address, key)
        if old.current:
            _unindex(self._owners, old.file, key)
        else:
            _unindex(self._copies, old.content_hash, key)

    def seq(self, key: _Key) -> int:
        return self._rows[key][0]

    def row(self, key: _Key) -> CrawlPage | None:
        held = self._rows.get(key)
        return None if held is None else held[1]

    def of(self, address: str) -> list[CrawlPage]:
        return list(self._by_address.get(address, {}).values())

    def owners(self, file: str) -> list[CrawlPage]:
        return list(self._owners.get(file, {}).values())

    def copies(self, content_hash: str) -> list[CrawlPage]:
        return list(self._copies.get(content_hash, {}).values())

    def all(self) -> list[CrawlPage]:
        return [page for _seq, page in self._rows.values()]


def _read(table: LanceTable | None, stamp: _Stamp | None = None) -> _View:
    """Every row of *table* at the version it has checked out, under *stamp* when given."""
    if table is None:
        return _View(None, [])
    return _View(_stamp(table) if stamp is None else stamp, table.to_arrow().to_pylist())


def _add(table: LanceTable, view: _View, pages: list[CrawlPage]) -> int:
    """Add *pages* in one commit above every row *view* holds; the version it made."""
    seq = view.next_seq
    rows = [{**asdict(page), _SEQ: seq + i} for i, page in enumerate(pages)]
    version: int = table.add(rows).version
    for i, page in enumerate(pages):
        view.put(page, seq + i)
    return version


class CrawlPages:
    """The crawl page table of one library. A write runs inside :meth:`held`.

    The crawl lock comes before the store's write lock and never after it, so a holder
    may remove index sources inside :meth:`held`.

    The crawl lock orders the holds, but not every commit: the import of an older
    release's files commits under the upgrade lock. So a hold checks the table's version
    and reads the rows again when another commit, of any process, came since, or when a
    commit of its own skipped a version. A hold keeps the rows in memory, with the table
    open, and its own writes update both.

    A LanceDB failure raises :class:`CrawlPagesError`, an ``OSError``, so a crawl skips
    the one page, as it skips a page the disk refuses.
    """

    def __init__(self, open_db: Callable[[], LanceDBConnection], lancedb_dir: Path) -> None:
        self._open_db = open_db
        self._lancedb_dir = lancedb_dir
        self._titles: tuple[_Stamp, dict[str, str]] | None = None
        # The table and its rows as the last hold left them; used only inside a hold.
        self._table: LanceTable | None = None
        self._view = _View(None, [])
        # The thread that holds the crawl lock through this object, or None.
        self._holder: int | None = None
        # The commits this store made since it last folded the table.
        self._unfolded = 0

    @contextmanager
    def held(self, timeout: float | None = LOCK_TIMEOUT) -> Iterator[None]:
        """Hold the crawl lock for the block; the library is open, and upgraded, first."""
        self._open_db()
        with crawl_pages_lock(self._lancedb_dir, timeout):
            with self._lancedb_failure():
                self._catch_up()
            self._holder = threading.get_ident()
            try:
                yield
            finally:
                self._holder = None

    @contextmanager
    def _lancedb_failure(self) -> Iterator[None]:
        """Raise a LanceDB failure as :class:`CrawlPagesError`.

        The rows in memory change only after a commit lands, so the next hold's version
        check reads them again when a failed write committed part of its work.
        """
        try:
            yield
        except RuntimeError as exc:
            raise CrawlPagesError(f"The crawl page table failed: {exc}") from exc

    def _catch_up(self) -> None:
        """Read the table again when a commit since the last hold changed it."""
        if self._table is not None:
            try:
                self._table.checkout_latest()
            except (RuntimeError, OSError):
                # A reset deleted the library under this store; the open table is gone.
                self._table = None
        if self._table is None:
            self._view = _View(None, [])
            self._table = open_committed(self._open_db(), CRAWL_PAGES_TABLE)
        if self._table is not None:
            stamp = _stamp(self._table)
            if stamp != self._view.stamp:
                self._view = _read(self._table, stamp)

    def _rows_now(self) -> _View:
        """The rows of the hold, or outside one the rows of the last commit."""
        if self._holder == threading.get_ident():
            return self._view
        return _read(open_committed(self._open_db(), CRAWL_PAGES_TABLE))

    def _committed(self, table: LanceTable, version: int) -> None:
        """Note that the hold's own write made *version* of *table*.

        A version the hold did not expect means a writer without the crawl lock, an
        older lilbee's files imported at store open, committed in between: read again.
        """
        expected = self._view.stamp is not None and version == self._view.stamp[1] + 1
        self._table = table
        self._unfolded += 1
        if expected and self._view.stamp is not None:
            self._view.stamp = (self._view.stamp[0], version)
        else:
            self._view = _read(table)

    def rows(self) -> list[CrawlPage]:
        """Every row of the table."""
        return self._rows_now().all()

    def current(self, address: str) -> CrawlPage | None:
        """The file the page at *address* owns, as the last commit holds it."""
        return next((row for row in self._rows_now().of(address) if row.current), None)

    def earlier(self, address: str) -> list[CrawlPage]:
        """The earlier copies of the page at *address*."""
        return [row for row in self._rows_now().of(address) if not row.current]

    def owners(self, file: str) -> list[CrawlPage]:
        """The pages whose current file is *file*."""
        return self._rows_now().owners(file)

    def copies_of(self, content_hash: str) -> list[CrawlPage]:
        """The earlier copies that hold the text *content_hash* names."""
        return self._rows_now().copies(content_hash)

    def claim(self, page: CrawlPage) -> None:
        """Write *page*; when it is current, the page's current file before it becomes earlier.

        One commit adds both rows. Caller holds :meth:`held`.
        """
        before = [
            replace(row, current=False)
            for row in self._view.of(page.address)
            if page.current and row.current and row.file != page.file
        ]
        with self._lancedb_failure():
            # The truth value of a table counts its rows, so the test is for None.
            table = self._table
            if table is None:
                table = ensure_table(self._open_db(), CRAWL_PAGES_TABLE, _schema())
            self._committed(table, _add(table, self._view, [page, *before]))

    def forget(self, page: CrawlPage) -> None:
        """Delete the rows of *page*'s key. Caller holds :meth:`held`."""
        if self._table is not None:
            with self._lancedb_failure():
                deleted = self._table.delete(_where(address=page.address, file=page.file))
                self._committed(self._table, deleted.version)
        self._view.drop(page)

    def rekey(
        self, canonical: Callable[[str], str], holds_text: Callable[[CrawlPage], bool]
    ) -> None:
        """Key each earlier copy by the address *canonical* gives. Caller holds :meth:`held`.

        An older lilbee keyed earlier copies by the address as the crawl was given it, so
        two spellings of one page can name one file. Of the copies that meet on one row,
        one whose file *holds_text* says holds its text is kept. A current row is never
        made an earlier copy.
        """
        table = self._table
        moved = [r for r in self._view.all() if not r.current and canonical(r.address) != r.address]
        if table is None or not moved:
            return
        kept: dict[_Key, CrawlPage] = {}
        for row in moved:
            copy = replace(row, address=canonical(row.address))
            key = (copy.address, copy.file)
            held = kept.get(key) or self._view.row(key)
            if held is None or (not held.current and not holds_text(held) and holds_text(copy)):
                kept[key] = copy
        with self._lancedb_failure():
            if kept:
                _add(table, self._view, list(kept.values()))
            table.delete(" OR ".join(f"({_where(address=r.address, file=r.file)})" for r in moved))
            self._view = _read(table)

    def compact(self) -> None:
        """Delete the rows later rows replaced, then fold the table's commits into one version.

        A version the fold replaces stays readable for ``_KEEP_VERSIONS``, so a reader of
        another process at that version still reads it. A store that committed nothing
        since its last fold, and holds no replaced rows, has nothing to fold. Caller holds
        :meth:`held`.
        """
        table = self._table
        if table is None or not (self._unfolded or self._view.replaced):
            return
        with self._lancedb_failure():
            keys = sorted(self._view.replaced)
            for start in range(0, len(keys), _KEYS_PER_DELETE):
                table.delete(
                    " OR ".join(
                        f"({_where(address=address, file=file)} AND {_SEQ} < {self._view.seq(key)})"
                        for key in keys[start : start + _KEYS_PER_DELETE]
                        for address, file in [key]
                        if self._view.row(key) is not None
                    )
                    or "false"
                )
            table.optimize(cleanup_older_than=_KEEP_VERSIONS)
            self._view.replaced.clear()
            # One row per key is left; a count that differs means a writer without the
            # crawl lock committed in between, so the rows are read again.
            if table.count_rows() == len(self._view.all()):
                self._view.stamp = _stamp(table)
            else:
                self._view = _read(table)
        self._unfolded = 0

    def compact_when_due(self) -> None:
        """Fold the table after every ``_FOLD_EVERY`` commits; a failure only warns.

        A long crawl folds as it goes, so the table never holds more than about that
        many fragments, and each new version stays small. Caller holds :meth:`held`.
        """
        if self._unfolded < _FOLD_EVERY:
            return
        try:
            self.compact()
        except CrawlPagesError:
            log.warning("Folding the crawl page table failed", exc_info=True)

    def titles(self) -> dict[str, str]:
        """The title of each crawled page that has one, by source name."""
        table = open_committed(self._open_db(), CRAWL_PAGES_TABLE)
        if table is None:
            return {}
        stamp = _stamp(table)
        if self._titles is None or self._titles[0] != stamp:
            rows = _read(table, stamp).all()
            self._titles = (
                stamp,
                {f"{WEB_PREFIX}{r.file}": r.title for r in rows if r.current and r.title},
            )
        return self._titles[1]


def legacy_crawl_files(data_dir: Path) -> list[Path]:
    """The crawl metadata file of an older lilbee, when it is still under *data_dir*."""
    path = data_dir / CRAWL_META_FILE
    return [path] if path.exists() else []


def _entry(address: str, data: Any) -> CrawlPage | None:
    """The earlier copy one legacy entry describes, or None when it is malformed.

    Every field an older lilbee wrote is a string; any other value is malformed.
    """
    strings = isinstance(data, dict) and all(isinstance(value, str) for value in data.values())
    try:
        if strings:
            return CrawlPage(address=address, current=False, **data)
    except TypeError:
        pass
    log.warning("Skipping a malformed crawl record entry of %s", address)
    return None


def _meta_pages(path: Path) -> list[CrawlPage] | None:
    """The earlier copies the crawl metadata at *path* names, keyed as it wrote them.

    None when the disk refuses the read, so the file stays for a later open.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        log.warning("Could not read crawl metadata %s: %s", path, exc)
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("Discarding unreadable crawl metadata %s: %s", path, exc)
        return []
    if not isinstance(raw, dict):
        return []
    entries = (_entry(str(key), data) for key, data in raw.items())
    return [page for page in entries if page is not None]


def import_legacy_crawl_files(db: LanceDBConnection, data_dir: Path) -> None:
    """Move the crawl metadata under *data_dir* into the table as earlier copies, then delete it.

    Store write lock held. A file the disk refuses to read stays, with a warning. One
    commit adds every entry the table has no row for; the file goes last, so an open
    killed before then, or a file the disk will not delete, commits nothing the next time.
    """
    path = data_dir / CRAWL_META_FILE
    entries = _meta_pages(path)
    if entries is None:
        return
    pages = {(page.address, page.file): page for page in entries}
    table = ensure_table(db, CRAWL_PAGES_TABLE, _schema())
    view = _read(table)
    new = [page for key, page in pages.items() if view.row(key) is None]
    if new:
        _add(table, view, new)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("Could not delete %s: %s", path, exc)
    log.info("Moved the crawl record of %s into the index", data_dir)
