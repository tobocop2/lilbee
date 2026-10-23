"""The crawl page ledger: the ``_crawl_pages`` table and the order a page save keeps."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import lancedb
import pytest
from lancedb.table import LanceTable

from lilbee.core.config import CRAWL_PAGES_TABLE, cfg
from lilbee.crawler import save as save_mod
from lilbee.crawler.events import _flush
from lilbee.crawler.models import CrawlResult
from lilbee.crawler.save import (
    CrawlRecord,
    content_hash,
    finish_crawl,
    load_crawl_record,
    save_page,
    url_to_filename,
)
from lilbee.data.ingest import pipeline
from lilbee.data.store import ChunkWrite, CrawlPage, CrawlPages, Store
from lilbee.data.store import crawl_pages as crawl_pages_mod
from lilbee.data.store.crawl_pages import import_legacy_crawl_files, legacy_crawl_files
from tests._crawl_meta import CrawlMeta, claim_pages

URL = "https://example.com/page"
NAME = url_to_filename(URL)
SOURCE = f"_web/{NAME}"
# How long one save waits, in a race, for the other save to claim the page first.
RACE_WAIT_S = 0.5
# The lock timeout a test gives a save, and how long another save holds the lock past it.
SHORT_TIMEOUT_S = 0.05
HOLD_S = 0.5
# The commits between folds a test gives a crawl.
FOLD_EVERY = 3
LANCE_IO_ERROR = "lance error: LanceError(IO): Permission denied (os error 13)"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    yield Store(cfg)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _save(store: Store, markdown: str, url: str = URL) -> Path | None:
    return save_page(CrawlResult(url=url, markdown=markdown), load_crawl_record(store), MagicMock())


def _page_file() -> Path:
    return cfg.documents_dir / "_web" / NAME


class TestSaveOrder:
    def test_the_claim_commits_before_the_file_is_written(self, store, monkeypatch):
        seen: list[CrawlPage | None] = []
        write = save_mod._replace_atomically

        def spy(path: Path, text: str) -> None:
            seen.append(Store(cfg).crawl_pages.current(save_mod.canonical_address(URL)))
            write(path, text)

        monkeypatch.setattr(save_mod, "_replace_atomically", spy)
        _save(store, "# P")
        assert seen[0] is not None
        assert seen[0].content_hash == content_hash("# P")

    def test_a_claim_that_fails_leaves_no_file(self, store, monkeypatch):
        monkeypatch.setattr(LanceTable, "add", _lance_refuses)
        with pytest.raises(OSError, match="Permission denied"):
            _save(store, "# P")
        assert not _page_file().exists()


def _lance_refuses(*_args: object, **_kwargs: object) -> None:
    """What LanceDB raises for a commit its folder refuses."""
    raise RuntimeError(LANCE_IO_ERROR)


_LANCE_ADD = LanceTable.add


def _lance_commits_then_refuses(table: LanceTable, *args: object, **kwargs: object) -> None:
    """A commit that lands, then raises as if the writer stopped right after it."""
    _LANCE_ADD(table, *args, **kwargs)
    raise RuntimeError(LANCE_IO_ERROR)


class TestStoreFailure:
    def test_a_table_write_lancedb_refuses_skips_only_that_page(self, store, monkeypatch):
        """The crawl skips a page whose claim fails, as it skips one the disk refuses."""
        add = LanceTable.add
        refused: list[bool] = []

        def refuse_once(table: LanceTable, *args: object, **kwargs: object) -> object:
            if not refused:
                refused.append(True)
                _lance_refuses()
            return add(table, *args, **kwargs)

        monkeypatch.setattr(LanceTable, "add", refuse_once)
        with pytest.raises(OSError):
            _save(store, "# P")
        assert _save(store, "# P") == _page_file()
        assert store.crawl_pages.current(save_mod.canonical_address(URL)) is not None

    def test_a_crawl_goes_on_past_a_page_lancedb_refuses(self, store, monkeypatch):
        add = LanceTable.add
        calls: list[int] = []

        def refuse_first(table: LanceTable, *args: object, **kwargs: object) -> object:
            calls.append(1)
            if len(calls) == 1:
                _lance_refuses()
            return add(table, *args, **kwargs)

        monkeypatch.setattr(LanceTable, "add", refuse_first)
        record = load_crawl_record(store)
        other = "https://example.com/other"

        def flush(result: CrawlResult) -> None:
            save_page(result, record, MagicMock())

        asyncio.run(_flush(flush, CrawlResult(url=URL, markdown="# P")))
        asyncio.run(_flush(flush, CrawlResult(url=other, markdown="# O")))
        assert [row.address for row in store.crawl_pages.rows()] == [other]


class TestLockWait:
    def test_a_save_waits_for_the_crawl_lock_past_the_lock_timeout(self, store, monkeypatch):
        """A page is never skipped because another save held the crawl lock too long."""
        real = crawl_pages_mod.crawl_pages_lock

        def short_lock(lancedb_dir: Path, timeout: float | None) -> object:
            return real(lancedb_dir, None if timeout is None else SHORT_TIMEOUT_S)

        record = load_crawl_record(store)
        monkeypatch.setattr(crawl_pages_mod, "crawl_pages_lock", short_lock)
        held, done = threading.Event(), threading.Event()

        def hold() -> None:
            with Store(cfg).crawl_pages.held(timeout=None):
                held.set()
                done.wait(HOLD_S)

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait(HOLD_S)
        try:
            assert save_page(CrawlResult(url=URL, markdown="# P"), record, MagicMock())
        finally:
            done.set()
            holder.join()


class TestTwoCrawls:
    def test_the_last_claim_describes_the_last_file(self, store, monkeypatch):
        """One save waits in its file write for the other; the row and the file still agree."""
        claimed = threading.Event()
        write = save_mod._replace_atomically

        def slow_write(path: Path, text: str) -> None:
            if text == "# A":
                claimed.wait(RACE_WAIT_S)
            write(path, text)

        monkeypatch.setattr(save_mod, "_replace_atomically", slow_write)
        claim = store.crawl_pages.claim

        def noted_claim(page: CrawlPage) -> None:
            claim(page)
            if page.content_hash == content_hash("# B"):
                claimed.set()

        monkeypatch.setattr(store.crawl_pages, "claim", noted_claim)
        a = threading.Thread(target=_save, args=(store, "# A"))
        a.start()
        _save(store, "# B")
        a.join()
        row = Store(cfg).crawl_pages.current(save_mod.canonical_address(URL))
        assert row is not None
        assert row.content_hash == content_hash(_page_file().read_text(encoding="utf-8"))


class TestRebuild:
    def test_a_changed_page_saves_after_drop_all(self, store):
        _save(store, "# Old")
        store.drop_all()
        assert _save(store, "# New") == _page_file()
        assert _page_file().read_text(encoding="utf-8") == "# New"

    def test_drop_all_keeps_the_ledger(self, store):
        _save(store, "# Old")
        store.drop_all()
        row = store.crawl_pages.current(save_mod.canonical_address(URL))
        assert row is not None
        assert row.content_hash == content_hash("# Old")

    def test_indexing_the_page_leaves_its_row_unchanged(self, store):
        _save(store, "# Old")
        before = store.crawl_pages.current(save_mod.canonical_address(URL))
        store.write_chunks_batch([ChunkWrite(SOURCE, "f", [], needs_cleanup=False)])
        assert store.crawl_pages.current(save_mod.canonical_address(URL)) == before


class TestLegacyImport:
    def _legacy(self) -> None:
        earlier = {"file": "old/index.md", "content_hash": "e", "crawled_at": "t"}
        meta = {"https://EXAMPLE.com/page": earlier}
        (cfg.data_dir / "crawl_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_the_crawl_metadata_imports_once_and_goes(self, store):
        self._legacy()
        pages = Store(cfg).crawl_pages
        assert [p.file for p in pages.earlier("https://EXAMPLE.com/page")] == ["old/index.md"]
        assert not (cfg.data_dir / "crawl_meta.json").exists()

    def test_an_import_killed_before_the_delete_imports_onto_the_same_rows(self, store):
        self._legacy()
        Store(cfg).get_db()
        self._legacy()
        pages = Store(cfg).crawl_pages
        assert len(pages.rows()) == 1

    def test_a_file_that_cannot_be_deleted_adds_one_commit_over_two_opens(
        self, store, monkeypatch, caplog
    ):
        self._legacy()
        meta = cfg.data_dir / "crawl_meta.json"
        unlink = Path.unlink

        def refuse(path: Path, missing_ok: bool = False) -> None:
            if path == meta:
                raise PermissionError(13, "Permission denied", str(path))
            unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", refuse)
        Store(cfg).get_db()
        first = _raw_table().version
        Store(cfg).get_db()
        assert _raw_table().version == first
        assert meta.exists()
        assert "Could not delete" in caplog.text

    def test_an_import_leaves_the_current_page_current(self, store):
        claim_pages({URL: CrawlMeta("example.com/new.md", "n", "t")})
        self._legacy()
        pages = Store(cfg).crawl_pages
        assert pages.current(URL) == CrawlPage(URL, "example.com/new.md", "n", "t")
        assert [p.file for p in pages.earlier("https://EXAMPLE.com/page")] == ["old/index.md"]

    def test_an_unreadable_crawl_metadata_file_warns_and_stays(self, store, monkeypatch, caplog):
        self._legacy()
        meta = cfg.data_dir / "crawl_meta.json"
        read_text = Path.read_text

        def refuse(path: Path, encoding: str = "utf-8") -> str:
            if path == meta:
                raise PermissionError(13, "Permission denied", str(path))
            return read_text(path, encoding=encoding)

        monkeypatch.setattr(Path, "read_text", refuse)
        assert Store(cfg).crawl_pages.rows() == []
        assert "Permission denied" in caplog.text
        assert meta.exists()

    def test_the_crawl_metadata_goes_after_the_import_commits(self, store, monkeypatch):
        self._legacy()
        meta = cfg.data_dir / "crawl_meta.json"
        present_at_commit: list[bool] = []
        add = LanceTable.add

        def add_seeing_the_file(table: LanceTable, *args: object, **kwargs: object) -> object:
            if table.name == CRAWL_PAGES_TABLE:
                present_at_commit.append(meta.exists())
            return add(table, *args, **kwargs)

        monkeypatch.setattr(LanceTable, "add", add_seeing_the_file)
        Store(cfg).get_db()
        assert present_at_commit == [True]
        assert not meta.exists()

    @pytest.mark.parametrize("field", ["file", "content_hash", "crawled_at", "title"])
    @pytest.mark.parametrize("value", [None, 3, ["a.md"]], ids=["null", "number", "list"])
    def test_an_entry_with_a_field_not_a_string_is_skipped(
        self, store, caplog, field: str, value: object
    ):
        good = {"file": "a.com/good.md", "content_hash": "g", "crawled_at": "t"}
        meta = {"https://a.com/bad": {**good, field: value}, "https://a.com/good": good}
        (cfg.data_dir / "crawl_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        pages = Store(cfg).crawl_pages
        assert [row.address for row in pages.rows()] == ["https://a.com/good"]
        assert "https://a.com/bad" in caplog.text

    @pytest.mark.parametrize("entry", ["a.com/bad.md", None, 3], ids=["string", "null", "number"])
    def test_an_entry_not_an_object_is_skipped(self, store, caplog, entry: object):
        good = {"file": "a.com/good.md", "content_hash": "g", "crawled_at": "t"}
        meta = {"https://a.com/bad": entry, "https://a.com/good": good}
        (cfg.data_dir / "crawl_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        pages = Store(cfg).crawl_pages
        assert [row.address for row in pages.rows()] == ["https://a.com/good"]
        assert "https://a.com/bad" in caplog.text

    def test_a_page_record_file_is_neither_read_nor_deleted(self, store):
        line = {"url": URL, "file": NAME, "content_hash": "h", "crawled_at": "t"}
        record = cfg.data_dir / "crawl_pages.jsonl"
        record.write_text(json.dumps(line) + "\n", encoding="utf-8")
        self._legacy()
        pages = Store(cfg).crawl_pages
        assert [p.file for p in pages.earlier("https://EXAMPLE.com/page")] == ["old/index.md"]
        assert pages.current(save_mod.canonical_address(URL)) is None
        assert record.exists()

    def test_a_page_record_file_alone_needs_no_upgrade(self, store):
        record = cfg.data_dir / "crawl_pages.jsonl"
        record.write_text("{}\n", encoding="utf-8")
        assert legacy_crawl_files(cfg.data_dir) == []


class TestTwoStoresSaving:
    def test_a_store_sees_the_claim_another_store_made_since_its_last_save(self, store):
        other = Store(cfg)
        assert _save(store, "# v1") == _page_file()
        assert _save(other, "# v2") == _page_file()
        assert _save(store, "# v3") == _page_file()
        assert store.crawl_pages.current(save_mod.canonical_address(URL)).content_hash == (
            content_hash("# v3")
        )


class TestSyncAfterRetire:
    def test_a_flush_of_a_retired_file_adds_no_source(self, store):
        written = store.write_chunks_batch(
            [ChunkWrite(SOURCE, "f", [], needs_cleanup=False)], present=lambda name: False
        )
        assert written == 0
        assert store.missing_sources([SOURCE]) == {SOURCE}


class TestTwoStores:
    def test_a_claim_in_one_store_is_read_by_another_at_once(self, store):
        other = Store(cfg)
        other.crawl_pages.current(save_mod.canonical_address(URL))
        _save(store, "# P")
        row = other.crawl_pages.current(save_mod.canonical_address(URL))
        assert row is not None


class TestReset:
    def test_a_store_saves_again_after_a_reset_deleted_its_table(self, store):
        record = load_crawl_record(store)
        save_page(CrawlResult(url=URL, markdown="# Old"), record, MagicMock())
        shutil.rmtree(cfg.lancedb_dir)
        _page_file().unlink()
        other = "https://example.com/other"
        assert save_page(CrawlResult(url=other, markdown="# New"), record, MagicMock())
        assert [row.address for row in store.crawl_pages.rows()] == [other]


class TestTitles:
    def test_the_title_of_a_page_comes_from_its_row(self, store):
        save_page(
            CrawlResult(url=URL, markdown="# P", title="Page title"),
            load_crawl_record(store),
            MagicMock(),
        )
        assert store.crawl_pages.titles() == {SOURCE: "Page title"}


def _raw_table() -> LanceTable:
    """The crawl page table as another reader opens it."""
    table: LanceTable = lancedb.connect(cfg.lancedb_dir).open_table(CRAWL_PAGES_TABLE)
    return table


def _save_pages(store: Store, count: int, text: str) -> CrawlRecord:
    record = load_crawl_record(store)
    for i in range(count):
        save_page(CrawlResult(url=f"{URL}/{i}", markdown=f"# {text} {i}"), record, MagicMock())
    return record


class TestCompaction:
    def test_finish_crawl_folds_the_crawl_into_one_fragment(self, store):
        finish_crawl(_save_pages(store, 3, "a"))
        finish_crawl(_save_pages(store, 3, "b"))
        table = _raw_table()
        assert table.stats()["fragment_stats"]["num_fragments"] == 1
        assert table.count_rows() == 3

    def test_a_reader_at_an_older_version_still_reads_after_compaction(self, store):
        record = _save_pages(store, 3, "a")
        reader = _raw_table()
        _save_pages(store, 3, "b")
        finish_crawl(record)
        assert len(reader.search().limit(None).to_list()) == 3

    def test_a_long_crawl_folds_its_commits_as_it_goes(self, store, monkeypatch):
        monkeypatch.setattr(crawl_pages_mod, "_FOLD_EVERY", FOLD_EVERY)
        _save_pages(store, FOLD_EVERY * 3, "a")
        assert _raw_table().stats()["fragment_stats"]["num_fragments"] <= FOLD_EVERY + 1

    def test_a_fold_reads_the_rows_again_when_a_writer_outside_the_lock_added_one(self, store):
        pages = store.crawl_pages
        with pages.held():
            pages.claim(CrawlPage("https://a.com/1", "a.com/1.md", "h1", "t"))
            import_legacy_crawl_files(store.get_db(), _legacy_record("https://b.com/1"))
            pages.compact()
            assert pages.current("https://a.com/1") is not None
            assert [row.file for row in pages.earlier("https://b.com/1")] == ["b.com/1.md"]

    def test_a_fold_that_fails_only_warns(self, store, monkeypatch, caplog):
        monkeypatch.setattr(crawl_pages_mod, "_FOLD_EVERY", 1)
        monkeypatch.setattr(LanceTable, "optimize", _lance_refuses)
        assert _save(store, "# P") == _page_file()
        assert "Folding the crawl page table failed" in caplog.text


class TestRowOrder:
    def test_the_highest_seq_of_a_key_counts_whatever_the_scan_order(self):
        row = {"address": URL, "file": NAME, "crawled_at": "t", "replaces": "", "title": ""}
        rows = [
            {**row, "content_hash": "new", "current": True, "seq": 2},
            {**row, "content_hash": "old", "current": True, "seq": 1},
        ]
        view = crawl_pages_mod._View(None, rows)
        assert view.row((URL, NAME)) == CrawlPage(URL, NAME, "new", "t")
        assert view.replaced == {(URL, NAME)}
        assert view.next_seq == 3


class TestResave:
    def test_a_resaved_page_reads_its_last_text(self, store):
        _save_pages(store, 2, "a")
        _save_pages(store, 2, "b")
        rows = {row.address: row.content_hash for row in Store(cfg).crawl_pages.rows()}
        assert rows == {f"{URL}/{i}": content_hash(f"# b {i}") for i in range(2)}

    def test_a_resave_keeps_its_title_after_the_fold(self, store):
        record = load_crawl_record(store)
        save_page(CrawlResult(url=URL, markdown="# a", title="T"), record, MagicMock())
        save_page(CrawlResult(url=URL, markdown="# b"), record, MagicMock())
        finish_crawl(record)
        assert Store(cfg).crawl_pages.titles() == {SOURCE: "T"}


class TestReadsPerSave:
    """A save reads of the table only what its open handle does not already know."""

    @pytest.fixture
    def reads(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        counts = {"schema": 0, "count_rows": 0, "optimize": 0}
        schema, count_rows = LanceTable.schema, LanceTable.count_rows
        optimize = LanceTable.optimize

        def read_schema(table: LanceTable) -> object:
            counts["schema"] += 1
            return schema.fget(table)

        def count(table: LanceTable, *args: object) -> int:
            counts["count_rows"] += 1
            return count_rows(table, *args)

        def fold(table: LanceTable, **kwargs: object) -> object:
            counts["optimize"] += 1
            return optimize(table, **kwargs)

        monkeypatch.setattr(LanceTable, "schema", property(read_schema))
        monkeypatch.setattr(LanceTable, "count_rows", count)
        monkeypatch.setattr(LanceTable, "optimize", fold)
        return counts

    def test_a_save_reads_the_table_id_once_and_counts_no_rows(self, store, reads):
        record = load_crawl_record(store)
        save_page(CrawlResult(url=URL, markdown="# One"), record, MagicMock())
        reads.update(schema=0, count_rows=0)
        assert save_page(CrawlResult(url=f"{URL}/2", markdown="# Two"), record, MagicMock())
        assert (reads["schema"], reads["count_rows"]) == (1, 0)

    def test_a_crawl_that_committed_nothing_folds_nothing(self, store, reads):
        finish_crawl(_save_pages(store, 2, "a"))
        record = _save_pages(store, 2, "a")
        reads.update(optimize=0)
        finish_crawl(record)
        assert reads["optimize"] == 0

    def test_a_crawl_that_committed_folds(self, store, reads):
        finish_crawl(_save_pages(store, 2, "a"))
        record = _save_pages(store, 2, "b")
        reads.update(optimize=0)
        finish_crawl(record)
        assert reads["optimize"] == 1

    def test_titles_of_a_new_version_read_the_table_id_once(self, store, reads):
        pages = store.crawl_pages
        with pages.held():
            pages.claim(CrawlPage("https://b.com/1", "b.com/1.md", "b", "t", title="B"))
        reads.update(schema=0, count_rows=0)
        assert pages.titles() == {"_web/b.com/1.md": "B"}
        assert (reads["schema"], reads["count_rows"]) == (1, 0)

    def test_a_save_after_another_store_committed_reads_the_table_id_once(self, store, reads):
        record = load_crawl_record(store)
        save_page(CrawlResult(url=URL, markdown="# One"), record, MagicMock())
        other = Store(cfg).crawl_pages
        with other.held():
            other.claim(CrawlPage("https://b.com/1", "b.com/1.md", "b", "t"))
        reads.update(schema=0, count_rows=0)
        assert save_page(CrawlResult(url=f"{URL}/2", markdown="# Two"), record, MagicMock())
        assert (reads["schema"], reads["count_rows"]) == (1, 0)
        assert store.crawl_pages.current("https://b.com/1") is not None


class TestResetToTheSameVersion:
    def test_a_store_sees_the_rows_of_a_reset_that_reached_its_version(self, store):
        """Another process resets the library and saves as many pages; this store reads them."""
        pages = store.crawl_pages
        with pages.held():
            pages.claim(CrawlPage("https://old.com/1", "old.com/1.md", "h1", "t", title="Old"))
            pages.claim(CrawlPage("https://old.com/2", "old.com/2.md", "h2", "t", title="Old"))
        assert pages.titles()
        shutil.rmtree(cfg.lancedb_dir)
        other = Store(cfg).crawl_pages
        with other.held():
            other.claim(CrawlPage("https://new.com/1", "new.com/1.md", "n1", "t", title="New"))
            other.claim(CrawlPage("https://new.com/2", "new.com/2.md", "n2", "t", title="New"))
        with pages.held():
            assert sorted(row.address for row in pages.rows()) == [
                "https://new.com/1",
                "https://new.com/2",
            ]
        assert set(pages.titles()) == {"_web/new.com/1.md", "_web/new.com/2.md"}


class TestCommitOutsideTheLock:
    def test_a_hold_reads_again_when_a_commit_skipped_a_version(self, store):
        """An older lilbee's files imported during a crawl show in the crawl's next save."""
        pages = store.crawl_pages
        with pages.held():
            pages.claim(CrawlPage("https://a.com/1", "a.com/1.md", "h1", "t"))
            import_legacy_crawl_files(store.get_db(), _legacy_record("https://b.com/1"))
            pages.claim(CrawlPage("https://a.com/2", "a.com/2.md", "h2", "t"))
            assert [row.file for row in pages.earlier("https://b.com/1")] == ["b.com/1.md"]


class TestCommitBetweenHolds:
    def test_a_hold_reads_the_rows_an_import_added_since_the_last_hold(self, store):
        """An older lilbee's files imported between two saves show in the second save."""
        pages = store.crawl_pages
        with pages.held():
            pages.claim(CrawlPage("https://a.com/1", "a.com/1.md", "h1", "t"))
        import_legacy_crawl_files(store.get_db(), _legacy_record("https://b.com/1"))
        with pages.held():
            assert [row.file for row in pages.earlier("https://b.com/1")] == ["b.com/1.md"]

    def test_a_commit_cut_off_after_it_landed_is_seen_by_another_store(self, store, monkeypatch):
        """A writer stopped right after its commit still shows in the next hold."""
        pages = store.crawl_pages
        with pages.held():
            pages.claim(CrawlPage("https://a.com/1", "a.com/1.md", "h1", "t"))
        other = Store(cfg).crawl_pages
        with monkeypatch.context() as patch:
            patch.setattr(LanceTable, "add", _lance_commits_then_refuses)
            with other.held(), pytest.raises(crawl_pages_mod.CrawlPagesError):
                other.claim(CrawlPage("https://b.com/1", "b.com/1.md", "h2", "t"))
        with pages.held():
            assert pages.current("https://b.com/1") is not None


class TestHoldAfterAFailure:
    """A failed write leaves the rows in memory as the table's version says they are."""

    def _claim_failing(
        self, store: Store, monkeypatch: pytest.MonkeyPatch, fail: Callable[..., None]
    ) -> None:
        pages = store.crawl_pages
        with pages.held():
            pages.claim(CrawlPage("https://a.com/1", "a.com/1.md", "h1", "t"))
        with monkeypatch.context() as patch:
            patch.setattr(LanceTable, "add", fail)
            with pages.held(), pytest.raises(crawl_pages_mod.CrawlPagesError):
                pages.claim(CrawlPage("https://a.com/1", "a.com/2.md", "h2", "t"))

    def test_a_commit_that_landed_before_its_failure_is_read_by_the_next_hold(
        self, store, monkeypatch
    ):
        self._claim_failing(store, monkeypatch, _lance_commits_then_refuses)
        pages = store.crawl_pages
        with pages.held():
            assert pages.current("https://a.com/1").file == "a.com/2.md"
            assert [row.file for row in pages.earlier("https://a.com/1")] == ["a.com/1.md"]

    def test_a_commit_that_never_landed_leaves_the_rows_as_they_were(self, store, monkeypatch):
        self._claim_failing(store, monkeypatch, _lance_refuses)
        pages = store.crawl_pages
        with pages.held():
            pages.claim(CrawlPage("https://a.com/1", "a.com/3.md", "h3", "t"))
            assert pages.current("https://a.com/1").file == "a.com/3.md"
        assert [row.file for row in Store(cfg).crawl_pages.earlier("https://a.com/1")] == [
            "a.com/1.md"
        ]


def _legacy_record(address: str) -> Path:
    folder = cfg.data_dir / "legacy"
    folder.mkdir()
    meta = {address: {"file": "b.com/1.md", "content_hash": "b", "crawled_at": "t"}}
    (folder / "crawl_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


class TestTitlesPerSync:
    def test_a_sync_plan_reads_the_titles_once(self, store, monkeypatch):
        web = cfg.documents_dir / "_web"
        web.mkdir()
        files = {}
        for i in range(3):
            (web / f"{i}.md").write_text("# P", encoding="utf-8")
            files[f"_web/{i}.md"] = web / f"{i}.md"
        reads: list[int] = []
        titles = CrawlPages.titles

        def counted(pages: CrawlPages) -> dict[str, str]:
            reads.append(1)
            return titles(pages)

        monkeypatch.setattr(CrawlPages, "titles", counted)
        monkeypatch.setattr(
            "lilbee.data.ingest.pipeline.get_services", lambda: MagicMock(store=store)
        )
        records = {name: {"filename": name, "file_hash": "x", "title": ""} for name in files}
        pipeline._plan_file_changes(files, records, cancel=None)
        assert len(reads) == 1
