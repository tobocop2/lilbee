"""Every writer of ``_sources`` keys a row by one rule and finds the row an earlier write made."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pytest

from lilbee.core.config import CHUNKS_TABLE, CITATIONS_TABLE, SOURCES_TABLE, cfg
from lilbee.data.store import (
    REMOVED_REASON,
    ChunkType,
    ChunkWrite,
    HeldOut,
    HeldRecord,
    SourceStat,
    SourceStatBackfill,
    SourceType,
    Store,
)
from tests.test_store import _axis_record, _citation, _early_source, _md5

_OLD_REMOVED = "removed via remove (re-add the source or run retry-skipped to restore)"
_FAILED = HeldRecord(HeldOut.FAILED, "hf", "no text extracted (0 chunks)")


@pytest.fixture()
def config(tmp_path):
    """A config over a fresh library, with no registry."""
    return cfg.model_copy(
        update={"data_root": tmp_path, "lancedb_dir": tmp_path / "lancedb", "linked_roots": {}}
    )


def _records(source: str, chunk_type: str = "raw") -> list[dict]:
    return [
        {
            "source": source,
            "content_type": "text",
            "chunk_type": chunk_type,
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": f"{source} body",
            "chunk_index": 0,
            "vector": [0.1] * cfg.embedding_dim,
        }
    ]


def _rows(store: Store) -> list[dict]:
    """Every source row as stored."""
    return store.open_table(SOURCES_TABLE).search().limit(None).to_list()


def _shape(store: Store) -> list[tuple]:
    """What each source row holds apart from the time it was written."""
    kept = ("source_id", "path", "source_type", "held_out", "held_hash", "file_hash", "chunk_count")
    return sorted((tuple(row[name] for name in kept) for row in _rows(store)), key=str)


def _batch(store: Store, name: str, source_type: SourceType) -> None:
    store.write_chunks_batch(
        [ChunkWrite(name, "h", _records(name), needs_cleanup=False, source_type=source_type)]
    )


def _upsert(store: Store, name: str, source_type: SourceType) -> None:
    store.upsert_source(name, "h", 1, source_type)


def _replace(store: Store, name: str, source_type: SourceType) -> None:
    chunk_type = ChunkType.WIKI if source_type is SourceType.WIKI else ChunkType.RAW
    store.replace_chunks(name, chunk_type, _records(name, chunk_type))


def _stats(store: Store, name: str, source_type: SourceType) -> None:
    store.upsert_source(name, "h", 1, source_type)
    [record] = store.get_sources_where(lambda shown: shown == name)
    store.update_source_stats([SourceStatBackfill(record, SourceStat(1, 2, 3))])


def _hold(store: Store, name: str, source_type: SourceType) -> None:
    store.hold_failed({name: _FAILED})


def _cite(store: Store, name: str, source_type: SourceType) -> None:
    store.add_citations([_citation(name)])


def _page_text(store: Store, name: str, source_type: SourceType) -> None:
    store.add_page_texts([{"source": name, "page": 1, "text": "t", "content_type": "text"}])


_Writer = Callable[[Store, str, SourceType], None]
_ALL_TYPES = (SourceType.DOCUMENT, SourceType.IMPORTED, SourceType.WIKI)
_WRITERS: list[tuple[_Writer, tuple[SourceType, ...]]] = [
    (_batch, (SourceType.DOCUMENT, SourceType.IMPORTED)),
    (_upsert, _ALL_TYPES),
    (_replace, (SourceType.DOCUMENT, SourceType.WIKI)),
    (_stats, (SourceType.DOCUMENT, SourceType.IMPORTED)),
    (_hold, (SourceType.DOCUMENT,)),
    (_cite, (SourceType.DOCUMENT,)),
    (_page_text, (SourceType.DOCUMENT,)),
]
# What ``_sources.path`` holds for ``notes/a.md``, by type and by whether ``notes`` is registered.
_STORED = {
    (SourceType.DOCUMENT, True): "{root}/a.md",
    (SourceType.DOCUMENT, False): "notes/a.md",
    (SourceType.IMPORTED, True): "notes/a.md",
    (SourceType.IMPORTED, False): "notes/a.md",
    (SourceType.WIKI, True): "notes/a.md",
    (SourceType.WIKI, False): "notes/a.md",
}
_CASES = [
    pytest.param(
        writer, source_type, registered, id=f"{writer.__name__}-{source_type}-{registered}"
    )
    for writer, types in _WRITERS
    for source_type in types
    for registered in (True, False)
]


class TestOneKeyRule:
    def test_the_table_covers_every_writer(self):
        assert len(_CASES) == 2 * sum(len(types) for _, types in _WRITERS) == 24

    @pytest.mark.parametrize(("writer", "source_type", "registered"), _CASES)
    def test_every_writer_stores_the_path_the_rule_gives(
        self, config, tmp_path, writer, source_type, registered
    ):
        root = (tmp_path / "src" / "notes").as_posix()
        if registered:
            config.linked_roots = {"notes": root}
        store = Store(config)

        writer(store, "notes/a.md", source_type)

        assert [row["path"] for row in _rows(store)] == [
            _STORED[source_type, registered].format(root=root)
        ]


class TestTheWriterStatesTheType:
    def test_a_batched_document_under_a_folder_labelled_wiki_has_one_row(self, config, tmp_path):
        root = tmp_path / "home" / "wiki"
        config.linked_roots = {"wiki": str(root)}
        store = Store(config)

        _batch(store, "wiki/notes.md", SourceType.DOCUMENT)

        [row] = _rows(store)
        assert (row["path"], row["source_type"]) == ((root / "notes.md").as_posix(), "document")
        assert store.source_ids(["wiki/notes.md"]) == {"wiki/notes.md": row["source_id"]}

    def test_a_wiki_page_and_a_document_of_one_name_have_two_ids(self, config, tmp_path):
        root = tmp_path / "home" / "wiki"
        config.linked_roots = {"wiki": str(root)}
        store = Store(config)

        _batch(store, "wiki/summaries/a.md", SourceType.DOCUMENT)
        _replace(store, "wiki/summaries/a.md", SourceType.WIKI)

        rows = {row["source_type"]: row for row in _rows(store)}
        assert sorted(rows) == ["document", "wiki"]
        assert rows["document"]["path"] == (root / "summaries" / "a.md").as_posix()
        assert rows["wiki"]["path"] == "wiki/summaries/a.md"
        assert rows["document"]["source_id"] != rows["wiki"]["source_id"]


class TestAWriteFindsTheLabelKeyedRow:
    def test_a_hold_out_onto_a_label_keyed_row_fills_it(self, config, tmp_path):
        store = Store(config)
        _batch(store, "notes/a.md", SourceType.DOCUMENT)
        [before] = _rows(store)
        root = tmp_path / "src" / "notes"
        config.linked_roots = {"notes": str(root)}

        store.hold_failed({"notes/a.md": _FAILED})

        [row] = _rows(store)
        assert row["source_id"] == before["source_id"]
        assert row["path"] == (root / "a.md").as_posix()
        assert (row["held_out"], row["held_hash"], row["file_hash"]) == ("failed", "hf", "h")
        assert store.held_records() == {"notes/a.md": _FAILED}


def _write_legacy(root: Path, markers: dict, reasons: dict) -> None:
    """The record files an older lilbee kept beside the index."""
    (root / "skipped_sources.json").write_text(json.dumps(markers), encoding="utf-8")
    (root / "skip_reasons.json").write_text(json.dumps(reasons), encoding="utf-8")
    (root / "skipped_sources.json.lock").write_text("", encoding="utf-8")


def _main_library(config, tmp_path) -> Path:
    """A library main wrote: keyed by label path, a removed cited source, its removal record.

    Returns the root of ``notes``, registered on *config*.
    """
    import lancedb

    page = "wiki/summaries/a.md"
    chunks = [
        {**_axis_record(name, axis, config.embedding_dim), "chunk_type": kind, "title": None}
        for axis, (name, kind) in enumerate([("notes/a.md", "raw"), (page, "wiki")])
    ]
    sources = [{**_early_source("notes/a.md", "ha"), "title": None}]
    tables = {
        SOURCES_TABLE: pa.Table.from_pylist(sources),
        CHUNKS_TABLE: pa.Table.from_pylist(chunks),
        CITATIONS_TABLE: pa.Table.from_pylist(
            [{**_citation("notes/cited.md"), "wiki_source": page}]
        ),
    }
    db = lancedb.connect(str(config.lancedb_dir))
    for name, rows in tables.items():
        db.create_table(name, rows)
    _write_legacy(tmp_path, {"notes/cited.md": "hc"}, {"notes/cited.md": _OLD_REMOVED})
    root = tmp_path / "src" / "notes"
    config.linked_roots = {"notes": str(root)}
    return root


class TestTheImportOfTheOlderRecords:
    def test_a_removed_cited_source_keeps_one_row_and_its_citation(self, config, tmp_path):
        root = _main_library(config, tmp_path)

        store = Store(config)

        cited = [row for row in _rows(store) if row["path"] == (root / "cited.md").as_posix()]
        assert [row["source_id"] for row in cited] == [_md5("notes/cited.md")]
        assert len(_rows(store)) == 3
        assert [
            row["citation_key"] for row in store.get_citations_for_source("notes/cited.md")
        ] == ["src1"]
        assert store.held_records() == {
            "notes/cited.md": HeldRecord(HeldOut.REMOVED, "hc", REMOVED_REASON)
        }

    def test_an_open_stopped_before_the_files_go_is_finished_to_the_same_rows(
        self, config, tmp_path, monkeypatch
    ):
        _main_library(config, tmp_path)
        monkeypatch.setattr("lilbee.data.store.core.remove_legacy_files", lambda root: None)
        first = Store(config)
        rows = sorted((row["source_id"], row["path"]) for row in _rows(first))
        shape = _shape(first)
        first.close()
        monkeypatch.undo()

        again = Store(config)

        assert sorted((row["source_id"], row["path"]) for row in _rows(again)) == rows
        assert _shape(again) == shape
        assert not (tmp_path / "skipped_sources.json").exists()

    def test_the_import_before_and_after_the_fill_writes_equal_rows(self, config, tmp_path):
        _main_library(config, tmp_path)
        before = _shape(Store(config))
        later = tmp_path / "later"
        later.mkdir()
        late = config.model_copy(update={"data_root": later, "lancedb_dir": later / "lancedb"})
        _main_library(late, later)
        legacy = [later / name for name in ("skipped_sources.json", "skip_reasons.json")]
        saved = {path: path.read_text(encoding="utf-8") for path in legacy}
        for path in legacy:
            path.unlink()
        Store(late).close()
        for path, text in saved.items():
            path.write_text(text, encoding="utf-8")

        after = _shape(Store(late))

        root, late_root = (tmp_path / "src").as_posix(), (later / "src").as_posix()
        assert sorted(str(row).replace(late_root, root) for row in after) == sorted(
            str(row) for row in before
        )

    def test_an_indexed_document_with_a_removal_record_stays_listed(self, config, tmp_path):
        root = tmp_path / "src" / "notes"
        config.linked_roots = {"notes": str(root)}
        Store(config).write_chunks_batch(
            [ChunkWrite("notes/a.md", "h", _records("notes/a.md"), needs_cleanup=False)]
        )
        _write_legacy(tmp_path, {"notes/a.md": "h"}, {"notes/a.md": _OLD_REMOVED})

        store = Store(config)

        assert [row["filename"] for row in store.get_sources()] == ["notes/a.md"]
        assert store.held_records() == {}
        assert len(_rows(store)) == 1


class TestARootThatLeavesOntoAHeldKey:
    def test_its_row_folds_into_the_held_row_and_the_citations_move(self, config, tmp_path):
        root = tmp_path / "src" / "notes"
        config.linked_roots = {"notes": str(root)}
        store = Store(config)
        _batch(store, "notes/x.md", SourceType.DOCUMENT)
        store.add_citations([_citation("notes/x.md", key="live")])
        unregistered = Store(config.model_copy(update={"linked_roots": {}}))
        unregistered.remove_documents(
            [], {"notes/x.md": HeldRecord(HeldOut.REMOVED, "hx", REMOVED_REASON)}
        )
        unregistered.add_citations([_citation("notes/x.md", key="held")])
        assert len(_rows(store)) == 2

        store.name_by_label(["notes"])
        config.linked_roots = {}

        [row] = _rows(store)
        assert row["path"] == "notes/x.md"
        keys = sorted(
            cited["citation_key"] for cited in store.get_citations_for_source("notes/x.md")
        )
        assert keys == ["held", "live"]
