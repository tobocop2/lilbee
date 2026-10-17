"""The held-out records of ``_sources``, and the import of the older record files."""

from __future__ import annotations

import json

import pytest

from lilbee.core.config import SOURCES_TABLE, cfg
from lilbee.data.store import (
    REMOVED_REASON,
    ChunkWrite,
    HeldOut,
    HeldRecord,
    SourceMove,
    SourceType,
    Store,
    held_import,
)
from lilbee.data.store.core import _held_unmigrated
from lilbee.data.store.lance_helpers import ensure_table
from lilbee.data.store.schema import _sources_schema

_REASON = "no text extracted (0 chunks)"


@pytest.fixture()
def held_config(tmp_path):
    """A Config whose data root and index are both under *tmp_path*."""
    return cfg.model_copy(update={"data_root": tmp_path, "lancedb_dir": tmp_path / "lancedb"})


@pytest.fixture()
def store(held_config):
    return Store(held_config)


def _records(source: str, n: int = 2) -> list[dict]:
    return [
        {
            "source": source,
            "content_type": "text",
            "chunk_type": "raw",
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": f"{source} chunk {i}",
            "chunk_index": i,
            "vector": [0.1] * cfg.embedding_dim,
        }
        for i in range(n)
    ]


def _index(store: Store, *names: str, file_hash: str = "h") -> None:
    store.write_chunks_batch(
        [ChunkWrite(name, file_hash, _records(name), needs_cleanup=False) for name in names]
    )


def _failed(digest: str, reason: str = _REASON) -> HeldRecord:
    return HeldRecord(HeldOut.FAILED, digest, reason)


def _removed(digest: str) -> HeldRecord:
    return HeldRecord(HeldOut.REMOVED, digest, REMOVED_REASON)


def _listed(store: Store) -> list[str]:
    return sorted(source["filename"] for source in store.get_sources())


def _cite(store: Store, name: str) -> None:
    store.add_citations(
        [
            {
                "wiki_source": "wiki/entities/x.md",
                "wiki_chunk_index": 0,
                "citation_key": "src1",
                "claim_type": "fact",
                "source_filename": name,
                "source_hash": "h",
                "page_start": 0,
                "page_end": 0,
                "line_start": 1,
                "line_end": 2,
                "excerpt": "e",
                "created_at": "",
            }
        ]
    )


class TestHoldFailed:
    def test_a_failure_with_no_document_is_held_and_not_listed(self, store):
        store.hold_failed({"scan.pdf": _failed("h1")})
        assert store.held_records() == {"scan.pdf": _failed("h1")}
        assert _listed(store) == []

    def test_a_failed_edit_keeps_the_indexed_document_listed(self, store):
        _index(store, "a.md", file_hash="old")
        store.hold_failed({"a.md": _failed("new")})
        assert store.held_records() == {"a.md": _failed("new")}
        assert _listed(store) == ["a.md"]
        assert [source["file_hash"] for source in store.get_sources()] == ["old"]

    def test_a_failure_of_a_removed_file_holds_no_document(self, store):
        _index(store, "a.md")
        store.remove_documents(["a.md"], {"a.md": _removed("h")})
        store.hold_failed({"a.md": _failed("edited")})
        assert store.held_records() == {"a.md": _failed("edited")}
        assert _listed(store) == []

    def test_holding_nothing_writes_nothing(self, store):
        store.hold_failed({})
        assert store.open_table(SOURCES_TABLE) is None

    def test_an_ingest_of_a_held_path_clears_its_record_and_keeps_its_id(self, store):
        store.hold_failed({"a.md": _failed("h1")})
        before = store.source_ids(["a.md"], SourceType.DOCUMENT)
        _index(store, "a.md", file_hash="h2")
        assert store.held_records() == {}
        assert store.source_ids(["a.md"], SourceType.DOCUMENT) == before
        assert _listed(store) == ["a.md"]


class TestRemoveWithRecords:
    def test_a_removal_drops_the_chunks_and_holds_the_path(self, store):
        _index(store, "a.md", "b.md")
        result = store.remove_documents(["a.md"], {"a.md": _removed("h")})
        assert result.removed == ["a.md"]
        assert store.held_records() == {"a.md": _removed("h")}
        assert _listed(store) == ["b.md"]
        assert store.get_chunks_by_source("a.md") == []

    def test_a_removal_without_a_record_leaves_no_row(self, store):
        _index(store, "a.md")
        store.remove_documents(["a.md"])
        assert store.held_records() == {}
        assert store.source_ids(["a.md"], SourceType.DOCUMENT) == {}

    def test_a_held_failure_named_by_a_removal_becomes_a_removal(self, store):
        store.hold_failed({"scan.pdf": _failed("h1")})
        result = store.remove_documents(["scan.pdf"], {"scan.pdf": _removed("h1")})
        assert result.not_found == ["scan.pdf"]
        assert store.held_records() == {"scan.pdf": _removed("h1")}


class TestRelease:
    def test_a_released_failure_with_no_document_leaves_no_row(self, store):
        store.hold_failed({"scan.pdf": _failed("h1")})
        store.release_held(["scan.pdf"])
        assert store.held_records() == {}
        assert store.source_ids(["scan.pdf"], SourceType.DOCUMENT) == {}

    def test_a_released_failed_edit_keeps_its_document(self, store):
        _index(store, "a.md", file_hash="old")
        store.hold_failed({"a.md": _failed("new")})
        store.release_held(["a.md"])
        assert store.held_records() == {}
        assert _listed(store) == ["a.md"]

    def test_a_released_removal_a_citation_names_keeps_its_path_only(self, store):
        _index(store, "a.md")
        _cite(store, "a.md")
        store.remove_documents(["a.md"], {"a.md": _removed("h")})
        store.release_held(["a.md"])
        assert store.held_records() == {}
        assert _listed(store) == []
        assert set(store.source_ids(["a.md"], SourceType.DOCUMENT)) == {"a.md"}

    def test_releasing_nothing_takes_no_lock(self, store):
        store.release_held([])
        assert store.open_table(SOURCES_TABLE) is None

    def test_release_failed_keeps_removals_and_names_what_it_released(self, store):
        store.hold_failed({"b.pdf": _failed("h2"), "a.pdf": _failed("h1")})
        store.remove_documents([], {"gone.md": _removed("h3")})
        assert store.release_failed() == ["a.pdf", "b.pdf"]
        assert store.held_records() == {"gone.md": _removed("h3")}

    def test_release_failed_on_an_empty_library(self, store):
        assert store.release_failed() == []


class TestMovesCarryFailures:
    def test_a_moved_failure_keeps_its_record_at_the_new_path(self, store):
        store.hold_failed({"a/empty.md": _failed("e")})
        step = store.relocate_sources([SourceMove(("a/empty.md",), "c/empty.md", None)])
        assert step.taken == {"c/empty.md": "a/empty.md"}
        assert store.held_records() == {"c/empty.md": _failed("e")}

    def test_a_removal_does_not_move(self, store):
        store.remove_documents([], {"a/gone.md": _removed("g")})
        step = store.relocate_sources([SourceMove(("a/gone.md",), "c/gone.md", None)])
        assert step.taken == {}
        assert store.held_records() == {"a/gone.md": _removed("g")}

    def test_sources_by_hash_offers_a_failure_at_the_hash_it_failed_at(self, store):
        _index(store, "a.md", file_hash="old")
        store.hold_failed({"a.md": _failed("new"), "b.pdf": _failed("hb")})
        version = store.sources_version()
        assert version is not None
        found = store.sources_by_hash(["old", "new", "hb"], version=version)
        assert found == {"old": ["a.md"], "new": ["a.md"], "hb": ["b.pdf"]}


def _write_legacy(root, markers, reasons=None, kinds=None):
    (root / "skipped_sources.json").write_text(json.dumps(markers), encoding="utf-8")
    if reasons is not None:
        (root / "skip_reasons.json").write_text(json.dumps(reasons), encoding="utf-8")
    if kinds is not None:
        (root / "skip_kinds.json").write_text(json.dumps(kinds), encoding="utf-8")
    (root / "skipped_sources.json.lock").write_text("", encoding="utf-8")


_OLD_REMOVED = "removed via remove (re-add the source or run retry-skipped to restore)"


class TestImportOfTheOlderFiles:
    def test_the_open_moves_every_record_into_the_index_and_deletes_the_files(
        self, held_config, tmp_path
    ):
        _write_legacy(
            tmp_path,
            {"scan.pdf": "h1", "gone.md": "h2", "kept.md": "h3"},
            {"scan.pdf": _REASON, "gone.md": _OLD_REMOVED, "kept.md": "user"},
            {"kept.md": {"kind": "removed", "hash": "h3", "reason": "user"}},
        )
        store = Store(held_config)
        assert store.held_records() == {
            "scan.pdf": _failed("h1"),
            "gone.md": _removed("h2"),
            "kept.md": _removed("h3"),
        }
        assert held_import.legacy_files(tmp_path) == []

    def test_a_write_the_older_lilbee_did_not_finish_is_read_from_its_pending_file(
        self, held_config, tmp_path
    ):
        _write_legacy(tmp_path, {"stale.md": "h0"})
        after = {
            "skipped_sources.json": json.dumps({"new.md": "h9"}),
            "skip_reasons.json": json.dumps({"new.md": _REASON}),
            "skip_kinds.json": json.dumps({}),
        }
        pending = tmp_path / "skip_records.pending.json"
        pending.write_text(json.dumps({"after": after}), encoding="utf-8")
        assert Store(held_config).held_records() == {"new.md": _failed("h9")}
        assert not pending.exists()

    def test_a_removal_record_of_an_indexed_file_is_not_imported(self, held_config, tmp_path):
        store = Store(held_config)
        _index(store, "a.md")
        _write_legacy(tmp_path, {"a.md": "h", "b.pdf": "hb"}, {"a.md": _OLD_REMOVED})
        reopened = Store(held_config)
        assert reopened.held_records() == {"b.pdf": _failed("hb", "held out by an earlier sync")}
        assert _listed(reopened) == ["a.md"]

    def test_an_import_killed_before_the_files_went_runs_again_to_the_same_rows(
        self, held_config, tmp_path, monkeypatch
    ):
        _write_legacy(tmp_path, {"scan.pdf": "h1"}, {"scan.pdf": _REASON})
        monkeypatch.setattr("lilbee.data.store.core.remove_legacy_files", lambda root: None)
        first = Store(held_config).held_records()
        monkeypatch.undo()
        again = Store(held_config)
        assert again.held_records() == first == {"scan.pdf": _failed("h1")}
        assert [row["path"] for row in again.open_table(SOURCES_TABLE).to_arrow().to_pylist()] == [
            "scan.pdf"
        ]
        assert held_import.legacy_files(tmp_path) == []

    def test_an_index_without_the_record_columns_gains_them(self, held_config, tmp_path):
        import lancedb

        db = lancedb.connect(str(held_config.lancedb_dir))
        schema = _sources_schema()
        older = schema.remove(schema.get_field_index("held_reason")).remove(
            schema.get_field_index("held_hash")
        )
        ensure_table(db, SOURCES_TABLE, older)
        assert _held_unmigrated(db, tmp_path)
        store = Store(held_config)
        store.hold_failed({"a.pdf": _failed("h")})
        assert store.held_records() == {"a.pdf": _failed("h")}
        assert not _held_unmigrated(store.get_db(), tmp_path)

    def test_a_new_library_has_nothing_to_import(self, held_config, tmp_path):
        import lancedb

        assert not _held_unmigrated(lancedb.connect(str(held_config.lancedb_dir)), tmp_path)


class TestLegacyReader:
    def test_unreadable_and_untyped_files_read_as_empty(self, tmp_path):
        (tmp_path / "skipped_sources.json").write_text("{not json", encoding="utf-8")
        assert held_import.read_legacy_records(tmp_path) == {}
        (tmp_path / "skipped_sources.json").write_text("[1, 2]", encoding="utf-8")
        assert held_import.read_legacy_records(tmp_path) == {}
        (tmp_path / "skipped_sources.json").write_bytes(b"\xff\xfe")
        assert held_import.read_legacy_records(tmp_path) == {}

    def test_a_marker_that_is_not_text_is_skipped(self, tmp_path):
        _write_legacy(tmp_path, {"a.md": 3, "b.md": "hb"})
        assert set(held_import.read_legacy_records(tmp_path)) == {"b.md"}

    def test_a_stored_kind_for_another_hash_reads_by_its_reason(self, tmp_path):
        stale = {"kind": "removed", "hash": "old", "reason": _REASON}
        _write_legacy(tmp_path, {"a.md": "new"}, {"a.md": _REASON}, {"a.md": stale})
        assert held_import.read_legacy_records(tmp_path) == {"a.md": _failed("new")}

    def test_a_stored_kind_lilbee_does_not_know_reads_by_its_reason(self, tmp_path):
        odd = {"kind": "odd", "hash": "h", "reason": _OLD_REMOVED}
        _write_legacy(tmp_path, {"a.md": "h"}, {"a.md": _OLD_REMOVED}, {"a.md": odd})
        assert held_import.read_legacy_records(tmp_path) == {"a.md": _removed("h")}

    def test_a_pending_file_without_every_text_leaves_the_files_as_they_are(self, tmp_path):
        _write_legacy(tmp_path, {"a.md": "h"})
        pending = {"after": {"skipped_sources.json": "{}"}}
        (tmp_path / "skip_records.pending.json").write_text(json.dumps(pending), encoding="utf-8")
        assert set(held_import.read_legacy_records(tmp_path)) == {"a.md"}

    def test_a_file_that_cannot_be_deleted_is_logged(self, tmp_path, monkeypatch, caplog):
        _write_legacy(tmp_path, {"a.md": "h"})

        def _refuse(self, missing_ok=False):
            raise PermissionError("read-only")

        monkeypatch.setattr("pathlib.Path.unlink", _refuse)
        held_import.remove_legacy_files(tmp_path)
        assert "Could not delete" in caplog.text
        assert (tmp_path / "skipped_sources.json").exists()
