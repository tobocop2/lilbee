"""``lilbee status`` stays small and fast however many sources the library holds."""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import UTC, datetime
from unittest.mock import patch

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.app.status import STATUS_RECENT_LIMIT, gather_status
from lilbee.cli.app import app
from lilbee.core.config import cfg
from lilbee.data.store import Store
from tests.conftest import make_mock_services

MILLION = 1_000_000
CHUNK_ROWS = 100_000
KINDS = ("text", "pdf", "code")
BASE_EPOCH = 1_760_000_000

MAX_STATUS_SECONDS = 30.0
MAX_STATUS_RSS_GROWTH_BYTES = 200 * 1024 * 1024


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _ingested_at(i: int) -> str:
    return datetime.fromtimestamp(BASE_EPOCH + i, UTC).isoformat()


def _fill(store: Store, count: int) -> None:
    """Write *count* source rows, newest ingest last by index, inserted newest chunk first."""
    table = store._sources_table()
    for start in reversed(range(0, count, CHUNK_ROWS)):
        ids = range(start, min(start + CHUNK_ROWS, count))
        rows = {
            "source_id": [f"{i:032x}" for i in ids],
            "path": [f"corpus/doc-{i:09d}.txt" for i in ids],
            "file_hash": [f"{i:064x}" for i in ids],
            "ingested_at": [_ingested_at(i) for i in ids],
            "chunk_count": pa.array([2 + i % 3 for i in ids], pa.int32()),
            "source_type": [KINDS[i % len(KINDS)] for i in ids],
            "size_bytes": pa.array([i for i in ids], pa.int64()),
            "mtime_ns": pa.array([i for i in ids], pa.int64()),
            "stat_captured_ns": pa.array([i for i in ids], pa.int64()),
            "title": pa.array([None for _ in ids], pa.utf8()),
            "authors": pa.array([None for _ in ids], pa.utf8()),
            "created_at": pa.array([None for _ in ids], pa.utf8()),
            **_not_held(len(ids)),
        }
        table.add(pa.table(rows, schema=table.schema))


def _not_held(count: int) -> dict[str, pa.Array]:
    """The held-out columns of *count* source rows that no sync holds out."""
    return {
        name: pa.array([None] * count, pa.utf8())
        for name in ("held_out", "held_hash", "held_reason")
    }


def _chunks_total(count: int) -> int:
    return sum(2 + i % 3 for i in range(count))


def _use_store(store: Store):
    svc_mod.set_services(make_mock_services(store=store))


@pytest.fixture(autouse=True)
def _reset_services():
    yield
    svc_mod.set_services(None)


@pytest.fixture(scope="module")
def million_store(tmp_path_factory):
    config = cfg.model_copy(update={"lancedb_dir": tmp_path_factory.mktemp("million") / "lancedb"})
    store = Store(config)
    _fill(store, MILLION)
    return store


@pytest.fixture
def small_store(tmp_path):
    store = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb_small"}))
    _fill(store, 3_000)
    return store


class TestStatusOnAMillionSources:
    def test_gather_status_is_bounded_in_time_and_memory(self, million_store):
        _use_store(million_store)
        rss_before = _peak_rss_bytes()
        started = time.monotonic()

        status = gather_status()

        elapsed = time.monotonic() - started
        growth = _peak_rss_bytes() - rss_before
        assert status.document_count == MILLION
        assert status.total_chunks == _chunks_total(MILLION)
        assert len(status.sources) == STATUS_RECENT_LIMIT
        assert elapsed < MAX_STATUS_SECONDS
        assert growth < MAX_STATUS_RSS_GROWTH_BYTES

    def test_status_names_the_newest_sources_and_counts_every_kind(self, million_store):
        _use_store(million_store)

        status = gather_status()

        newest = [f"corpus/doc-{MILLION - 1 - n:09d}.txt" for n in range(STATUS_RECENT_LIMIT)]
        assert [s.filename for s in status.sources] == newest
        assert status.source_kinds == {
            kind: len(range(i, MILLION, len(KINDS))) for i, kind in enumerate(KINDS)
        }

    def test_json_output_does_not_grow_with_the_library(self, million_store):
        _use_store(million_store)

        result = CliRunner().invoke(app, ["--json", "status"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["document_count"] == MILLION
        assert len(payload["sources"]) == STATUS_RECENT_LIMIT
        assert len(result.output) < 20_000


class TestHttpStatusOnAMillionSources:
    async def test_the_status_route_carries_the_summary_not_the_list(self, million_store):
        from lilbee.server import handlers

        _use_store(million_store)

        response = await handlers.status()

        assert response.document_count == MILLION
        assert len(response.sources) == STATUS_RECENT_LIMIT
        assert response.source_kinds == {
            kind: len(range(i, MILLION, len(KINDS))) for i, kind in enumerate(KINDS)
        }


class TestStatusOutputShape:
    def test_human_output_lists_only_the_newest_sources(self, small_store):
        _use_store(small_store)

        result = CliRunner().invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        shown = [line for line in result.output.splitlines() if "corpus/doc-" in line]
        assert len(shown) == STATUS_RECENT_LIMIT
        assert "corpus/doc-000002999.txt" in result.output
        assert "corpus/doc-000000000.txt" not in result.output
        assert "3000 documents" in result.output.replace(",", "")

    def test_human_output_points_at_the_full_listing(self, small_store):
        _use_store(small_store)

        result = CliRunner().invoke(app, ["status"])

        assert "lilbee status --list-sources" in result.output

    def test_human_output_counts_sources_by_kind(self, small_store):
        _use_store(small_store)

        result = CliRunner().invoke(app, ["status"])

        assert "code 1000, pdf 1000, text 1000" in result.output

    def test_list_sources_streams_every_source_once(self, small_store):
        _use_store(small_store)

        result = CliRunner().invoke(app, ["status", "--list-sources"])

        assert result.exit_code == 0, result.output
        names = result.output.splitlines()
        assert len(names) == 3_000
        assert len(set(names)) == 3_000

    def test_list_sources_in_json_mode_prints_one_object_per_line(self, small_store):
        _use_store(small_store)

        result = CliRunner().invoke(app, ["--json", "status", "--list-sources"])

        assert result.exit_code == 0, result.output
        rows = [json.loads(line) for line in result.output.splitlines()]
        assert len(rows) == 3_000
        assert set(rows[0]) == {"filename", "file_hash", "chunk_count", "ingested_at"}


class TestSourceSummary:
    def test_a_library_without_a_source_table_summarizes_to_nothing(self, tmp_path):
        store = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "empty"}))

        summary = store.source_summary(STATUS_RECENT_LIMIT)

        assert (summary.document_count, summary.total_chunks) == (0, 0)
        assert summary.kinds == {}
        assert summary.newest == []
        assert list(store.iter_source_heads()) == []

    def test_sources_ingested_at_the_same_time_come_back_by_filename(self, tmp_path):
        store = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "ties"}))
        table = store._sources_table()
        names = ["c.txt", "a.txt", "b.txt", "d.txt"]
        table.add(
            pa.table(
                {
                    "source_id": ["1", "2", "3", "4"],
                    "path": names,
                    "file_hash": ["h"] * 4,
                    "ingested_at": ["2026-01-01T00:00:00"] * 4,
                    "chunk_count": pa.array([1] * 4, pa.int32()),
                    "source_type": ["text"] * 4,
                    "size_bytes": pa.array([0] * 4, pa.int64()),
                    "mtime_ns": pa.array([0] * 4, pa.int64()),
                    "stat_captured_ns": pa.array([0] * 4, pa.int64()),
                    "title": pa.array([None] * 4, pa.utf8()),
                    "authors": pa.array([None] * 4, pa.utf8()),
                    "created_at": pa.array([None] * 4, pa.utf8()),
                    **_not_held(4),
                },
                schema=table.schema,
            )
        )

        summary = store.source_summary(3)

        assert [head["filename"] for head in summary.newest] == ["a.txt", "b.txt", "c.txt"]
        assert summary.document_count == 4

    def test_only_listed_sources_count_each_by_the_name_it_shows(self, tmp_path):
        notes = tmp_path / "notes"
        config = cfg.model_copy(
            update={"lancedb_dir": tmp_path / "listed", "linked_roots": {"notes": str(notes)}}
        )
        store = Store(config)
        table = store._sources_table()
        rows = [
            ("1", f"{notes.as_posix()}/a.md", "h1", "document", None),
            ("2", f"{notes.as_posix()}/b.md", "h2", "document", "failed"),
            ("3", "wiki/concepts/c.md", "h3", "wiki", None),
            ("4", f"{notes.as_posix()}/gone.md", None, "document", "removed"),
        ]
        table.add(
            pa.table(
                {
                    "source_id": [row[0] for row in rows],
                    "path": [row[1] for row in rows],
                    "file_hash": [row[2] for row in rows],
                    "ingested_at": [f"2026-01-01T00:00:0{n}" for n in range(len(rows))],
                    "chunk_count": pa.array([3] * len(rows), pa.int32()),
                    "source_type": [row[3] for row in rows],
                    "size_bytes": pa.array([0] * len(rows), pa.int64()),
                    "mtime_ns": pa.array([0] * len(rows), pa.int64()),
                    "stat_captured_ns": pa.array([0] * len(rows), pa.int64()),
                    "title": pa.array([None] * len(rows), pa.utf8()),
                    "authors": pa.array([None] * len(rows), pa.utf8()),
                    "created_at": pa.array([None] * len(rows), pa.utf8()),
                    "held_out": pa.array([row[4] for row in rows], pa.utf8()),
                    "held_hash": pa.array([None] * len(rows), pa.utf8()),
                    "held_reason": pa.array([None] * len(rows), pa.utf8()),
                },
                schema=table.schema,
            )
        )

        summary = store.source_summary(STATUS_RECENT_LIMIT)

        assert (summary.document_count, summary.total_chunks) == (2, 6)
        assert summary.kinds == {"document": 2}
        assert [head["filename"] for head in summary.newest] == ["notes/b.md", "notes/a.md"]
        assert sorted(head["filename"] for head in store.iter_source_heads()) == [
            "notes/a.md",
            "notes/b.md",
        ]
        assert summary.document_count == store.count_sources()

    def test_the_newest_come_from_every_batch_not_only_the_last(self, small_store):
        with patch("lilbee.data.store.core._SOURCE_SCAN_BATCH_ROWS", 7):
            summary = small_store.source_summary(3)

        assert [head["filename"] for head in summary.newest] == [
            "corpus/doc-000002999.txt",
            "corpus/doc-000002998.txt",
            "corpus/doc-000002997.txt",
        ]
        assert summary.document_count == 3_000
