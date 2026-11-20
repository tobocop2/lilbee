"""An optimize on the chunks, sources and concept tables keeps the version a reader holds."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from lancedb.table import LanceTable

from lilbee.core.config import (
    CHUNKS_TABLE,
    CONCEPT_EDGES_TABLE,
    CONCEPT_NODES_TABLE,
    SOURCES_TABLE,
    cfg,
)
from lilbee.data.store import Store, lance_helpers
from lilbee.retrieval.concepts import ConceptGraph
from lilbee.retrieval.concepts.schema import _concept_edges_schema, _concept_nodes_schema
from tests.test_store import _make_indexable_records, _make_records

# The library prunes at 7 days when no window is named, and the store's floor is minutes;
# the stand-ins are short so a test can idle past both.
_STAND_IN_DEFAULT = timedelta(seconds=2)
_SHORT_FLOOR = timedelta(seconds=1)
_IDLE_SECONDS = 3
_INDEXABLE_ROWS = 256


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(lance_helpers, "_KEEP_VERSIONS", _SHORT_FLOOR)
    return Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb_test"}))


@pytest.fixture()
def unwindowed(monkeypatch):
    """Optimize calls that name no window, which the library prunes at its default."""
    real_optimize = LanceTable.optimize
    calls: list[str] = []

    def optimize(self, *args, **kwargs):
        if kwargs.get("cleanup_older_than") is None:
            calls.append(self.name)
            kwargs["cleanup_older_than"] = _STAND_IN_DEFAULT
        return real_optimize(self, *args, **kwargs)

    monkeypatch.setattr(LanceTable, "optimize", optimize)
    return calls


def _reader(store: Store, name: str) -> LanceTable:
    """A reader held on the latest version of *name*, idle past the floor and the stand-in."""
    reader = store.open_table(name)
    assert reader is not None
    reader.checkout(reader.version)
    time.sleep(_IDLE_SECONDS)
    return reader


def _reopened_rows(store: Store, reader: LanceTable) -> int:
    """Rows of the reader's version, opened again as a reader in another process would.

    Lance keeps young data files after a prune, so the open reader reads either way; the
    version itself is what the prune deletes, and a new handle cannot open a deleted one.
    """
    fresh = store.open_table(reader.name)
    assert fresh is not None
    fresh.checkout(reader.version)
    return fresh.to_arrow().num_rows


class TestChunksTable:
    def test_fts_optimize_keeps_the_version_a_reader_holds(self, store, unwindowed):
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        store.add_chunks(_make_records())
        reader = _reader(store, CHUNKS_TABLE)
        rows = reader.count_rows()
        # The sync writes on: Lance keeps the latest version on its own, not an older one.
        store.add_chunks(_make_records())
        store.ensure_fts_index()
        assert reader.to_arrow().num_rows == rows
        assert _reopened_rows(store, reader) == rows
        assert unwindowed == []

    def test_vector_optimize_keeps_the_version_a_reader_holds(self, store, unwindowed):
        config = store._config
        config.ann_index_threshold = 50
        store.add_chunks(_make_indexable_records(_INDEXABLE_ROWS, config.embedding_dim))
        store.ensure_fts_index()
        assert store.ensure_vector_index() is True
        store.add_chunks(_make_indexable_records(_INDEXABLE_ROWS, config.embedding_dim))
        reader = _reader(store, CHUNKS_TABLE)
        rows = reader.count_rows()
        # The sync writes on: Lance keeps the latest version on its own, not an older one.
        store.add_chunks(_make_indexable_records(_INDEXABLE_ROWS, config.embedding_dim))
        assert store.ensure_vector_index() is True
        assert reader.to_arrow().num_rows == rows
        assert _reopened_rows(store, reader) == rows
        assert unwindowed == []


class TestSourcesTable:
    def test_optimize_keeps_the_version_a_reader_holds(self, store, unwindowed):
        for index in range(3):
            store.upsert_source(f"doc{index}.md", f"hash{index}", 1)
        reader = _reader(store, SOURCES_TABLE)
        rows = reader.count_rows()
        store.optimize_sources()
        assert reader.to_arrow().num_rows == rows
        assert _reopened_rows(store, reader) == rows
        assert unwindowed == []


class TestConceptTables:
    def test_compact_keeps_the_version_a_reader_holds(self, store, unwindowed):
        for index in range(3):
            store.clear_and_add(
                CONCEPT_NODES_TABLE,
                _concept_nodes_schema(),
                [{"concept": f"c{index}", "cluster_id": 0, "degree": 1}],
                f"concept = 'c{index}'",
            )
            store.clear_and_add(
                CONCEPT_EDGES_TABLE,
                _concept_edges_schema(),
                [{"source": f"c{index}", "target": "d", "weight": 1.0}],
                f"source = 'c{index}'",
            )
        nodes = _reader(store, CONCEPT_NODES_TABLE)
        edges = store.open_table(CONCEPT_EDGES_TABLE)
        assert edges is not None
        edges.checkout(edges.version)
        readers = {CONCEPT_NODES_TABLE: nodes, CONCEPT_EDGES_TABLE: edges}
        rows = {name: reader.count_rows() for name, reader in readers.items()}
        ConceptGraph(store._config, store).compact_tables()
        assert {name: reader.to_arrow().num_rows for name, reader in readers.items()} == rows
        assert {name: _reopened_rows(store, reader) for name, reader in readers.items()} == rows
        assert unwindowed == []


class TestTheRealFloor:
    """No fixture shortens the floor here: the window is the one the product ships."""

    def test_the_floor_is_at_least_a_minute(self):
        assert lance_helpers._KEEP_VERSIONS.total_seconds() >= 60

    def test_an_optimize_with_no_kept_version_names_the_real_floor(self):
        table = MagicMock()
        lance_helpers.prune_old_versions(table)
        window = table.optimize.call_args.kwargs["cleanup_older_than"]
        assert window >= timedelta(minutes=1)

    def test_an_optimize_with_a_fresh_kept_version_names_a_window_of_at_least_a_minute(self):
        table = MagicMock()
        table.list_versions.return_value = [{"version": 3, "timestamp": datetime.now()}]
        lance_helpers.prune_old_versions(table, keep_version=3)
        window = table.optimize.call_args.kwargs["cleanup_older_than"]
        assert window >= timedelta(minutes=1)
