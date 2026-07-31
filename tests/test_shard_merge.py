"""Folding per-GPU ingest shards into one index."""

from __future__ import annotations

import lancedb
import pyarrow as pa
import pytest

from lilbee.core.config import cfg
from lilbee.core.config.defaults import CHUNKS_TABLE, META_TABLE, SOURCES_TABLE
from lilbee.data.store import Store
from lilbee.data.store.shard_merge import merge_shards


@pytest.fixture()
def store(tmp_path):
    """A parent store to merge into."""
    return Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "merged"}))


def _chunk_rows(sources: list[str]) -> pa.Table:
    return pa.table(
        {
            "source": sources,
            "chunk": [f"body of {source}" for source in sources],
            "vector": [[float(index)] * cfg.embedding_dim for index in range(len(sources))],
        }
    )


def _write_shard(path, sources: list[str]) -> None:
    """A worker's finished store: its chunks, its source rows, its own meta row."""
    database = lancedb.connect(str(path))
    database.create_table(CHUNKS_TABLE, _chunk_rows(sources))
    database.create_table(SOURCES_TABLE, pa.table({"filename": sources}))
    database.create_table(META_TABLE, pa.table({"embedding_model": ["shard-model"]}))


class TestWholeShardMerge:
    def test_every_shard_s_rows_land_in_the_one_index(self, tmp_path, store):
        _write_shard(tmp_path / "w0", ["a.txt", "b.txt"])
        _write_shard(tmp_path / "w1", ["c.txt"])
        merged = merge_shards(store, [tmp_path / "w0", tmp_path / "w1"])
        assert merged[CHUNKS_TABLE] == 3
        assert store.open_table(CHUNKS_TABLE).count_rows() == 3
        assert store.open_table(SOURCES_TABLE).count_rows() == 3

    def test_the_merged_store_keeps_one_meta_row_from_its_own_config(self, tmp_path, store):
        """Two shards' meta rows would land beside each other and describe the index twice."""
        _write_shard(tmp_path / "w0", ["a.txt"])
        _write_shard(tmp_path / "w1", ["b.txt"])
        merge_shards(store, [tmp_path / "w0", tmp_path / "w1"])
        meta = store.open_table(META_TABLE)
        assert meta.count_rows() == 1
        assert store.get_meta()["embedding_model"] == cfg.embedding_model


class TestScopedMerge:
    def test_only_the_touched_sources_are_taken(self, tmp_path, store):
        _write_shard(tmp_path / "w0", ["a.txt", "b.txt"])
        merged = merge_shards(store, [tmp_path / "w0"], sources={"b.txt"})
        assert merged[CHUNKS_TABLE] == 1
        assert store.open_table(CHUNKS_TABLE).to_arrow()["source"].to_pylist() == ["b.txt"]

    def test_re_merging_a_source_replaces_its_rows_instead_of_doubling_them(self, tmp_path, store):
        """A second sync must not append a second copy of everything it already holds."""
        _write_shard(tmp_path / "w0", ["a.txt"])
        merge_shards(store, [tmp_path / "w0"])
        merge_shards(store, [tmp_path / "w0"], sources={"a.txt"})
        assert store.open_table(CHUNKS_TABLE).count_rows() == 1

    def test_a_table_with_no_source_column_is_left_to_the_corpus_wide_passes(self, tmp_path, store):
        database = lancedb.connect(str(tmp_path / "w0"))
        database.create_table("concept_nodes", pa.table({"label": ["braking"]}))
        merged = merge_shards(store, [tmp_path / "w0"], sources={"a.txt"})
        assert merged["concept_nodes"] == 0

    def test_names_are_taken_in_batches(self, tmp_path, store, monkeypatch):
        """One predicate per sync would be a megabytes-long SQL string on a big corpus."""
        monkeypatch.setattr("lilbee.data.store.shard_merge._NAMES_PER_PREDICATE", 2)
        sources = [f"f{index}.txt" for index in range(5)]
        _write_shard(tmp_path / "w0", sources)
        merged = merge_shards(store, [tmp_path / "w0"], sources=set(sources))
        assert merged[CHUNKS_TABLE] == 5

    def test_a_quoted_source_name_does_not_break_the_predicate(self, tmp_path, store):
        _write_shard(tmp_path / "w0", ["it's here.txt"])
        merged = merge_shards(store, [tmp_path / "w0"], sources={"it's here.txt"})
        assert merged[CHUNKS_TABLE] == 1


class TestBatching:
    def test_a_shard_is_streamed_rather_than_loaded_whole(self, tmp_path, store, monkeypatch):
        """A chunks row carries its vector, so a whole shard table does not fit in memory."""
        monkeypatch.setattr("lilbee.data.store.shard_merge._MERGE_BATCH_ROWS", 2)
        appends = []
        original = Store.absorb_rows

        def counting_absorb(self, name, rows):
            appends.append(rows.num_rows)
            return original(self, name, rows)

        monkeypatch.setattr(Store, "absorb_rows", counting_absorb)
        _write_shard(tmp_path / "w0", [f"f{index}.txt" for index in range(5)])
        merge_shards(store, [tmp_path / "w0"])
        assert max(appends) <= 2
