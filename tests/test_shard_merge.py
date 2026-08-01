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


class TestFragmentAdoption:
    """A full merge takes over the shards' chunk files instead of rewriting them.

    The chunks table carries the vectors, so copying it is the whole cost of a
    merge, and the shard stores are kept as resume state: a copied corpus is on
    disk twice. Adoption hard-links the fragments and commits metadata, so the
    bytes exist once with two names.
    """

    def test_chunk_rows_are_never_read_during_a_full_merge(self, tmp_path, store, monkeypatch):
        # absorb_rows is the row-copy path. Chunks must not go through it.
        _write_shard(tmp_path / "w0", ["a.txt", "b.txt"])
        _write_shard(tmp_path / "w1", ["c.txt"])
        absorbed: list[str] = []
        real = store.absorb_rows

        def spy(name, rows):
            absorbed.append(name)
            return real(name, rows)

        monkeypatch.setattr(store, "absorb_rows", spy)
        merged = merge_shards(store, [tmp_path / "w0", tmp_path / "w1"])
        assert merged[CHUNKS_TABLE] == 3
        assert store.open_table(CHUNKS_TABLE).count_rows() == 3
        assert CHUNKS_TABLE not in absorbed
        # The small tables still copy: they carry no vectors.
        assert SOURCES_TABLE in absorbed

    def test_the_data_files_are_shared_rather_than_duplicated(self, tmp_path, store):
        # The point of adoption. A copy leaves two independent sets of bytes; a
        # hard link leaves one, which every link counts as its own name for.
        _write_shard(tmp_path / "w0", ["a.txt", "b.txt"])
        merge_shards(store, [tmp_path / "w0"])
        shard_files = sorted((tmp_path / "w0" / f"{CHUNKS_TABLE}.lance" / "data").iterdir())
        assert shard_files
        for path in shard_files:
            assert path.stat().st_nlink > 1, f"{path.name} was copied, not linked"

    def test_an_adopted_index_matches_what_copying_the_rows_produces(self, tmp_path):
        # The acceptance question: same rows, same sources, same content.
        _write_shard(tmp_path / "w0", ["a.txt", "b.txt"])
        _write_shard(tmp_path / "w1", ["c.txt"])
        shards = [tmp_path / "w0", tmp_path / "w1"]

        adopted_store = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "adopted"}))
        merge_shards(adopted_store, shards)

        copied_store = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "copied"}))
        merge_shards(copied_store, shards, sources={"a.txt", "b.txt", "c.txt"})

        def rows(store):
            table = store.open_table(CHUNKS_TABLE).search().limit(None).to_arrow()
            return sorted(
                (r["source"], r["chunk"]) for r in table.select(["source", "chunk"]).to_pylist()
            )

        assert rows(adopted_store) == rows(copied_store)
        assert len(rows(adopted_store)) == 3

    def test_a_scoped_resync_still_copies_rows(self, tmp_path, store, monkeypatch):
        # A named-source merge cannot adopt: a fragment holds touched and
        # untouched rows together, so taking it whole would drag in the rest.
        _write_shard(tmp_path / "w0", ["a.txt", "b.txt"])
        absorbed: list[str] = []
        real = store.absorb_rows
        monkeypatch.setattr(store, "absorb_rows", lambda n, r: (absorbed.append(n), real(n, r))[1])
        merge_shards(store, [tmp_path / "w0"], sources={"a.txt"})
        assert CHUNKS_TABLE in absorbed

    def test_a_shard_with_no_chunks_table_is_not_adopted(self, tmp_path, store):
        # A worker that indexed nothing writes its source and meta rows but no
        # chunks table; there is no fragment to take over and the merge goes on.
        database = lancedb.connect(str(tmp_path / "w0"))
        database.create_table(SOURCES_TABLE, pa.table({"filename": ["a.txt"]}))
        database.create_table(META_TABLE, pa.table({"embedding_model": ["shard-model"]}))
        merged = merge_shards(store, [tmp_path / "w0"])
        assert CHUNKS_TABLE not in merged
        assert store.open_table(SOURCES_TABLE).count_rows() == 1

    def test_a_shard_with_a_different_schema_is_refused(self, tmp_path, store):
        # The dangerous case. Adoption commits fragment metadata and never passes
        # rows through a writer, so a mismatched vector width is not rejected by
        # anything downstream: before this guard the merge reported success and
        # the index then panicked inside Arrow on the next read.
        _write_shard(tmp_path / "w0", ["a.txt"])
        merge_shards(store, [tmp_path / "w0"])

        narrow = lancedb.connect(str(tmp_path / "w1"))
        narrow.create_table(
            CHUNKS_TABLE,
            pa.table(
                {
                    "source": ["b.txt"],
                    "chunk": ["body of b.txt"],
                    "vector": [[1.0] * 16],
                }
            ),
        )
        with pytest.raises(ValueError, match="does not match"):
            merge_shards(store, [tmp_path / "w1"])
        # And the index it refused to touch is intact and still readable.
        table = store.open_table(CHUNKS_TABLE)
        assert table.count_rows() == 1
        assert table.search().limit(None).to_arrow().num_rows == 1

    def test_a_failed_link_leaves_no_files_behind(self, tmp_path, store):
        # A link made before the failure is a file the manifest never names, so
        # nothing would ever remove it. Fail on the second shard, after the first
        # has already been linked.
        import os

        _write_shard(tmp_path / "w0", ["a.txt"])
        _write_shard(tmp_path / "w1", ["b.txt"])
        real_link = os.link
        calls = {"n": 0}

        def fail_after_first(src, dst, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError("Invalid cross-device link")
            return real_link(src, dst, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "link", fail_after_first)
            merged = merge_shards(store, [tmp_path / "w0", tmp_path / "w1"])
        # The link that succeeded before the failure must be gone: nothing names
        # it, so it would sit in the table's directory forever.
        data = tmp_path / "merged" / f"{CHUNKS_TABLE}.lance" / "data"
        shard_names = {
            path.name
            for shard in ("w0", "w1")
            for path in (tmp_path / shard / f"{CHUNKS_TABLE}.lance" / "data").iterdir()
        }
        assert not {p.name for p in data.iterdir()} & shard_names
        # And the fallback copy still landed every row exactly once.
        assert merged[CHUNKS_TABLE] == 2
        assert store.open_table(CHUNKS_TABLE).count_rows() == 2

    def test_adopting_nothing_is_a_no_op(self, store):
        # A guard on a public method: merge_shards never calls it with an empty
        # list, but the method does not get to assume its only caller.
        assert store.adopt_fragments(CHUNKS_TABLE, []) == 0

    def test_a_shard_whose_chunks_table_is_empty_contributes_nothing(self, tmp_path, store):
        # A worker that indexed nothing still creates its tables. There is no
        # fragment to take over, and the merge has to carry on regardless.
        database = lancedb.connect(str(tmp_path / "w0"))
        database.create_table(CHUNKS_TABLE, schema=_chunk_rows([]).schema)
        database.create_table(SOURCES_TABLE, pa.table({"filename": ["a.txt"]}))
        merged = merge_shards(store, [tmp_path / "w0"])
        assert merged[CHUNKS_TABLE] == 0
        assert store.open_table(SOURCES_TABLE).count_rows() == 1

    def test_a_shard_that_cannot_be_linked_falls_back_to_copying(self, tmp_path, store):
        # Hard links cannot cross a filesystem, and a data file name could already
        # be taken. Either way the merge has to complete, just more slowly.
        import os

        _write_shard(tmp_path / "w0", ["a.txt", "b.txt"])

        def refuse(*_args, **_kwargs):
            raise OSError("Invalid cross-device link")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "link", refuse)
            merged = merge_shards(store, [tmp_path / "w0"])
        assert merged[CHUNKS_TABLE] == 2
        assert store.open_table(CHUNKS_TABLE).count_rows() == 2


class TestWholeShardMergeAtScale:
    def test_a_shard_larger_than_any_default_query_limit_lands_whole(self, tmp_path, store):
        """A capped scan would drop the tail of every shard and say nothing."""
        sources = [f"f{index:05d}.txt" for index in range(1200)]
        _write_shard(tmp_path / "w0", sources)
        merged = merge_shards(store, [tmp_path / "w0"])
        assert merged[CHUNKS_TABLE] == 1200
        assert store.open_table(CHUNKS_TABLE).count_rows() == 1200


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


class TestReconciliation:
    def test_an_index_short_of_the_workers_says_so(self, tmp_path, store, caplog):
        """A source a worker holds and the index does not must not stay missing silently."""
        _write_shard(tmp_path / "w0", ["a.txt", "b.txt"])
        merge_shards(store, [tmp_path / "w0"], sources={"a.txt"})
        assert "against 2 across the ingest workers" in caplog.text

    def test_a_complete_merge_is_quiet(self, tmp_path, store, caplog):
        _write_shard(tmp_path / "w0", ["a.txt"])
        merge_shards(store, [tmp_path / "w0"])
        assert "across the ingest workers" not in caplog.text
