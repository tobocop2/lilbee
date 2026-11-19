"""Tests for the ``_memories`` table: CRUD, dedup, eviction, recall, and rebuild."""

import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import numpy.typing as npt
import pytest

from lilbee.core.config import MEMORIES_TABLE, cfg
from lilbee.data.store import (
    LOCAL_OWNER,
    EmbeddingModelMismatchError,
    MemoryKind,
    MemoryRow,
    MemorySource,
    Store,
    agent_owner,
    human_recall_predicate,
    is_agent_owner,
    lance_helpers,
)

LOCAL_PREDICATE = f"owner = '{LOCAL_OWNER}'"


class TestOwnerHelpers:
    def test_agent_owner_round_trips_through_is_agent_owner(self):
        owner = agent_owner("opencode")
        assert owner == "agent:opencode"
        assert is_agent_owner(owner) is True

    def test_local_owner_is_not_an_agent(self):
        assert is_agent_owner(LOCAL_OWNER) is False


class TestMemorySourceHelpers:
    def test_memory_source_prefixes_the_id(self):
        from lilbee.data.store import memory_source

        assert memory_source("abc") == "memory:abc"

    def test_is_memory_source_matches_prefix_only(self):
        from lilbee.data.store import is_memory_source

        assert is_memory_source("memory:abc") is True
        assert is_memory_source("doc.md") is False


@pytest.fixture()
def test_config(tmp_path):
    """A Config pointing at a temp LanceDB dir."""
    return cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb_test"})


@pytest.fixture()
def store(test_config):
    """A Store backed by the temp config."""
    return Store(test_config)


def _unit_vector(dim: int, axis: int) -> list[float]:
    """A unit vector along *axis* so cosine distance between distinct axes is 1.0."""
    vec = [0.0] * dim
    vec[axis] = 1.0
    return vec


def _unit_array(dim: int, axis: int) -> npt.NDArray[np.float32]:
    """:func:`_unit_vector` as the float32 array the embedder hands the store."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[axis] = 1.0
    return vec


def _memory(
    store: Store,
    *,
    text: str = "a memory",
    kind: MemoryKind = MemoryKind.FACT,
    owner: str = LOCAL_OWNER,
    source: MemorySource = MemorySource.MANUAL,
    axis: int = 0,
    created_at: str | None = None,
    memory_id: str = "",
) -> MemoryRow:
    now = created_at or datetime.now(UTC).isoformat()
    return MemoryRow(
        id=memory_id or f"id-{text}-{owner}-{axis}",
        owner=owner,
        shared=False,
        kind=kind,
        source=source,
        text=text,
        vector=_unit_vector(store._config.embedding_dim, axis),
        created_at=now,
        updated_at=now,
    )


class TestAddAndGet:
    def test_add_then_get_round_trips(self, store):
        store.add_memory(_memory(store, text="prefers rust"))
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)
        assert len(got) == 1
        assert got[0].text == "prefers rust"
        assert got[0].kind is MemoryKind.FACT

    def test_get_filters_by_kind(self, store):
        store.add_memory(_memory(store, text="be terse", kind=MemoryKind.PREFERENCE, axis=0))
        store.add_memory(_memory(store, text="uses lancedb", kind=MemoryKind.FACT, axis=1))
        prefs = store.get_memories(owner_predicate=LOCAL_PREDICATE, kind=MemoryKind.PREFERENCE)
        assert [m.text for m in prefs] == ["be terse"]

    def test_get_missing_table_returns_empty(self, store):
        assert store.get_memories(owner_predicate=LOCAL_PREDICATE) == []

    def test_dimension_mismatch_raises(self, store):
        bad = _memory(store)
        bad.vector = [0.1, 0.2]
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add_memory(bad)


class TestDedup:
    def test_identical_vector_same_owner_kind_updates_in_place(self, store):
        store.add_memory(_memory(store, text="original", axis=0, memory_id="first"))
        store.add_memory(_memory(store, text="updated", axis=0, memory_id="second"))
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)
        assert len(got) == 1
        assert got[0].text == "updated"
        assert got[0].id == "first"  # kept the original id

    def test_different_owner_not_deduped(self, store):
        store.add_memory(_memory(store, text="mine", owner=LOCAL_OWNER, axis=0))
        store.add_memory(_memory(store, text="theirs", owner=agent_owner("opencode"), axis=0))
        local = store.get_memories(owner_predicate=LOCAL_PREDICATE)
        agent = store.get_memories(owner_predicate=f"owner = '{agent_owner('opencode')}'")
        assert len(local) == 1
        assert len(agent) == 1

    def test_different_kind_not_deduped(self, store):
        store.add_memory(_memory(store, text="f", kind=MemoryKind.FACT, axis=0))
        store.add_memory(_memory(store, text="p", kind=MemoryKind.PREFERENCE, axis=0))
        assert len(store.get_memories(owner_predicate=LOCAL_PREDICATE)) == 2


class TestEviction:
    def test_oldest_evicted_past_cap(self, store):
        store._config.memory_max_per_owner = 3
        for i in range(4):
            store.add_memory(
                _memory(
                    store,
                    text=f"m{i}",
                    axis=i,
                    created_at=f"2026-06-0{i + 1}T00:00:00+00:00",
                    memory_id=f"m{i}",
                )
            )
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)
        assert len(got) == 3
        assert "m0" not in {m.id for m in got}  # oldest gone


class TestSearchMemories:
    def test_recalls_near_excludes_far(self, store):
        store.add_memory(_memory(store, text="near", axis=0))
        store.add_memory(_memory(store, text="far", axis=1))
        query = _unit_vector(store._config.embedding_dim, 0)
        hits = store.search_memories(
            query, owner_predicate=LOCAL_PREDICATE, top_k=5, max_distance=0.6
        )
        assert [m.text for m in hits] == ["near"]

    def test_excludes_preferences(self, store):
        store.add_memory(_memory(store, text="pref", kind=MemoryKind.PREFERENCE, axis=0))
        query = _unit_vector(store._config.embedding_dim, 0)
        hits = store.search_memories(
            query, owner_predicate=LOCAL_PREDICATE, top_k=5, max_distance=0.6
        )
        assert hits == []

    def test_owner_predicate_scopes_results(self, store):
        store.add_memory(_memory(store, text="mine", owner=LOCAL_OWNER, axis=0))
        store.add_memory(_memory(store, text="theirs", owner=agent_owner("x"), axis=0))
        query = _unit_vector(store._config.embedding_dim, 0)
        hits = store.search_memories(
            query, owner_predicate=LOCAL_PREDICATE, top_k=5, max_distance=0.6
        )
        assert [m.text for m in hits] == ["mine"]

    def test_top_k_zero_returns_empty(self, store):
        store.add_memory(_memory(store, text="x", axis=0))
        query = _unit_vector(store._config.embedding_dim, 0)
        assert (
            store.search_memories(query, owner_predicate=LOCAL_PREDICATE, top_k=0, max_distance=1.0)
            == []
        )

    def test_missing_table_returns_empty(self, store):
        query = _unit_vector(store._config.embedding_dim, 0)
        assert (
            store.search_memories(query, owner_predicate=LOCAL_PREDICATE, top_k=5, max_distance=1.0)
            == []
        )


class TestUpdateAndDelete:
    def test_update_toggles_shared(self, store):
        store.add_memory(_memory(store, text="x", memory_id="u1", axis=0))
        assert store.update_memory("u1", shared=True, owner=LOCAL_OWNER) is True
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)[0]
        assert got.shared is True

    def test_update_missing_returns_false(self, store):
        store.add_memory(_memory(store, text="x", axis=0))
        assert store.update_memory("nope", shared=True, owner=LOCAL_OWNER) is False

    def test_update_no_table_returns_false(self, store):
        assert store.update_memory("nope", shared=True, owner=LOCAL_OWNER) is False

    def test_delete_removes(self, store):
        store.add_memory(_memory(store, text="x", memory_id="d1", axis=0))
        assert store.delete_memory("d1", owner=LOCAL_OWNER) is True
        assert store.get_memories(owner_predicate=LOCAL_PREDICATE) == []


class TestSwallowedDeleteFailOpen:
    """A swallowed LanceDB delete must not corrupt state or report false success."""

    @staticmethod
    def _fail_delete(monkeypatch):
        # Simulate _safe_delete_unlocked swallowing a delete failure (returns False
        # without removing anything), as it does on a real LanceDB delete error.
        monkeypatch.setattr("lilbee.data.store.core._safe_delete_unlocked", lambda *a, **k: False)

    def test_delete_memory_reports_failure(self, store, monkeypatch):
        store.add_memory(_memory(store, text="x", memory_id="d1", axis=0))
        self._fail_delete(monkeypatch)
        assert store.delete_memory("d1", owner=LOCAL_OWNER) is False
        assert len(store.get_memories(owner_predicate=LOCAL_PREDICATE)) == 1

    def test_update_memory_failed_delete_adds_no_duplicate(self, store, monkeypatch):
        store.add_memory(_memory(store, text="x", memory_id="u1", axis=0))
        self._fail_delete(monkeypatch)
        assert store.update_memory("u1", shared=True, owner=LOCAL_OWNER) is False
        assert len(store.get_memories(owner_predicate=LOCAL_PREDICATE)) == 1

    def test_add_memory_failed_dedup_delete_keeps_distinct_ids(self, store, monkeypatch):
        store.add_memory(_memory(store, text="orig", axis=0, memory_id="first"))
        self._fail_delete(monkeypatch)
        # Same vector/owner/kind -> dedup path; the dedup delete fails, so the new
        # row must keep a fresh id rather than collide with the surviving original.
        store.add_memory(_memory(store, text="new", axis=0, memory_id="second"))
        rows = store.get_memories(owner_predicate=LOCAL_PREDICATE)
        ids = [r.id for r in rows]
        assert len(ids) == len(set(ids))  # no id collision

    def test_get_meta_returns_newest_when_duplicate_rows_exist(self, store, monkeypatch):
        import lilbee.data.store.core as core

        dim = store._config.embedding_dim
        # Stamp explicit, far-apart timestamps (not wall-clock) so the assertion
        # is deterministic: the stale row is written first, the current row last.
        times = iter(["2020-01-01T00:00:00+00:00", "2030-01-01T00:00:00+00:00"])

        class _FakeDatetime:
            @staticmethod
            def now(tz=None):
                return datetime.fromisoformat(next(times))

        monkeypatch.setattr(core, "datetime", _FakeDatetime)
        store._write_meta_unlocked(embedding_model="model-stale", embedding_dim=dim)
        self._fail_delete(monkeypatch)  # second write's delete is swallowed -> 2 rows
        store._write_meta_unlocked(embedding_model="model-current", embedding_dim=dim)
        meta = store.get_meta()
        assert meta is not None
        assert meta["embedding_model"] == "model-current"


class TestHumanRecallPredicate:
    """The human's view includes local memories plus agent-shared ones."""

    def test_human_sees_local_and_shared_agent_memories(self, store):
        store.add_memory(_memory(store, text="mine", owner=LOCAL_OWNER, axis=0))
        shared = _memory(store, text="shared by agent", owner=agent_owner("a"), axis=1)
        shared.shared = True
        store.add_memory(shared)
        store.add_memory(_memory(store, text="agent private", owner=agent_owner("a"), axis=2))
        rows = store.get_memories(owner_predicate=human_recall_predicate())
        assert {r.text for r in rows} == {"mine", "shared by agent"}


class TestOwnerScopedMutation:
    """delete_memory/update_memory must reject ids the caller does not own."""

    def test_agent_cannot_delete_local_memory(self, store):
        store.add_memory(_memory(store, text="human", owner=LOCAL_OWNER, memory_id="m1", axis=0))
        deleted = store.delete_memory("m1", owner=agent_owner("evil"))
        assert deleted is False
        assert [m.text for m in store.get_memories(owner_predicate=LOCAL_PREDICATE)] == ["human"]

    def test_agent_cannot_delete_other_agents_memory(self, store):
        store.add_memory(
            _memory(store, text="a", owner=agent_owner("alice"), memory_id="m1", axis=0)
        )
        deleted = store.delete_memory("m1", owner=agent_owner("bob"))
        assert deleted is False
        alice = store.get_memories(owner_predicate=f"owner = '{agent_owner('alice')}'")
        assert [m.text for m in alice] == ["a"]

    def test_owner_can_delete_own_memory(self, store):
        store.add_memory(
            _memory(store, text="a", owner=agent_owner("alice"), memory_id="m1", axis=0)
        )
        assert store.delete_memory("m1", owner=agent_owner("alice")) is True
        assert store.get_memories(owner_predicate=f"owner = '{agent_owner('alice')}'") == []

    def test_agent_cannot_flip_shared_on_local_memory(self, store):
        store.add_memory(_memory(store, text="human", owner=LOCAL_OWNER, memory_id="m1", axis=0))
        updated = store.update_memory("m1", shared=True, owner=agent_owner("evil"))
        assert updated is False
        assert store.get_memories(owner_predicate=LOCAL_PREDICATE)[0].shared is False


class TestDropAllPreservesMemory:
    def test_rebuild_keeps_memory_drops_chunks(self, store):
        store.add_memory(_memory(store, text="kept", axis=0))
        store.add_chunks(
            [
                {
                    "source": "d.md",
                    "content_type": "text",
                    "chunk_type": "raw",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "doc text",
                    "chunk_index": 0,
                    "vector": _unit_vector(store._config.embedding_dim, 2),
                }
            ]
        )
        store.drop_all()
        assert store.has_chunks() is False
        assert [m.text for m in store.get_memories(owner_predicate=LOCAL_PREDICATE)] == ["kept"]


class TestRebuildEmbeddings:
    def test_no_table_returns_zero(self, store):
        assert store.rebuild_memory_embeddings(lambda texts: []) == 0

    def test_empty_table_returns_zero(self, store):
        store.add_memory(_memory(store, text="x", axis=0))
        store.delete_memory(
            store.get_memories(owner_predicate=LOCAL_PREDICATE)[0].id, owner=LOCAL_OWNER
        )
        assert store.rebuild_memory_embeddings(lambda texts: []) == 0

    def test_recreates_at_new_dim_preserving_text(self, store):
        store.add_memory(_memory(store, text="survives", axis=0))
        new_dim = 4
        store._config.embedding_dim = new_dim

        def embed(texts: list[str]) -> list[npt.NDArray[np.float32]]:
            return [np.full(new_dim, 0.5, dtype=np.float32) for _ in texts]

        count = store.rebuild_memory_embeddings(embed)
        assert count == 1
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)
        assert got[0].text == "survives"
        assert len(got[0].vector) == new_dim

    def test_same_dim_updates_vectors(self, store):
        store.add_memory(_memory(store, text="x", axis=0))
        dim = store._config.embedding_dim

        def embed(texts: list[str]) -> list[npt.NDArray[np.float32]]:
            return [_unit_array(dim, 3) for _ in texts]

        store.rebuild_memory_embeddings(embed)
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)[0]
        assert got.vector[3] == 1.0
        assert isinstance(got.vector, list)  # the row serializes as JSON, never as an ndarray

    def test_snapshot_and_embed_run_under_write_lock(self, store):
        # The read+embed must hold the lock, or a concurrent add_memory
        # committing in the embed window is erased by the unconditional drop_table.
        from lilbee.runtime.lock import _write_mutex

        store.add_memory(_memory(store, text="x", axis=0))
        dim = store._config.embedding_dim
        locked_during: list[bool] = []

        def embed(texts: list[str]) -> list[npt.NDArray[np.float32]]:
            locked_during.append(_write_mutex.locked())
            return [_unit_array(dim, 3) for _ in texts]

        store.rebuild_memory_embeddings(embed)
        assert locked_during == [True]  # embed ran while the write lock was held


_ADD_FROM_OTHER_PROCESS = """
import sys
from datetime import UTC, datetime
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.data.store import LOCAL_OWNER, MemoryKind, MemoryRow, MemorySource, Store

store = Store(cfg.model_copy(update={"lancedb_dir": Path(sys.argv[1])}))
now = datetime.now(UTC).isoformat()
print("ready", flush=True)
store.add_memory(
    MemoryRow(
        id="id-other-process",
        owner=LOCAL_OWNER,
        shared=False,
        kind=MemoryKind.FACT,
        source=MemorySource.MANUAL,
        text="from another process",
        vector=[0.0] * store._config.embedding_dim,
        created_at=now,
        updated_at=now,
    )
)
"""


# The production floor is minutes; the tests shorten it to this so an idle can pass it.
_SHORT_FLOOR = timedelta(seconds=1)
_PAST_THE_WINDOW = 2  # seconds: longer than the shortened floor


def _table_bytes(store: Store) -> int:
    """Bytes the memories table occupies on disk, old versions included."""
    table_dir = store._config.lancedb_dir / f"{MEMORIES_TABLE}.lance"
    return sum(path.stat().st_size for path in table_dir.rglob("*") if path.is_file())


class TestRebuildPrunesVersions:
    """Each rebuild replaces the table; the replaced versions must not pile up on disk."""

    @pytest.fixture(autouse=True)
    def short_floor(self, monkeypatch):
        monkeypatch.setattr(lance_helpers, "_KEEP_VERSIONS", _SHORT_FLOOR)

    @pytest.fixture()
    def stocked(self, store):
        for axis in range(40):
            store.add_memory(_memory(store, text=f"memory {axis} " + "x" * 2000, axis=axis))
        return store

    def test_rebuilds_spaced_past_the_window_keep_the_disk_flat(self, stocked, monkeypatch):
        # The production slack is 60 s; 1 s keeps the test fast.
        monkeypatch.setattr(lance_helpers, "_CLOCK_SLACK", timedelta(seconds=1))
        dim = stocked._config.embedding_dim

        def rebuild() -> int:
            time.sleep(1.2)
            stocked.rebuild_memory_embeddings(lambda texts: [_unit_array(dim, 3) for _ in texts])
            return _table_bytes(stocked)

        rebuild()
        steady = rebuild()
        for _ in range(4):
            rebuild()
        assert steady > 0
        assert _table_bytes(stocked) <= steady * 1.1

    def test_reader_opened_before_a_rebuild_still_reads_after_it(self, stocked):
        # The table sat idle past the window, so the version a rebuild replaces is older than it.
        time.sleep(_PAST_THE_WINDOW)
        reader = stocked.open_table(MEMORIES_TABLE)
        assert reader is not None
        dim = stocked._config.embedding_dim
        stocked.rebuild_memory_embeddings(lambda texts: [_unit_array(dim, 3) for _ in texts])
        assert reader.to_arrow().num_rows == 40

    def test_reader_still_reads_after_a_slow_prune(self, stocked, monkeypatch):
        # The prune starts 2 s after its clock read; the slack covers that delay.
        time.sleep(_PAST_THE_WINDOW)
        reader = stocked.open_table(MEMORIES_TABLE)
        assert reader is not None
        table_type = type(reader)
        real_optimize = table_type.optimize
        windows: list[timedelta] = []

        def slow(self, *args, **kwargs):
            windows.append(kwargs["cleanup_older_than"])
            time.sleep(2)
            return real_optimize(self, *args, **kwargs)

        monkeypatch.setattr(table_type, "optimize", slow)
        dim = stocked._config.embedding_dim
        stocked.rebuild_memory_embeddings(lambda texts: [_unit_array(dim, 3) for _ in texts])
        assert len(windows) == 1
        assert windows[0] > timedelta(seconds=_PAST_THE_WINDOW)
        assert reader.to_arrow().num_rows == 40

    def test_a_failed_prune_after_the_overwrite_keeps_the_rebuilt_rows(
        self, stocked, monkeypatch, caplog
    ):
        reader = stocked.open_table(MEMORIES_TABLE)
        assert reader is not None

        def failing(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(type(reader), "optimize", failing)
        dim = stocked._config.embedding_dim
        with caplog.at_level("WARNING", logger="lilbee.data.store.core"):
            count = stocked.rebuild_memory_embeddings(
                lambda texts: [_unit_array(dim, 3) for _ in texts]
            )
        assert count == 40
        memories = stocked.get_memories(owner_predicate=LOCAL_PREDICATE)
        assert len(memories) == 40
        assert {m.vector.index(1.0) for m in memories} == {3}
        assert any(
            MEMORIES_TABLE in record.getMessage() and "disk full" in record.getMessage()
            for record in caplog.records
        )

    def test_prune_runs_under_the_write_lock(self, stocked, monkeypatch):
        from lilbee.data.store import core
        from lilbee.runtime.lock import _write_mutex

        real_prune = core.prune_old_versions
        locked_during: list[bool] = []

        def spy(*args, **kwargs):
            locked_during.append(_write_mutex.locked())
            real_prune(*args, **kwargs)

        monkeypatch.setattr(core, "prune_old_versions", spy)
        dim = stocked._config.embedding_dim
        stocked.rebuild_memory_embeddings(lambda texts: [_unit_array(dim, 3) for _ in texts])
        assert locked_during == [True]

    def test_memory_added_by_another_process_during_a_rebuild_is_kept(self, stocked, tmp_path):
        script = tmp_path / "add_from_other_process.py"
        script.write_text(_ADD_FROM_OTHER_PROCESS, encoding="utf-8")
        dim = stocked._config.embedding_dim
        other = subprocess.Popen(
            [sys.executable, str(script), str(stocked._config.lancedb_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        blocked_during: list[bool] = []

        def embed(texts: list[str]) -> list[npt.NDArray[np.float32]]:
            assert other.stdout is not None
            assert other.stdout.readline().strip() == "ready"
            time.sleep(1.0)
            # The other process asked for the write lock; it must still be waiting for it.
            blocked_during.append(other.poll() is None)
            return [_unit_array(dim, 3) for _ in texts]

        try:
            stocked.rebuild_memory_embeddings(embed)
            _, stderr = other.communicate(timeout=120)
        finally:
            other.kill()
        assert blocked_during == [True]
        assert other.returncode == 0, stderr
        texts = {m.text for m in stocked.get_memories(owner_predicate=LOCAL_PREDICATE)}
        assert "from another process" in texts
        assert len(texts) == 41


_NOW = 1_800_000_000.0  # a fixed epoch for the prune clock
_DAY = 86400.0


class _RecordingTable:
    """A table that lists the versions it is given and records each call the prune makes."""

    def __init__(self, ages: dict[int, float]) -> None:
        # Stamps as LanceDB lists them: naive local times built from the manifest epoch.
        self._versions = [
            {"version": version, "timestamp": datetime.fromtimestamp(_NOW - age)}
            for version, age in ages.items()
        ]
        self.calls: list[tuple[str, timedelta | None]] = []

    def list_versions(self) -> list[dict[str, object]]:
        self.calls.append(("list_versions", None))
        return self._versions

    def optimize(self, *, cleanup_older_than: timedelta) -> None:
        self.calls.append(("optimize", cleanup_older_than))


class TestPruneWindow:
    """The single optimize of a prune names a window that keeps the keep version."""

    @pytest.fixture(autouse=True)
    def fixed_clock(self, monkeypatch):
        monkeypatch.setattr(lance_helpers.time, "time", lambda: _NOW)

    def _prune(self, ages: dict[int, float], keep_version: int | None) -> list[tuple]:
        table = _RecordingTable(ages)
        lance_helpers.prune_old_versions(table, keep_version=keep_version)
        return table.calls

    def test_an_eight_day_old_keep_version_sets_the_window(self):
        calls = self._prune({1: 9 * _DAY, 2: 8 * _DAY, 3: 1.0}, keep_version=2)
        window = timedelta(seconds=8 * _DAY) + lance_helpers._CLOCK_SLACK
        assert calls == [("list_versions", None), ("optimize", window)]

    def test_a_newer_version_stamped_earlier_sets_the_window(self):
        # The clock stepped back between the replaced version and the overwrite.
        calls = self._prune({1: 900.0, 2: 300.0, 3: 400.0}, keep_version=2)
        assert calls[-1] == ("optimize", timedelta(seconds=400) + lance_helpers._CLOCK_SLACK)

    def test_a_version_older_than_the_keep_version_does_not_set_the_window(self):
        calls = self._prune({1: 3 * _DAY, 2: 300.0, 3: 1.0}, keep_version=2)
        assert calls[-1] == ("optimize", timedelta(seconds=300) + lance_helpers._CLOCK_SLACK)

    def test_a_keep_version_stamped_ahead_of_the_clock_gets_the_floor_window(self):
        # The clock stepped back after the overwrite: the age is negative.
        calls = self._prune({1: -120.0, 2: -121.0}, keep_version=1)
        assert calls[-1] == ("optimize", lance_helpers._KEEP_VERSIONS)

    def test_no_keep_version_folds_at_the_floor_window_without_listing(self):
        calls = self._prune({1: 3 * _DAY, 2: 1.0}, keep_version=None)
        assert calls == [("optimize", lance_helpers._KEEP_VERSIONS)]


_KILL_DURING_SWAP = """
import os
import sys
from pathlib import Path

import lancedb.db as lancedb_db
import lancedb.table as lancedb_table
import numpy as np
import pyarrow as pa

from lilbee.core.config import cfg
from lilbee.data.store import Store

store = Store(cfg.model_copy(update={"lancedb_dir": Path(sys.argv[1])}))
real_sanitize = lancedb_db.sanitize_create_table


def killing_sanitize(*args, **kwargs):
    data, schema = real_sanitize(*args, **kwargs)
    if not isinstance(data, pa.RecordBatchReader):
        return data, schema

    def batches():
        for batch in data:
            yield batch
            os._exit(9)

    return pa.RecordBatchReader.from_batches(schema, batches()), schema


def killing_add(*args, **kwargs):
    os._exit(9)


lancedb_db.sanitize_create_table = killing_sanitize
lancedb_table.LanceTable.add = killing_add
dim = store._config.embedding_dim
store.rebuild_memory_embeddings(lambda texts: [np.ones(dim, dtype=np.float32) for _ in texts])
"""


def _vectors_by_text(store) -> dict[str, list[float]]:
    return {m.text: list(m.vector) for m in store.get_memories(owner_predicate=LOCAL_PREDICATE)}


class TestRebuildKeepsMemories:
    """A failure while the vectors are rewritten leaves every memory with its old vector."""

    @pytest.fixture()
    def kept(self, store):
        for axis, text in enumerate(["first", "second", "third"]):
            store.add_memory(_memory(store, text=text, axis=axis))
        return _vectors_by_text(store)

    def test_wrong_dimension_keeps_old_vectors(self, store, kept):
        with pytest.raises(ValueError, match="dimension 5, but the embedding_dim setting is"):
            store.rebuild_memory_embeddings(
                lambda texts: [np.ones(5, dtype=np.float32)] * len(texts)
            )
        assert _vectors_by_text(store) == kept

    def test_wrong_count_keeps_old_vectors(self, store, kept):
        dim = store._config.embedding_dim
        with pytest.raises(ValueError, match="1 vectors for 3 memories"):
            store.rebuild_memory_embeddings(lambda texts: [_unit_array(dim, 1)])
        assert _vectors_by_text(store) == kept

    def test_embedder_failure_keeps_old_vectors(self, store, kept):
        def embed(texts: list[str]) -> list[npt.NDArray[np.float32]]:
            raise RuntimeError("embedder down")

        with pytest.raises(RuntimeError, match="embedder down"):
            store.rebuild_memory_embeddings(embed)
        assert _vectors_by_text(store) == kept

    def test_failed_table_write_keeps_old_vectors(self, store, kept, monkeypatch):
        dim = store._config.embedding_dim

        def failing_create(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(type(store.get_db()), "create_table", failing_create)
        with pytest.raises(OSError, match="disk full"):
            store.rebuild_memory_embeddings(lambda texts: [_unit_array(dim, 3) for _ in texts])
        monkeypatch.undo()
        assert _vectors_by_text(store) == kept

    def test_process_killed_mid_rewrite_keeps_old_vectors(self, store, kept, tmp_path):
        script = tmp_path / "kill_during_swap.py"
        script.write_text(_KILL_DURING_SWAP, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(script), str(store._config.lancedb_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=120,
        )
        assert done.returncode == 9, done.stderr
        reopened = Store(store._config.model_copy())
        assert _vectors_by_text(reopened) == kept

    def test_successful_swap_replaces_every_vector(self, store, kept):
        dim = store._config.embedding_dim
        assert (
            store.rebuild_memory_embeddings(lambda texts: [_unit_array(dim, 7) for _ in texts]) == 3
        )
        assert set(_vectors_by_text(store)) == set(kept)
        assert all(vec[7] == 1.0 for vec in _vectors_by_text(store).values())


class TestEmbeddingCompat:
    def test_add_after_model_change_raises(self, store):
        store.add_memory(_memory(store, text="x", axis=0))  # writes _meta under current model
        store._config.embedding_model = "different/embed-model"
        with pytest.raises(EmbeddingModelMismatchError):
            store.add_memory(_memory(store, text="y", axis=1))

    def test_search_after_model_change_raises(self, store):
        store.add_memory(_memory(store, text="x", axis=0))
        store._config.embedding_model = "different/embed-model"
        with pytest.raises(EmbeddingModelMismatchError):
            store.search_memories(
                _unit_vector(store._config.embedding_dim, 0),
                owner_predicate=LOCAL_PREDICATE,
                top_k=5,
                max_distance=1.0,
            )


def test_memories_table_name_constant():
    assert MEMORIES_TABLE == "_memories"
