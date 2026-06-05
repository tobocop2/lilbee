"""Tests for the ``_memories`` table: CRUD, dedup, eviction, recall, and rebuild."""

from datetime import UTC, datetime

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
    is_agent_owner,
)

LOCAL_PREDICATE = f"owner = '{LOCAL_OWNER}'"


class TestOwnerHelpers:
    def test_agent_owner_round_trips_through_is_agent_owner(self):
        owner = agent_owner("opencode")
        assert owner == "agent:opencode"
        assert is_agent_owner(owner) is True

    def test_local_owner_is_not_an_agent(self):
        assert is_agent_owner(LOCAL_OWNER) is False


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


def _memory(
    store: Store,
    *,
    text: str = "a memory",
    kind: MemoryKind = MemoryKind.FACT,
    owner: str = LOCAL_OWNER,
    confirmed: bool = True,
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
        confirmed=confirmed,
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

    def test_get_confirmed_only(self, store):
        store.add_memory(_memory(store, text="confirmed", confirmed=True, axis=0))
        store.add_memory(_memory(store, text="pending", confirmed=False, axis=1))
        confirmed = store.get_memories(owner_predicate=LOCAL_PREDICATE, confirmed_only=True)
        assert [m.text for m in confirmed] == ["confirmed"]

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

    def test_excludes_unconfirmed_and_preferences(self, store):
        store.add_memory(_memory(store, text="unconfirmed", confirmed=False, axis=0))
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
    def test_update_toggles_shared_and_confirmed(self, store):
        store.add_memory(_memory(store, text="x", confirmed=False, memory_id="u1", axis=0))
        assert store.update_memory("u1", shared=True, confirmed=True) is True
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)[0]
        assert got.shared is True
        assert got.confirmed is True

    def test_update_missing_returns_false(self, store):
        store.add_memory(_memory(store, text="x", axis=0))
        assert store.update_memory("nope") is False

    def test_update_no_table_returns_false(self, store):
        assert store.update_memory("nope") is False

    def test_delete_removes(self, store):
        store.add_memory(_memory(store, text="x", memory_id="d1", axis=0))
        store.delete_memory("d1")
        assert store.get_memories(owner_predicate=LOCAL_PREDICATE) == []


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
        store.delete_memory(store.get_memories(owner_predicate=LOCAL_PREDICATE)[0].id)
        assert store.rebuild_memory_embeddings(lambda texts: []) == 0

    def test_recreates_at_new_dim_preserving_text(self, store):
        store.add_memory(_memory(store, text="survives", axis=0))
        new_dim = 4
        store._config.embedding_dim = new_dim

        def embed(texts: list[str]) -> list[list[float]]:
            return [[0.5] * new_dim for _ in texts]

        count = store.rebuild_memory_embeddings(embed)
        assert count == 1
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)
        assert got[0].text == "survives"
        assert len(got[0].vector) == new_dim

    def test_same_dim_updates_vectors(self, store):
        store.add_memory(_memory(store, text="x", axis=0))
        dim = store._config.embedding_dim

        def embed(texts: list[str]) -> list[list[float]]:
            return [_unit_vector(dim, 3) for _ in texts]

        store.rebuild_memory_embeddings(embed)
        got = store.get_memories(owner_predicate=LOCAL_PREDICATE)[0]
        assert got.vector[3] == 1.0


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
