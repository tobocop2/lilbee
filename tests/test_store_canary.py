"""Canary vectors: the store stops search and writes when the embedder gives other vectors."""

import logging

import lancedb
import numpy as np
import pyarrow as pa
import pytest

from lilbee.core.config import META_TABLE, cfg
from lilbee.core.health_warnings import WarningCode
from lilbee.data.store import (
    EmbeddingEngineChangedError,
    EmbeddingModelMismatchError,
    MemoryKind,
    MemoryRow,
    Store,
)
from lilbee.data.store.canary import (
    CANARY_MAX_DISTANCE,
    CANARY_TEXTS,
    CanaryProbe,
    CanaryState,
    canary_distance,
)
from lilbee.data.store.schema import _meta_schema
from lilbee.data.store.types import ChunkWrite
from lilbee.providers.base import ProviderError
from lilbee.runtime.lock import write_lock
from tests.test_store import _make_records

DIM = 16
T = CANARY_MAX_DISTANCE


class Engine:
    """An embedder whose canary vectors sit a set cosine distance from a fixed space."""

    def __init__(self, trigger: str = "A", distance: float = 0.0) -> None:
        self.trigger_name: str | None = trigger
        self.distance = distance
        self.moved: set[int] | None = None
        self.calls = 0
        self.failure: Exception | None = None

    def trigger(self) -> str | None:
        return self.trigger_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        out = []
        for i, _ in enumerate(texts):
            rng = np.random.default_rng(i)
            base = rng.normal(size=DIM)
            base /= np.linalg.norm(base)
            other = rng.normal(size=DIM)
            other -= other.dot(base) * base
            other /= np.linalg.norm(other)
            d = self.distance if self.moved is None or i in self.moved else 0.0
            cos = 1.0 - d
            out.append((cos * base + np.sqrt(1.0 - cos * cos) * other).tolist())
        return out

    def probe(self) -> CanaryProbe:
        return CanaryProbe(embed=self.embed, trigger=self.trigger)


@pytest.fixture()
def test_config(tmp_path):
    return cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb_canary"})


@pytest.fixture()
def engine():
    return Engine()


@pytest.fixture()
def store(test_config, engine):
    return Store(test_config, canary_probe=engine.probe())


def _records_of(source):
    return [{**record, "source": source} for record in _make_records()]


def _reopen(test_config, engine):
    """A new process over the same library: a new store, no verdict in memory."""
    return Store(test_config, canary_probe=engine.probe())


def _memory(config, axis=0):
    vec = [0.0] * config.embedding_dim
    vec[axis] = 1.0
    return MemoryRow(
        id=f"m{axis}",
        owner="local",
        shared=False,
        kind=MemoryKind.FACT,
        source="manual",
        text="fact",
        vector=vec,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


class TestCanaryDistance:
    def test_identical_vectors_are_at_distance_zero(self):
        vectors = Engine().embed(list(CANARY_TEXTS))

        assert canary_distance(vectors, vectors) == pytest.approx(0.0, abs=1e-9)

    def test_the_largest_distance_of_any_canary_decides(self):
        base = Engine()
        moved = Engine(distance=0.2)
        moved.moved = {4}

        distance = canary_distance(base.embed(list(CANARY_TEXTS)), moved.embed(list(CANARY_TEXTS)))

        assert distance == pytest.approx(0.2, abs=1e-6)

    def test_there_are_twelve_canaries(self):
        assert len(CANARY_TEXTS) == 12
        assert len(set(CANARY_TEXTS)) == 12


class TestBaseline:
    def test_the_first_write_stores_a_verified_baseline(self, store, engine):
        store.add_chunks(_make_records())

        state = store.canary_state()

        assert state is not None
        assert state.verified is True
        assert state.trigger == "A"
        assert len(state.vectors) == len(CANARY_TEXTS)

    def test_the_same_embedder_embeds_no_canary(self, store, test_config, engine):
        store.add_chunks(_make_records())
        engine.calls = 0

        assert store.canary_mismatch() is None
        assert _reopen(test_config, engine).canary_mismatch() is None
        assert engine.calls == 0

    def test_a_store_with_no_probe_never_checks(self, test_config):
        plain = Store(test_config)
        plain.add_chunks(_make_records())

        assert plain.canary_state() is None
        assert plain.canary_mismatch() is None

    def test_an_embedder_with_no_trigger_is_not_checked(self, store, engine):
        store.add_chunks(_make_records())
        engine.trigger_name = None
        engine.calls = 0

        assert store.canary_mismatch() is None
        assert engine.calls == 0

    def test_a_baseline_of_another_length_is_no_baseline(self, store, test_config):
        store.add_chunks(_make_records())
        with write_lock():
            store._write_meta_unlocked(
                embedding_model=test_config.embedding_model,
                embedding_dim=test_config.embedding_dim,
                canary=CanaryState([[1.0, 0.0]], "A", True),
            )

        assert store.canary_state() is None

    def test_a_meta_rewrite_keeps_the_baseline(self, store, test_config):
        store.add_chunks(_make_records())
        before = store.canary_state()

        with write_lock():
            store._write_meta_unlocked(
                embedding_model=test_config.embedding_model,
                embedding_dim=test_config.embedding_dim,
            )

        assert store.canary_state() == before

    def test_an_embedder_that_is_down_at_the_first_write_leaves_no_baseline(
        self, test_config, engine
    ):
        engine.failure = ProviderError("down")
        down = _reopen(test_config, engine)

        down.add_chunks(_make_records())

        assert down.get_meta() is not None
        assert down.canary_state() is None


class TestChangedEmbedder:
    def test_a_far_embedder_stops_search(self, store, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 2 * T

        with pytest.raises(EmbeddingEngineChangedError, match="different vectors"):
            store.search([0.1] * store._config.embedding_dim)

    def test_one_canary_past_the_threshold_is_enough(self, store, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance, engine.moved = "B", 4 * T, {7}

        assert store.canary_mismatch() is not None

    @pytest.mark.parametrize("factor", [0.0, 0.5, 0.98])
    def test_a_near_embedder_passes(self, store, engine, factor):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", factor * T

        assert store.canary_mismatch() is None

    @pytest.mark.parametrize("factor", [1.02, 1.5, 4.0])
    def test_a_far_embedder_stops(self, store, engine, factor):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", factor * T

        assert store.canary_mismatch() is not None

    def test_a_pass_is_remembered_by_the_next_process(self, store, test_config, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 0.5 * T
        assert store.canary_mismatch() is None
        engine.calls = 0

        assert _reopen(test_config, engine).canary_mismatch() is None
        assert engine.calls == 0

    def test_a_pass_keeps_the_first_baseline(self, store, test_config, engine):
        """B passes at 0.9T and C sits 1.1T from A: C stops, though it is near B."""
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 0.9 * T
        assert store.canary_mismatch() is None

        engine.trigger_name, engine.distance = "C", 1.1 * T

        assert _reopen(test_config, engine).canary_mismatch() is not None

    def test_the_verdict_is_taken_once_per_embedder(self, store, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 2 * T
        first = store.canary_mismatch()
        engine.calls = 0

        assert store.canary_mismatch() is first
        assert engine.calls == 0

    def test_a_restart_under_the_stopped_embedder_checks_again(self, store, test_config, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 2 * T
        assert store.canary_mismatch() is not None
        engine.calls = 0

        assert _reopen(test_config, engine).canary_mismatch() is not None
        assert engine.calls == 1

    def test_switching_back_serves_with_no_check(self, store, test_config, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 2 * T
        assert store.canary_mismatch() is not None
        engine.trigger_name, engine.distance = "A", 0.0
        engine.calls = 0

        assert _reopen(test_config, engine).canary_mismatch() is None
        assert engine.calls == 0

    def test_an_embedder_that_is_down_during_the_check_is_not_a_stop(self, store, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 2 * T
        engine.failure = OSError("refused")

        assert store.canary_mismatch() is None

        engine.failure = None
        assert store.canary_mismatch() is not None

    def test_a_rebuild_under_the_new_embedder_makes_it_the_baseline(
        self, store, test_config, engine
    ):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 2 * T
        assert store.canary_mismatch() is not None

        store.drop_all()
        rebuilt = _reopen(test_config, engine)
        rebuilt.add_chunks(_make_records())
        engine.calls = 0

        assert rebuilt.canary_mismatch() is None
        assert _reopen(test_config, engine).canary_mismatch() is None
        assert engine.calls == 0

    def test_the_stop_is_an_embedding_mismatch_that_names_no_adopt_fix(self, store, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 2 * T

        mismatch = store.embedding_mismatch()

        assert isinstance(mismatch, EmbeddingModelMismatchError)
        assert mismatch.dims_match is False
        assert mismatch.describe().adoptable is False
        assert "no verified baseline" not in str(mismatch)

    def test_the_embedder_name_check_comes_first(self, store, test_config, engine):
        store.add_chunks(_make_records())
        test_config.embedding_model = "acme/other-GGUF/other.gguf"
        engine.trigger_name, engine.distance = "B", 2 * T
        engine.calls = 0

        mismatch = store.embedding_mismatch()

        assert not isinstance(mismatch, EmbeddingEngineChangedError)
        assert engine.calls == 0


class TestGates:
    @pytest.fixture()
    def stopped(self, store, engine):
        store.add_chunks(_make_records())
        before = store.open_table("chunks").count_rows()
        engine.trigger_name, engine.distance = "B", 2 * T
        return store, before

    def test_add_chunks_stops(self, stopped):
        store, before = stopped

        with pytest.raises(EmbeddingEngineChangedError):
            store.add_chunks(_make_records())

        assert store.open_table("chunks").count_rows() == before

    def test_replace_chunks_stops(self, stopped):
        store, before = stopped

        with pytest.raises(EmbeddingEngineChangedError):
            store.replace_chunks("doc0.md", "raw", _make_records())

        assert store.open_table("chunks").count_rows() == before

    def test_a_batched_write_stops(self, stopped):
        store, before = stopped
        item = ChunkWrite("new.md", "hash", _records_of("new.md"), needs_cleanup=False)

        with pytest.raises(EmbeddingEngineChangedError):
            store.write_chunks_batch([item])

        assert store.open_table("chunks").count_rows() == before

    def test_a_memory_write_stops(self, stopped):
        store, _ = stopped

        with pytest.raises(EmbeddingEngineChangedError):
            store.add_memory(_memory(store._config))

        assert store.open_table("_memories") is None

    def test_memory_search_stops(self, store, engine):
        store.add_memory(_memory(store._config))
        engine.trigger_name, engine.distance = "B", 2 * T

        with pytest.raises(EmbeddingEngineChangedError):
            store.search_memories(
                _memory(store._config).vector,
                owner_predicate="owner = 'local'",
                top_k=3,
                max_distance=1.0,
            )

    def test_the_first_memory_write_stores_a_verified_baseline(self, store):
        store.add_memory(_memory(store._config))

        state = store.canary_state()

        assert state is not None
        assert state.verified is True

    def test_the_first_batched_write_stores_a_verified_baseline(self, store):
        store.write_chunks_batch([ChunkWrite("a.md", "h", _records_of("a.md"), needs_cleanup=False)])

        state = store.canary_state()

        assert state is not None
        assert state.verified is True


def _make_v3_library(config):
    """A library the schema 3 code wrote: its meta row holds no canary columns."""
    Store(config).add_chunks(_make_records())
    legacy = pa.schema([_meta_schema().field(n) for n in _meta_schema().names[:4]])
    row = {
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "schema_version": 3,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    db = lancedb.connect(str(config.lancedb_dir))
    db.drop_table(META_TABLE)
    db.create_table(META_TABLE, pa.Table.from_pylist([row], schema=legacy))


class TestOldIndex:
    def test_the_first_open_stores_an_unverified_baseline_and_does_not_stop(
        self, test_config, engine, caplog
    ):
        _make_v3_library(test_config)
        old = _reopen(test_config, engine)

        with caplog.at_level(logging.WARNING):
            assert old.search([0.1] * test_config.embedding_dim) is not None

        state = old.canary_state()
        assert state is not None
        assert state.verified is False
        assert old.get_meta()["schema_version"] == 4
        assert "canary check starts" in caplog.text

    def test_the_second_process_embeds_no_canary(self, test_config, engine):
        _make_v3_library(test_config)
        assert _reopen(test_config, engine).canary_mismatch() is None
        engine.calls = 0

        assert _reopen(test_config, engine).canary_mismatch() is None
        assert engine.calls == 0

    def test_a_later_swap_stops_and_says_there_is_no_verified_baseline(
        self, test_config, engine
    ):
        _make_v3_library(test_config)
        assert _reopen(test_config, engine).canary_mismatch() is None
        engine.trigger_name, engine.distance = "B", 2 * T

        mismatch = _reopen(test_config, engine).canary_mismatch()

        assert mismatch is not None
        assert "no verified baseline" in str(mismatch)

    def test_an_embedder_that_is_down_writes_no_baseline(self, test_config, engine):
        _make_v3_library(test_config)
        engine.failure = ProviderError("down")

        assert _reopen(test_config, engine).canary_mismatch() is None
        assert Store(test_config).get_meta()["schema_version"] == 3

    def test_an_index_with_no_meta_row_is_not_checked(self, store, engine):
        engine.calls = 0

        assert store.canary_mismatch() is None
        assert engine.calls == 0


class TestHealthWarning:
    def test_an_engine_change_is_a_health_warning_with_the_stop_message(self, store, engine):
        store.add_chunks(_make_records())
        engine.trigger_name, engine.distance = "B", 2 * T

        warnings = store.health_warnings()

        stale = [w for w in warnings if w.code is WarningCode.INDEX_EMBEDDING_MISMATCH]
        assert len(stale) == 1
        assert "different vectors" in stale[0].message
        assert "lilbee rebuild" in stale[0].remedy

    def test_no_change_is_no_warning(self, store):
        store.add_chunks(_make_records())

        assert not [
            w for w in store.health_warnings() if w.code is WarningCode.INDEX_EMBEDDING_MISMATCH
        ]


class TestEngineChangedMessage:
    def test_the_message_names_the_fix(self):
        error = EmbeddingEngineChangedError(model="m", dim=8, distance=0.123, verified=True)

        assert "0.123" in str(error)
        assert "lilbee rebuild" in str(error)
        assert "earlier engine" in str(error)
        assert "verified baseline" not in str(error)
