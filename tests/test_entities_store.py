"""Tests for the entities table: writes, scans, replacement, compatibility."""

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import Store


@pytest.fixture()
def store(tmp_path):
    config = cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb_test"})
    return Store(config)


def _entity(entity, type_, value, source, chunk_index=0, page=1):
    return {
        "entity": entity,
        "type": type_,
        "normalized_value": value,
        "source": source,
        "page": page,
        "chunk_index": chunk_index,
        "confidence": 1.0,
    }


def _chunk_record(source, idx, text, dim):
    return {
        "source": source,
        "content_type": "text",
        "chunk_type": "raw",
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "chunk": text,
        "chunk_index": idx,
        "vector": [0.1] * dim,
    }


class TestEntityWritesAndScans:
    def test_value_counts(self, store):
        store.add_entities(
            [
                _entity("PX4471", "part_number", "px4471", "a.txt", 0),
                _entity("PX4471", "part_number", "px4471", "b.txt", 0),
                _entity("PX9001", "part_number", "px9001", "b.txt", 1),
                _entity("Fresno", "depot", "fresno", "a.txt", 0),
            ]
        )
        mentions, distinct = store.entity_value_counts("part_number")
        assert (mentions, distinct) == (3, 2)
        assert store.entity_value_counts("vessel") == (0, 0)

    def test_empty_write_is_a_noop(self, store):
        assert store.add_entities([]) == 0

    def test_association_scan_skips_unrelated_types(self, store):
        store.add_entities(
            [
                _entity("S1", "shipment", "s1", "a.txt", 0),
                _entity("PX4471", "part_number", "px4471", "a.txt", 0),
                _entity("Fresno", "depot", "fresno", "a.txt", 0),
            ]
        )
        counts = store.entity_association_counts("shipment", grouped_by="part_number")
        assert counts == {"px4471": 1}

    def test_association_counts_by_chunk_cooccurrence(self, store):
        store.add_entities(
            [
                # chunk a.txt#0: shipment S1 with part PX4471
                _entity("S1", "shipment", "s1", "a.txt", 0),
                _entity("PX4471", "part_number", "px4471", "a.txt", 0),
                # chunk a.txt#1: shipments S2, S3 with part PX4471
                _entity("S2", "shipment", "s2", "a.txt", 1),
                _entity("S3", "shipment", "s3", "a.txt", 1),
                _entity("PX4471", "part_number", "px4471", "a.txt", 1),
                # chunk b.txt#0: shipment S1 again with part PX9001
                _entity("S1", "shipment", "s1", "b.txt", 0),
                _entity("PX9001", "part_number", "px9001", "b.txt", 0),
            ]
        )
        counts = store.entity_association_counts("shipment", grouped_by="part_number")
        assert counts == {"px4471": 3, "px9001": 1}

    def test_source_replacement_deletes_entity_rows(self, store):
        """Removing a source must delete its entity rows like its chunks."""
        dim = store._config.embedding_dim
        store.add_chunks([_chunk_record("a.txt", 0, "part PX4471", dim)])
        store.upsert_source("a.txt", "hash-a", chunk_count=1)
        store.add_entities(
            [
                _entity("PX4471", "part_number", "px4471", "a.txt", 0),
                _entity("PX9001", "part_number", "px9001", "b.txt", 0),
            ]
        )
        result = store.remove_documents(["a.txt"])
        assert result.removed == ["a.txt"]
        # a.txt's entity rows are gone; the other source's row survives.
        mentions, distinct = store.entity_value_counts("part_number")
        assert (mentions, distinct) == (1, 1)


class TestCompatibility:
    def test_scans_are_empty_on_store_without_entities_table(self, store):
        """A pre-entities store (no table) reads as nothing extracted."""
        assert store.entity_value_counts("anything") == (0, 0)
        assert store.entity_association_counts("a", grouped_by="b") == {}

    def test_existing_tables_untouched_by_entity_writes(self, store):
        dim = store._config.embedding_dim
        store.add_chunks([_chunk_record("a.txt", 0, "hello world", dim)])
        before = store.count_chunks()
        store.add_entities([_entity("X1", "part_number", "x1", "a.txt", 0)])
        assert store.count_chunks() == before
        results = store.search([0.1] * dim, top_k=1)
        assert results and results[0].source == "a.txt"
