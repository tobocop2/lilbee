"""Tests for the automatic entity lifecycle run by sync."""

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from lilbee.data.store import Store
from lilbee.retrieval.entities import (
    EntitySchema,
    EntityType,
    ExtractorKind,
    load_schema,
    save_schema,
)
from lilbee.retrieval.entities.lifecycle import ensure_entities


@pytest.fixture()
def isolated(tmp_path):
    """A real Store on a tmp data root with entity extraction enabled."""
    from tests.conftest import make_mock_services

    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.entity_extraction = True
    store = Store(cfg)
    services = make_mock_services(store=store)
    svc_mod.set_services(services)
    yield store, services
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _index_chunks(store, texts, source="catalog.txt"):
    dim = cfg.embedding_dim
    store.add_chunks(
        [
            {
                "source": source,
                "content_type": "text",
                "chunk_type": "raw",
                "page_start": 1,
                "page_end": 1,
                "line_start": 0,
                "line_end": 0,
                "chunk": text,
                "chunk_index": i,
                "vector": [0.1] * dim,
            }
            for i, text in enumerate(texts)
        ]
    )


def _index_sources(store, count, *, start=0, text="part PX4471"):
    """Index *count* distinct sources so count_sources() sees a real corpus."""
    for i in range(start, start + count):
        name = f"doc{i}.txt"
        _index_chunks(store, [text], source=name)
        store.upsert_source(name, f"hash{i}", chunk_count=1)


def _part_schema():
    return EntitySchema(
        types=[EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}")]
    )


def _applied(store) -> bool:
    state = store.entity_schema_state()
    return state is not None and state["applied"]


_INDUCED_JSON = (
    '{"types": [{"name": "part_number", "kind": "regex", '
    '"pattern": "PX\\\\d{4}", "description": "ids", "synonyms": []}]}'
)

# A re-induction that proposes the known part_number type plus a new dock type.
_REINDUCED_JSON = (
    '{"types": [{"name": "part_number", "kind": "regex", '
    '"pattern": "PX\\\\d{4}", "description": "ids", "synonyms": []}, '
    '{"name": "dock", "kind": "regex", "pattern": "D-\\\\d{2}", '
    '"description": "docks", "synonyms": []}]}'
)


class TestEnsureEntities:
    def test_off_is_a_noop(self, isolated):
        _store, services = isolated
        cfg.entity_extraction = False
        ensure_entities()
        services.provider.chat.assert_not_called()

    def test_first_run_induces_and_extracts(self, isolated):
        """Enabling the setting is the whole interaction: one sync pass
        induces the schema, persists it into the index, and extracts."""
        store, services = isolated
        _index_chunks(store, ["part PX4471 shipped", "parts PX9001 and PX4471"])
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        assert store.entity_schema_state() is not None
        assert _applied(store)
        mentions, distinct = store.entity_value_counts("part_number")
        assert (mentions, distinct) == (3, 2)

    def test_nothing_indexed_defers_induction(self, isolated):
        store, services = isolated
        ensure_entities()
        services.provider.chat.assert_not_called()
        assert store.entity_schema_state() is None

    def test_unusable_induction_retries_next_sync(self, isolated):
        store, services = isolated
        _index_chunks(store, ["part PX4471"])
        services.provider.chat.return_value = SimpleNamespace(text="no json")
        ensure_entities()
        assert store.entity_schema_state() is None
        # Next sync with a working model succeeds.
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)

    def test_applied_schema_is_a_noop(self, isolated):
        store, services = isolated
        _index_chunks(store, ["part PX4471"])
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        with mock.patch("lilbee.retrieval.entities.lifecycle._full_pass") as full_pass:
            ensure_entities()
        full_pass.assert_not_called()

    def test_unreadable_persisted_schema_reinduces(self, isolated):
        """A corrupt persisted row is machine state gone wrong; the lifecycle
        replaces it with a freshly induced schema instead of failing sync."""
        store, services = isolated
        _index_chunks(store, ["part PX4471"])
        store.save_entity_schema("{not json", applied=False, source_count=1)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        assert _applied(store)
        assert store.entity_value_counts("part_number") == (1, 1)

    def test_scoped_config_governs_lifecycle(self, isolated):
        """The library API binds its config via config_scope without touching
        the process-global cfg; the lifecycle must read the scoped config or
        Lilbee(config=...) silently skips entity extraction."""
        from lilbee.core.config import config_scope

        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), store, applied=False, source_count=1)
        scoped = cfg.model_copy()
        cfg.entity_extraction = False  # global says off; the scope says on
        scoped.entity_extraction = True
        with config_scope(scoped):
            ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)

    def test_cancelled_pass_restarts_next_sync(self, isolated):
        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), store, applied=False, source_count=1)
        cancelled = threading.Event()
        cancelled.set()
        ensure_entities(cancel=cancelled)
        assert not _applied(store)
        ensure_entities()
        assert _applied(store)
        assert store.entity_value_counts("part_number") == (1, 1)

    def test_interrupted_pass_never_double_counts(self, isolated):
        """The full pass clears prior rows first, so redoing an unapplied
        schema replaces rows instead of appending duplicates."""
        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), store, applied=False, source_count=1)
        ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)
        # Simulate an interrupted pass recorded as unapplied: redo replaces.
        save_schema(_part_schema(), store, applied=False, source_count=1)
        ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)  # not doubled

    def test_spacy_and_llm_kinds_wire_their_tools(self, isolated):
        store, services = isolated
        _index_chunks(store, ["part PX4471 shipped"])
        save_schema(
            EntitySchema(
                types=[
                    EntityType(name="person", kind=ExtractorKind.SPACY, pattern="PERSON"),
                    EntityType(name="vessel", kind=ExtractorKind.LLM, description="ships"),
                    EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}"),
                ]
            ),
            store,
            applied=False,
            source_count=1,
        )
        services.provider.chat.return_value = SimpleNamespace(text="{}")
        fake_nlp = mock.MagicMock(return_value=mock.MagicMock(ents=[]))
        with (
            mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True),
            mock.patch("lilbee.retrieval.concepts.nlp.load_spacy_pipeline", return_value=fake_nlp),
        ):
            ensure_entities()
        fake_nlp.assert_called()
        services.provider.chat.assert_called()
        assert store.entity_value_counts("part_number") == (1, 1)

    def test_spacy_model_import_error_degrades(self, isolated, caplog):
        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(
            EntitySchema(
                types=[
                    EntityType(name="person", kind=ExtractorKind.SPACY, pattern="PERSON"),
                    EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}"),
                ]
            ),
            store,
            applied=False,
            source_count=1,
        )
        with (
            mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True),
            mock.patch(
                "lilbee.retrieval.concepts.nlp.load_spacy_pipeline",
                side_effect=ImportError("no model"),
            ),
            caplog.at_level("WARNING"),
        ):
            ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)  # regex still ran
        assert any("spaCy model unavailable" in r.message for r in caplog.records)

    def test_empty_chunks_table_skips_pass(self, isolated):
        store, _services = isolated
        save_schema(_part_schema(), store, applied=False, source_count=1)
        ensure_entities()
        assert not _applied(store)

    def test_emptied_chunks_table_defers_induction(self, isolated):
        """A chunks table whose rows were all removed samples nothing."""
        store, services = isolated
        _index_chunks(store, ["part PX4471"])
        store.upsert_source("catalog.txt", "h", chunk_count=1)
        store.remove_documents(["catalog.txt"])
        ensure_entities()
        services.provider.chat.assert_not_called()


class TestSchemaEvolution:
    """The taxonomy keeps up with the corpus without anyone touching it: a
    library that grows a new kind of document gains the types to answer about
    it, and one that merely grows costs a sample and no re-extraction."""

    def test_new_document_family_adds_types_and_reextracts(self, isolated):
        store, services = isolated
        _index_sources(store, 12)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        assert [t.name for t in load_schema(store).types] == ["part_number"]
        assert store.entity_value_counts("dock") == (0, 0)

        # The corpus grows past the drift threshold with documents carrying a
        # kind of identifier the first sample never saw.
        _index_sources(store, 8, start=12, text="part PX9001 near dock D-77")
        services.provider.chat.return_value = SimpleNamespace(text=_REINDUCED_JSON)
        ensure_entities()

        schema = load_schema(store)
        assert [t.name for t in schema.types] == ["part_number", "dock"]
        assert store.entity_value_counts("dock") == (8, 1)  # the new type was extracted
        assert _applied(store)
        state = store.entity_schema_state()
        assert state["source_count"] == 20  # drift baseline moved to the new size

    def test_growth_without_new_types_skips_reextraction(self, isolated):
        """Re-induction that proposes nothing new must not pay for a pass."""
        store, services = isolated
        _index_sources(store, 12)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        _index_sources(store, 8, start=12)

        with mock.patch("lilbee.retrieval.entities.lifecycle._full_pass") as full_pass:
            ensure_entities()
        full_pass.assert_not_called()
        assert [t.name for t in load_schema(store).types] == ["part_number"]
        # The new size is recorded, so the next sync does not re-induce again.
        assert store.entity_schema_state()["source_count"] == 20
        with mock.patch("lilbee.retrieval.entities.lifecycle.induce_schema") as induce:
            ensure_entities()
        induce.assert_not_called()

    def test_ordinary_growth_does_not_reinduce(self, isolated):
        """A drip-feed of documents stays under the threshold: those files are
        already extracted at ingest under the existing schema."""
        store, services = isolated
        _index_sources(store, 12)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        _index_sources(store, 3, start=12)  # 15 < 12 * 1.5

        with mock.patch("lilbee.retrieval.entities.lifecycle.induce_schema") as induce:
            ensure_entities()
        induce.assert_not_called()

    def test_small_corpus_reinduces_on_any_growth(self, isolated):
        """Under the floor a growth ratio is meaningless, and the first sample
        necessarily saw very little, so any new document revisits the schema."""
        store, services = isolated
        _index_sources(store, 2)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        _index_sources(store, 1, start=2, text="dock D-77")
        services.provider.chat.return_value = SimpleNamespace(text=_REINDUCED_JSON)
        ensure_entities()
        assert [t.name for t in load_schema(store).types] == ["part_number", "dock"]

    def test_shrinking_corpus_never_reinduces(self, isolated):
        store, services = isolated
        _index_sources(store, 12)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        store.remove_documents(["doc0.txt", "doc1.txt"])
        with mock.patch("lilbee.retrieval.entities.lifecycle.induce_schema") as induce:
            ensure_entities()
        induce.assert_not_called()

    def test_unusable_reinduction_leaves_the_schema_intact(self, isolated):
        """A failed re-induction must not lose the working taxonomy, and must
        not record the new size (so the next sync retries)."""
        store, services = isolated
        _index_sources(store, 12)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        _index_sources(store, 8, start=12)
        services.provider.chat.return_value = SimpleNamespace(text="no json")
        ensure_entities()
        assert [t.name for t in load_schema(store).types] == ["part_number"]
        assert store.entity_schema_state()["source_count"] == 12  # retry next sync

    def test_reinduction_ignores_renames_of_known_extractors(self, isolated):
        """A model that renames a known pattern must not double the taxonomy:
        the type is the extractor, not the label it was given this time."""
        store, services = isolated
        _index_sources(store, 12)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        _index_sources(store, 8, start=12)
        renamed = _INDUCED_JSON.replace('"part_number"', '"catalog_id"')
        services.provider.chat.return_value = SimpleNamespace(text=renamed)
        ensure_entities()
        assert [t.name for t in load_schema(store).types] == ["part_number"]

    def test_cancelled_reextraction_restarts_next_sync(self, isolated):
        """An evolved schema whose pass is interrupted stays unapplied, so the
        next sync completes it rather than leaving the new type unextracted."""
        store, services = isolated
        _index_sources(store, 12)
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        _index_sources(store, 8, start=12, text="part PX9001 near dock D-77")
        services.provider.chat.return_value = SimpleNamespace(text=_REINDUCED_JSON)
        cancelled = threading.Event()
        cancelled.set()
        ensure_entities(cancel=cancelled)
        assert not _applied(store)
        assert [t.name for t in load_schema(store).types] == ["part_number", "dock"]

        ensure_entities()  # no re-induction needed; the pass just completes
        assert _applied(store)
        assert store.entity_value_counts("dock") == (8, 1)


class TestMergeSchemas:
    def test_appends_only_unseen_extractors(self):
        from lilbee.retrieval.entities import merge_schemas

        existing = _part_schema()
        induced = EntitySchema(
            types=[
                EntityType(name="renamed", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}"),
                EntityType(name="dock", kind=ExtractorKind.REGEX, pattern=r"D-\d{2}"),
            ]
        )
        merged = merge_schemas(existing, induced)
        assert [t.name for t in merged.types] == ["part_number", "dock"]

    def test_name_collision_is_not_re_added(self):
        """Same name, different pattern: keep the established one rather than
        carry two types answering to the same noun."""
        from lilbee.retrieval.entities import merge_schemas

        induced = EntitySchema(
            types=[EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"ZZ\d{4}")]
        )
        merged = merge_schemas(_part_schema(), induced)
        assert [(t.name, t.pattern) for t in merged.types] == [("part_number", r"PX\d{4}")]

    def test_nothing_new_returns_the_existing_schema(self):
        from lilbee.retrieval.entities import merge_schemas

        existing = _part_schema()
        assert merge_schemas(existing, _part_schema()) is existing

    def test_llm_kinds_dedupe_by_description(self):
        from lilbee.retrieval.entities import merge_schemas

        existing = EntitySchema(
            types=[EntityType(name="vessel", kind=ExtractorKind.LLM, description="Ships")]
        )
        induced = EntitySchema(
            types=[EntityType(name="boat", kind=ExtractorKind.LLM, description="ships")]
        )
        assert merge_schemas(existing, induced) is existing


class TestCorpusDrifted:
    def test_growth_below_the_factor_is_not_drift(self):
        from lilbee.retrieval.entities.lifecycle import _corpus_drifted

        assert not _corpus_drifted(100, 149)
        assert _corpus_drifted(100, 150)

    def test_small_corpora_drift_on_any_growth(self):
        from lilbee.retrieval.entities.lifecycle import _corpus_drifted

        assert _corpus_drifted(2, 3)
        assert not _corpus_drifted(2, 2)

    def test_shrinking_is_never_drift(self):
        from lilbee.retrieval.entities.lifecycle import _corpus_drifted

        assert not _corpus_drifted(100, 40)


class TestStatusEntities:
    def test_status_reports_types_and_rows(self, isolated):
        from lilbee.app.status import gather_status

        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), store, applied=False, source_count=1)
        ensure_entities()
        result = gather_status()
        assert result.entities is not None
        assert result.entities.types == ["part_number"]
        assert result.entities.rows == 1

    def test_status_omits_section_when_off(self, isolated):
        from lilbee.app.status import gather_status

        cfg.entity_extraction = False
        result = gather_status()
        assert result.entities is None
