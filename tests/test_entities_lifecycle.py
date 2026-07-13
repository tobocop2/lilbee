"""Tests for the automatic entity lifecycle run by sync."""

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from lilbee.data.store import Store
from lilbee.retrieval.entities import EntitySchema, EntityType, ExtractorKind, save_schema
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


def _index_chunks(store, texts):
    dim = cfg.embedding_dim
    store.add_chunks(
        [
            {
                "source": "catalog.txt",
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


def _part_schema():
    return EntitySchema(
        types=[EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}")]
    )


def _applied(store) -> bool:
    state = store.entity_schema_state()
    return state is not None and state[1]


_INDUCED_JSON = (
    '{"types": [{"name": "part_number", "kind": "regex", '
    '"pattern": "PX\\\\d{4}", "description": "ids", "synonyms": []}]}'
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
        store.save_entity_schema("{not json", applied=False)
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
        save_schema(_part_schema(), store)
        scoped = cfg.model_copy()
        cfg.entity_extraction = False  # global says off; the scope says on
        scoped.entity_extraction = True
        with config_scope(scoped):
            ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)

    def test_cancelled_pass_restarts_next_sync(self, isolated):
        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), store)
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
        save_schema(_part_schema(), store)
        ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)
        # Simulate an interrupted pass recorded as unapplied: redo replaces.
        save_schema(_part_schema(), store, applied=False)
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
        )
        services.provider.chat.return_value = SimpleNamespace(text="{}")
        fake_nlp = mock.MagicMock(return_value=mock.MagicMock(ents=[]))
        with (
            mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True),
            mock.patch("lilbee.retrieval.concepts.nlp._ensure_spacy_model", return_value=fake_nlp),
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
        )
        with (
            mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True),
            mock.patch(
                "lilbee.retrieval.concepts.nlp._ensure_spacy_model",
                side_effect=ImportError("no model"),
            ),
            caplog.at_level("WARNING"),
        ):
            ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)  # regex still ran
        assert any("spaCy model unavailable" in r.message for r in caplog.records)

    def test_empty_chunks_table_skips_pass(self, isolated):
        store, _services = isolated
        save_schema(_part_schema(), store)
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


class TestStatusEntities:
    def test_status_reports_types_and_rows(self, isolated):
        from lilbee.app.status import gather_status

        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), store)
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
