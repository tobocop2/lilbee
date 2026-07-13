"""Tests for the automatic entity lifecycle run by sync."""

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from lilbee.data.store import Store
from lilbee.retrieval.entities import EntitySchema, EntityType, ExtractorKind, save_schema
from lilbee.retrieval.entities.lifecycle import _APPLIED_MARKER, ensure_entities


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
        induces the schema, saves it, and extracts across the index."""
        store, services = isolated
        _index_chunks(store, ["part PX4471 shipped", "parts PX9001 and PX4471"])
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        assert (cfg.data_dir / "entity_schema.json").is_file()
        assert (cfg.data_dir / _APPLIED_MARKER).is_file()
        mentions, distinct = store.entity_value_counts("part_number")
        assert (mentions, distinct) == (3, 2)

    def test_nothing_indexed_defers_induction(self, isolated):
        _store, services = isolated
        ensure_entities()
        services.provider.chat.assert_not_called()
        assert not (cfg.data_dir / "entity_schema.json").is_file()

    def test_unusable_induction_retries_next_sync(self, isolated):
        store, services = isolated
        _index_chunks(store, ["part PX4471"])
        services.provider.chat.return_value = SimpleNamespace(text="no json")
        ensure_entities()
        assert not (cfg.data_dir / "entity_schema.json").is_file()
        # Next sync with a working model succeeds.
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)

    def test_up_to_date_schema_is_a_noop(self, isolated):
        store, services = isolated
        _index_chunks(store, ["part PX4471"])
        services.provider.chat.return_value = SimpleNamespace(text=_INDUCED_JSON)
        ensure_entities()
        with mock.patch("lilbee.retrieval.entities.lifecycle._full_pass") as full_pass:
            ensure_entities()
        full_pass.assert_not_called()

    def test_edited_schema_reextracts_without_double_counting(self, isolated):
        """The manage-after-the-fact path: edit entity_schema.json, the next
        sync detects the digest change and re-runs a replace-semantics pass."""
        store, _services = isolated
        _index_chunks(store, ["part PX4471 near dock D-77"])
        save_schema(_part_schema(), cfg.data_dir)
        ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)
        # Edit: add a second type. Digest changes; next sync re-applies.
        save_schema(
            EntitySchema(
                types=[
                    EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}"),
                    EntityType(name="dock", kind=ExtractorKind.REGEX, pattern=r"D-\d{2}"),
                ]
            ),
            cfg.data_dir,
        )
        ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)  # not doubled
        assert store.entity_value_counts("dock") == (1, 1)

    def test_scoped_config_governs_lifecycle(self, isolated):
        """The library API binds its config via config_scope without touching
        the process-global cfg; the lifecycle must read the scoped config or
        Lilbee(config=...) silently skips entity extraction."""
        from lilbee.core.config import config_scope

        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), cfg.data_dir)
        scoped = cfg.model_copy()
        cfg.entity_extraction = False  # global says off; the scope says on
        scoped.entity_extraction = True
        with config_scope(scoped):
            ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)

    def test_cancelled_pass_restarts_next_sync(self, isolated):
        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), cfg.data_dir)
        cancelled = threading.Event()
        cancelled.set()
        ensure_entities(cancel=cancelled)
        assert not (cfg.data_dir / _APPLIED_MARKER).is_file()
        ensure_entities()
        assert (cfg.data_dir / _APPLIED_MARKER).is_file()
        assert store.entity_value_counts("part_number") == (1, 1)

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
            cfg.data_dir,
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
            cfg.data_dir,
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
        _store, _services = isolated
        save_schema(_part_schema(), cfg.data_dir)
        ensure_entities()
        assert not (cfg.data_dir / _APPLIED_MARKER).is_file()

    def test_emptied_chunks_table_defers_induction(self, isolated):
        """A chunks table whose rows were all removed samples nothing."""
        store, services = isolated
        _index_chunks(store, ["part PX4471"])
        store.upsert_source("catalog.txt", "h", chunk_count=1)
        store.remove_documents(["catalog.txt"])
        ensure_entities()
        services.provider.chat.assert_not_called()

    def test_blank_applied_marker_reads_as_unapplied(self, isolated):
        """A truncated marker file must trigger a (safe, idempotent) re-pass."""
        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), cfg.data_dir)
        (cfg.data_dir / _APPLIED_MARKER).write_text("")
        ensure_entities()
        assert store.entity_value_counts("part_number") == (1, 1)
        assert (cfg.data_dir / _APPLIED_MARKER).read_text().strip()

    def test_missing_schema_digest_never_records_marker(self, isolated):
        """_record_applied(None) is the no-op guard for a vanished schema file."""
        from lilbee.retrieval.entities.lifecycle import _record_applied, _schema_digest

        assert _schema_digest() is None
        _record_applied(None)
        assert not (cfg.data_dir / _APPLIED_MARKER).is_file()


class TestStatusEntities:
    def test_status_reports_types_and_rows(self, isolated):
        from lilbee.app.status import gather_status

        store, _services = isolated
        _index_chunks(store, ["part PX4471"])
        save_schema(_part_schema(), cfg.data_dir)
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
