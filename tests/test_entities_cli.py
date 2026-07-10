"""End-to-end tests for the ``lilbee entities`` command."""

import pytest
from typer.testing import CliRunner

import lilbee.app.services as svc_mod
import lilbee.cli.commands  # noqa: F401  (registers commands on the app)
from lilbee.cli.app import app
from lilbee.core.config import cfg
from lilbee.data.store import Store
from lilbee.retrieval.entities import EntitySchema, EntityType, ExtractorKind, save_schema

runner = CliRunner()


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """A real Store on a tmp data root, pinned via --data-dir on every invoke.

    The CLI callback re-resolves the data root, so tests pass the flag
    explicitly instead of trusting pre-set singleton paths.
    """
    from tests.conftest import make_mock_services

    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(cfg)
    services = make_mock_services(store=store)
    svc_mod.set_services(services)
    yield store
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _invoke(action, tmp_root):
    return runner.invoke(app, ["entities", action, "--data-dir", str(tmp_root)])


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


def _invoke_json(action, tmp_root):
    return runner.invoke(app, ["--json", "entities", action, "--data-dir", str(tmp_root)])


class TestEntitiesCommand:
    def test_json_status_and_backfill_and_induce(self, isolated):
        import json as jsonlib
        from types import SimpleNamespace

        _index_chunks(isolated, ["part PX4471"])
        services = svc_mod.get_services()
        services.provider.chat.return_value = SimpleNamespace(
            text='{"types": [{"name": "part_number", "kind": "regex", '
            '"pattern": "PX\\\\d{4}", "description": "ids", "synonyms": []}]}'
        )
        induced = _invoke_json("induce", cfg.data_root)
        assert induced.exit_code == 0
        assert jsonlib.loads(induced.output)["action"] == "induce"
        backfilled = _invoke_json("backfill", cfg.data_root)
        assert jsonlib.loads(backfilled.output)["rows"] == 1
        status = _invoke_json("status", cfg.data_root)
        payload = jsonlib.loads(status.output)
        assert payload["rows"] == 1
        assert payload["types"] == ["part_number"]

    def test_backfill_with_spacy_and_llm_types(self, isolated):
        from types import SimpleNamespace
        from unittest import mock as umock

        _index_chunks(isolated, ["part PX4471 shipped"])
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
        services = svc_mod.get_services()
        services.provider.chat.return_value = SimpleNamespace(text="{}")
        fake_nlp = umock.MagicMock(return_value=umock.MagicMock(ents=[]))
        with (
            umock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True),
            umock.patch("lilbee.retrieval.concepts.nlp._ensure_spacy_model", return_value=fake_nlp),
        ):
            result = _invoke("backfill", cfg.data_root)
        assert result.exit_code == 0
        assert "1 entity rows" in result.output

    def test_backfill_with_nothing_indexed(self, isolated):
        save_schema(
            EntitySchema(
                types=[EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}")]
            ),
            cfg.data_dir,
        )
        result = _invoke("backfill", cfg.data_root)
        assert result.exit_code == 1
        assert "sync documents" in result.output

    def test_induce_with_emptied_table(self, isolated):
        """A chunks table whose rows were all removed samples nothing."""
        _index_chunks(isolated, ["part PX4471"])
        isolated.upsert_source("catalog.txt", "h", chunk_count=1)
        isolated.remove_documents(["catalog.txt"])
        result = _invoke("induce", cfg.data_root)
        assert result.exit_code == 1
        assert "sync documents" in result.output

    def test_status_before_anything(self, isolated):
        result = _invoke("status", cfg.data_root)
        assert result.exit_code == 0
        assert "No entity schema" in result.output

    def test_backfill_requires_schema(self, isolated):
        result = _invoke("backfill", cfg.data_root)
        assert result.exit_code == 1
        assert "induce" in result.output

    def test_backfill_extracts_over_stored_chunks(self, isolated):
        """The no-re-ingest path: rows come from the chunks table alone."""
        _index_chunks(isolated, ["part PX4471 shipped", "parts PX9001 and PX4471"])
        save_schema(
            EntitySchema(
                types=[EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}")]
            ),
            cfg.data_dir,
        )
        result = _invoke("backfill", cfg.data_root)
        assert result.exit_code == 0
        assert "3 entity rows" in result.output
        mentions, distinct = isolated.entity_value_counts("part_number")
        assert (mentions, distinct) == (3, 2)

    def test_status_after_backfill(self, isolated):
        _index_chunks(isolated, ["part PX4471"])
        save_schema(
            EntitySchema(
                types=[EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}")]
            ),
            cfg.data_dir,
        )
        _invoke("backfill", cfg.data_root)
        result = _invoke("status", cfg.data_root)
        assert result.exit_code == 0
        assert "part_number" in result.output
        assert "Extracted rows: 1" in result.output

    def test_induce_writes_reviewable_schema(self, isolated, monkeypatch):
        from types import SimpleNamespace

        _index_chunks(isolated, ["part PX4471 shipped from the depot"])
        services = svc_mod.get_services()
        services.provider.chat.return_value = SimpleNamespace(
            text='{"types": [{"name": "part_number", "kind": "regex", '
            '"pattern": "PX\\\\d{4}", "description": "ids", "synonyms": []}]}'
        )
        result = _invoke("induce", cfg.data_root)
        assert result.exit_code == 0
        assert "review and edit" in result.output
        assert (cfg.data_dir / "entity_schema.json").is_file()

    def test_induce_with_unusable_model_output(self, isolated):
        from types import SimpleNamespace

        _index_chunks(isolated, ["part PX4471"])
        svc_mod.get_services().provider.chat.return_value = SimpleNamespace(text="no json")
        result = _invoke("induce", cfg.data_root)
        assert result.exit_code == 1
        assert "nothing usable" in result.output

    def test_unknown_action(self, isolated):
        result = _invoke("frobnicate", cfg.data_root)
        assert result.exit_code == 2

    def test_induce_with_empty_index(self, isolated):
        result = _invoke("induce", cfg.data_root)
        assert result.exit_code == 1
        assert "sync documents" in result.output
