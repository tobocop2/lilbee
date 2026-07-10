"""Tests for the entity-extraction ingest stage and flush."""

import asyncio
from unittest import mock

import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from lilbee.data.ingest.pipeline import _build_entity_records, _flush_entity_rows
from lilbee.data.ingest.types import _IngestResult
from lilbee.retrieval.entities import EntitySchema, EntityType, ExtractorKind, save_schema


@pytest.fixture()
def isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.entity_extraction = False
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def mock_svc(isolated_cfg):
    from tests.conftest import make_mock_services

    services = make_mock_services()
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


def _records():
    return [
        {
            "source": "a.txt",
            "content_type": "text",
            "chunk_type": "raw",
            "page_start": 1,
            "page_end": 1,
            "line_start": 0,
            "line_end": 0,
            "chunk": "parts PX4471 and PX9001",
            "chunk_index": 0,
            "vector": [0.1],
        }
    ]


def _part_schema():
    return EntitySchema(
        types=[EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}")]
    )


class TestBuildEntityRecords:
    def test_gate_off_is_none(self, mock_svc):
        assert asyncio.run(_build_entity_records(_records(), "a.txt")) is None

    def test_no_schema_is_none(self, mock_svc):
        cfg.entity_extraction = True
        assert asyncio.run(_build_entity_records(_records(), "a.txt")) is None

    def test_extracts_rows_with_reviewed_schema(self, mock_svc):
        cfg.entity_extraction = True
        save_schema(_part_schema(), cfg.data_dir)
        rows = asyncio.run(_build_entity_records(_records(), "a.txt"))
        assert rows is not None
        assert {r["normalized_value"] for r in rows} == {"px4471", "px9001"}
        assert all(r["source"] == "a.txt" for r in rows)

    def test_extraction_failure_degrades_to_none(self, mock_svc):
        cfg.entity_extraction = True
        save_schema(_part_schema(), cfg.data_dir)
        with mock.patch(
            "lilbee.retrieval.entities.extract_entities", side_effect=RuntimeError("boom")
        ):
            assert asyncio.run(_build_entity_records(_records(), "a.txt")) is None

    def test_empty_records_is_none(self, mock_svc):
        cfg.entity_extraction = True
        assert asyncio.run(_build_entity_records([], "a.txt")) is None


class TestExtractorToolWiring:
    def _schema_with(self, kinds):
        types = []
        if "spacy" in kinds:
            types.append(EntityType(name="person", kind=ExtractorKind.SPACY, pattern="PERSON"))
        if "llm" in kinds:
            types.append(EntityType(name="vessel", kind=ExtractorKind.LLM, description="ships"))
        types.append(EntityType(name="part_number", kind=ExtractorKind.REGEX, pattern=r"PX\d{4}"))
        return EntitySchema(types=types)

    def test_spacy_types_load_the_model_when_available(self, mock_svc):
        cfg.entity_extraction = True
        save_schema(self._schema_with({"spacy"}), cfg.data_dir)
        fake_nlp = mock.MagicMock(return_value=mock.MagicMock(ents=[]))
        with (
            mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True),
            mock.patch("lilbee.retrieval.concepts.nlp._ensure_spacy_model", return_value=fake_nlp),
        ):
            rows = asyncio.run(_build_entity_records(_records(), "a.txt"))
        assert rows is not None
        fake_nlp.assert_called()

    def test_spacy_model_import_error_degrades(self, mock_svc, caplog):
        cfg.entity_extraction = True
        save_schema(self._schema_with({"spacy"}), cfg.data_dir)
        with (
            mock.patch("lilbee.retrieval.concepts.concepts_available", return_value=True),
            mock.patch(
                "lilbee.retrieval.concepts.nlp._ensure_spacy_model",
                side_effect=ImportError("no model"),
            ),
            caplog.at_level("WARNING"),
        ):
            rows = asyncio.run(_build_entity_records(_records(), "a.txt"))
        assert rows is not None  # regex type still extracted
        assert any("spaCy model unavailable" in r.message for r in caplog.records)

    def test_llm_types_fetch_the_provider(self, mock_svc):
        cfg.entity_extraction = True
        save_schema(self._schema_with({"llm"}), cfg.data_dir)
        mock_svc.provider.chat.return_value = mock.MagicMock(text="{}")
        rows = asyncio.run(_build_entity_records(_records(), "a.txt"))
        assert rows is not None
        mock_svc.provider.chat.assert_called()


class TestFlushEntityRows:
    def _result(self, rows):
        return _IngestResult(
            name="a.txt", path=cfg.data_dir / "a.txt", chunk_count=1, error=None, entity_rows=rows
        )

    def test_writes_merged_rows(self, mock_svc):
        row = {
            "entity": "PX4471",
            "type": "part_number",
            "normalized_value": "px4471",
            "source": "a.txt",
            "page": 1,
            "chunk_index": 0,
            "confidence": 1.0,
        }
        _flush_entity_rows([self._result([row]), self._result(None)])
        mock_svc.store.add_entities.assert_called_once_with([row])

    def test_no_rows_no_write(self, mock_svc):
        _flush_entity_rows([self._result(None), self._result([])])
        mock_svc.store.add_entities.assert_not_called()

    def test_write_failure_is_logged_not_raised(self, mock_svc, caplog):
        mock_svc.store.add_entities.side_effect = RuntimeError("locked")
        row = {
            "entity": "x",
            "type": "t",
            "normalized_value": "x",
            "source": "a.txt",
            "page": 1,
            "chunk_index": 0,
            "confidence": 1.0,
        }
        with caplog.at_level("WARNING"):
            _flush_entity_rows([self._result([row])])
        assert any("Entity indexing failed" in r.message for r in caplog.records)
