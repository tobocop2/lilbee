"""Tests for the `lilbee export` / `lilbee import` CLI commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.data.store import Store
from tests.conftest import make_mock_services

runner = CliRunner()


class _StubEmbedder:
    truncated_total = 0

    def embed_batch(self, texts, **_kwargs):
        return [[0.1] * cfg.embedding_dim for _ in texts]


@pytest.fixture()
def store(tmp_path):
    real = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb"}))
    svc_mod.set_services(make_mock_services(store=real, embedder=_StubEmbedder()))
    yield real
    svc_mod.set_services(None)


def _seed(store, source="doc.pdf"):
    store.add_page_texts(
        [
            {"source": source, "page": 1, "text": "page one", "content_type": "pdf"},
            {"source": source, "page": 2, "text": "page two", "content_type": "pdf"},
        ]
    )
    store.upsert_source(source, "h", 2)


class TestExportCommand:
    def test_export_writes_file(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.parquet"
        result = runner.invoke(app, ["--json", "export", str(out)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload == {
            "command": "export",
            "format": "parquet",
            "output": str(out),
            "pages": 2,
            "sources": 1,
        }
        assert out.exists()

    def test_export_jsonl_by_suffix(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.jsonl"
        result = runner.invoke(app, ["export", str(out)])
        assert result.exit_code == 0, result.output
        assert "Wrote 2 pages" in result.output
        assert out.exists()

    def test_export_empty_store_fails(self, store, tmp_path):
        result = runner.invoke(app, ["--json", "export", str(tmp_path / "pages.parquet")])
        assert result.exit_code == 1
        assert "error" in json.loads(result.output)

    def test_export_empty_store_human_output(self, store, tmp_path):
        result = runner.invoke(app, ["export", str(tmp_path / "pages.parquet")])
        assert result.exit_code == 1
        assert "Nothing to export" in result.output

    def test_export_bad_format_fails(self, store, tmp_path):
        _seed(store)
        result = runner.invoke(
            app, ["--json", "export", str(tmp_path / "pages.txt"), "--format", "csv"]
        )
        assert result.exit_code == 1
        assert "Unsupported format" in json.loads(result.output)["error"]

    def test_export_unknown_source_fails(self, store, tmp_path):
        _seed(store)
        result = runner.invoke(
            app, ["--json", "export", str(tmp_path / "p.parquet"), "--source", "missing.pdf"]
        )
        assert result.exit_code == 1
        assert "Source not found" in json.loads(result.output)["error"]


class TestImportCommand:
    def test_round_trip(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.parquet"
        assert runner.invoke(app, ["export", str(out)]).exit_code == 0

        # Fresh store for the import target.
        target = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb2"}))
        svc_mod.set_services(make_mock_services(store=target, embedder=_StubEmbedder()))

        result = runner.invoke(app, ["--json", "import", str(out)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["command"] == "import"
        assert payload["sources"] == ["doc.pdf"]
        assert payload["pages"] == 2
        assert payload["chunks"] > 0
        assert {r["page"] for r in target.get_page_texts("doc.pdf")} == {1, 2}

    def test_round_trip_human_output(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.jsonl"
        assert runner.invoke(app, ["export", str(out)]).exit_code == 0

        target = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb2"}))
        svc_mod.set_services(make_mock_services(store=target, embedder=_StubEmbedder()))

        result = runner.invoke(app, ["import", str(out)])
        assert result.exit_code == 0, result.output
        assert "Imported 1 source(s)" in result.output

    def test_import_embedder_mismatch_fails(self, store, tmp_path):
        out = tmp_path / "pages.jsonl"
        out.write_text('{"source":"doc.pdf","page":1,"text":"body","content_type":"pdf"}\n')
        # Seed the store under one embedder, then drift the model so import is refused.
        store.add_chunks(
            [
                {
                    "source": "seed.pdf",
                    "content_type": "pdf",
                    "chunk_type": "raw",
                    "page_start": 1,
                    "page_end": 1,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "seed",
                    "chunk_index": 0,
                    "vector": [0.1] * cfg.embedding_dim,
                }
            ]
        )
        store._config.embedding_model = "ollama/other-model:v1"
        result = runner.invoke(app, ["--json", "import", str(out)])
        assert result.exit_code == 1
        assert "error" in json.loads(result.output)

    def test_import_missing_file_fails(self, store, tmp_path):
        result = runner.invoke(app, ["--json", "import", str(tmp_path / "nope.parquet")])
        assert result.exit_code == 1
        assert "Dataset not found" in json.loads(result.output)["error"]

    def test_import_empty_dataset_fails(self, store, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = runner.invoke(app, ["--json", "import", str(empty)])
        assert result.exit_code == 1
        assert "no pages" in json.loads(result.output)["error"]
