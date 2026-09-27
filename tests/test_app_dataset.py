"""Tests for the surface-neutral export/import use cases in app/dataset.py."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from lilbee.app import services as svc_mod
from lilbee.app.dataset import (
    DatasetError,
    export_to_bytes,
    export_to_path,
    import_from_bytes,
    import_from_path,
)
from lilbee.core.config import cfg
from lilbee.data.export import DatasetFormat, ImportResult
from lilbee.data.store import Store
from tests.conftest import make_mock_services


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


class TestExportToPath:
    def test_writes_file_and_summary(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.parquet"
        summary = export_to_path(out, "", None)
        assert out.exists()
        assert summary.model_dump() == {
            "command": "export",
            "format": "parquet",
            "output": str(out),
            "pages": 2,
            "sources": 1,
        }

    def test_named_source_only(self, store, tmp_path):
        _seed(store, "a.pdf")
        _seed(store, "b.pdf")
        summary = export_to_path(tmp_path / "pages.parquet", "", "b.pdf")
        assert summary.sources == 1
        assert summary.pages == 2

    def test_bad_format_raises(self, store, tmp_path):
        with pytest.raises(DatasetError, match="Unsupported format"):
            export_to_path(tmp_path / "pages.txt", "csv", None)

    def test_unknown_source_raises(self, store, tmp_path):
        _seed(store)
        with pytest.raises(DatasetError, match=r"Source not found: missing\.pdf"):
            export_to_path(tmp_path / "pages.parquet", "", "missing.pdf")

    def test_empty_store_raises(self, store, tmp_path):
        with pytest.raises(DatasetError, match="Nothing to export"):
            export_to_path(tmp_path / "pages.parquet", "", None)


class TestExportToBytes:
    def test_defaults_to_parquet(self, store):
        _seed(store)
        payload = export_to_bytes("", None)
        assert payload.fmt == DatasetFormat.PARQUET
        assert payload.pages == 2
        assert payload.sources == 1
        assert payload.data.startswith(b"PAR1")

    def test_jsonl_bytes(self, store):
        _seed(store)
        payload = export_to_bytes("jsonl", None)
        rows = [json.loads(line) for line in payload.data.decode().splitlines()]
        assert {r["page"] for r in rows} == {1, 2}

    def test_bad_format_raises(self, store):
        _seed(store)
        with pytest.raises(DatasetError, match="Unsupported format"):
            export_to_bytes("csv", None)

    def test_empty_store_raises(self, store):
        with pytest.raises(DatasetError, match="Nothing to export"):
            export_to_bytes("", None)


class TestImportFromPath:
    async def test_round_trip(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.jsonl"
        export_to_path(out, "", None)

        target = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb2"}))
        svc_mod.set_services(make_mock_services(store=target, embedder=_StubEmbedder()))

        summary = await import_from_path(out, "")
        assert summary.command == "import"
        assert summary.sources == ["doc.pdf"]
        assert summary.pages == 2
        assert summary.chunks > 0
        assert {r["page"] for r in target.get_page_texts("doc.pdf")} == {1, 2}

    async def test_missing_file_raises(self, store, tmp_path):
        with pytest.raises(DatasetError, match="Dataset not found"):
            await import_from_path(tmp_path / "nope.parquet", "")

    async def test_empty_dataset_raises(self, store, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        with pytest.raises(DatasetError, match="no pages"):
            await import_from_path(empty, "")

    async def test_embedder_mismatch_raises(self, store, tmp_path):
        dataset = tmp_path / "pages.jsonl"
        dataset.write_text('{"source":"doc.pdf","page":1,"text":"body","content_type":"pdf"}\n')
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
        with pytest.raises(DatasetError, match="embedding"):
            await import_from_path(dataset, "")


class TestImportFromBytes:
    async def test_round_trip(self, store):
        _seed(store)
        payload = export_to_bytes("jsonl", None)
        store.delete_by_source("doc.pdf")

        summary = await import_from_bytes(payload.data, "jsonl")
        assert summary.sources == ["doc.pdf"]
        assert summary.pages == 2

    async def test_unicode_parquet_round_trip(self, store):
        text = "café 蜂蜜 🐝"
        store.add_page_texts(
            [{"source": "uni.pdf", "page": 1, "text": text, "content_type": "pdf"}]
        )
        store.upsert_source("uni.pdf", "h", 1)
        payload = export_to_bytes("parquet", None)
        store.delete_by_source("uni.pdf")

        await import_from_bytes(payload.data, "parquet")
        assert store.get_page_texts("uni.pdf")[0]["text"] == text

    async def test_format_required(self, store):
        with pytest.raises(DatasetError, match="format is required"):
            await import_from_bytes(b"{}", "")

    async def test_bad_payload_raises(self, store):
        with pytest.raises(DatasetError, match="missing required"):
            await import_from_bytes(b'{"not": "a row"}\n', "jsonl")

    async def test_on_progress_reaches_embedder(self, store):
        from lilbee.runtime.progress import EmbedEvent, EventType

        class _EmittingEmbedder:
            truncated_total = 0

            def embed_batch(self, texts, source="", on_progress=None, **_kwargs):
                if on_progress is not None:
                    on_progress(
                        EventType.EMBED,
                        EmbedEvent(file=source, chunk=len(texts), total_chunks=len(texts)),
                    )
                return [[0.1] * cfg.embedding_dim for _ in texts]

        _seed(store)
        payload = export_to_bytes("jsonl", None)
        svc_mod.set_services(make_mock_services(store=store, embedder=_EmittingEmbedder()))

        events = []
        await import_from_bytes(payload.data, "jsonl", on_progress=lambda et, d: events.append(et))
        assert EventType.EMBED in events


def _cli_reset(data_root) -> subprocess.CompletedProcess[str]:
    """Run a real ``lilbee reset --yes`` in its own process against *data_root*."""
    return subprocess.run(
        [sys.executable, "-m", "lilbee", "reset", "--yes"],
        env={**os.environ, "LILBEE_DATA": str(data_root)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


class TestImportHoldsTheSyncMark:
    """An import holds the data root's sync mark, so a reset in any process refuses."""

    async def test_a_cli_reset_in_another_process_is_refused_mid_import(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.jsonl"
        export_to_path(out, "", None)
        kept = cfg.data_root / "documents" / "kept.txt"
        kept.parent.mkdir(parents=True)
        kept.write_text("stays through the refused reset", encoding="utf-8")
        started = asyncio.Event()
        release = asyncio.Event()

        async def _blocked_import(_store, rows, **_kwargs):
            started.set()
            await release.wait()
            return ImportResult(sources=["doc.pdf"], pages=len(rows), chunks=1)

        with patch("lilbee.app.dataset.import_dataset", new=_blocked_import):
            importing = asyncio.create_task(import_from_path(out, ""))
            await asyncio.wait_for(started.wait(), 5)
            refused = await asyncio.to_thread(_cli_reset, cfg.data_root)
            release.set()
            summary = await asyncio.wait_for(importing, 5)

        assert refused.returncode == 1, refused.stderr
        assert "A sync or import is running on this library" in refused.stdout
        assert kept.exists()
        assert summary.pages == 2
        after = await asyncio.to_thread(_cli_reset, cfg.data_root)
        assert after.returncode == 0, after.stderr
        assert not kept.exists()
