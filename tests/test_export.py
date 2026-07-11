"""Tests for the per-page text dataset export/import (lilbee.data.export)."""

from pathlib import Path

import pyarrow as pa
import pytest

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.data.export import (
    DatasetFormat,
    build_page_dataset,
    import_dataset,
    load_page_dataset,
    resolve_format,
    write_dataset,
)
from lilbee.data.store import EmbeddingModelMismatchError, SourceType, Store
from tests.conftest import make_mock_services


@pytest.fixture()
def test_config(tmp_path):
    """A Config pointed at a temp LanceDB directory (mutable by tests)."""
    return cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb"})


@pytest.fixture()
def store(test_config):
    """A real Store on a temp LanceDB directory."""
    return Store(test_config)


@pytest.fixture()
def services(store):
    """Install a services container with the real store and a stub embedder."""
    embedder = _StubEmbedder()
    svc_mod.set_services(make_mock_services(store=store, embedder=embedder))
    yield store
    svc_mod.set_services(None)


class _StubEmbedder:
    """Deterministic embedder: every chunk maps to a fixed-dim vector."""

    truncated_total = 0

    def embed_batch(self, texts, **_kwargs):
        return [[0.1] * cfg.embedding_dim for _ in texts]


def _chunk(source, page, chunk_index, text):
    return {
        "source": source,
        "content_type": "pdf",
        "chunk_type": "raw",
        "page_start": page,
        "page_end": page,
        "line_start": 0,
        "line_end": 0,
        "chunk": text,
        "chunk_index": chunk_index,
        "vector": [0.1] * cfg.embedding_dim,
    }


def _page(source, page, text, content_type="pdf"):
    return {"source": source, "page": page, "text": text, "content_type": content_type}


class TestResolveFormat:
    def test_explicit_value_wins(self):
        assert resolve_format("jsonl", Path("x.parquet")) == DatasetFormat.JSONL

    def test_suffix_parquet(self):
        assert resolve_format("", Path("data.parquet")) == DatasetFormat.PARQUET

    def test_suffix_jsonl(self):
        assert resolve_format("", Path("data.jsonl")) == DatasetFormat.JSONL

    def test_bad_explicit_value(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            resolve_format("csv", Path("data.parquet"))

    def test_unknown_suffix(self):
        with pytest.raises(ValueError, match="Could not infer format"):
            resolve_format("", Path("data.txt"))


class TestWriteRoundTrip:
    @pytest.mark.parametrize("fmt", [DatasetFormat.PARQUET, DatasetFormat.JSONL])
    def test_round_trip(self, tmp_path, fmt):
        table = pa.Table.from_pylist([_page("a.pdf", 1, "hello"), _page("a.pdf", 2, "world")])
        path = tmp_path / f"pages.{fmt}"
        write_dataset(table, path, fmt)
        loaded = load_page_dataset(path, fmt)
        assert [(r["source"], r["page"], r["text"]) for r in loaded] == [
            ("a.pdf", 1, "hello"),
            ("a.pdf", 2, "world"),
        ]

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="Dataset not found"):
            load_page_dataset(tmp_path / "nope.parquet", DatasetFormat.PARQUET)

    def test_jsonl_skips_blank_lines(self, tmp_path):
        path = tmp_path / "pages.jsonl"
        path.write_text('{"source":"a.pdf","page":1,"text":"x","content_type":"pdf"}\n\n')
        assert len(load_page_dataset(path, DatasetFormat.JSONL)) == 1

    def test_bad_row_raises(self, tmp_path):
        path = tmp_path / "pages.jsonl"
        path.write_text('{"source":"a.pdf"}\n')
        with pytest.raises(ValueError, match="missing required"):
            load_page_dataset(path, DatasetFormat.JSONL)


class TestBuildPageDataset:
    def test_clean_path_uses_captured_text(self, store):
        store.add_page_texts([_page("a.pdf", 1, "clean one"), _page("a.pdf", 2, "clean two")])
        store.upsert_source("a.pdf", "h", 2)
        rows = build_page_dataset(store).to_pylist()
        assert [(r["page"], r["text"]) for r in rows] == [(1, "clean one"), (2, "clean two")]

    def test_fallback_reconstructs_from_chunks(self, store):
        # No page texts captured; reconstruct from chunks, ordered by chunk_index.
        store.add_chunks([_chunk("b.pdf", 1, 1, "second"), _chunk("b.pdf", 1, 0, "first")])
        store.upsert_source("b.pdf", "h", 2)
        rows = build_page_dataset(store).to_pylist()
        assert len(rows) == 1
        assert rows[0]["page"] == 1
        assert rows[0]["text"] == "first\nsecond"

    def test_source_filter(self, store):
        store.add_page_texts([_page("a.pdf", 1, "a-text")])
        store.add_page_texts([_page("b.pdf", 1, "b-text")])
        store.upsert_source("a.pdf", "h", 1)
        store.upsert_source("b.pdf", "h", 1)
        rows = build_page_dataset(store, source="b.pdf").to_pylist()
        assert {r["source"] for r in rows} == {"b.pdf"}

    def test_single_source_reconstructs_when_not_captured(self, store):
        # Exporting one source that has chunks but no captured page text
        # reconstructs just that source from its chunks.
        store.add_chunks([_chunk("only-chunks.pdf", 1, 0, "rebuilt body")])
        store.upsert_source("only-chunks.pdf", "h", 1)
        rows = build_page_dataset(store, source="only-chunks.pdf").to_pylist()
        assert [(r["source"], r["text"]) for r in rows] == [("only-chunks.pdf", "rebuilt body")]

    def test_all_sources_scans_page_texts_once(self, store, monkeypatch):
        # Regression (bb-bqg): the all-sources export must read the page-text
        # table in a single Arrow scan, not one filtered query per source. The
        # per-source loop was O(sources) and hung for >25 min on a 346k-source
        # store; the fixed-per-query overhead, not data size, dominated.
        for i in range(5):
            store.add_page_texts([_page(f"s{i}.pdf", 1, f"t{i}")])
            store.upsert_source(f"s{i}.pdf", "h", 1)
        calls: list[str | None] = []
        real = store.page_texts_arrow

        def spy(source=None):
            calls.append(source)
            return real(source)

        monkeypatch.setattr(store, "page_texts_arrow", spy)
        table = build_page_dataset(store)
        assert table.num_rows == 5
        # Exactly one full-table scan (source=None); never a per-source query.
        assert calls == [None]

    def test_orphaned_page_text_without_source_is_excluded(self, store):
        # A page-text row whose source has no tracking record (e.g. the source was
        # removed) is left out: the scan is restricted to get_sources(), matching
        # the universe the old per-source loop walked.
        store.add_page_texts([_page("real.pdf", 1, "kept")])
        store.upsert_source("real.pdf", "h", 1)
        store.add_page_texts([_page("orphan.pdf", 1, "dropped")])  # no upsert_source
        rows = build_page_dataset(store).to_pylist()
        assert {r["source"] for r in rows} == {"real.pdf"}

    def test_mixed_captured_and_reconstructed_sources(self, store):
        # Captured sources come from the single scan; only sources with no
        # captured text fall back to per-source chunk reconstruction.
        store.add_page_texts([_page("cap.pdf", 1, "captured")])
        store.upsert_source("cap.pdf", "h", 1)
        store.add_chunks([_chunk("recon.pdf", 1, 0, "rebuilt")])
        store.upsert_source("recon.pdf", "h", 1)
        rows = build_page_dataset(store).to_pylist()
        assert [(r["source"], r["text"]) for r in rows] == [
            ("cap.pdf", "captured"),
            ("recon.pdf", "rebuilt"),
        ]

    def test_sorted_by_source_and_page(self, store):
        store.add_page_texts(
            [_page("b.pdf", 1, "b1"), _page("a.pdf", 2, "a2"), _page("a.pdf", 1, "a1")]
        )
        store.upsert_source("a.pdf", "h", 2)
        store.upsert_source("b.pdf", "h", 1)
        rows = build_page_dataset(store).to_pylist()
        assert [(r["source"], r["page"]) for r in rows] == [
            ("a.pdf", 1),
            ("a.pdf", 2),
            ("b.pdf", 1),
        ]


class TestImportDataset:
    async def test_round_trip_into_fresh_store(self, services):
        store = services
        rows = [_page("doc.pdf", 1, "page one body"), _page("doc.pdf", 2, "page two body")]
        result = await import_dataset(store, rows)

        assert result.sources == ["doc.pdf"]
        assert result.pages == 2
        assert result.chunks > 0
        # Page texts are preserved with their page numbers.
        assert {r["page"] for r in store.get_page_texts("doc.pdf")} == {1, 2}
        # Source is recorded as detached/imported.
        sources = store.get_sources()
        assert sources[0]["source_type"] == SourceType.IMPORTED
        # Chunks landed and carry the page number.
        chunks = store.get_chunks_by_source("doc.pdf")
        assert chunks and all(c.page_start in {1, 2} for c in chunks)

    async def test_reimport_replaces(self, services):
        store = services
        await import_dataset(store, [_page("doc.pdf", 1, "first version")])
        await import_dataset(store, [_page("doc.pdf", 1, "second version")])
        assert len(store.get_page_texts("doc.pdf")) == 1

    async def test_dim_mismatch_raises(self, services, test_config):
        store = services
        # Seed a different embedding identity so the gate refuses the import.
        store.add_chunks([_chunk("seed.pdf", 1, 0, "seed")])
        test_config.embedding_model = "ollama/other-model:v1"
        with pytest.raises(EmbeddingModelMismatchError):
            await import_dataset(store, [_page("doc.pdf", 1, "body")])

    async def test_import_uses_single_atomic_batch_write(self, services, monkeypatch):
        # Each source must be written via one locked write_chunks_batch
        # (cleanup + chunks + page texts + source row), not four separate unlocked
        # writes that could destroy the source if one fails mid-way.
        store = services
        batch_calls = 0
        real_batch = store.write_chunks_batch

        def counting_batch(items):
            nonlocal batch_calls
            batch_calls += 1
            return real_batch(items)

        legacy: list[str] = []
        monkeypatch.setattr(store, "write_chunks_batch", counting_batch)
        monkeypatch.setattr(store, "delete_by_source", lambda *a, **k: legacy.append("delete"))
        monkeypatch.setattr(store, "add_chunks", lambda *a, **k: legacy.append("add_chunks"))
        monkeypatch.setattr(store, "add_page_texts", lambda *a, **k: legacy.append("page_texts"))
        monkeypatch.setattr(store, "upsert_source", lambda *a, **k: legacy.append("upsert"))

        await import_dataset(store, [_page("doc.pdf", 1, "body")])

        assert batch_calls == 1
        assert legacy == []  # none of the old per-op unlocked writes were used
        # The atomic write still landed the source as IMPORTED.
        assert store.get_sources()[0]["source_type"] == SourceType.IMPORTED
