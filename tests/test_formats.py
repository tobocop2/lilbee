"""Real-file format tests — full sync() pipeline with actual files on disk.

All document formats go through kreuzberg. Code files still use tree-sitter.
Embeddings are mocked (no Ollama needed). kreuzberg is mocked for document
extraction since we're testing the pipeline, not kreuzberg itself.
"""

from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock

import pytest

import lilbee.config as cfg
import lilbee.store as store_mod


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths to temp dir for every test."""
    docs = tmp_path / "documents"
    docs.mkdir()
    data = tmp_path / "data" / "lancedb"

    orig_docs, orig_db, orig_data = cfg.DOCUMENTS_DIR, cfg.LANCEDB_DIR, cfg.DATA_DIR

    cfg.DOCUMENTS_DIR = docs
    cfg.DATA_DIR = tmp_path / "data"
    cfg.LANCEDB_DIR = data
    store_mod.LANCEDB_DIR = data

    yield docs

    cfg.DOCUMENTS_DIR = orig_docs
    cfg.DATA_DIR = orig_data
    cfg.LANCEDB_DIR = orig_db
    store_mod.LANCEDB_DIR = orig_db


def _fake_embed_batch(texts):
    return [[0.1] * 768 for _ in texts]


def _fake_embed(text):
    return [0.1] * 768


def _make_kreuzberg_result(text="Extracted content. " * 10, num_chunks=1):
    """Build a mock kreuzberg ExtractionResult."""
    chunks = []
    for i in range(num_chunks):
        chunk_text = text[i * len(text) // num_chunks : (i + 1) * len(text) // num_chunks]
        chunk = mock.MagicMock()
        chunk.content = chunk_text
        chunk.metadata = {
            "byte_start": 0,
            "byte_end": len(chunk_text),
            "chunk_index": i,
            "total_chunks": num_chunks,
            "token_count": None,
        }
        chunks.append(chunk)
    result = mock.MagicMock()
    result.chunks = chunks
    result.content = text
    return result


# ---------------------------------------------------------------------------
# Document formats (all go through kreuzberg)
# ---------------------------------------------------------------------------


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch(
    "kreuzberg.extract_file",
    new_callable=AsyncMock,
    return_value=_make_kreuzberg_result(),
)
class TestSyncDocx:
    async def test_docx_discovered_and_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "sample.docx").write_bytes(b"fake docx content")
        from lilbee.ingest import sync

        result = await sync()
        assert "sample.docx" in result.added


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch(
    "kreuzberg.extract_file",
    new_callable=AsyncMock,
    return_value=_make_kreuzberg_result(),
)
class TestSyncXlsx:
    async def test_xlsx_discovered_and_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "data.xlsx").write_bytes(b"fake xlsx content")
        from lilbee.ingest import sync

        result = await sync()
        assert "data.xlsx" in result.added


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch(
    "kreuzberg.extract_file",
    new_callable=AsyncMock,
    return_value=_make_kreuzberg_result(),
)
class TestSyncPptx:
    async def test_pptx_discovered_and_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "slides.pptx").write_bytes(b"fake pptx content")
        from lilbee.ingest import sync

        result = await sync()
        assert "slides.pptx" in result.added


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch(
    "kreuzberg.extract_file",
    new_callable=AsyncMock,
    return_value=_make_kreuzberg_result(),
)
class TestSyncEpub:
    async def test_epub_discovered_and_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "book.epub").write_bytes(b"fake epub content")
        from lilbee.ingest import sync

        result = await sync()
        assert "book.epub" in result.added


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch(
    "kreuzberg.extract_file",
    new_callable=AsyncMock,
    return_value=_make_kreuzberg_result(),
)
class TestSyncImage:
    async def test_image_discovered_and_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "scan.png").write_bytes(b"fake png content")
        from lilbee.ingest import sync

        result = await sync()
        assert "scan.png" in result.added


# ---------------------------------------------------------------------------
# Code — all supported languages through full sync pipeline
# ---------------------------------------------------------------------------

_CODE_FIXTURES: dict[str, tuple[str, str]] = {
    "greet.py": (
        ".py",
        'def greet(name):\n    """Say hello."""\n    return f"Hello, {name}!"\n',
    ),
    "greet.js": (
        ".js",
        'function greet(name) {\n    return "Hello, " + name + "!";\n}\n',
    ),
    "greet.ts": (
        ".ts",
        "function greet(name: string): string {\n    return `Hello, ${name}!`;\n}\n",
    ),
    "greet.go": (
        ".go",
        'package main\n\nfunc greet(name string) string {\n\treturn "Hello, " + name\n}\n',
    ),
    "greet.rs": (
        ".rs",
        'fn greet(name: &str) -> String {\n    format!("Hello, {}!", name)\n}\n',
    ),
    "Greet.java": (
        ".java",
        "public class Greet {\n"
        "    public static String greet(String name) {\n"
        '        return "Hello, " + name + "!";\n'
        "    }\n"
        "}\n",
    ),
    "greet.c": (
        ".c",
        "#include <stdio.h>\n\n"
        'void greet(const char* name) {\n    printf("Hello, %s!\\n", name);\n}\n',
    ),
    "greet.cpp": (
        ".cpp",
        "#include <string>\n\n"
        "std::string greet(const std::string& name) {\n"
        '    return "Hello, " + name + "!";\n'
        "}\n",
    ),
}


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
class TestSyncCode:
    @pytest.mark.parametrize("filename,fixture", list(_CODE_FIXTURES.items()))
    async def test_code_file_syncs(self, _eb, _e, _vm, isolated_env, filename, fixture):
        _ext, content = fixture
        (isolated_env / filename).write_text(content)

        from lilbee.ingest import sync

        result = await sync()
        assert filename in result.added


# ---------------------------------------------------------------------------
# CSV / TSV
# ---------------------------------------------------------------------------


@mock.patch("lilbee.embedder.validate_model")
@mock.patch("lilbee.embedder.embed", side_effect=_fake_embed)
@mock.patch("lilbee.embedder.embed_batch", side_effect=_fake_embed_batch)
@mock.patch(
    "kreuzberg.extract_file",
    new_callable=AsyncMock,
    return_value=_make_kreuzberg_result(),
)
class TestSyncCsvTsv:
    async def test_csv_discovered_and_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "people.csv").write_text("name,city\nAlice,Montreal\n")
        from lilbee.ingest import sync

        result = await sync()
        assert "people.csv" in result.added

    async def test_tsv_discovered_and_ingested(self, _kf, _eb, _e, _vm, isolated_env):
        (isolated_env / "products.tsv").write_text("id\tproduct\n1\tWidget\n")
        from lilbee.ingest import sync

        result = await sync()
        assert "products.tsv" in result.added
