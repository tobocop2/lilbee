"""Tests for wiki page generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.config import cfg
from lilbee.store import SearchChunk, Store
from lilbee.citation import ParsedCitation
from lilbee.wiki_gen import (
    _chunks_to_text,
    _parse_faithfulness_score,
    _resolve_citations,
    generate_summary_page,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.wiki = True
    cfg.wiki_dir = "wiki"
    cfg.wiki_faithfulness_threshold = 0.7
    cfg.chat_model = "test-model"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _make_chunk(text: str, source: str = "doc.md", **kwargs) -> SearchChunk:
    defaults = {
        "source": source,
        "content_type": "text",
        "chunk_type": "raw",
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "chunk": text,
        "chunk_index": 0,
        "vector": [0.1],
    }
    defaults.update(kwargs)
    return SearchChunk(**defaults)


class TestChunksToText:
    def test_basic_formatting(self):
        chunks = [_make_chunk("Hello world"), _make_chunk("Second chunk", chunk_index=1)]
        result = _chunks_to_text(chunks)
        assert "[Chunk 1]:" in result
        assert "Hello world" in result
        assert "[Chunk 2]:" in result

    def test_includes_page_location(self):
        chunks = [_make_chunk("PDF content", page_start=5)]
        result = _chunks_to_text(chunks)
        assert "(page 5)" in result

    def test_includes_line_location(self):
        chunks = [_make_chunk("Code", line_start=10, line_end=20)]
        result = _chunks_to_text(chunks)
        assert "(lines 10-20)" in result


class TestParseFaithfulnessScore:
    def test_valid_score(self):
        assert _parse_faithfulness_score("0.85") == 0.85

    def test_clamps_high(self):
        assert _parse_faithfulness_score("1.5") == 1.0

    def test_clamps_low(self):
        assert _parse_faithfulness_score("-0.5") == 0.0

    def test_multiline_extracts_first_number(self):
        assert _parse_faithfulness_score("Score:\n0.72\nDone") == 0.72

    def test_unparseable_returns_zero(self):
        assert _parse_faithfulness_score("I think it's good") == 0.0

    def test_empty_returns_zero(self):
        assert _parse_faithfulness_score("") == 0.0


class TestResolveCitations:
    def test_resolves_excerpt_to_chunk_location(self):
        chunks = [_make_chunk("Python supports typing.", page_start=3, page_end=3)]
        parsed = [ParsedCitation("src1", 'doc.pdf, excerpt: "Python supports typing."', 1)]
        records = _resolve_citations(parsed, "doc.pdf", "hash123", chunks)
        assert len(records) == 1
        assert records[0]["page_start"] == 3
        assert records[0]["claim_type"] == "fact"

    def test_inference_when_no_excerpt(self):
        chunks = [_make_chunk("Some text")]
        parsed = [ParsedCitation("src1", "doc.md, no excerpt here", 1)]
        records = _resolve_citations(parsed, "doc.md", "hash", chunks)
        assert records[0]["claim_type"] == "inference"

    def test_excerpt_not_found_gets_zero_locations(self):
        chunks = [_make_chunk("Different text entirely")]
        parsed = [ParsedCitation("src1", 'doc.md, excerpt: "Not in any chunk"', 1)]
        records = _resolve_citations(parsed, "doc.md", "hash", chunks)
        assert records[0]["page_start"] == 0
        assert records[0]["line_start"] == 0


class TestGenerateSummaryPage:
    def _mock_provider(self, wiki_text: str, faith_score: str = "0.85") -> MagicMock:
        provider = MagicMock()
        provider.chat.side_effect = [wiki_text, faith_score]
        return provider

    def _mock_store(self) -> MagicMock:
        store = MagicMock(spec=Store)
        store.add_citations.return_value = 0
        return store

    def test_generates_summary_page(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Python supports gradual typing.")
        chunks = [_make_chunk("Python supports gradual typing.")]

        wiki_text = (
            "# Doc Summary\n\n"
            "> Python supports gradual typing.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "Python supports gradual typing."'
        )
        provider = self._mock_provider(wiki_text)
        store = self._mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
        assert result.exists()
        assert "summaries" in str(result)
        content = result.read_text()
        assert "generated_by: test-model" in content
        assert "faithfulness_score: 0.85" in content
        store.add_citations.assert_called_once()

    def test_low_score_goes_to_drafts(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]

        wiki_text = (
            "# Draft\n\n"
            "> Content.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "Content"'
        )
        provider = self._mock_provider(wiki_text, faith_score="0.3")
        store = self._mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
        assert "drafts" in str(result)

    def test_empty_chunks_returns_none(self):
        provider = MagicMock()
        store = self._mock_store()
        result = generate_summary_page("doc.md", [], provider, store)
        assert result is None
        provider.chat.assert_not_called()

    def test_llm_failure_returns_none(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = MagicMock()
        provider.chat.side_effect = ConnectionError("LLM down")
        store = self._mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is None

    def test_no_valid_citations_returns_none(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]

        # Wiki text with citation whose excerpt doesn't match any chunk
        wiki_text = (
            "# Bad\n\n"
            "> Fabricated claim.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "This text is not in any chunk at all"'
        )
        provider = self._mock_provider(wiki_text)
        store = self._mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is None

    def test_faithfulness_check_failure_uses_zero(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]

        wiki_text = (
            "# Test\n\n"
            "> Content.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "Content"'
        )
        provider = MagicMock()
        provider.chat.side_effect = [wiki_text, ConnectionError("LLM down")]
        store = self._mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
        assert "drafts" in str(result)  # score=0.0 < threshold=0.7
