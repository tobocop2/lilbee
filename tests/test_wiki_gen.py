"""Tests for wiki page generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.config import cfg
from lilbee.store import SearchChunk, Store
from lilbee.wiki.citation import ParsedCitation
from lilbee.wiki.gen import (
    _check_faithfulness,
    _chunks_in_page_range,
    _chunks_to_text,
    _collect_inner_children,
    _content_change_ratio,
    _diff_summary,
    _divert_to_drafts,
    _extract_excerpt,
    _find_cached_leaf,
    _find_excerpt_source,
    _format_children_for_reduce,
    _generate_inner_node,
    _generate_synthesis_page,
    _group_chunks_by_page,
    _inner_node_target,
    _is_draft_path,
    _leaf_hash,
    _leaves_in_range,
    _load_document_structure,
    _match_citation_source,
    _page_slug,
    _parse_faithfulness_score,
    _read_page_body,
    _resolve_citations,
    _resolve_multi_source_citations,
    _source_slug,
    _truncate_chunks_to_budget,
    _verify_citations,
    _wiki_nodes_bottom_up,
    generate_summary_page,
    generate_synthesis_pages,
)
from lilbee.wiki.shared import make_slug
from lilbee.wiki.structure import WikiNode


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


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


def _mock_provider(wiki_text: str, faith_score: str = "0.85") -> MagicMock:
    provider = MagicMock()
    provider.chat.side_effect = [wiki_text, faith_score]
    return provider


def _mock_store() -> MagicMock:
    store = MagicMock(spec=Store)
    store.add_citations.return_value = 0
    # Tree reductions look up the persisted DocumentStructure; most tests
    # don't stage one, so default to "no structure persisted".
    store.get_document_structure.return_value = None
    return store


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


class TestTruncateChunksToBudget:
    def test_small_chunks_unchanged(self):
        """Chunks that fit within budget are returned as-is."""
        chunks = [_make_chunk("short text")]
        result = _truncate_chunks_to_budget(chunks, cfg)
        assert result == chunks

    def test_truncates_when_exceeding_budget(self):
        """Large chunk sets are truncated to fit the context window."""
        cfg.num_ctx = 100  # 100 tokens * 0.75 * 4 chars = 300 chars budget
        big_text = "x" * 200  # 200 chars each, only one fits in 300
        chunks = [_make_chunk(big_text, chunk_index=i) for i in range(5)]
        result = _truncate_chunks_to_budget(chunks, cfg)
        assert len(result) == 1

    def test_always_keeps_at_least_one_chunk(self):
        """Even if the first chunk exceeds the budget, it is kept."""
        cfg.num_ctx = 10  # tiny budget: 10 * 0.75 * 4 = 30 chars
        huge_chunk = _make_chunk("x" * 10000)
        result = _truncate_chunks_to_budget([huge_chunk], cfg)
        assert len(result) == 1

    def test_uses_default_context_when_num_ctx_none(self):
        """Falls back to default context window when num_ctx is not set."""
        cfg.num_ctx = None
        # Default 8192 * 0.75 * 4 = 24576 chars budget
        small_chunks = [_make_chunk("hello", chunk_index=i) for i in range(10)]
        result = _truncate_chunks_to_budget(small_chunks, cfg)
        assert len(result) == 10  # all fit easily

    def test_logs_warning_on_truncation(self, caplog: pytest.LogCaptureFixture):
        """A warning is logged when chunks are truncated."""
        cfg.num_ctx = 100
        chunks = [_make_chunk("x" * 200, chunk_index=i) for i in range(5)]
        with caplog.at_level("WARNING", logger="lilbee.wiki.gen"):
            _truncate_chunks_to_budget(chunks, cfg)
        assert "Truncated chunks from 5 to 1" in caplog.text


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


class TestExtractExcerpt:
    def test_normal_quoted_excerpt(self):
        assert _extract_excerpt('doc.md, excerpt: "Python supports typing."') == (
            "Python supports typing."
        )

    def test_no_excerpt_marker(self):
        assert _extract_excerpt("doc.md, no excerpt here") == ""

    def test_unclosed_quote_returns_rest(self):
        assert _extract_excerpt('doc.md, excerpt: "trailing text') == "trailing text"


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


class TestVerifyCitations:
    def test_keeps_matching_excerpts(self):
        from lilbee.store import CitationRecord

        chunks = [_make_chunk("Python supports typing.")]
        recs: list[CitationRecord] = [
            {
                "wiki_source": "",
                "wiki_chunk_index": 0,
                "citation_key": "src1",
                "claim_type": "fact",
                "source_filename": "doc.md",
                "source_hash": "h",
                "page_start": 0,
                "page_end": 0,
                "line_start": 0,
                "line_end": 0,
                "excerpt": "Python supports typing.",
                "created_at": "now",
            }
        ]
        verified = _verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 1

    def test_drops_unmatched_excerpts(self):
        from lilbee.store import CitationRecord

        chunks = [_make_chunk("Different text")]
        recs: list[CitationRecord] = [
            {
                "wiki_source": "",
                "wiki_chunk_index": 0,
                "citation_key": "src1",
                "claim_type": "fact",
                "source_filename": "doc.md",
                "source_hash": "h",
                "page_start": 0,
                "page_end": 0,
                "line_start": 0,
                "line_end": 0,
                "excerpt": "Not in chunks",
                "created_at": "now",
            }
        ]
        verified = _verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 0

    def test_keeps_inference_citations(self):
        from lilbee.store import CitationRecord

        chunks = [_make_chunk("text")]
        recs: list[CitationRecord] = [
            {
                "wiki_source": "",
                "wiki_chunk_index": 0,
                "citation_key": "src1",
                "claim_type": "inference",
                "source_filename": "doc.md",
                "source_hash": "h",
                "page_start": 0,
                "page_end": 0,
                "line_start": 0,
                "line_end": 0,
                "excerpt": "",
                "created_at": "now",
            }
        ]
        verified = _verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 1

    def test_skips_wiki_sourced_citations(self):
        from lilbee.store import CitationRecord

        chunks = [_make_chunk("text")]
        recs: list[CitationRecord] = [
            {
                "wiki_source": "",
                "wiki_chunk_index": 0,
                "citation_key": "src1",
                "claim_type": "fact",
                "source_filename": "wiki/summaries/page.md",
                "source_hash": "h",
                "page_start": 0,
                "page_end": 0,
                "line_start": 0,
                "line_end": 0,
                "excerpt": "text",
                "created_at": "now",
            }
        ]
        verified = _verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 0


class TestCheckFaithfulness:
    def test_returns_score(self):
        provider = MagicMock()
        provider.chat.return_value = "0.85"
        score = _check_faithfulness("chunks", "wiki", provider, "test")
        assert score == 0.85

    def test_failure_returns_zero(self):
        provider = MagicMock()
        provider.chat.side_effect = ConnectionError("down")
        score = _check_faithfulness("chunks", "wiki", provider, "test")
        assert score == 0.0


class TestGenerateSummaryPage:
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
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        result = pages[0]
        assert result.exists()
        assert "summaries/doc/page-0000.md" in str(result).replace("\\", "/")
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
        provider = _mock_provider(wiki_text, faith_score="0.3")
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        assert "drafts/doc/page-0000.md" in str(pages[0]).replace("\\", "/")

    def test_empty_chunks_returns_empty_list(self):
        provider = MagicMock()
        store = _mock_store()
        pages = generate_summary_page("doc.md", [], provider, store)
        assert pages == []
        provider.chat.assert_not_called()

    def test_llm_failure_returns_empty_list(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = MagicMock()
        provider.chat.side_effect = ConnectionError("LLM down")
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert pages == []

    def test_no_valid_citations_returns_empty_list(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]

        wiki_text = (
            "# Bad\n\n"
            "> Fabricated claim.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "This text is not in any chunk at all"'
        )
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert pages == []

    def test_no_valid_citations_emits_failed_progress(self, tmp_path: Path):
        """Progress callback receives 'failed' stage when no citations verify."""
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]

        wiki_text = (
            "# Bad\n\n"
            "> Fabricated claim.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "This text is not in any chunk at all"'
        )
        provider = _mock_provider(wiki_text)
        store = _mock_store()
        events: list[tuple[str, dict[str, object]]] = []

        def on_progress(stage: str, data: dict[str, object]) -> None:
            events.append((stage, data))

        generate_summary_page("doc.md", chunks, provider, store, on_progress=on_progress)
        failed_events = [(s, d) for s, d in events if s == "failed"]
        assert len(failed_events) == 1
        assert "citation" in str(failed_events[0][1]["error"]).lower()

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
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        assert "drafts" in str(pages[0])  # score=0.0 < threshold=0.7

    def test_llm_returns_empty_string(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = _mock_provider("   ")  # whitespace-only -> empty after strip
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert pages == []

    def test_provider_error_returns_empty_list(self, tmp_path: Path):
        """ProviderError from chat() is caught and returns an empty list."""
        from lilbee.providers.base import ProviderError

        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = MagicMock()
        provider.chat.side_effect = ProviderError("model not found", provider="litellm")
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert pages == []

    def test_unexpected_exception_returns_empty_list(self, tmp_path: Path):
        """Unexpected exceptions (ValueError, KeyError, etc.) are caught."""
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = MagicMock()
        provider.chat.side_effect = ValueError("context window exceeded")
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert pages == []

    def test_llm_failure_emits_failed_progress(self, tmp_path: Path):
        """Progress callback receives 'failed' stage with error on LLM failure."""
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("GPU OOM")
        store = _mock_store()
        events: list[tuple[str, dict[str, object]]] = []

        def on_progress(stage: str, data: dict[str, object]) -> None:
            events.append((stage, data))

        generate_summary_page("doc.md", chunks, provider, store, on_progress=on_progress)
        failed_events = [(s, d) for s, d in events if s == "failed"]
        assert len(failed_events) == 1
        assert "GPU OOM" in str(failed_events[0][1]["error"])

    def test_empty_response_emits_failed_progress(self, tmp_path: Path):
        """Progress callback receives 'failed' stage when model returns empty."""
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = _mock_provider("   ")
        store = _mock_store()
        events: list[tuple[str, dict[str, object]]] = []

        def on_progress(stage: str, data: dict[str, object]) -> None:
            events.append((stage, data))

        generate_summary_page("doc.md", chunks, provider, store, on_progress=on_progress)
        failed_events = [(s, d) for s, d in events if s == "failed"]
        assert len(failed_events) == 1
        assert "empty" in str(failed_events[0][1]["error"]).lower()

    def test_faithfulness_provider_error_uses_zero(self, tmp_path: Path):
        """ProviderError during faithfulness check returns score 0.0."""
        from lilbee.providers.base import ProviderError

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
        provider.chat.side_effect = [
            wiki_text,
            ProviderError("timeout", provider="litellm"),
        ]
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        assert "drafts" in str(pages[0])  # score=0.0 < threshold=0.7

    def test_inference_citations_pass_verification(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content here.")
        chunks = [_make_chunk("Content here.")]

        wiki_text = (
            "# Summary\n\n"
            "This is an observation.[*inference*]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, no excerpt"
        )
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        store.add_citations.assert_called_once()

    def test_prune_raw_deletes_source_chunks(self, tmp_path: Path):
        cfg.wiki_prune_raw = True
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
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        store.delete_by_source.assert_called_once_with("doc.md")

    def test_citations_cleared_before_adding(self, tmp_path: Path):
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
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        generate_summary_page("doc.md", chunks, provider, store)
        store.delete_citations_for_wiki.assert_called_once()
        store.add_citations.assert_called_once()

    def test_think_tags_stripped_from_wiki_output(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Python supports gradual typing.")
        chunks = [_make_chunk("Python supports gradual typing.")]

        wiki_text_with_think = (
            "<think>\nLet me reason about this...\n</think>\n"
            "# Doc Summary\n\n"
            "> Python supports gradual typing.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "Python supports gradual typing."'
        )
        provider = _mock_provider(wiki_text_with_think)
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        content = pages[0].read_text()
        assert "<think>" not in content
        assert "Let me reason" not in content
        assert "# Doc Summary" in content

    def test_multi_page_source_produces_one_file_per_page(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.pdf"
        source.write_text("Two pages of content.")
        chunks = [
            _make_chunk("Alpha page one.", page_start=1, page_end=1, chunk_index=0),
            _make_chunk("Alpha page one.", page_start=1, page_end=1, chunk_index=1),
            _make_chunk("Beta page two.", page_start=2, page_end=2, chunk_index=2),
        ]
        wiki_p1 = (
            "# P1\n\n"
            "> Alpha page one.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.pdf, excerpt: "Alpha page one."'
        )
        wiki_p2 = (
            "# P2\n\n"
            "> Beta page two.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.pdf, excerpt: "Beta page two."'
        )
        provider = MagicMock()
        provider.chat.side_effect = [wiki_p1, "0.9", wiki_p2, "0.9"]
        store = _mock_store()

        pages = generate_summary_page("doc.pdf", chunks, provider, store)
        assert len(pages) == 2
        paths = {str(p).replace("\\", "/") for p in pages}
        assert any("summaries/doc/page-0001.md" in p for p in paths)
        assert any("summaries/doc/page-0002.md" in p for p in paths)

    def test_partial_multi_page_failure_returns_successful_pages(self, tmp_path: Path):
        """One failing page doesn't drop the rest of the source."""
        source = tmp_path / "documents" / "doc.pdf"
        source.write_text("Two pages of content.")
        chunks = [
            _make_chunk("Alpha page one.", page_start=1, page_end=1, chunk_index=0),
            _make_chunk("Beta page two.", page_start=2, page_end=2, chunk_index=1),
        ]
        wiki_p1 = (
            "# P1\n\n"
            "> Alpha page one.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.pdf, excerpt: "Alpha page one."'
        )
        # Page 2's model call returns empty, which skips that page.
        provider = MagicMock()
        provider.chat.side_effect = [wiki_p1, "0.9", "   "]
        store = _mock_store()

        pages = generate_summary_page("doc.pdf", chunks, provider, store)
        assert len(pages) == 1
        assert "summaries/doc/page-0001.md" in str(pages[0]).replace("\\", "/")

    def test_same_page_chunks_collapse_into_single_file(self, tmp_path: Path):
        """Two chunks sharing page_start produce exactly one page file."""
        source = tmp_path / "documents" / "doc.pdf"
        source.write_text("Two chunks on one page.")
        chunks = [
            _make_chunk("Same page fact.", page_start=3, page_end=3, chunk_index=0),
            _make_chunk("Same page fact.", page_start=3, page_end=3, chunk_index=1),
        ]
        wiki_text = (
            "# Page three\n\n"
            "> Same page fact.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.pdf, excerpt: "Same page fact."'
        )
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        pages = generate_summary_page("doc.pdf", chunks, provider, store)
        assert len(pages) == 1
        assert "summaries/doc/page-0003.md" in str(pages[0]).replace("\\", "/")

    def test_frontmatter_carries_leaf_hash(self, tmp_path: Path):
        """Fresh generation writes leaf_hash so next rebuild can cache-hit."""
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
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        content = pages[0].read_text(encoding="utf-8")
        expected = _leaf_hash(chunks)
        assert f"leaf_hash: {expected}" in content

    def test_unchanged_chunks_skip_llm_on_rerun(self, tmp_path: Path):
        """Second sync with the same chunks hits the cache and does not call the provider."""
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
        provider1 = _mock_provider(wiki_text)
        store = _mock_store()
        first = generate_summary_page("doc.md", chunks, provider1, store)
        assert len(first) == 1
        first_path = first[0]

        # Second run: fresh provider that would blow up if asked to generate.
        provider2 = MagicMock()
        provider2.chat.side_effect = AssertionError("should not be called on cache hit")
        store2 = _mock_store()
        stages: list[str] = []

        def on_progress(stage: str, data: dict[str, object]) -> None:
            stages.append(stage)

        second = generate_summary_page("doc.md", chunks, provider2, store2, on_progress=on_progress)
        assert second == [first_path]
        provider2.chat.assert_not_called()
        store2.add_citations.assert_not_called()
        assert "cached" in stages

    def test_changed_chunks_invalidate_cache(self, tmp_path: Path):
        """If chunk content changes, the provider is called again."""
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Original content.")
        chunks1 = [_make_chunk("Original fact.")]

        wiki_text1 = (
            "# V1\n\n"
            "> Original fact.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "Original fact."'
        )
        generate_summary_page("doc.md", chunks1, _mock_provider(wiki_text1), _mock_store())

        chunks2 = [_make_chunk("Edited fact.")]
        wiki_text2 = (
            "# V2\n\n"
            "> Edited fact.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "Edited fact."'
        )
        provider2 = _mock_provider(wiki_text2)
        store2 = _mock_store()

        pages = generate_summary_page("doc.md", chunks2, provider2, store2)
        assert len(pages) == 1
        # Provider got both the generate and faithfulness calls.
        assert provider2.chat.call_count == 2
        content = pages[0].read_text(encoding="utf-8")
        assert "Edited fact." in content
        assert f"leaf_hash: {_leaf_hash(chunks2)}" in content


class TestGroupChunksByPage:
    def test_empty_returns_empty(self):
        assert _group_chunks_by_page([]) == []

    def test_single_page_preserves_chunk_order(self):
        chunks = [
            _make_chunk("a", page_start=1, chunk_index=0),
            _make_chunk("b", page_start=1, chunk_index=1),
        ]
        result = _group_chunks_by_page(chunks)
        assert len(result) == 1
        page_num, group = result[0]
        assert page_num == 1
        assert [c.chunk for c in group] == ["a", "b"]

    def test_sorts_by_page_number(self):
        chunks = [
            _make_chunk("z", page_start=5, chunk_index=0),
            _make_chunk("a", page_start=1, chunk_index=1),
            _make_chunk("m", page_start=3, chunk_index=2),
        ]
        result = _group_chunks_by_page(chunks)
        assert [page for page, _ in result] == [1, 3, 5]

    def test_non_contiguous_pages_kept_separately(self):
        chunks = [
            _make_chunk("a", page_start=1, chunk_index=0),
            _make_chunk("b", page_start=7, chunk_index=1),
        ]
        result = _group_chunks_by_page(chunks)
        assert [page for page, _ in result] == [1, 7]

    def test_non_paginated_source_single_bucket(self):
        """Chunks with page_start=0 (markdown, code, HTML) collapse to one entry."""
        chunks = [_make_chunk(f"c{i}", chunk_index=i) for i in range(4)]
        result = _group_chunks_by_page(chunks)
        assert len(result) == 1
        assert result[0][0] == 0
        assert len(result[0][1]) == 4


class TestPageSlug:
    def test_zero_padded_width_four(self):
        assert _page_slug("cv-manual", 1) == "cv-manual/page-0001"

    def test_page_zero(self):
        assert _page_slug("doc", 0) == "doc/page-0000"

    def test_large_page_number(self):
        assert _page_slug("book", 12345) == "book/page-12345"

    def test_preserves_double_dash_in_source_slug(self):
        assert _page_slug("nested--source", 42) == "nested--source/page-0042"


class TestLeafHash:
    def test_empty_returns_hash_of_empty(self):
        """Deterministic hash even for no chunks."""
        h = _leaf_hash([])
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex digest

    def test_same_chunks_same_hash(self):
        a = [_make_chunk("one"), _make_chunk("two", chunk_index=1)]
        b = [_make_chunk("one"), _make_chunk("two", chunk_index=1)]
        assert _leaf_hash(a) == _leaf_hash(b)

    def test_order_sensitive(self):
        a = [_make_chunk("one"), _make_chunk("two", chunk_index=1)]
        b = [_make_chunk("two", chunk_index=1), _make_chunk("one")]
        assert _leaf_hash(a) != _leaf_hash(b)

    def test_content_change_changes_hash(self):
        a = [_make_chunk("one")]
        b = [_make_chunk("two")]
        assert _leaf_hash(a) != _leaf_hash(b)

    def test_null_separator_prevents_concat_collision(self):
        """Chunk boundaries must affect the hash, not just the concatenated bytes."""
        a = [_make_chunk("ab"), _make_chunk("c", chunk_index=1)]
        b = [_make_chunk("a"), _make_chunk("bc", chunk_index=1)]
        assert _leaf_hash(a) != _leaf_hash(b)


class TestFindCachedLeaf:
    def _write(self, tmp_path: Path, subdir: str, slug: str, leaf_hash: str) -> Path:
        path = tmp_path / subdir / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        fm = f"---\nleaf_hash: {leaf_hash}\n---\nBody.\n" if leaf_hash else "Body.\n"
        path.write_text(fm, encoding="utf-8")
        return path

    def test_no_file_returns_none(self, tmp_path: Path):
        assert _find_cached_leaf(tmp_path, "src/page-0001", "h") is None

    def test_returns_summaries_path_on_match(self, tmp_path: Path):
        p = self._write(tmp_path, "summaries", "src/page-0001", "abcd")
        assert _find_cached_leaf(tmp_path, "src/page-0001", "abcd") == p

    def test_returns_drafts_path_on_match(self, tmp_path: Path):
        p = self._write(tmp_path, "drafts", "src/page-0001", "abcd")
        assert _find_cached_leaf(tmp_path, "src/page-0001", "abcd") == p

    def test_summaries_wins_when_both_match(self, tmp_path: Path):
        s = self._write(tmp_path, "summaries", "src/page-0001", "abcd")
        self._write(tmp_path, "drafts", "src/page-0001", "abcd")
        assert _find_cached_leaf(tmp_path, "src/page-0001", "abcd") == s

    def test_mismatch_returns_none(self, tmp_path: Path):
        self._write(tmp_path, "summaries", "src/page-0001", "abcd")
        assert _find_cached_leaf(tmp_path, "src/page-0001", "different") is None

    def test_missing_hash_in_frontmatter_is_not_a_match(self, tmp_path: Path):
        self._write(tmp_path, "summaries", "src/page-0001", "")
        assert _find_cached_leaf(tmp_path, "src/page-0001", "abcd") is None


class TestSourceSlug:
    def test_strips_extension(self):
        assert _source_slug("cv-manual.pdf") == "cv-manual"

    def test_replaces_slash_with_double_dash(self):
        assert _source_slug("folder/doc.md") == "folder--doc"

    def test_no_extension_kept(self):
        assert _source_slug("README") == "README"


def _wiki_node(
    slug: str = "01-chapter",
    parent_slug: str | None = None,
    depth: int = 1,
    ordinal: int = 0,
    title: str = "Chapter",
    page_start: int = 1,
    page_end: int = 1,
    kind: str = "chapter",
) -> WikiNode:
    return WikiNode(
        slug=slug,
        parent_slug=parent_slug,
        depth=depth,
        ordinal=ordinal,
        title=title,
        page_start=page_start,
        page_end=page_end,
        kind=kind,
        kreuzberg_node_id=f"node-{slug}",
    )


class TestChunksInPageRange:
    def test_inclusive_endpoints(self):
        chunks = [
            _make_chunk("a", page_start=1),
            _make_chunk("b", page_start=3, chunk_index=1),
            _make_chunk("c", page_start=5, chunk_index=2),
        ]
        result = _chunks_in_page_range(chunks, 1, 3)
        assert [c.chunk for c in result] == ["a", "b"]

    def test_empty_range_returns_empty(self):
        chunks = [_make_chunk("a", page_start=1)]
        assert _chunks_in_page_range(chunks, 5, 3) == []

    def test_no_matches_returns_empty(self):
        chunks = [_make_chunk("a", page_start=10)]
        assert _chunks_in_page_range(chunks, 1, 5) == []


class TestLeavesInRange:
    def test_returns_matching_paths_sorted(self, tmp_path: Path):
        paths = {3: tmp_path / "p3.md", 1: tmp_path / "p1.md", 5: tmp_path / "p5.md"}
        result = _leaves_in_range(paths, 1, 4)
        # Sorted by page number so the reduce prompt sees pages in order.
        assert result == [tmp_path / "p1.md", tmp_path / "p3.md"]

    def test_empty_range(self, tmp_path: Path):
        paths = {1: tmp_path / "p.md"}
        assert _leaves_in_range(paths, 5, 3) == []


class TestReadPageBody:
    def test_strips_frontmatter(self, tmp_path: Path):
        p = tmp_path / "p.md"
        p.write_text("---\ntitle: T\n---\nBody text.\n", encoding="utf-8")
        assert _read_page_body(p).rstrip() == "Body text."

    def test_no_frontmatter_returns_all(self, tmp_path: Path):
        p = tmp_path / "p.md"
        p.write_text("# Heading\nBody.\n", encoding="utf-8")
        assert _read_page_body(p) == "# Heading\nBody.\n"

    def test_unclosed_frontmatter_returns_all(self, tmp_path: Path):
        p = tmp_path / "p.md"
        p.write_text("---\ntitle: unclosed\nmore text\n", encoding="utf-8")
        assert _read_page_body(p).startswith("---")


class TestFormatChildrenForReduce:
    def test_joins_bodies_with_separator(self, tmp_path: Path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("---\ntitle: A\n---\nBody A.\n")
        b.write_text("---\ntitle: B\n---\nBody B.\n")
        result = _format_children_for_reduce([a, b])
        assert "Body A." in result
        assert "Body B." in result
        assert "[From a]" in result
        assert "[From b]" in result
        assert "\n\n---\n\n" in result

    def test_skips_empty_bodies(self, tmp_path: Path):
        a = tmp_path / "a.md"
        a.write_text("---\ntitle: A\n---\n\n", encoding="utf-8")
        result = _format_children_for_reduce([a])
        assert result == ""


class TestIsDraftPath:
    def test_draft_directory(self, tmp_path: Path):
        assert _is_draft_path(tmp_path / "drafts" / "x.md", tmp_path) is True

    def test_summaries_directory(self, tmp_path: Path):
        assert _is_draft_path(tmp_path / "summaries" / "x.md", tmp_path) is False

    def test_unrelated_path_is_not_draft(self, tmp_path: Path):
        assert _is_draft_path(Path("/elsewhere/x.md"), tmp_path) is False


class TestWikiNodesBottomUp:
    def test_deepest_first(self):
        nodes = [
            _wiki_node(slug="01-a", depth=1),
            _wiki_node(slug="01-a/01-b", parent_slug="01-a", depth=2),
            _wiki_node(slug="01-a/01-b/01-c", parent_slug="01-a/01-b", depth=3),
        ]
        ordered = _wiki_nodes_bottom_up(nodes)
        assert [n.depth for n in ordered] == [3, 2, 1]


class TestInnerNodeTarget:
    def test_builds_nested_index_path(self, tmp_path: Path):
        got = _inner_node_target(tmp_path, "summaries", "cv-manual", "01-chapter/02-section")
        assert got == (
            tmp_path / "summaries" / "cv-manual" / "01-chapter" / "02-section" / "index.md"
        )


class TestLoadDocumentStructure:
    def test_no_record_returns_empty(self):
        store = MagicMock(spec=Store)
        store.get_document_structure.return_value = None
        assert _load_document_structure(store, "doc.pdf") == []

    def test_non_dict_record_returns_empty(self):
        store = MagicMock(spec=Store)
        store.get_document_structure.return_value = "not a dict"
        assert _load_document_structure(store, "doc.pdf") == []

    def test_invalid_json_returns_empty(self):
        store = MagicMock(spec=Store)
        store.get_document_structure.return_value = {"document_json": "{bad"}
        assert _load_document_structure(store, "doc.pdf") == []

    def test_missing_json_field_returns_empty(self):
        store = MagicMock(spec=Store)
        store.get_document_structure.return_value = {"other": "fields"}
        assert _load_document_structure(store, "doc.pdf") == []

    def test_walks_valid_structure(self):
        import json as _json

        store = MagicMock(spec=Store)
        store.get_document_structure.return_value = {
            "document_json": _json.dumps(
                {
                    "nodes": [
                        {
                            "id": "h",
                            "content": {"node_type": "heading", "level": 1, "text": "Chapter"},
                            "page": 1,
                            "children": [],
                        }
                    ]
                }
            )
        }
        result = _load_document_structure(store, "doc.pdf")
        assert len(result) == 1
        assert result[0].title == "Chapter"


class TestCollectInnerChildren:
    def test_uses_wiki_children_when_present(self, tmp_path: Path):
        parent = _wiki_node(slug="01-a", depth=1)
        child = _wiki_node(slug="01-a/01-b", parent_slug="01-a", depth=2)
        child_path = tmp_path / "child.md"
        child_path.write_text("body")
        paths, partial = _collect_inner_children(
            parent, [parent, child], {child.slug: child_path}, {}
        )
        assert paths == [child_path]
        assert partial == []

    def test_missing_wiki_child_marks_partial(self, tmp_path: Path):
        parent = _wiki_node(slug="01-a", depth=1)
        c1 = _wiki_node(slug="01-a/01-b", parent_slug="01-a", depth=2, ordinal=0)
        c2 = _wiki_node(slug="01-a/02-c", parent_slug="01-a", depth=2, ordinal=1)
        good = tmp_path / "good.md"
        good.write_text("body")
        paths, partial = _collect_inner_children(parent, [parent, c1, c2], {c1.slug: good}, {})
        assert paths == [good]
        assert partial == ["01-a/02-c"]

    def test_falls_back_to_page_leaves(self, tmp_path: Path):
        leaf_node = _wiki_node(
            slug="01-page-section",
            depth=2,
            parent_slug="01-a",
            page_start=1,
            page_end=3,
        )
        leaves = {
            1: tmp_path / "p1.md",
            2: tmp_path / "p2.md",
            5: tmp_path / "p5.md",
        }
        paths, partial = _collect_inner_children(leaf_node, [leaf_node], {}, leaves)
        assert paths == [tmp_path / "p1.md", tmp_path / "p2.md"]
        assert partial == []


class TestGenerateInnerNode:
    def _setup(self, isolated_env: Path):
        source = isolated_env / "documents" / "doc.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("content")
        leaf_path = isolated_env / "wiki" / "summaries" / "doc" / "page-0001.md"
        leaf_path.parent.mkdir(parents=True)
        leaf_path.write_text(
            "---\ntitle: P1\nfaithfulness_score: 0.90\n---\n> Leaf fact.[^src1]\n\nDetails.\n",
            encoding="utf-8",
        )
        return leaf_path

    def test_writes_index_md(self, isolated_env: Path):
        leaf_path = self._setup(isolated_env)
        chunks = [_make_chunk("Leaf fact.", page_start=1, page_end=1)]
        provider = MagicMock()
        provider.chat.side_effect = ["# Section\n\n> Section body.\n", "0.9"]
        node = _wiki_node(slug="01-chapter", page_start=1, page_end=1)
        out = _generate_inner_node(
            node=node,
            source_name="doc.pdf",
            source_slug="doc",
            children_paths=[leaf_path],
            raw_chunks=chunks,
            partial_paths=[],
            provider=provider,
            config=cfg,
            on_progress=None,
        )
        assert out is not None
        assert out.name == "index.md"
        assert "summaries/doc/01-chapter/index.md" in str(out).replace("\\", "/")
        content = out.read_text(encoding="utf-8")
        assert "faithfulness_score: 0.90" in content
        assert "kind: chapter" in content

    def test_low_score_goes_to_drafts(self, isolated_env: Path):
        leaf_path = self._setup(isolated_env)
        chunks = [_make_chunk("Leaf fact.", page_start=1, page_end=1)]
        provider = MagicMock()
        provider.chat.side_effect = ["# S\nbody\n", "0.3"]
        node = _wiki_node(slug="01-chapter", page_start=1, page_end=1)
        out = _generate_inner_node(
            node=node,
            source_name="doc.pdf",
            source_slug="doc",
            children_paths=[leaf_path],
            raw_chunks=chunks,
            partial_paths=[],
            provider=provider,
            config=cfg,
            on_progress=None,
        )
        assert out is not None
        assert "drafts/doc/01-chapter" in str(out).replace("\\", "/")

    def test_partial_marks_frontmatter(self, isolated_env: Path):
        leaf_path = self._setup(isolated_env)
        chunks = [_make_chunk("Leaf fact.", page_start=1, page_end=1)]
        provider = MagicMock()
        provider.chat.side_effect = ["# S\nbody\n", "0.9"]
        node = _wiki_node(slug="01-chapter", page_start=1, page_end=1)
        out = _generate_inner_node(
            node=node,
            source_name="doc.pdf",
            source_slug="doc",
            children_paths=[leaf_path],
            raw_chunks=chunks,
            partial_paths=["01-chapter/02-missing"],
            provider=provider,
            config=cfg,
            on_progress=None,
        )
        assert out is not None
        content = out.read_text(encoding="utf-8")
        assert "partial: true" in content
        assert "01-chapter/02-missing" in content

    def test_no_children_returns_none(self, isolated_env: Path):
        provider = MagicMock()
        node = _wiki_node(slug="01-chapter")
        out = _generate_inner_node(
            node=node,
            source_name="doc.pdf",
            source_slug="doc",
            children_paths=[],
            raw_chunks=[],
            partial_paths=[],
            provider=provider,
            config=cfg,
            on_progress=None,
        )
        assert out is None
        provider.chat.assert_not_called()

    def test_empty_child_bodies_returns_none(self, isolated_env: Path, tmp_path: Path):
        empty = tmp_path / "empty.md"
        empty.write_text("---\ntitle: E\n---\n\n", encoding="utf-8")
        node = _wiki_node(slug="01-chapter")
        provider = MagicMock()
        out = _generate_inner_node(
            node=node,
            source_name="doc.pdf",
            source_slug="doc",
            children_paths=[empty],
            raw_chunks=[],
            partial_paths=[],
            provider=provider,
            config=cfg,
            on_progress=None,
        )
        assert out is None
        provider.chat.assert_not_called()

    def test_llm_failure_returns_none(self, isolated_env: Path):
        leaf_path = self._setup(isolated_env)
        chunks = [_make_chunk("Leaf fact.", page_start=1, page_end=1)]
        provider = MagicMock()
        provider.chat.side_effect = ConnectionError("LLM down")
        node = _wiki_node(slug="01-chapter", page_start=1, page_end=1)
        stages: list[str] = []
        out = _generate_inner_node(
            node=node,
            source_name="doc.pdf",
            source_slug="doc",
            children_paths=[leaf_path],
            raw_chunks=chunks,
            partial_paths=[],
            provider=provider,
            config=cfg,
            on_progress=lambda stage, data: stages.append(stage),
        )
        assert out is None
        assert "failed" in stages

    def test_empty_llm_response_returns_none(self, isolated_env: Path):
        leaf_path = self._setup(isolated_env)
        chunks = [_make_chunk("Leaf fact.", page_start=1, page_end=1)]
        provider = MagicMock()
        provider.chat.side_effect = ["   ", "0.9"]
        node = _wiki_node(slug="01-chapter", page_start=1, page_end=1)
        out = _generate_inner_node(
            node=node,
            source_name="doc.pdf",
            source_slug="doc",
            children_paths=[leaf_path],
            raw_chunks=chunks,
            partial_paths=[],
            provider=provider,
            config=cfg,
            on_progress=None,
        )
        assert out is None

    def test_faithfulness_falls_back_to_children_when_no_chunks_in_range(self, isolated_env: Path):
        """When a node's raw-chunk range is empty, fall back to grounding in children."""
        leaf_path = self._setup(isolated_env)
        provider = MagicMock()
        provider.chat.side_effect = ["# S\nbody\n", "0.9"]
        node = _wiki_node(slug="01-chapter", page_start=10, page_end=20)
        out = _generate_inner_node(
            node=node,
            source_name="doc.pdf",
            source_slug="doc",
            children_paths=[leaf_path],
            raw_chunks=[_make_chunk("different page", page_start=1)],
            partial_paths=[],
            provider=provider,
            config=cfg,
            on_progress=None,
        )
        assert out is not None
        # Two chat calls even when the range is empty -- generate + faithfulness.
        assert provider.chat.call_count == 2


class TestGenerateSummaryPageTreeIntegration:
    """End-to-end: structure persisted -> leaves + inner nodes land on disk."""

    def test_leaves_only_when_no_structure(self, isolated_env: Path):
        source = isolated_env / "documents" / "doc.pdf"
        source.write_text("content")
        chunks = [_make_chunk("Leaf fact.", page_start=1, page_end=1)]
        wiki_text = (
            "# P1\n\n> Leaf fact.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.pdf, excerpt: "Leaf fact."'
        )
        provider = _mock_provider(wiki_text)
        store = _mock_store()  # get_document_structure returns None
        pages = generate_summary_page("doc.pdf", chunks, provider, store)
        assert len(pages) == 1
        assert "page-0001.md" in str(pages[0])

    def test_structure_triggers_inner_node(self, isolated_env: Path):
        import json as _json

        source = isolated_env / "documents" / "doc.pdf"
        source.write_text("content")
        chunks = [_make_chunk("Leaf fact.", page_start=1, page_end=1)]
        leaf_wiki = (
            "# P1\n\n> Leaf fact.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.pdf, excerpt: "Leaf fact."'
        )
        # Provider: leaf gen + leaf faithfulness + inner gen + inner faithfulness.
        provider = MagicMock()
        provider.chat.side_effect = [leaf_wiki, "0.9", "# Section\n\nBody.\n", "0.9"]
        store = _mock_store()
        store.get_document_structure.return_value = {
            "document_json": _json.dumps(
                {
                    "nodes": [
                        {
                            "id": "h",
                            "content": {"node_type": "heading", "level": 1, "text": "Chapter"},
                            "page": 1,
                            "children": [],
                        }
                    ]
                }
            )
        }
        pages = generate_summary_page("doc.pdf", chunks, provider, store)
        assert len(pages) == 2
        paths = [str(p).replace("\\", "/") for p in pages]
        assert any("page-0001.md" in p for p in paths)
        assert any("01-chapter/index.md" in p for p in paths)


class TestMakeSlug:
    def test_spaces_to_dashes(self):
        assert make_slug("gradual typing") == "gradual-typing"

    def test_slashes_to_double_dashes(self):
        assert make_slug("path/to/concept") == "path--to--concept"

    def test_lowercase(self):
        assert make_slug("Python Types") == "python-types"

    def test_strips_special_characters(self):
        assert make_slug("hello! world?") == "hello-world"

    def test_preserves_hyphens(self):
        assert make_slug("well-known") == "well-known"


class TestMatchCitationSource:
    def test_matches_filename_in_ref(self):
        sources = ["doc1.md", "doc2.md"]
        assert _match_citation_source("doc2.md, excerpt: ...", sources) == "doc2.md"

    def test_no_match_returns_empty(self):
        assert _match_citation_source("unknown.md, ...", ["doc.md"]) == ""


class TestFindExcerptSource:
    def test_finds_source_containing_excerpt(self):
        chunks = {
            "a.md": [_make_chunk("Alpha content", source="a.md")],
            "b.md": [_make_chunk("Beta content", source="b.md")],
        }
        assert _find_excerpt_source("Beta content", chunks) == "b.md"

    def test_empty_excerpt_returns_empty(self):
        assert _find_excerpt_source("", {"a.md": [_make_chunk("text")]}) == ""

    def test_not_found_returns_empty(self):
        chunks = {"a.md": [_make_chunk("Unrelated")]}
        assert _find_excerpt_source("Missing text", chunks) == ""


class TestResolveMultiSourceCitations:
    def test_resolves_to_correct_source(self):
        chunks_a = [_make_chunk("Alpha fact.", source="a.md", page_start=1, page_end=1)]
        chunks_b = [_make_chunk("Beta fact.", source="b.md")]
        parsed = [
            ParsedCitation("src1", 'a.md, excerpt: "Alpha fact."', 1),
            ParsedCitation("src2", 'b.md, excerpt: "Beta fact."', 2),
        ]
        records = _resolve_multi_source_citations(
            parsed,
            ["a.md", "b.md"],
            {"a.md": "h1", "b.md": "h2"},
            {"a.md": chunks_a, "b.md": chunks_b},
        )
        assert len(records) == 2
        assert records[0]["source_filename"] == "a.md"
        assert records[0]["page_start"] == 1
        assert records[1]["source_filename"] == "b.md"

    def test_falls_back_to_excerpt_search(self):
        chunks = {"a.md": [_make_chunk("Special text", source="a.md")]}
        parsed = [ParsedCitation("src1", 'excerpt: "Special text"', 1)]
        records = _resolve_multi_source_citations(
            parsed,
            ["a.md"],
            {"a.md": "h"},
            chunks,
        )
        assert records[0]["source_filename"] == "a.md"

    def test_falls_back_to_first_source(self):
        parsed = [ParsedCitation("src1", 'excerpt: "Not found anywhere"', 1)]
        records = _resolve_multi_source_citations(
            parsed,
            ["fallback.md"],
            {},
            {},
        )
        assert records[0]["source_filename"] == "fallback.md"


def _synthesis_wiki_text(sources: list[str]) -> str:
    """Build a valid synthesis wiki text with citations to the given sources."""
    lines = ["# Synthesis\n"]
    cite_lines = [
        "---",
        "<!-- citations (auto-generated from _citations table -- do not edit) -->",
    ]
    for i, src in enumerate(sources, 1):
        lines.append(f"> Fact from {src}.[^src{i}]\n")
        cite_lines.append(f'[^src{i}]: {src}, excerpt: "Fact from {src}."')
    return "\n".join(lines) + "\n" + "\n".join(cite_lines)


class TestGenerateSynthesisPage:
    def test_generates_synthesis_page(self, tmp_path: Path):
        sources = ["a.md", "b.md", "c.md"]
        for name in sources:
            (tmp_path / "documents" / name).write_text(f"Fact from {name}.")

        chunks_by_source = {
            name: [_make_chunk(f"Fact from {name}.", source=name)] for name in sources
        }
        wiki_text = _synthesis_wiki_text(sources)
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        result = _generate_synthesis_page(
            "gradual typing",
            sources,
            chunks_by_source,
            provider,
            store,
            cfg,
        )
        assert result is not None
        assert result.exists()
        assert "synthesis" in str(result)
        assert result.name == "gradual-typing.md"
        content = result.read_text()
        assert "generated_by: test-model" in content
        assert 'sources: ["a.md", "b.md", "c.md"]' in content
        assert "faithfulness_score: 0.85" in content
        store.add_citations.assert_called_once()

    def test_low_score_goes_to_drafts(self, tmp_path: Path):
        sources = ["a.md", "b.md", "c.md"]
        for name in sources:
            (tmp_path / "documents" / name).write_text(f"Fact from {name}.")

        chunks_by_source = {
            name: [_make_chunk(f"Fact from {name}.", source=name)] for name in sources
        }
        wiki_text = _synthesis_wiki_text(sources)
        provider = _mock_provider(wiki_text, faith_score="0.3")
        store = _mock_store()

        result = _generate_synthesis_page(
            "topic",
            sources,
            chunks_by_source,
            provider,
            store,
            cfg,
        )
        assert result is not None
        assert "drafts" in str(result)

    def test_no_chunks_returns_none(self):
        provider = MagicMock()
        store = _mock_store()
        result = _generate_synthesis_page("topic", ["a.md"], {}, provider, store, cfg)
        assert result is None
        provider.chat.assert_not_called()

    def test_llm_failure_returns_none(self, tmp_path: Path):
        chunks_by_source = {"a.md": [_make_chunk("text", source="a.md")]}
        provider = MagicMock()
        provider.chat.side_effect = ConnectionError("down")
        store = _mock_store()

        result = _generate_synthesis_page(
            "topic",
            ["a.md"],
            chunks_by_source,
            provider,
            store,
            cfg,
        )
        assert result is None

    def test_no_valid_citations_returns_none(self, tmp_path: Path):
        chunks_by_source = {"a.md": [_make_chunk("real text", source="a.md")]}
        wiki_text = (
            "# Bad\n\n"
            "> Fabricated.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: a.md, excerpt: "Not in any chunk at all"'
        )
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        result = _generate_synthesis_page(
            "topic",
            ["a.md"],
            chunks_by_source,
            provider,
            store,
            cfg,
        )
        assert result is None

    def test_faithfulness_failure_uses_zero(self, tmp_path: Path):
        sources = ["a.md"]
        (tmp_path / "documents" / "a.md").write_text("Fact from a.md.")
        chunks_by_source = {"a.md": [_make_chunk("Fact from a.md.", source="a.md")]}
        wiki_text = _synthesis_wiki_text(sources)
        provider = MagicMock()
        provider.chat.side_effect = [wiki_text, ConnectionError("down")]
        store = _mock_store()

        result = _generate_synthesis_page(
            "topic",
            sources,
            chunks_by_source,
            provider,
            store,
            cfg,
        )
        assert result is not None
        assert "drafts" in str(result)

    def test_llm_returns_empty_string(self, tmp_path: Path):
        chunks_by_source = {"a.md": [_make_chunk("text", source="a.md")]}
        provider = _mock_provider("   ")
        store = _mock_store()

        result = _generate_synthesis_page(
            "topic",
            ["a.md"],
            chunks_by_source,
            provider,
            store,
            cfg,
        )
        assert result is None

    def test_inference_citations_pass_verification(self, tmp_path: Path):
        sources = ["a.md"]
        (tmp_path / "documents" / "a.md").write_text("Fact from a.md.")
        chunks_by_source = {"a.md": [_make_chunk("Fact from a.md.", source="a.md")]}
        wiki_text = (
            "# Synthesis\n\n"
            "A cross-source observation.[*inference*]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: a.md, no excerpt"
        )
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        result = _generate_synthesis_page(
            "topic",
            sources,
            chunks_by_source,
            provider,
            store,
            cfg,
        )
        assert result is not None
        store.add_citations.assert_called_once()


class _FakeClusterer:
    """Test double implementing the SourceClusterer protocol."""

    def __init__(self, clusters: list) -> None:
        self._clusters = clusters

    def available(self) -> bool:
        return True

    def get_clusters(self, min_sources: int = 3):
        return list(self._clusters)


class TestGenerateSynthesisPages:
    def test_no_clusters_returns_empty(self, tmp_path: Path):
        store = _mock_store()
        provider = MagicMock()
        clusterer = _FakeClusterer([])
        result = generate_synthesis_pages(provider, store, clusterer)
        assert result == []

    def test_skips_clusters_with_insufficient_chunks(self, tmp_path: Path):
        from lilbee.clustering import SourceCluster

        store = _mock_store()
        # Only 2 sources have chunks (need 3)
        store.get_chunks_by_source.side_effect = lambda name: (
            [_make_chunk("text", source=name)] if name != "c.md" else []
        )
        provider = MagicMock()
        clusterer = _FakeClusterer(
            [
                SourceCluster(
                    cluster_id="x", label="topic", sources=frozenset({"a.md", "b.md", "c.md"})
                )
            ]
        )

        result = generate_synthesis_pages(provider, store, clusterer)
        assert result == []
        provider.chat.assert_not_called()

    def test_generates_page_for_qualifying_cluster(self, tmp_path: Path):
        from lilbee.clustering import SourceCluster

        sources = ["a.md", "b.md", "c.md"]
        for name in sources:
            (tmp_path / "documents" / name).write_text(f"Fact from {name}.")

        store = _mock_store()
        store.get_chunks_by_source.side_effect = lambda name: [
            _make_chunk(f"Fact from {name}.", source=name)
        ]

        wiki_text = _synthesis_wiki_text(sources)
        provider = _mock_provider(wiki_text)
        clusterer = _FakeClusterer(
            [
                SourceCluster(
                    cluster_id="x",
                    label="gradual typing",
                    sources=frozenset(sources),
                )
            ]
        )

        result = generate_synthesis_pages(provider, store, clusterer)
        assert len(result) == 1
        assert result[0].exists()
        assert "synthesis" in str(result[0]) or "drafts" in str(result[0])

    def test_failed_page_generation_omitted(self, tmp_path: Path):
        from lilbee.clustering import SourceCluster

        store = _mock_store()
        # Returning empty chunks for every source means no cluster qualifies.
        store.get_chunks_by_source.side_effect = lambda name: []
        provider = _mock_provider("")
        clusterer = _FakeClusterer(
            [
                SourceCluster(
                    cluster_id="x",
                    label="topic",
                    sources=frozenset({"a.md", "b.md", "c.md"}),
                )
            ]
        )

        result = generate_synthesis_pages(provider, store, clusterer)
        assert result == []


class TestContentChangeRatio:
    def test_identical_texts(self):
        assert _content_change_ratio("a\nb\nc", "a\nb\nc") == 0.0

    def test_completely_different(self):
        assert _content_change_ratio("a\nb\nc", "x\ny\nz") == 1.0

    def test_partial_change(self):
        old = "line1\nline2\nline3\nline4"
        new = "line1\nchanged\nline3\nline4"
        ratio = _content_change_ratio(old, new)
        assert 0.0 < ratio < 1.0

    def test_empty_old(self):
        # empty -> something = 100% change
        assert _content_change_ratio("", "new content") == 1.0

    def test_empty_both(self):
        assert _content_change_ratio("", "") == 0.0


class TestDiffSummary:
    def test_produces_unified_diff(self):
        result = _diff_summary("old line", "new line")
        assert "---" in result or "-old line" in result

    def test_truncates_long_diff(self):
        old = "\n".join(f"line{i}" for i in range(50))
        new = "\n".join(f"changed{i}" for i in range(50))
        result = _diff_summary(old, new)
        assert "more lines" in result


class TestDivertToDrafts:
    def test_writes_draft_with_note(self, tmp_path: Path):
        drafts_dir = tmp_path / "drafts"
        content = "# New Page\n\nNew content."
        result = _divert_to_drafts(content, drafts_dir, "my-page", 0.45, "diff text")
        assert result.exists()
        assert result.parent == drafts_dir
        text = result.read_text()
        assert "DRIFT" in text
        assert "45%" in text
        assert "human review" in text
        assert content in text


class TestSummaryDriftDetection:
    """Drift detection during summary page regeneration."""

    def test_drift_diverts_to_drafts(self, tmp_path: Path):
        """When >30% of content changes on a regeneration, the new version goes to drafts."""
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Python supports gradual typing.")
        chunks = [_make_chunk("Python supports gradual typing.")]

        # Pre-seed the per-page file at its new tree location with wildly different content.
        existing = tmp_path / "wiki" / "summaries" / "doc" / "page-0000.md"
        existing.parent.mkdir(parents=True)
        existing.write_text("---\ngenerated_by: old-model\n---\n\nCompletely different content.\n")

        wiki_text = (
            "# Doc Summary\n\n"
            "> Python supports gradual typing.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "Python supports gradual typing."'
        )
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages) == 1
        assert "drafts" in str(pages[0])
        # Original page should be unchanged.
        assert "Completely different content" in existing.read_text()
        # Draft should have drift note.
        draft_text = pages[0].read_text()
        assert "DRIFT" in draft_text

    def test_small_change_overwrites(self, tmp_path: Path):
        """When content barely changes, existing page is overwritten normally."""
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

        provider = _mock_provider(wiki_text, faith_score="0.85")
        store = _mock_store()

        # First generation
        pages1 = generate_summary_page("doc.md", chunks, provider, store)
        assert len(pages1) == 1
        assert "summaries" in str(pages1[0])

        # Regenerate with same content; provider returns same text.
        provider2 = _mock_provider(wiki_text, faith_score="0.85")
        store2 = _mock_store()
        pages2 = generate_summary_page("doc.md", chunks, provider2, store2)
        # Small diff (only timestamp) should overwrite, not divert.
        assert len(pages2) == 1
        # Should still be in summaries (not drafts) since content is nearly identical.
        assert "summaries" in str(pages2[0])


class TestSynthesisDriftDetection:
    """Drift detection during synthesis page regeneration."""

    def test_drift_diverts_synthesis_to_drafts(self, tmp_path: Path):
        """Synthesis pages also get drift-checked."""
        sources = ["a.md", "b.md", "c.md"]
        for name in sources:
            (tmp_path / "documents" / name).write_text(f"Fact from {name}.")

        # Write an existing synthesis page with very different content
        synthesis_dir = tmp_path / "wiki" / "synthesis"
        synthesis_dir.mkdir(parents=True)
        existing = synthesis_dir / "gradual-typing.md"
        existing.write_text("---\ngenerated_by: old\n---\n\nTotally different synthesis.\n")

        chunks_by_source = {
            name: [_make_chunk(f"Fact from {name}.", source=name)] for name in sources
        }
        wiki_text = _synthesis_wiki_text(sources)
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        result = _generate_synthesis_page(
            "gradual typing", sources, chunks_by_source, provider, store, cfg
        )
        assert result is not None
        assert "drafts" in str(result)
        # Original should be unchanged
        assert "Totally different synthesis" in existing.read_text()


class TestProgressCallback:
    """Test the on_progress callback in generate_summary_page."""

    def test_callback_receives_stages(self, tmp_path: Path):
        """on_progress is called with preparing, generating, faithfulness_check stages."""
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
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        stages: list[str] = []

        def on_progress(stage: str, data: dict) -> None:
            stages.append(stage)

        pages = generate_summary_page("doc.md", chunks, provider, store, on_progress=on_progress)
        assert len(pages) == 1
        assert "preparing" in stages
        assert "generating" in stages
        assert "faithfulness_check" in stages

    def test_callback_none_is_safe(self, tmp_path: Path):
        """on_progress=None (default) does not raise."""
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
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        pages = generate_summary_page("doc.md", chunks, provider, store, on_progress=None)
        assert len(pages) == 1
