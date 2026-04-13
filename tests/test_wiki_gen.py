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
    _chunks_to_text,
    _content_change_ratio,
    _diff_summary,
    _divert_to_drafts,
    _extract_excerpt,
    _find_excerpt_source,
    _generate_synthesis_page,
    _match_citation_source,
    _parse_faithfulness_score,
    _resolve_citations,
    _resolve_multi_source_citations,
    _truncate_chunks_to_budget,
    _verify_citations,
    generate_summary_page,
    generate_synthesis_pages,
)
from lilbee.wiki.shared import make_slug


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
        provider = _mock_provider(wiki_text, faith_score="0.3")
        store = _mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
        assert "drafts" in str(result)

    def test_empty_chunks_returns_none(self):
        provider = MagicMock()
        store = _mock_store()
        result = generate_summary_page("doc.md", [], provider, store)
        assert result is None
        provider.chat.assert_not_called()

    def test_llm_failure_returns_none(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = MagicMock()
        provider.chat.side_effect = ConnectionError("LLM down")
        store = _mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is None

    def test_no_valid_citations_returns_none(self, tmp_path: Path):
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
        store = _mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
        assert "drafts" in str(result)  # score=0.0 < threshold=0.7

    def test_llm_returns_empty_string(self, tmp_path: Path):
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Content")
        chunks = [_make_chunk("Content")]
        provider = _mock_provider("   ")  # whitespace-only -> empty after strip
        store = _mock_store()

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is None

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

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
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

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
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

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
        content = result.read_text()
        assert "<think>" not in content
        assert "Let me reason" not in content
        assert "# Doc Summary" in content


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
        """When >30% of content changes, new version goes to drafts."""
        source = tmp_path / "documents" / "doc.md"
        source.write_text("Python supports gradual typing.")
        chunks = [_make_chunk("Python supports gradual typing.")]

        # Write an existing page with very different content
        wiki_root = tmp_path / "wiki" / "summaries"
        wiki_root.mkdir(parents=True)
        existing = wiki_root / "doc.md"
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

        result = generate_summary_page("doc.md", chunks, provider, store)
        assert result is not None
        assert "drafts" in str(result)
        # Original page should be unchanged
        assert "Completely different content" in existing.read_text()
        # Draft should have drift note
        draft_text = result.read_text()
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

        # Write an existing page with nearly identical content (only timestamp differs)
        provider = _mock_provider(wiki_text, faith_score="0.85")
        store = _mock_store()

        # First generation
        result1 = generate_summary_page("doc.md", chunks, provider, store)
        assert result1 is not None
        assert "summaries" in str(result1)

        # Regenerate with same content — provider returns same text
        provider2 = _mock_provider(wiki_text, faith_score="0.85")
        store2 = _mock_store()
        result2 = generate_summary_page("doc.md", chunks, provider2, store2)
        # Small diff (only timestamp) should overwrite, not divert
        assert result2 is not None
        # Should still be in summaries (not drafts) since content is nearly identical
        assert "summaries" in str(result2)


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

        result = generate_summary_page("doc.md", chunks, provider, store, on_progress=on_progress)
        assert result is not None
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

        result = generate_summary_page("doc.md", chunks, provider, store, on_progress=None)
        assert result is not None
