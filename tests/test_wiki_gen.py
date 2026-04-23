"""Tests for wiki page generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.config import CHUNKS_TABLE, cfg
from lilbee.store import CHUNK_TYPE_WIKI, SearchChunk, Store
from lilbee.wiki.citation import ParsedCitation
from lilbee.wiki.gen import (
    _check_faithfulness,
    _chunks_to_text,
    _content_change_ratio,
    _diff_summary,
    _divert_to_drafts,
    _extract_excerpt,
    _find_cached_leaf,
    _find_excerpt_source,
    _generate_synthesis_page,
    _group_chunks_by_page,
    _leaf_hash,
    _match_citation_source,
    _parse_faithfulness_score,
    _resolve_citations,
    _resolve_multi_source_citations,
    _truncate_chunks_to_budget,
    _verify_citations,
    generate_synthesis_pages,
    index_wiki_page,
)
from lilbee.wiki.shared import (
    CONCEPTS_SUBDIR,
    DRAFTS_SUBDIR,
    PageTarget,
    make_slug,
)


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


@pytest.fixture(autouse=True)
def _stub_wiki_index_services(monkeypatch):
    """Stub ``get_services`` inside ``wiki.gen`` so tests that drive
    ``_persist_and_finalize`` don't hit the real provider when the new
    wiki-body indexer runs. ``TestWikiIndexing`` re-patches explicitly
    to exercise the indexer's own assertions.
    """
    svc = MagicMock()
    svc.embedder.embed_batch.side_effect = lambda texts, **kw: [
        [0.1] * cfg.embedding_dim for _ in texts
    ]
    monkeypatch.setattr("lilbee.wiki.gen.get_services", lambda: svc)
    return svc


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


def _mock_provider(
    wiki_text: str,
    faith_score: str = "0.85",
    capabilities: list[str] | None = None,
) -> MagicMock:
    provider = MagicMock()
    provider.chat.side_effect = [wiki_text, faith_score]
    provider.get_capabilities.return_value = (
        list(capabilities) if capabilities is not None else ["completion"]
    )
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

    def test_decodes_escaped_newlines(self):
        """Models that emit ``\\n`` as a backslash-n escape get the real newline back."""
        result = _extract_excerpt('doc.md, excerpt: "Warning\\nWhen you see this symbol"')
        assert result == "Warning\nWhen you see this symbol"

    def test_decodes_escaped_tab_and_backslash_and_quote(self):
        result = _extract_excerpt('doc.md, excerpt: "a\\tb\\\\c')  # unclosed so raw path runs too
        assert result == "a\tb\\c"

    def test_leaves_unknown_escape_untouched(self):
        """An unrecognized escape (``\\x``) is kept verbatim, not mangled."""
        result = _extract_excerpt('doc.md, excerpt: "hex \\x41"')
        assert result == "hex \\x41"


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

    def test_keeps_excerpts_that_differ_only_in_whitespace(self):
        """Source with a mid-sentence newline still matches an LLM quote that collapsed it."""
        from lilbee.store import CitationRecord

        chunks = [
            _make_chunk(
                "Congratulations on acquiring your new Ford Motor Company product.\n"
                "Please take the time to get well acquainted with your vehicle."
            )
        ]
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
                "excerpt": (
                    "Ford Motor Company product. Please take the time "
                    "to get well acquainted with your vehicle."
                ),
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


class TestBuildWikiMessages:
    def test_thinking_capability_prepends_no_think_directive(self):
        from lilbee.wiki.gen import _build_wiki_messages

        provider = MagicMock()
        provider.get_capabilities.return_value = ["completion", "thinking"]
        cfg.chat_model = "any-model"

        messages = _build_wiki_messages("Summarize these chunks.", provider, cfg)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.startswith("/no_think\n\n")
        assert "Summarize these chunks." in content
        provider.get_capabilities.assert_called_once()

    def test_no_thinking_capability_passes_prompt_through(self):
        from lilbee.wiki.gen import _build_wiki_messages

        provider = MagicMock()
        provider.get_capabilities.return_value = ["completion"]
        cfg.chat_model = "any-model"

        messages = _build_wiki_messages("Summarize these chunks.", provider, cfg)
        assert messages == [{"role": "user", "content": "Summarize these chunks."}]

    def test_empty_capabilities_passes_prompt_through(self):
        """Backends that don't report capabilities leave the prompt untouched."""
        from lilbee.wiki.gen import _build_wiki_messages

        provider = MagicMock()
        provider.get_capabilities.return_value = []
        cfg.chat_model = "any-model"

        messages = _build_wiki_messages("Summarize these chunks.", provider, cfg)
        assert messages == [{"role": "user", "content": "Summarize these chunks."}]


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

    def test_passes_faithfulness_max_tokens_cap(self):
        """Faithfulness chat is capped by cfg.wiki_faithfulness_max_tokens.

        Without this cap a reasoning model (Qwen3, DeepSeek-R1) can burn
        the whole context window thinking before emitting the number.
        """
        provider = MagicMock()
        provider.chat.return_value = "0.85"
        cfg.wiki_faithfulness_max_tokens = 42
        _check_faithfulness("chunks", "wiki", provider, "test", cfg)
        _, kwargs = provider.chat.call_args
        assert kwargs["options"]["max_tokens"] == 42

    def test_uses_wiki_temperature(self):
        """Faithfulness chat uses the lower wiki temperature, not the chat default."""
        provider = MagicMock()
        provider.chat.return_value = "0.85"
        cfg.wiki_temperature = 0.05
        _check_faithfulness("chunks", "wiki", provider, "test", cfg)
        _, kwargs = provider.chat.call_args
        assert kwargs["options"]["temperature"] == 0.05


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


class TestWikiIndexing:
    """``index_wiki_page`` chunks, embeds and writes wiki page bodies."""

    @staticmethod
    def _target(subdir: str = CONCEPTS_SUBDIR, slug: str = "brakes") -> PageTarget:
        wiki_root = cfg.data_root / cfg.wiki_dir
        return PageTarget(
            wiki_root=wiki_root,
            subdir=subdir,
            slug=slug,
            wiki_source=f"{cfg.wiki_dir}/{subdir}/{slug}.md",
            page_type=subdir.rstrip("s"),  # summaries -> summary, concepts -> concept
            label=slug,
        )

    @staticmethod
    def _services_mock(vector_dim: int | None = None) -> MagicMock:
        dim = vector_dim if vector_dim is not None else cfg.embedding_dim
        svc = MagicMock()
        svc.embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * dim for _ in texts]
        return svc

    @staticmethod
    def _content(body: str) -> str:
        return (
            "---\ntitle: Brakes\ntype: concept\n---\n\n"
            f"{body}\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "brake fluid"\n'
        )

    def test_concept_page_writes_wiki_chunks(self):
        store = MagicMock(spec=Store)
        target = self._target()
        content = self._content("Brakes convert kinetic energy to heat through friction pads.")

        with (
            patch("lilbee.wiki.gen.get_services", return_value=self._services_mock()),
            patch(
                "lilbee.wiki.gen.chunk_text",
                return_value=["Brakes convert kinetic energy to heat through friction pads."],
            ),
        ):
            index_wiki_page(content, target.wiki_source, store)

        store.clear_table.assert_called_once()
        call_args = store.clear_table.call_args
        assert call_args.args[0] == CHUNKS_TABLE
        predicate = call_args.args[1]
        assert target.wiki_source in predicate
        assert CHUNK_TYPE_WIKI in predicate

        store.add_chunks.assert_called_once()
        records = store.add_chunks.call_args.args[0]
        assert len(records) == 1
        rec = records[0]
        assert rec["chunk_type"] == CHUNK_TYPE_WIKI
        assert rec["source"] == target.wiki_source
        assert rec["content_type"] == "text"
        # page/line positions follow the markdown ingest convention: all zero
        assert rec["page_start"] == 0
        assert rec["line_start"] == 0
        assert "friction pads" in rec["chunk"]
        # Frontmatter and citation block are stripped from what gets chunked
        assert "title: Brakes" not in rec["chunk"]
        assert "src1" not in rec["chunk"]

    def test_drafts_subdir_is_not_indexed(self):
        """Drafts never enter the search pool. No clear, no add."""
        store = MagicMock(spec=Store)
        target = self._target(subdir=DRAFTS_SUBDIR)

        with (
            patch("lilbee.wiki.gen.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.gen.chunk_text", return_value=["body"]),
        ):
            index_wiki_page(self._content("body"), target.wiki_source, store)

        store.clear_table.assert_not_called()
        store.add_chunks.assert_not_called()

    def test_malformed_wiki_source_logs_warning_and_skips(
        self, caplog: pytest.LogCaptureFixture
    ):
        """A ``wiki_source`` without a subdir component is logged and
        skipped rather than silently writing to the store or crashing.
        """
        store = MagicMock(spec=Store)
        caplog.set_level("WARNING", logger="lilbee.wiki.gen")
        result = index_wiki_page(self._content("body"), "malformed", store)
        assert result == 0
        store.clear_table.assert_not_called()
        assert any("malformed wiki_source" in r.message for r in caplog.records)

    def test_empty_body_clears_stale_but_adds_nothing(self):
        """A page whose body is empty after frontmatter+citation stripping invalidates
        old rows but writes none, so stale wiki rows from a prior generation are removed.
        """
        store = MagicMock(spec=Store)
        target = self._target()
        # Body is pure whitespace. After extract_body + strip, nothing remains
        content = (
            "---\ntitle: Empty\n---\n\n   \n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "x"\n'
        )

        with (
            patch("lilbee.wiki.gen.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.gen.chunk_text") as chunker,
        ):
            index_wiki_page(content, target.wiki_source, store)

        store.clear_table.assert_called_once()
        chunker.assert_not_called()
        store.add_chunks.assert_not_called()

    def test_chunker_returns_empty_skips_add(self):
        """If chunk_text returns no chunks, invalidate stale rows and return."""
        store = MagicMock(spec=Store)
        target = self._target()

        with (
            patch("lilbee.wiki.gen.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.gen.chunk_text", return_value=[]),
        ):
            index_wiki_page(self._content("some body"), target.wiki_source, store)

        store.clear_table.assert_called_once()
        store.add_chunks.assert_not_called()

    def test_regen_invalidates_before_writing(self):
        """Second call still clears first, then adds. No accumulation."""
        store = MagicMock(spec=Store)
        target = self._target()

        call_order: list[str] = []
        store.clear_table.side_effect = lambda *a, **kw: call_order.append("clear")
        store.add_chunks.side_effect = lambda records: call_order.append("add") or len(records)

        with (
            patch("lilbee.wiki.gen.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.gen.chunk_text", return_value=["one chunk"]),
        ):
            index_wiki_page(self._content("first body"), target.wiki_source, store)
            index_wiki_page(self._content("second body"), target.wiki_source, store)

        assert call_order == ["clear", "add", "clear", "add"]
