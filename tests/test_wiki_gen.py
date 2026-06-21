"""Tests for wiki page generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import CHUNKS_TABLE, cfg
from lilbee.core.text import make_slug
from lilbee.data.store import ChunkType, SearchChunk, Store
from lilbee.wiki.batch import (
    _group_chunks_by_page,
    _unwrap_archived_links,
    archive_legacy_concept_pages,
)
from lilbee.wiki.cache import _find_cached_leaf, _leaf_hash
from lilbee.wiki.citation import ParsedCitation
from lilbee.wiki.citations import (
    _extract_excerpt,
    _find_excerpt_source,
    _match_citation_source,
    _resolve_citations,
    resolve_multi_source_citations,
    verify_citations,
)
from lilbee.wiki.entity_extractor import ChunkRef, EntityKind, ExtractedEntity
from lilbee.wiki.generation import generate_synthesis_pages
from lilbee.wiki.page import chunks_to_text, index_wiki_page, truncate_chunks_to_budget
from lilbee.wiki.persistence import divert_to_drafts
from lilbee.wiki.quality import (
    _embedding_faithfulness_score,
    _mean_vector,
    check_faithfulness,
    content_change_ratio,
    diff_summary,
)
from lilbee.wiki.shared import (
    PageTarget,
    WikiSubdir,
)
from lilbee.wiki.synthesis import generate_synthesis_page, group_entities_by_primary_source


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


@pytest.fixture(autouse=True)
def _stub_wiki_index_services(monkeypatch):
    """Stub ``get_services`` inside the wiki page + quality modules so tests
    that drive ``persist_and_finalize`` don't hit the real provider when the
    wiki-body indexer or the embedding faithfulness scorer runs.
    ``TestWikiIndexing`` re-patches explicitly to exercise the indexer's
    own assertions.
    """
    svc = MagicMock()
    svc.embedder.embed_batch.side_effect = lambda texts, **kw: [
        [0.1] * cfg.embedding_dim for _ in texts
    ]
    monkeypatch.setattr("lilbee.wiki.page.get_services", lambda: svc)
    monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
    return svc


def _make_chunk(text: str, source: str = "doc.md", **kwargs) -> SearchChunk:
    # Default chunk vectors match ``cfg.embedding_dim`` so the
    # stub embedder's body vector (also cfg.embedding_dim) is
    # dimension-compatible with the chunk vectors when the
    # batched path computes cosine similarity.
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
        "vector": [0.1] * cfg.embedding_dim,
    }
    defaults.update(kwargs)
    return SearchChunk(**defaults)


def _mock_provider(
    wiki_text: str,
    faith_score: str = "0.85",
    capabilities: list[str] | None = None,
) -> MagicMock:
    from lilbee.providers.base import ChatResult, FinishReason

    def _result(text: str) -> ChatResult:
        return ChatResult(text=text, tool_calls=(), finish_reason=FinishReason.STOP)

    provider = MagicMock()
    provider.chat.side_effect = [_result(wiki_text), _result(faith_score)]
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
        result = chunks_to_text(chunks)
        assert "[Chunk 1]:" in result
        assert "Hello world" in result
        assert "[Chunk 2]:" in result

    def test_includes_page_location(self):
        chunks = [_make_chunk("PDF content", page_start=5)]
        result = chunks_to_text(chunks)
        assert "(page 5)" in result

    def test_includes_line_location(self):
        chunks = [_make_chunk("Code", line_start=10, line_end=20)]
        result = chunks_to_text(chunks)
        assert "(lines 10-20)" in result


class TestTruncateChunksToBudget:
    def test_small_chunks_unchanged(self):
        """Chunks that fit within budget are returned as-is."""
        chunks = [_make_chunk("short text")]
        result = truncate_chunks_to_budget(chunks, cfg)
        assert result == chunks

    def test_truncates_when_exceeding_budget(self):
        """Large chunk sets are truncated to fit the context window."""
        cfg.num_ctx = 100  # 100 tokens * 0.75 * 4 chars = 300 chars budget
        big_text = "x" * 200  # 200 chars each, only one fits in 300
        chunks = [_make_chunk(big_text, chunk_index=i) for i in range(5)]
        result = truncate_chunks_to_budget(chunks, cfg)
        assert len(result) == 1

    def test_always_keeps_at_least_one_chunk(self):
        """Even if the first chunk exceeds the budget, it is kept."""
        cfg.num_ctx = 10  # tiny budget: 10 * 0.75 * 4 = 30 chars
        huge_chunk = _make_chunk("x" * 10000)
        result = truncate_chunks_to_budget([huge_chunk], cfg)
        assert len(result) == 1

    def test_uses_default_context_when_num_ctx_none(self):
        """Falls back to default context window when num_ctx is not set."""
        cfg.num_ctx = None
        # Default 8192 * 0.75 * 4 = 24576 chars budget
        small_chunks = [_make_chunk("hello", chunk_index=i) for i in range(10)]
        result = truncate_chunks_to_budget(small_chunks, cfg)
        assert len(result) == 10  # all fit easily

    def test_logs_warning_on_truncation(self, caplog: pytest.LogCaptureFixture):
        """A warning is logged when chunks are truncated."""
        cfg.num_ctx = 100
        chunks = [_make_chunk("x" * 200, chunk_index=i) for i in range(5)]
        with caplog.at_level("WARNING", logger="lilbee.wiki.page"):
            truncate_chunks_to_budget(chunks, cfg)
        assert "Truncated chunks from 5 to 1" in caplog.text


class TestEmbeddingFaithfulness:
    """Deterministic cosine-similarity faithfulness scoring."""

    def test_mean_vector_of_empty_list_is_empty(self):
        assert _mean_vector([]) == []

    def test_mean_vector_averages_componentwise(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        assert _mean_vector(vectors) == [2.0, 3.0]

    def test_score_is_dot_for_normalized_vectors(self):
        # two identical unit vectors => score 1.0
        unit = [1.0, 0.0, 0.0]
        assert _embedding_faithfulness_score(unit, [unit]) == pytest.approx(1.0)

    def test_score_clamped_to_zero_for_orthogonal(self):
        assert _embedding_faithfulness_score([1.0, 0.0], [[0.0, 1.0]]) == pytest.approx(0.0)

    def test_score_clamped_to_zero_for_anti_correlated(self):
        score = _embedding_faithfulness_score([1.0, 0.0], [[-1.0, 0.0]])
        assert score == pytest.approx(0.0)

    def test_score_zero_for_missing_inputs(self):
        assert _embedding_faithfulness_score([], [[1.0]]) == 0.0
        assert _embedding_faithfulness_score([1.0], []) == 0.0

    def test_score_zero_on_dim_mismatch_and_warns(self, caplog: pytest.LogCaptureFixture):
        """Off-shape body vector vs source-mean vector returns 0.0 with a warning."""
        caplog.set_level("WARNING", logger="lilbee.wiki.quality")
        score = _embedding_faithfulness_score([1.0, 0.0], [[1.0, 0.0, 0.0]])
        assert score == 0.0
        assert any("does not match source vector dim" in r.message for r in caplog.records)


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
        from lilbee.data.store import CitationRecord

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
        verified = verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 1

    def test_keeps_excerpts_that_differ_only_in_whitespace(self):
        """Source with a mid-sentence newline still matches an LLM quote that collapsed it."""
        from lilbee.data.store import CitationRecord

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
        verified = verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 1

    def test_drops_unmatched_excerpts(self):
        from lilbee.data.store import CitationRecord

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
        verified = verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 0

    def test_keeps_inference_citations(self):
        from lilbee.data.store import CitationRecord

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
        verified = verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 1

    def test_skips_wiki_sourced_citations(self):
        from lilbee.data.store import CitationRecord

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
        verified = verify_citations(recs, chunks, "test", cfg)
        assert len(verified) == 0


class TestBuildWikiMessages:
    def test_thinking_capability_prepends_no_think_directive(self):
        from lilbee.wiki.page import build_wiki_messages

        provider = MagicMock()
        provider.get_capabilities.return_value = ["completion", "thinking"]
        cfg.chat_model = "ollama/any-model:latest"

        messages = build_wiki_messages("Summarize these chunks.", provider, cfg)

        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.startswith("/no_think\n\n")
        assert "Summarize these chunks." in content
        provider.get_capabilities.assert_called_once()

    def test_no_thinking_capability_passes_prompt_through(self):
        from lilbee.wiki.page import build_wiki_messages

        provider = MagicMock()
        provider.get_capabilities.return_value = ["completion"]
        cfg.chat_model = "ollama/any-model:latest"

        messages = build_wiki_messages("Summarize these chunks.", provider, cfg)
        assert messages == [{"role": "user", "content": "Summarize these chunks."}]

    def test_empty_capabilities_passes_prompt_through(self):
        """Backends that don't report capabilities leave the prompt untouched."""
        from lilbee.wiki.page import build_wiki_messages

        provider = MagicMock()
        provider.get_capabilities.return_value = []
        cfg.chat_model = "ollama/any-model:latest"

        messages = build_wiki_messages("Summarize these chunks.", provider, cfg)
        assert messages == [{"role": "user", "content": "Summarize these chunks."}]


_COHERENT_WIKI = "# Test\n\nThe test concept refers to the thing under test."


def _chunk_with_vector(vector: list[float], text: str = "t", **kw) -> SearchChunk:
    return _make_chunk(text, vector=vector, **kw)


class TestCheckFaithfulness:
    """Embedding-based faithfulness scoring.

    The cosine-similarity path uses the chunk's own .vector (set by
    LanceDB for every SearchChunk) and an embedder call for the body
    side only.
    """

    def test_uses_chunk_vectors_no_embed_batch_for_chunks(self, monkeypatch):
        """The body embeds once; chunks reuse their stored .vector."""
        svc = MagicMock()
        svc.embedder.embed_batch.return_value = [[1.0, 0.0]]
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        chunks = [_chunk_with_vector([1.0, 0.0])]
        score = check_faithfulness(chunks, _COHERENT_WIKI, "test")
        assert score == pytest.approx(1.0)
        # Only ONE call, and its argument is the body text list: no
        # chunks_text batch embedding.
        assert svc.embedder.embed_batch.call_count == 1
        body_arg = svc.embedder.embed_batch.call_args[0][0]
        assert isinstance(body_arg, list) and len(body_arg) == 1

    def test_below_threshold_score_stays_low(self, monkeypatch):
        """A body that points away from the chunk mean scores at/near zero."""
        svc = MagicMock()
        svc.embedder.embed_batch.return_value = [[0.0, 1.0]]
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        chunks = [_chunk_with_vector([1.0, 0.0])]
        score = check_faithfulness(chunks, _COHERENT_WIKI, "test")
        assert score == pytest.approx(0.0)

    def test_coherence_failure_returns_zero(self, monkeypatch):
        """B3: a structurally broken H1 returns 0.0 without embedding."""
        svc = MagicMock()
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        wiki = "# | | bad\n\nbody"
        chunks = [_chunk_with_vector([1.0])]
        score = check_faithfulness(chunks, wiki, "bad")
        assert score == 0.0
        svc.embedder.embed_batch.assert_not_called()

    def test_missing_concept_mention_returns_zero(self, monkeypatch):
        svc = MagicMock()
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        wiki = "# Brakes\n\nThis page talks about tires and wheels."
        chunks = [_chunk_with_vector([1.0])]
        score = check_faithfulness(chunks, wiki, "brakes")
        assert score == 0.0

    def test_missing_h1_returns_zero(self, monkeypatch):
        svc = MagicMock()
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        wiki = "No heading here, just prose about brakes."
        chunks = [_chunk_with_vector([1.0])]
        score = check_faithfulness(chunks, wiki, "brakes")
        assert score == 0.0

    def test_coherent_page_uses_embedding_score(self, monkeypatch):
        svc = MagicMock()
        svc.embedder.embed_batch.return_value = [[1.0, 0.0]]
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        wiki = "# Chevrolet\n\nChevrolet is a manufacturer of vehicles."
        chunks = [_chunk_with_vector([1.0, 0.0])]
        score = check_faithfulness(chunks, wiki, "Chevrolet")
        assert score == pytest.approx(1.0)

    def test_display_name_cleanup_before_comparison(self, monkeypatch):
        """Structural chars in the label don't block coherence matching."""
        svc = MagicMock()
        svc.embedder.embed_batch.return_value = [[1.0, 0.0]]
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        wiki = "# Designer\n\nThe designer of the Caprice was Irv Rybicki."
        chunks = [_chunk_with_vector([1.0, 0.0])]
        score = check_faithfulness(chunks, wiki, "| | designer")
        assert score == pytest.approx(1.0)

    def test_embedder_failure_returns_zero(self, monkeypatch):
        svc = MagicMock()
        svc.embedder.embed_batch.side_effect = RuntimeError("down")
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        chunks = [_chunk_with_vector([1.0, 0.0])]
        score = check_faithfulness(chunks, _COHERENT_WIKI, "test")
        assert score == 0.0

    def test_empty_source_vectors_returns_zero(self, monkeypatch):
        svc = MagicMock()
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        score = check_faithfulness([], _COHERENT_WIKI, "test")
        assert score == 0.0

    def test_empty_display_label_returns_zero(self, monkeypatch):
        """clean_label_for_display → empty string → coherence False."""
        svc = MagicMock()
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        chunks = [_chunk_with_vector([1.0])]
        # A label made entirely of structural chars reduces to empty
        # display under clean_label_for_display, hitting the guard
        # in _title_content_coherence.
        score = check_faithfulness(chunks, "# anything\n\nbody", "|||")
        assert score == 0.0
        # Embedder is never called because coherence already failed.
        svc.embedder.embed_batch.assert_not_called()

    def test_empty_body_after_citation_strip_returns_zero(self, monkeypatch):
        """A page whose body is entirely citation block collapses to empty."""
        svc = MagicMock()
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        # Construct a wiki text whose body (after strip_citation_block)
        # is blank. ``## footnotes`` is the citation block header used
        # by render_citation_block, so strip_citation_block removes
        # everything from it downward. The H1 alone remains above it.
        # The title-coherence check still needs the body to mention the
        # display name; we craft a page where H1 passes the coherence
        # gate (display in heading, display in body just once in the
        # paragraph) so we reach the embedder stage, then swap in a
        # wiki whose citation strip zeroes the body.
        wiki = (
            "# test\n\n"
            "The test concept applies here.\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: doc.md, excerpt: "x"\n'
        )
        # Monkeypatch strip_citation_block to return an empty body so
        # we exercise the empty-body guard directly without fighting
        # the title-coherence gate semantics.
        monkeypatch.setattr("lilbee.wiki.quality.strip_citation_block", lambda _: "   ")
        chunks = [_chunk_with_vector([1.0])]
        score = check_faithfulness(chunks, wiki, "test")
        assert score == 0.0
        svc.embedder.embed_batch.assert_not_called()

    def test_empty_body_vectors_returns_zero(self, monkeypatch):
        """Embedder returned an empty list → score 0.0 without crashing."""
        svc = MagicMock()
        svc.embedder.embed_batch.return_value = []
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        chunks = [_chunk_with_vector([1.0, 0.0])]
        score = check_faithfulness(chunks, _COHERENT_WIKI, "test")
        assert score == 0.0


class TestTitleContentCoherence:
    """B3 deterministic gate unchanged; kept for regression coverage."""

    def test_coherence_failure_returns_zero_score(self, monkeypatch):
        svc = MagicMock()
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        wiki = "# | | designer\n\nThe designer refers to an individual."
        chunks = [_chunk_with_vector([1.0])]
        score = check_faithfulness(chunks, wiki, "designer")
        assert score == 0.0

    def test_logs_info_on_coherence_failure(self, monkeypatch, caplog: pytest.LogCaptureFixture):
        svc = MagicMock()
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        caplog.set_level("INFO", logger="lilbee.wiki.quality")
        chunks = [_chunk_with_vector([1.0])]
        check_faithfulness(chunks, "# bad\n\nbody", "brakes")
        assert any("coherence failed" in r.message for r in caplog.records)


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
        records = resolve_multi_source_citations(
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
        records = resolve_multi_source_citations(
            parsed,
            ["a.md"],
            {"a.md": "h"},
            chunks,
        )
        assert records[0]["source_filename"] == "a.md"

    def test_falls_back_to_first_source(self):
        parsed = [ParsedCitation("src1", 'excerpt: "Not found anywhere"', 1)]
        records = resolve_multi_source_citations(
            parsed,
            ["fallback.md"],
            {},
            {},
        )
        assert records[0]["source_filename"] == "fallback.md"


def _synthesis_wiki_text(sources: list[str], topic: str | None = None) -> str:
    """Build a valid synthesis wiki text with citations to the given sources.

    Heading defaults to ``# <topic>`` so the B3 title/body coherence
    gate passes; override ``topic`` when a test wants to exercise a
    mismatched-heading path.
    """
    heading = topic or "topic"
    lines = [f"# {heading}\n", f"This page is about {heading}.\n"]
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
        wiki_text = _synthesis_wiki_text(sources, topic="gradual typing")
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        result = generate_synthesis_page(
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
        assert f"generated_by: {cfg.chat_model}" in content
        assert 'sources: ["a.md", "b.md", "c.md"]' in content
        # Faithfulness is a cosine-similarity score between the body
        # embedding and the mean of the source chunk vectors. Matching
        # stub vectors produce a score of 1.00 (identical).
        assert "faithfulness_score: 1.00" in content
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

        result = generate_synthesis_page(
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
        result = generate_synthesis_page("topic", ["a.md"], {}, provider, store, cfg)
        assert result is None
        provider.chat.assert_not_called()

    def test_llm_failure_returns_none(self, tmp_path: Path):
        chunks_by_source = {"a.md": [_make_chunk("text", source="a.md")]}
        provider = MagicMock()
        provider.chat.side_effect = ConnectionError("down")
        store = _mock_store()

        result = generate_synthesis_page(
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

        result = generate_synthesis_page(
            "topic",
            ["a.md"],
            chunks_by_source,
            provider,
            store,
            cfg,
        )
        assert result is None

    def test_faithfulness_failure_uses_zero(self, tmp_path: Path, _stub_wiki_index_services):
        """Body-embedding failure routes to drafts (score 0.0)."""
        sources = ["a.md"]
        (tmp_path / "documents" / "a.md").write_text("Fact from a.md.")
        chunks_by_source = {"a.md": [_make_chunk("Fact from a.md.", source="a.md")]}
        wiki_text = _synthesis_wiki_text(sources)
        provider = _mock_provider(wiki_text)
        store = _mock_store()
        # The body-side embed call crashes so check_faithfulness
        # returns 0.0 and the page routes to drafts.
        _stub_wiki_index_services.embedder.embed_batch.side_effect = ConnectionError("down")

        result = generate_synthesis_page(
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

        result = generate_synthesis_page(
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

        result = generate_synthesis_page(
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
        from lilbee.retrieval.clustering import SourceCluster

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
        from lilbee.retrieval.clustering import SourceCluster

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
        from lilbee.retrieval.clustering import SourceCluster

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
        assert content_change_ratio("a\nb\nc", "a\nb\nc") == 0.0

    def test_completely_different(self):
        assert content_change_ratio("a\nb\nc", "x\ny\nz") == 1.0

    def test_partial_change(self):
        old = "line1\nline2\nline3\nline4"
        new = "line1\nchanged\nline3\nline4"
        ratio = content_change_ratio(old, new)
        assert 0.0 < ratio < 1.0

    def test_empty_old(self):
        # empty -> something = 100% change
        assert content_change_ratio("", "new content") == 1.0

    def test_empty_both(self):
        assert content_change_ratio("", "") == 0.0


class TestDiffSummary:
    def test_produces_unified_diff(self):
        result = diff_summary("old line", "new line")
        assert "---" in result or "-old line" in result

    def test_truncates_long_diff(self):
        old = "\n".join(f"line{i}" for i in range(50))
        new = "\n".join(f"changed{i}" for i in range(50))
        result = diff_summary(old, new)
        assert "more lines" in result


class TestDivertToDrafts:
    def test_writes_draft_with_note(self, tmp_path: Path):
        drafts_dir = tmp_path / "drafts"
        content = "# New Page\n\nNew content."
        result = divert_to_drafts(content, drafts_dir, "my-page", 0.45, "diff text", "concepts")
        assert result.exists()
        assert result.parent == drafts_dir
        text = result.read_text()
        assert "DRIFT" in text
        assert "45%" in text
        assert "human review" in text
        # The origin subdir rides the marker so accept restores the page to concepts/.
        assert "origin: concepts" in text
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
        wiki_text = _synthesis_wiki_text(sources, topic="gradual typing")
        provider = _mock_provider(wiki_text)
        store = _mock_store()

        result = generate_synthesis_page(
            "gradual typing", sources, chunks_by_source, provider, store, cfg
        )
        assert result is not None
        assert "drafts" in str(result)
        # Original should be unchanged
        assert "Totally different synthesis" in existing.read_text()


class TestWikiIndexing:
    """``index_wiki_page`` chunks, embeds and writes wiki page bodies."""

    @staticmethod
    def _target(subdir: str = WikiSubdir.CONCEPTS, slug: str = "brakes") -> PageTarget:
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
            patch("lilbee.wiki.page.get_services", return_value=self._services_mock()),
            patch(
                "lilbee.wiki.page.chunk_text",
                return_value=["Brakes convert kinetic energy to heat through friction pads."],
            ),
        ):
            index_wiki_page(content, target.wiki_source, store)

        store.clear_table.assert_called_once()
        call_args = store.clear_table.call_args
        assert call_args.args[0] == CHUNKS_TABLE
        predicate = call_args.args[1]
        assert target.wiki_source in predicate
        assert ChunkType.WIKI in predicate

        store.add_chunks.assert_called_once()
        records = store.add_chunks.call_args.args[0]
        assert len(records) == 1
        rec = records[0]
        assert rec["chunk_type"] == ChunkType.WIKI
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
        target = self._target(subdir=WikiSubdir.DRAFTS)

        with (
            patch("lilbee.wiki.page.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.page.chunk_text", return_value=["body"]),
        ):
            index_wiki_page(self._content("body"), target.wiki_source, store)

        store.clear_table.assert_not_called()
        store.add_chunks.assert_not_called()

    def test_malformed_wiki_source_logs_warning_and_skips(self, caplog: pytest.LogCaptureFixture):
        """A ``wiki_source`` without a subdir component is logged and
        skipped rather than silently writing to the store or crashing.
        """
        store = MagicMock(spec=Store)
        caplog.set_level("WARNING", logger="lilbee.wiki.page")
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
            patch("lilbee.wiki.page.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.page.chunk_text") as chunker,
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
            patch("lilbee.wiki.page.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.page.chunk_text", return_value=[]),
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
            patch("lilbee.wiki.page.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.page.chunk_text", return_value=["one chunk"]),
        ):
            index_wiki_page(self._content("first body"), target.wiki_source, store)
            index_wiki_page(self._content("second body"), target.wiki_source, store)

        assert call_order == ["clear", "add", "clear", "add"]


class TestBuildFrontmatter:
    """B2: frontmatter carries a provenance block when chunks are provided."""

    def test_no_chunks_omits_provenance(self):
        from lilbee.wiki.page import build_frontmatter

        fm = build_frontmatter(cfg, ["doc.md"], 0.9)
        assert "provenance:" not in fm

    def test_with_chunks_renders_provenance_block(self):
        from lilbee.wiki.page import build_frontmatter

        chunks = [
            _make_chunk("body a", source="doc.md", chunk_index=0),
            _make_chunk("body b", source="doc.md", chunk_index=1),
        ]
        fm = build_frontmatter(cfg, ["doc.md"], 0.85, chunks=chunks)
        assert "provenance:" in fm
        assert f"extraction_method: {cfg.wiki_entity_mode.value}" in fm
        # yaml.safe_dump emits block style; unquoted scalars on safe names.
        assert "source: doc.md" in fm
        assert "chunk_index: 0" in fm
        assert "chunk_index: 1" in fm

    def test_provenance_round_trips_through_parse_frontmatter(self):
        from lilbee.wiki.page import build_frontmatter
        from lilbee.wiki.shared import parse_frontmatter

        chunks = [_make_chunk("a", source="foo.pdf", chunk_index=5)]
        fm = build_frontmatter(cfg, ["foo.pdf"], 0.7, chunks=chunks)
        parsed = parse_frontmatter(fm + "body\n")
        assert parsed["provenance"]["extraction_method"] == cfg.wiki_entity_mode.value
        assert parsed["provenance"]["chunks"] == [{"source": "foo.pdf", "chunk_index": 5}]

    def test_existing_frontmatter_without_provenance_still_parses(self):
        """Pages generated before B2 have no provenance block; the
        parser must still return a dict (backwards-compat)."""
        from lilbee.wiki.shared import parse_frontmatter

        fm = (
            "---\n"
            "generated_by: qwen3:0.6b\n"
            "generated_at: 2026-01-01T00:00:00\n"
            'sources: ["doc.md"]\n'
            "faithfulness_score: 0.90\n"
            "---\n\n"
            "body\n"
        )
        parsed = parse_frontmatter(fm)
        assert parsed["faithfulness_score"] == 0.90
        assert "provenance" not in parsed

    def test_provenance_quotes_safely_on_pathological_chunk_source(self):
        """Chunk sources with ``"``, ``\\``, ``:``, or newlines must
        still yield valid YAML inside the provenance block. Before B2
        review this rendered inline as ``source: "<raw>"`` and broke
        the parser on any embedded quote; ``yaml.safe_dump`` escapes
        them correctly now.

        The outer ``sources:`` list is pre-existing hand-rolled YAML
        (not part of this PR); pass a benign value for the sources
        list so the test isolates the provenance-block behavior.
        """
        from lilbee.wiki.page import build_frontmatter
        from lilbee.wiki.shared import parse_frontmatter

        pathological = 'weird "name": with\\slash\n'
        chunks = [_make_chunk("body", source=pathological, chunk_index=0)]
        fm = build_frontmatter(cfg, ["benign.md"], 0.5, chunks=chunks)
        parsed = parse_frontmatter(fm + "body\n")
        assert parsed["provenance"]["chunks"] == [{"source": pathological, "chunk_index": 0}]


class TestGroupEntitiesByPrimarySource:
    def test_primary_source_by_mention_count(self):
        ent = ExtractedEntity(
            slug="x",
            kind=EntityKind.ENTITY,
            label="X",
            type_hint="PERSON",
            chunk_refs=(
                ChunkRef("a.md", 0),
                ChunkRef("a.md", 1),
                ChunkRef("b.md", 0),
            ),
        )
        grouped = group_entities_by_primary_source([ent])
        assert list(grouped) == ["a.md"]

    def test_lexicographic_tiebreak(self):
        ent = ExtractedEntity(
            slug="x",
            kind=EntityKind.ENTITY,
            label="X",
            type_hint="PERSON",
            chunk_refs=(ChunkRef("b.md", 0), ChunkRef("a.md", 0)),
        )
        grouped = group_entities_by_primary_source([ent])
        assert list(grouped) == ["a.md"]

    def test_empty_refs_dropped(self):
        ent = ExtractedEntity(
            slug="x",
            kind=EntityKind.ENTITY,
            label="X",
            type_hint="PERSON",
            chunk_refs=(),
        )
        assert group_entities_by_primary_source([ent]) == {}


class TestLegacyConceptsMigration:
    def test_archives_concept_pages(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "foo.md").write_text("original")
        data_dir = tmp_path / "data"
        archive_legacy_concept_pages(wiki_root, data_dir)
        assert not (wiki_root / "concepts" / "foo.md").exists()
        assert (wiki_root / "archive" / "concepts" / "foo.md").read_text() == "original"
        assert (data_dir / ".phase-d-migrated").exists()

    def test_idempotent(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "foo.md").write_text("original")
        data_dir = tmp_path / "data"
        archive_legacy_concept_pages(wiki_root, data_dir)
        # Second run: nothing to archive; should not touch disk state.
        (wiki_root / "concepts").mkdir(parents=True, exist_ok=True)
        (wiki_root / "concepts" / "new.md").write_text("fresh")
        archive_legacy_concept_pages(wiki_root, data_dir)
        # Freshly written page stayed put.
        assert (wiki_root / "concepts" / "new.md").exists()


class TestUnwrapArchivedLinks:
    def test_replaces_wiki_links_with_plain_text(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        (wiki_root / "entities").mkdir(parents=True)
        (wiki_root / "entities" / "henry.md").write_text(
            "# Henry\n\nRelated: [[foo]], [[bar]], keep [[not-archived]]."
        )
        _unwrap_archived_links(wiki_root, ["foo", "bar"])
        body = (wiki_root / "entities" / "henry.md").read_text()
        assert "[[foo]]" not in body
        assert "foo" in body
        assert "[[bar]]" not in body
        assert "[[not-archived]]" in body

    def test_no_archived_slugs_noop(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        (wiki_root / "entities").mkdir(parents=True)
        (wiki_root / "entities" / "henry.md").write_text("body")
        _unwrap_archived_links(wiki_root, [])
        assert (wiki_root / "entities" / "henry.md").read_text() == "body"


class TestRunFullBuild:
    def test_defaults_to_global_cfg_when_called_with_no_args(self, monkeypatch):
        """run_full_build() with no arg falls back to lilbee.config.cfg."""
        from lilbee.wiki.generation import run_full_build

        captured: dict[str, object] = {}

        def fake_get_services():
            svc = MagicMock()
            svc.store.get_sources.return_value = []
            return svc

        def fake_extractor(*a, **kw):
            ext = MagicMock()
            ext.extract.return_value = []
            return ext

        def fake_build_wiki(entities, provider, store, config, *, extract_concepts):
            captured["config"] = config
            return []

        monkeypatch.setattr("lilbee.wiki.generation.get_services", fake_get_services)
        monkeypatch.setattr("lilbee.wiki.generation.build_wiki", fake_build_wiki)
        monkeypatch.setattr("lilbee.wiki.generation.update_wiki_index", lambda *a, **kw: None)
        monkeypatch.setattr("lilbee.wiki.generation.append_wiki_log", lambda *a, **kw: None)
        monkeypatch.setattr("lilbee.wiki.entity_extractor.get_entity_extractor", fake_extractor)

        result = run_full_build()
        assert captured["config"] is cfg
        assert result == {"paths": [], "entities": 0, "count": 0}


class TestRunFullSynthesize:
    def test_defaults_to_global_cfg_when_called_with_no_args(self, monkeypatch, tmp_path):
        """run_full_synthesize() with no arg falls back to lilbee.config.cfg."""
        from lilbee.wiki.generation import run_full_synthesize

        captured: dict[str, object] = {}

        def fake_get_services():
            return MagicMock()

        def fake_generate(provider, store, clusterer, config):
            captured["config"] = config
            return [tmp_path / "wiki" / "synthesis" / "typing.md"]

        monkeypatch.setattr("lilbee.wiki.generation.get_services", fake_get_services)
        monkeypatch.setattr("lilbee.wiki.generation.generate_synthesis_pages", fake_generate)

        result = run_full_synthesize()
        assert captured["config"] is cfg
        assert result["count"] == 1
        assert result["paths"][0].endswith("typing.md")


class TestPersistAndFinalizeDrift:
    """A drift-diverted regen must not leak its unreviewed body into the index or
    citations under the published page's identity (bb-ziks.35)."""

    def test_diversion_skips_publish_indexing(self):
        from lilbee.wiki.persistence import persist_and_finalize

        store = MagicMock(spec=Store)
        target = TestWikiIndexing._target()
        published = target.wiki_root / target.subdir / f"{target.slug}.md"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text("Old published body, unrelated to the regen.", encoding="utf-8")

        old = cfg.wiki_drift_threshold
        cfg.wiki_drift_threshold = 0.1
        try:
            page_path = persist_and_finalize(
                "Brand new drifted body sharing nothing with the old page.",
                target,
                [],
                [],
                store,
                cfg,
            )
        finally:
            cfg.wiki_drift_threshold = old

        assert WikiSubdir.DRAFTS in page_path.parts
        assert "Old published body" in published.read_text()
        store.add_citations.assert_not_called()
        store.delete_citations_for_wiki.assert_not_called()
        store.clear_table.assert_not_called()
