"""Tests for wiki page generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lilbee.core.config import CHUNKS_TABLE, cfg
from lilbee.core.text import make_slug
from lilbee.data.store import ChunkType, SearchChunk, Store
from lilbee.data.store.core import _check_vector_dims
from lilbee.wiki.batch import (
    _unwrap_archived_links,
    archive_legacy_concept_pages,
)
from lilbee.wiki.citations import (
    ParsedCitation,
    _extract_excerpt,
    _find_excerpt_source,
    _match_citation_source,
    _resolve_citations,
    resolve_multi_source_citations,
    verify_citations,
)
from lilbee.wiki.entity_extractor import ChunkRef, EntityKind, ExtractedEntity
from lilbee.wiki.generation import generate_synthesis_pages
from lilbee.wiki.page import (
    WIKI_DEFAULT_SEED,
    chunks_to_text,
    index_wiki_page,
    prompt_overhead_tokens,
    truncate_chunks_to_budget,
    wiki_generation_options,
    write_page,
)
from lilbee.wiki.persistence import (
    delete_drift_draft_if_present,
    divert_to_drafts,
    write_pending_marker,
)
from lilbee.wiki.quality import (
    _embedding_faithfulness_score,
    _mean_vector,
    check_faithfulness,
    content_change_ratio,
    diff_summary,
)
from lilbee.wiki.shared import (
    PENDING_MARKER_KEYWORD_COLLISION,
    PENDING_MARKER_KEYWORD_PARSE,
    PageTarget,
    WikiSubdir,
)
from lilbee.wiki.stats import BuildStats
from lilbee.wiki.synthesis import generate_synthesis_page, group_entities_by_primary_source
from tests.conftest import make_citation


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
        np.full(cfg.embedding_dim, 0.1, dtype=np.float32) for _ in texts
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


def _mock_provider(wiki_text: str) -> MagicMock:
    """Provider whose single chat call returns *wiki_text*."""
    from lilbee.providers.base import ChatResult, FinishReason

    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text=wiki_text, tool_calls=(), finish_reason=FinishReason.STOP
    )
    provider.get_capabilities.return_value = ["completion"]
    return provider


def _orthogonal_body_vector() -> list[float]:
    """A body vector whose cosine against the uniform chunk vectors is <= 0."""
    half = cfg.embedding_dim // 2
    return [1.0] * half + [-1.0] * (cfg.embedding_dim - half)


def _mock_store() -> MagicMock:
    store = MagicMock(spec=Store)
    store.replace_chunks.side_effect = lambda records, predicate: len(records)
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
        # The output cap alone exceeds the window, so the budget is the
        # quarter-window floor: 25 tokens = 100 chars.
        cfg.num_ctx = 100
        big_text = "x" * 200  # 200 chars each, only the first is kept
        chunks = [_make_chunk(big_text, chunk_index=i) for i in range(5)]
        result = truncate_chunks_to_budget(chunks, cfg)
        assert len(result) == 1

    def test_always_keeps_at_least_one_chunk(self):
        """Even if the first chunk exceeds the budget, it is kept."""
        cfg.num_ctx = 10  # floor of a tiny window: 2 tokens = 8 chars
        huge_chunk = _make_chunk("x" * 10000)
        result = truncate_chunks_to_budget([huge_chunk], cfg)
        assert len(result) == 1

    def test_uses_default_context_when_num_ctx_none(self):
        """Falls back to default context window when num_ctx is not set."""
        cfg.num_ctx = None
        # Default 8192 less the output cap and the prompt overhead, times 4 chars.
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

    def test_kept_chunks_leave_room_for_the_prompt_and_the_generation(self):
        """The whole call has to fit: chunks + template + output cap <= num_ctx."""
        cfg.num_ctx = 8192
        cfg.wiki_summary_max_tokens = 2048
        chunks = [_make_chunk("x" * 4000, chunk_index=i) for i in range(20)]
        kept = truncate_chunks_to_budget(chunks, cfg)
        prompt_tokens = len(chunks_to_text(kept)) // 4
        total = prompt_tokens + cfg.wiki_summary_max_tokens + prompt_overhead_tokens(cfg)
        assert 0 < len(kept) < len(chunks)
        assert total <= cfg.num_ctx

    def test_a_bigger_output_cap_leaves_room_for_fewer_chunks(self):
        cfg.num_ctx = 8192
        chunks = [_make_chunk("x" * 500, chunk_index=i) for i in range(40)]
        cfg.wiki_summary_max_tokens = 256
        small_cap = truncate_chunks_to_budget(chunks, cfg)
        cfg.wiki_summary_max_tokens = 4096
        large_cap = truncate_chunks_to_budget(chunks, cfg)
        assert len(large_cap) < len(small_cap)

    def test_per_chunk_formatting_counts_against_the_budget(self):
        """Chunk numbering and separators are prompt tokens too."""
        cfg.num_ctx = 1200
        cfg.wiki_summary_max_tokens = 256
        chunks = [_make_chunk("x" * 100, chunk_index=i, page_start=i + 1) for i in range(40)]
        kept = truncate_chunks_to_budget(chunks, cfg)
        budget_chars = (cfg.num_ctx - cfg.wiki_summary_max_tokens - prompt_overhead_tokens(cfg)) * 4
        assert len(chunks_to_text(kept)) <= budget_chars

    def test_budget_floors_at_a_quarter_of_the_window(self):
        """An output cap larger than the whole window still leaves chunks room."""
        cfg.num_ctx = 512
        cfg.wiki_summary_max_tokens = 4096
        chunks = [_make_chunk("x" * 400, chunk_index=i) for i in range(10)]
        assert len(truncate_chunks_to_budget(chunks, cfg)) == 1

    def test_the_floor_never_raises_a_positive_budget(self):
        """A budget between zero and a quarter of the window is honest and small.
        Lifting it to the floor would overflow a window the real budget fits."""
        cfg.num_ctx = 4096
        cfg.wiki_summary_max_tokens = 3000
        available = cfg.num_ctx - cfg.wiki_summary_max_tokens - prompt_overhead_tokens(cfg)
        assert 0 < available < cfg.num_ctx * 0.25
        chunks = [_make_chunk("x" * 400, chunk_index=i) for i in range(20)]
        kept = truncate_chunks_to_budget(chunks, cfg)
        total = (
            len(chunks_to_text(kept)) // 4
            + cfg.wiki_summary_max_tokens
            + prompt_overhead_tokens(cfg)
        )
        assert 0 < len(kept) < len(chunks)
        assert total <= cfg.num_ctx

    def test_a_rendered_prompt_is_charged_instead_of_the_raw_template(self):
        """Per-call substitutions (concept instruction, entity list, source list)
        are not in the template, so a caller that rendered them passes their size."""
        cfg.num_ctx = 8192
        cfg.wiki_summary_max_tokens = 2048
        chunks = [_make_chunk("x" * 500, chunk_index=i) for i in range(40)]
        from_template = truncate_chunks_to_budget(chunks, cfg)
        rendered_chars = len(cfg.wiki_entity_batch_prompt) + 8000
        from_rendered = truncate_chunks_to_budget(chunks, cfg, rendered_chars)
        assert prompt_overhead_tokens(cfg, rendered_chars) > prompt_overhead_tokens(cfg)
        assert len(from_rendered) < len(from_template)


class TestWikiGenerationOptions:
    """Wiki calls sample deterministically so an unchanged corpus converges."""

    def test_uses_the_fixed_seed_when_the_user_set_none(self):
        cfg.seed = None
        assert wiki_generation_options(cfg)["seed"] == WIKI_DEFAULT_SEED

    def test_a_user_seed_wins_over_the_default(self):
        cfg.seed = 99
        assert wiki_generation_options(cfg)["seed"] == 99

    def test_applies_the_wiki_temperature_and_output_cap(self):
        options = wiki_generation_options(cfg)
        assert options["temperature"] == cfg.wiki_temperature
        assert options["max_tokens"] == cfg.wiki_summary_max_tokens


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

    def test_scores_the_float32_array_the_embedder_returns(self):
        # The body vector arrives from embed_batch as an ndarray; the source
        # vectors stay lancedb lists.
        body = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        assert _embedding_faithfulness_score(body, [[1.0, 0.0, 0.0]]) == pytest.approx(1.0)

    def test_score_zero_for_empty_body_array(self):
        assert _embedding_faithfulness_score(np.zeros(0, dtype=np.float32), [[1.0]]) == 0.0

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

    def test_missing_excerpt_is_still_a_fact_claim(self):
        """A footnote the model left unquoted is a parse failure, not an inference."""
        chunks = [_make_chunk("Some text")]
        parsed = [ParsedCitation("src1", "doc.md, no excerpt here", 1)]
        records = _resolve_citations(parsed, "doc.md", "hash", chunks)
        assert records[0]["claim_type"] == "fact"
        assert records[0]["excerpt"] == ""

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

    @pytest.mark.parametrize("claim_type", ["fact", "inference"])
    def test_drops_citations_without_an_excerpt(self, claim_type: str):
        """The gate has to be able to fail: an unquoted footnote verifies nothing."""
        chunks = [_make_chunk("text")]
        recs = [make_citation(excerpt="", claim_type=claim_type)]
        assert verify_citations(recs, chunks, "test", cfg) == []

    def test_drops_excerpt_stitched_across_two_chunks(self):
        """Joining the chunk pool into one string would match a quote no source carries."""
        chunks = [_make_chunk("chunk one end"), _make_chunk("start chunk two", chunk_index=1)]
        recs = [make_citation(excerpt="chunk one end start chunk two")]
        assert verify_citations(recs, chunks, "test", cfg) == []

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

    def test_matches_across_whitespace_differences(self):
        """Attribution uses the verification rule, so a re-wrapped quote still resolves."""
        chunks = {"a.md": [_make_chunk("Beta\n   content here", source="a.md")]}
        assert _find_excerpt_source("Beta content here", chunks) == "a.md"


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

    def test_drops_unattributable_citation(self, caplog: pytest.LogCaptureFixture):
        """Defaulting to the first source would invent the provenance."""
        caplog.set_level("WARNING", logger="lilbee.wiki.citations")
        parsed = [ParsedCitation("src1", 'excerpt: "Not found anywhere"', 1)]
        records = resolve_multi_source_citations(
            parsed,
            ["fallback.md"],
            {},
            {},
        )
        assert records == []
        assert any("Dropping citation src1" in r.message for r in caplog.records)


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
        assert WikiSubdir.SYNTHESIS in result.parts
        assert result.name == "gradual-typing.md"
        provider.chat.assert_called_once()
        content = result.read_text()
        assert f"generated_by: {cfg.chat_model}" in content
        assert 'sources: ["a.md", "b.md", "c.md"]' in content
        # Faithfulness is a cosine-similarity score between the body
        # embedding and the mean of the source chunk vectors. Matching
        # stub vectors produce a score of 1.00 (identical).
        assert "faithfulness_score: 1.00" in content
        store.replace_citations_for_wiki.assert_called_once()

    def test_low_score_goes_to_drafts(self, tmp_path: Path, monkeypatch):
        """A body that does not resemble its sources routes to drafts, not synthesis."""
        sources = ["a.md", "b.md", "c.md"]
        for name in sources:
            (tmp_path / "documents" / name).write_text(f"Fact from {name}.")

        svc = MagicMock()
        svc.embedder.embed_batch.return_value = [_orthogonal_body_vector()]
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)

        chunks_by_source = {
            name: [_make_chunk(f"Fact from {name}.", source=name)] for name in sources
        }
        wiki_text = _synthesis_wiki_text(sources)
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
        assert WikiSubdir.DRAFTS in result.parts
        assert WikiSubdir.SYNTHESIS not in result.parts

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
        assert WikiSubdir.DRAFTS in result.parts

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

    def test_excerpt_free_footnotes_do_not_publish(self, tmp_path: Path):
        """A page whose only footnote quotes nothing has nothing verified to publish on."""
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
        assert result is None
        store.replace_citations_for_wiki.assert_not_called()


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

        wiki_text = _synthesis_wiki_text(sources, topic="gradual typing")
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
        assert WikiSubdir.SYNTHESIS in result[0].parts

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
    @staticmethod
    def _divert(drafts_dir: Path, content: str, sources: list[str]) -> Path:
        return divert_to_drafts(
            content, drafts_dir, "my-page", 0.45, "diff text", "concepts", sources
        )

    def test_writes_draft_with_note(self, tmp_path: Path):
        drafts_dir = tmp_path / "drafts"
        content = "# New Page\n\nNew content."
        result = self._divert(drafts_dir, content, ["a.md"])
        assert result.exists()
        assert result.parent == drafts_dir
        text = result.read_text()
        assert "DRIFT" in text
        assert "45%" in text
        assert "human review" in text
        # The origin subdir rides the marker so accept restores the page to concepts/.
        assert "origin: concepts" in text
        assert content in text

    def test_same_source_rewrites_its_own_draft(self, tmp_path: Path):
        drafts_dir = tmp_path / "drafts"
        first = self._divert(drafts_dir, "# First\n", ["a.md"])
        second = self._divert(drafts_dir, "# Second\n", ["a.md"])
        assert second == first
        assert "# Second" in first.read_text()

    def test_other_source_lands_on_a_collision_draft(self, tmp_path: Path):
        """A second source's diverted page must not overwrite one awaiting review."""
        drafts_dir = tmp_path / "drafts"
        first = self._divert(drafts_dir, "# From a\n", ["a.md"])
        second = self._divert(drafts_dir, "# From b\n", ["b.md"])
        assert second != first
        assert second.name.startswith("my-page-collision-")
        assert "# From a" in first.read_text()
        assert "# From b" in second.read_text()
        # The collision marker is what the drafts surface classifies on.
        assert PENDING_MARKER_KEYWORD_COLLISION in second.read_text()

    def test_replaces_a_pending_marker_at_the_same_slug(self, tmp_path: Path):
        """A marker is a placeholder, not review content, so drift may claim the slug."""
        drafts_dir = tmp_path / "drafts"
        marker_path = write_pending_marker(
            drafts_dir, "my-page", f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source a.md -->"
        )
        result = self._divert(drafts_dir, "# Regenerated\n", ["b.md"])
        assert result == marker_path
        assert "# Regenerated" in result.read_text()


class TestWritePageDrift:
    """Rebuilds converge: only body prose counts as drift, and a published
    regen retires the proposal an earlier drift parked in drafts/."""

    @staticmethod
    def _page(body: str, timestamp: str) -> str:
        return (
            f"---\ngenerated_by: m\ngenerated_at: {timestamp}\n"
            f'sources: ["a.md"]\nfaithfulness_score: 0.90\n---\n\n{body}\n'
        )

    def test_frontmatter_churn_alone_does_not_divert(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        body = "# Brakes\n\n" + "\n".join(f"line {i}" for i in range(20))
        page = wiki_root / WikiSubdir.CONCEPTS / "brakes.md"
        page.parent.mkdir(parents=True)
        page.write_text(self._page(body, "2020-01-01T00:00:00+00:00"))
        # A zero threshold diverts on any body change at all, so publishing
        # here proves the timestamp and score churn was excluded.
        result = write_page(
            wiki_root,
            WikiSubdir.CONCEPTS,
            "brakes",
            self._page(body, "2026-07-28T12:00:00+00:00"),
            0.0,
            ["a.md"],
        )
        assert result == page

    def test_publishing_removes_a_superseded_drift_draft(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        stale = divert_to_drafts(
            "# Old proposal\n",
            wiki_root / WikiSubdir.DRAFTS,
            "brakes",
            0.5,
            "diff",
            WikiSubdir.CONCEPTS,
            ["a.md"],
        )
        assert stale.is_file()
        write_page(
            wiki_root,
            WikiSubdir.CONCEPTS,
            "brakes",
            self._page("# Brakes\n\nfresh body", "2026-07-28T12:00:00+00:00"),
            0.3,
            ["a.md"],
        )
        assert not stale.exists()

    def test_a_page_routed_to_drafts_survives_its_own_write(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        result = write_page(
            wiki_root,
            WikiSubdir.DRAFTS,
            "brakes",
            self._page("# Brakes\n\nlow score body", "2026-07-28T12:00:00+00:00"),
            0.3,
            ["a.md"],
        )
        assert result.is_file()


class TestWritePageToDrafts:
    """A drafts target is a proposal, not a published body: no drift ratio is
    computed, and the source set decides who may claim ``drafts/<slug>.md``."""

    @staticmethod
    def _page(body: str, sources: list[str]) -> str:
        return (
            "---\ngenerated_by: m\ngenerated_at: 2026-07-28T12:00:00+00:00\n"
            f"sources: {json.dumps(sorted(sources))}\nfaithfulness_score: 0.10\n---\n\n{body}\n"
        )

    def _write(self, wiki_root: Path, body: str, sources: list[str]) -> Path:
        return write_page(
            wiki_root, WikiSubdir.DRAFTS, "brakes", self._page(body, sources), 0.3, sources
        )

    def test_the_same_sources_supersede_their_own_draft(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        first = self._write(wiki_root, "# Brakes\n\nfirst proposal", ["a.md"])
        second = self._write(wiki_root, "# Brakes\n\nsecond proposal", ["a.md"])
        assert second == first
        text = first.read_text()
        assert "second proposal" in text
        assert "DRIFT" not in text
        assert f"origin: {WikiSubdir.DRAFTS}" not in text

    def test_a_different_source_set_lands_on_a_collision_draft(self, tmp_path: Path):
        """Overwriting here is the loss the collision path exists to prevent."""
        wiki_root = tmp_path / "wiki"
        first = self._write(wiki_root, "# Brakes\n\nfrom a", ["a.md"])
        second = self._write(wiki_root, "# Brakes\n\nfrom b", ["b.md"])
        assert second != first
        assert second.name.startswith("brakes-collision-")
        assert "from a" in first.read_text()
        assert PENDING_MARKER_KEYWORD_COLLISION in second.read_text()

    def test_a_pending_marker_at_the_slug_is_claimed(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        marker = write_pending_marker(
            wiki_root / WikiSubdir.DRAFTS,
            "brakes",
            f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source a.md -->",
        )
        result = self._write(wiki_root, "# Brakes\n\nreal content", ["b.md"])
        assert result == marker
        assert "real content" in result.read_text()

    def test_a_drift_draft_from_the_same_sources_is_superseded(self, tmp_path: Path):
        """A drift draft records its sources in the frontmatter it carries, so a
        later proposal from those sources replaces it in place."""
        wiki_root = tmp_path / "wiki"
        drift = divert_to_drafts(
            self._page("# Brakes\n\nearlier proposal", ["a.md"]),
            wiki_root / WikiSubdir.DRAFTS,
            "brakes",
            0.5,
            "diff",
            WikiSubdir.CONCEPTS,
            ["a.md"],
        )
        result = self._write(wiki_root, "# Brakes\n\nlater proposal", ["a.md"])
        assert result == drift
        text = drift.read_text()
        assert "later proposal" in text
        assert "DRIFT" not in text


class TestDeleteDriftDraftIfPresent:
    def test_returns_false_when_no_draft_exists(self, tmp_path: Path):
        assert delete_drift_draft_if_present(tmp_path, "missing") is False

    def test_leaves_a_low_faithfulness_draft_alone(self, tmp_path: Path):
        draft = tmp_path / "x.md"
        draft.write_text("---\nfaithfulness_score: 0.2\n---\n\nbody\n")
        assert delete_drift_draft_if_present(tmp_path, "x") is False
        assert draft.is_file()

    def test_removes_a_drift_draft(self, tmp_path: Path):
        draft = divert_to_drafts(
            "# body\n", tmp_path, "x", 0.5, "diff", WikiSubdir.CONCEPTS, ["a.md"]
        )
        assert delete_drift_draft_if_present(tmp_path, "x") is True
        assert not draft.exists()


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
        assert WikiSubdir.DRAFTS in result.parts
        # Original should be unchanged
        assert "Totally different synthesis" in existing.read_text()


def _assert_no_store_writes(store: MagicMock) -> None:
    """Assert no citation or chunk rows were written for this page."""
    store.replace_citations_for_wiki.assert_not_called()
    store.add_citations.assert_not_called()
    store.delete_citations_for_wiki.assert_not_called()
    store.replace_chunks.assert_not_called()
    store.clear_table.assert_not_called()


def _assert_cleared(store: MagicMock, wiki_source: str) -> None:
    """Assert the page's rows were cleared once, with no replacement written."""
    store.clear_table.assert_called_once()
    table, predicate = store.clear_table.call_args.args
    assert table == CHUNKS_TABLE
    assert wiki_source in predicate
    assert ChunkType.WIKI in predicate


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
            index_wiki_page(content, target.wiki_source, store, cfg)

        store.clear_table.assert_not_called()
        store.replace_chunks.assert_called_once()
        records, predicate = store.replace_chunks.call_args.args
        assert target.wiki_source in predicate
        assert ChunkType.WIKI in predicate
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
            index_wiki_page(self._content("body"), target.wiki_source, store, cfg)

        store.clear_table.assert_not_called()
        store.replace_chunks.assert_not_called()

    def test_nested_wiki_dir_still_resolves_the_subdir(self):
        """A wiki_dir carrying a separator keeps its pages in retrieval."""
        store = MagicMock(spec=Store)
        config = cfg.model_copy(update={"wiki_dir": "notes/wiki"})
        wiki_source = f"{config.wiki_dir}/{WikiSubdir.CONCEPTS}/brakes.md"

        with (
            patch("lilbee.wiki.page.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.page.chunk_text", return_value=["body"]),
        ):
            index_wiki_page(self._content("body"), wiki_source, store, config)

        store.replace_chunks.assert_called_once()
        _records, predicate = store.replace_chunks.call_args.args
        assert wiki_source in predicate

    def test_malformed_wiki_source_logs_warning_and_skips(self, caplog: pytest.LogCaptureFixture):
        """A ``wiki_source`` without a subdir component is logged and
        skipped rather than silently writing to the store or crashing.
        """
        store = MagicMock(spec=Store)
        caplog.set_level("WARNING", logger="lilbee.wiki.page")
        result = index_wiki_page(self._content("body"), "malformed", store, cfg)
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
            index_wiki_page(content, target.wiki_source, store, cfg)

        _assert_cleared(store, target.wiki_source)
        chunker.assert_not_called()
        store.replace_chunks.assert_not_called()

    def test_chunker_returns_empty_skips_add(self):
        """If chunk_text returns no chunks, invalidate stale rows and return."""
        store = MagicMock(spec=Store)
        target = self._target()

        with (
            patch("lilbee.wiki.page.get_services", return_value=self._services_mock()),
            patch("lilbee.wiki.page.chunk_text", return_value=[]),
        ):
            index_wiki_page(self._content("some body"), target.wiki_source, store, cfg)

        _assert_cleared(store, target.wiki_source)
        store.replace_chunks.assert_not_called()

    def test_wrong_dimension_vectors_hit_the_store_dimension_gate(self):
        """Wiki rows embedded at the wrong width fail loudly instead of being written."""
        store = MagicMock(spec=Store)
        store.replace_chunks.side_effect = lambda records, predicate: _check_vector_dims(
            records, cfg.embedding_dim
        )
        target = self._target()
        services = self._services_mock(vector_dim=cfg.embedding_dim + 1)

        with (
            patch("lilbee.wiki.page.get_services", return_value=services),
            patch("lilbee.wiki.page.chunk_text", return_value=["body"]),
            pytest.raises(ValueError, match="Vector dimension mismatch"),
        ):
            index_wiki_page(self._content("body"), target.wiki_source, store, cfg)

    def test_embedding_runs_before_any_store_write(self):
        """No crash window empties the page: chunking and embedding finish before the
        store is touched, and the swap itself is the store's single locked replace."""
        store = MagicMock(spec=Store)
        target = self._target()
        store.replace_chunks.side_effect = RuntimeError("embedder is called first")

        embedded: list[list[str]] = []
        services = self._services_mock()
        services.embedder.embed_batch.side_effect = lambda texts, **kw: (
            embedded.append(texts) or [[0.1] * cfg.embedding_dim for _ in texts]
        )

        with (
            patch("lilbee.wiki.page.get_services", return_value=services),
            patch("lilbee.wiki.page.chunk_text", return_value=["one chunk"]),
            pytest.raises(RuntimeError),
        ):
            index_wiki_page(self._content("first body"), target.wiki_source, store, cfg)

        assert embedded == [["one chunk"]]
        store.clear_table.assert_not_called()


class TestBuildFrontmatter:
    """B2: frontmatter carries a provenance block when chunks are provided."""

    def test_no_chunks_omits_provenance(self):
        from lilbee.wiki.page import build_frontmatter

        fm = build_frontmatter(cfg, ["doc.md"], 0.9)
        assert "provenance:" not in fm

    def test_source_with_quotes_and_backslashes_round_trips(self):
        """A filename with quotes/backslashes stays valid YAML, not corrupt frontmatter."""
        import yaml

        from lilbee.wiki.page import build_frontmatter

        nasty = 'weird"name\\path.txt'
        fm = build_frontmatter(cfg, [nasty, "plain.md"], 0.9)
        parsed = yaml.safe_load(fm.strip().strip("-").strip())
        assert set(parsed["sources"]) == {nasty, "plain.md"}

    def test_find_excerpt_location_matches_across_whitespace(self):
        """A verified excerpt finds its page/line even when the chunk wraps it
        across newlines (verify normalizes whitespace; location must too)."""
        from lilbee.wiki.citations import _find_excerpt_location

        chunk = _make_chunk(
            "the quick\nbrown   fox jumps", page_start=3, page_end=3, line_start=7, line_end=8
        )
        loc = _find_excerpt_location("the quick brown fox", [chunk])
        assert loc == (3, 3, 7, 8)

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

    def test_provenance_records_effective_mode_not_configured(self, monkeypatch):
        """When the configured entity mode falls back, provenance names the one that ran."""
        from lilbee.core.config.enums import WikiEntityMode
        from lilbee.wiki.page import build_frontmatter

        monkeypatch.setattr(cfg, "wiki_entity_mode", WikiEntityMode.LLM_TAGGED)
        chunks = [_make_chunk("body", source="doc.md", chunk_index=0)]
        fm = build_frontmatter(cfg, ["doc.md"], 0.85, chunks=chunks)
        # LLM_TAGGED isn't implemented; it falls back to ner_entities at run time.
        assert "extraction_method: ner_entities" in fm
        assert "llm_tagged" not in fm

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
    def test_archives_concept_pages_and_deletes_their_rows(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "foo.md").write_text("original")
        data_dir = tmp_path / "data"
        store = MagicMock(spec=Store)
        archive_legacy_concept_pages(wiki_root, data_dir, store, cfg)
        assert not (wiki_root / "concepts" / "foo.md").exists()
        assert (wiki_root / "archive" / "concepts" / "foo.md").read_text() == "original"
        assert (data_dir / ".phase-d-migrated").exists()
        # An archived page must stop serving its content from the index.
        wiki_source = f"{cfg.wiki_dir}/concepts/foo.md"
        store.delete_by_source.assert_called_once_with(wiki_source)
        store.delete_citations_for_wiki.assert_called_once_with(wiki_source)

    def test_idempotent(self, tmp_path: Path):
        wiki_root = tmp_path / "wiki"
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "foo.md").write_text("original")
        data_dir = tmp_path / "data"
        store = MagicMock(spec=Store)
        archive_legacy_concept_pages(wiki_root, data_dir, store, cfg)
        # Second run: nothing to archive; should not touch disk state.
        (wiki_root / "concepts").mkdir(parents=True, exist_ok=True)
        (wiki_root / "concepts" / "new.md").write_text("fresh")
        archive_legacy_concept_pages(wiki_root, data_dir, store, cfg)
        # Freshly written page stayed put.
        assert (wiki_root / "concepts" / "new.md").exists()
        store.delete_by_source.assert_called_once()


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

        def fake_build_wiki(
            entities, provider, store, config, *, extract_concepts, on_progress, stats
        ):
            captured["config"] = config
            return []

        monkeypatch.setattr("lilbee.wiki.generation.get_services", fake_get_services)
        monkeypatch.setattr("lilbee.wiki.generation.build_wiki", fake_build_wiki)
        monkeypatch.setattr("lilbee.wiki.generation.update_wiki_index", lambda *a, **kw: None)
        monkeypatch.setattr("lilbee.wiki.generation.append_wiki_log", lambda *a, **kw: None)
        monkeypatch.setattr("lilbee.wiki.generation.get_entity_extractor", fake_extractor)

        result = run_full_build()
        assert captured["config"] is cfg
        assert result["paths"] == []
        assert result["entities"] == 0
        assert result["count"] == 0
        assert result["stats"] == BuildStats().as_dict()


class TestRunFullSynthesize:
    def test_defaults_to_global_cfg_when_called_with_no_args(self, monkeypatch, tmp_path):
        """run_full_synthesize() with no arg falls back to lilbee.config.cfg."""
        from lilbee.wiki.generation import run_full_synthesize

        captured: dict[str, object] = {}

        def fake_get_services():
            return MagicMock()

        def fake_generate(provider, store, clusterer, config, on_progress, stats):
            captured["config"] = config
            return [tmp_path / "wiki" / "synthesis" / "typing.md"]

        monkeypatch.setattr("lilbee.wiki.generation.get_services", fake_get_services)
        monkeypatch.setattr("lilbee.wiki.generation.generate_synthesis_pages", fake_generate)

        result = run_full_synthesize()
        assert captured["config"] is cfg
        assert result["count"] == 1
        assert result["paths"][0].endswith("typing.md")


class TestRunSummaryStats:
    """Both run entry points report what their gates did, in the summary and log.md."""

    def test_build_summary_and_log_carry_the_gate_stats(self, monkeypatch):
        from lilbee.wiki.generation import run_full_build

        def fake_build_wiki(
            entities, provider, store, config, *, extract_concepts, on_progress, stats
        ):
            stats.record_published("wiki/concepts/brakes.md", 2)
            stats.record_drafted()
            stats.record_pending_marker()
            stats.record_citations(rendered=2, dropped=1)
            return [Path("wiki/concepts/brakes.md")]

        def fake_extractor(*a, **kw):
            ext = MagicMock()
            ext.extract.return_value = []
            return ext

        services = MagicMock()
        services.store.get_sources.return_value = []
        monkeypatch.setattr("lilbee.wiki.generation.get_services", lambda: services)
        monkeypatch.setattr("lilbee.wiki.generation.build_wiki", fake_build_wiki)
        monkeypatch.setattr("lilbee.wiki.generation.update_wiki_index", lambda *a, **kw: None)
        monkeypatch.setattr("lilbee.wiki.generation.get_entity_extractor", fake_extractor)

        result = run_full_build(cfg)
        assert result["stats"]["pages_published"] == 1
        assert result["stats"]["pages_drafted"] == 1
        assert result["stats"]["pending_markers"] == 1
        assert result["stats"]["citation_verify_rate"] == 2 / 3
        assert result["stats"]["verified_by_page"] == {"wiki/concepts/brakes.md": 2}

        log_text = (cfg.data_root / cfg.wiki_dir / "log.md").read_text()
        assert "1 published, 1 drafted, 1 markers, 2/3 citations verified" in log_text

    def test_synthesize_summary_and_log_carry_the_gate_stats(self, monkeypatch, tmp_path):
        from lilbee.wiki.generation import run_full_synthesize

        def fake_generate(provider, store, clusterer, config, on_progress, stats):
            stats.record_published("wiki/synthesis/typing.md", 3)
            stats.record_citations(rendered=3, dropped=0)
            return [tmp_path / "wiki" / "synthesis" / "typing.md"]

        monkeypatch.setattr("lilbee.wiki.generation.get_services", MagicMock())
        monkeypatch.setattr("lilbee.wiki.generation.generate_synthesis_pages", fake_generate)

        result = run_full_synthesize(cfg)
        assert result["stats"]["pages_published"] == 1
        assert result["stats"]["publish_rate"] == 1.0
        assert result["stats"]["verified_by_page"] == {"wiki/synthesis/typing.md": 3}

        log_text = (cfg.data_root / cfg.wiki_dir / "log.md").read_text()
        assert "synthesize | 1 synthesis pages" in log_text
        assert "1 published, 0 drafted, 0 markers, 3/3 citations verified" in log_text


class TestPersistAndFinalizeDrift:
    """A drift-diverted regen must not leak its unreviewed body into the index or
    citations under the published page's identity."""

    def test_diversion_skips_publish_indexing(self):
        from lilbee.wiki.persistence import persist_and_finalize

        store = MagicMock(spec=Store)
        target = TestWikiIndexing._target()
        published = target.wiki_root / target.subdir / f"{target.slug}.md"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text("Old published body, unrelated to the regen.", encoding="utf-8")

        stats = BuildStats()
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
                stats=stats,
            )
        finally:
            cfg.wiki_drift_threshold = old

        assert WikiSubdir.DRAFTS in page_path.parts
        assert "Old published body" in published.read_text()
        _assert_no_store_writes(store)
        assert (stats.pages_drafted, stats.pages_published) == (1, 0)


class TestPersistAndFinalizeDrafts:
    """A page routed to drafts carries no store state until it is accepted."""

    def test_draft_target_writes_the_file_and_nothing_else(self):
        from lilbee.wiki.persistence import persist_and_finalize

        store = MagicMock(spec=Store)
        target = TestWikiIndexing._target(subdir=WikiSubdir.DRAFTS)
        cfg.wiki_prune_raw = True

        page_path = persist_and_finalize(
            "# Brakes\n\nBody that failed the faithfulness gate.\n",
            target,
            [make_citation()],
            ["doc.md"],
            store,
            cfg,
        )

        assert page_path == target.wiki_root / WikiSubdir.DRAFTS / f"{target.slug}.md"
        assert page_path.read_text().startswith("# Brakes")
        _assert_no_store_writes(store)
        # A draft supersedes nothing, so its sources stay searchable.
        store.delete_by_source.assert_not_called()
        assert "pending review" in (target.wiki_root / "log.md").read_text()


class TestWritePendingMarker:
    def test_writes_the_marker_when_no_draft_exists(self, tmp_path: Path):
        marker = f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source a.md -->"
        path = write_pending_marker(tmp_path / "drafts", "brakes", marker, "---\nx: 1\n---\n")
        assert path.read_text() == f"{marker}\n\n---\nx: 1\n---\n"

    def test_refreshes_an_existing_marker(self, tmp_path: Path):
        drafts = tmp_path / "drafts"
        first = write_pending_marker(
            drafts, "brakes", f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source a.md -->"
        )
        second_marker = f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source b.md -->"
        assert write_pending_marker(drafts, "brakes", second_marker) == first
        assert "source b.md" in first.read_text()

    def test_keeps_a_draft_holding_real_content(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """A parse failure must not replace a page a human is reviewing with a marker.

        The withheld write reports None so the caller does not count a marker that
        never landed."""
        drafts = tmp_path / "drafts"
        review_path = divert_to_drafts(
            "# Brakes\n\nReviewed body.\n", drafts, "brakes", 0.4, "diff", "concepts", ["a.md"]
        )
        with caplog.at_level("WARNING", logger="lilbee.wiki.persistence"):
            result = write_pending_marker(
                drafts, "brakes", f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source a.md -->"
            )
        assert result is None
        assert "Reviewed body." in review_path.read_text()
        assert PENDING_MARKER_KEYWORD_PARSE not in review_path.read_text()
        assert "pending review" in caplog.text
