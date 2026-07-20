"""Tests for the RAG query pipeline (mocked: no live server needed)."""

from typing import ClassVar
from unittest import mock

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.core.config import cfg
from lilbee.data.store import ChunkType, MemoryRow, SearchChunk
from lilbee.providers.base import ChatResult, FinishReason
from lilbee.retrieval.query import (
    Searcher,
    build_context,
    filter_results,
    format_source,
    format_sources_block,
    sort_by_relevance,
    strip_llm_citations,
)
from lilbee.retrieval.query.dedup import _relevance_weight
from lilbee.retrieval.query.formatting import (
    StreamingCitationFilter,
    _extract_cited_indices,
    _format_citation,
    cited_subset,
    unique_sources,
)
from lilbee.retrieval.query.searcher import (
    EMPTY_LIBRARY,
    GROUNDED_REFUSAL,
    SEARCH_NEEDS_EMBEDDER,
    QueryMode,
)
from tests.conftest import make_citation


def _text_result(text: str) -> ChatResult:
    """Build a text-only ``ChatResult`` for non-streaming chat mocks."""
    return ChatResult(text=text, tool_calls=(), finish_reason=FinishReason.STOP)


@pytest.fixture(autouse=True)
def _disable_concepts():
    """Disable concept graph by default in query tests to avoid spaCy loads."""
    old = cfg.concept_graph
    cfg.concept_graph = False
    yield
    cfg.concept_graph = old


@pytest.fixture(autouse=True)
def _reset_chat_mode():
    """Pin chat_mode to 'search' so retrieval-tests are not skipped by stale state."""
    old = cfg.chat_mode
    cfg.chat_mode = "search"
    yield
    cfg.chat_mode = old


@pytest.fixture(autouse=True)
def _reset_show_reasoning():
    """Pin show_reasoning to False (model.py default) so think-tag-strip tests
    are not flipped by a writable config.toml that persisted True from a
    previous session."""
    old = cfg.show_reasoning
    cfg.show_reasoning = False
    yield
    cfg.show_reasoning = old


@pytest.fixture(autouse=True)
def _disable_hyde():
    """Pin hyde off so the HyDE call_args does not shadow the primary search
    in mock assertions that inspect call_args (single-call-only)."""
    old = cfg.hyde
    cfg.hyde = False
    yield
    cfg.hyde = old


@pytest.fixture(autouse=True)
def mock_svc():
    """Inject mock Services so tests never hit real backends."""
    from tests.conftest import make_mock_services

    services = make_mock_services()
    set_services(services)
    yield services
    set_services(None)


# Sentinel: derive the canonical score from distance unless overridden.
_AUTO_SCORE = object()


def _make_result(
    source="test.pdf",
    content_type="pdf",
    chunk_type="raw",
    page_start=1,
    page_end=1,
    line_start=0,
    line_end=0,
    chunk="some text",
    chunk_index=0,
    distance=0.5,
    relevance_score=None,
    bm25_score=None,
    rerank_score=None,
    vector=None,
    score=_AUTO_SCORE,
) -> SearchChunk:
    if score is _AUTO_SCORE:
        # Mirror the store contract: every search path sets the canonical
        # score. Pass score=None explicitly to build a legacy (pre-score) row.
        score = max(0.0, min(1.0, 1.0 - distance)) if distance is not None else None
    return SearchChunk(
        source=source,
        content_type=content_type,
        chunk_type=chunk_type,
        page_start=page_start,
        page_end=page_end,
        line_start=line_start,
        line_end=line_end,
        chunk=chunk,
        chunk_index=chunk_index,
        distance=distance,
        relevance_score=relevance_score,
        bm25_score=bm25_score,
        rerank_score=rerank_score,
        vector=vector or [0.1],
        score=score,
    )


class TestDisplaySourcePath:
    """source citations render absolute paths with ~ expansion."""

    def test_expands_under_documents_dir(self, tmp_path):
        from lilbee.retrieval.query import display_source_path

        cfg.documents_dir = tmp_path / "docs"
        result = display_source_path("_web/example.com/index.md")
        normalized = result.replace("\\", "/")
        assert str(tmp_path).replace("\\", "/") in normalized or normalized.startswith("~/")
        assert "_web/example.com/index.md" in normalized

    def test_substitutes_home_with_tilde(self, tmp_path, monkeypatch):
        from pathlib import Path as _Path

        from lilbee.retrieval.query import display_source_path

        # Force documents_dir under the home directory so ~ substitution fires.
        cfg.documents_dir = _Path.home() / ".lilbee-fixes-content-test" / "docs"
        result = display_source_path("note.md")
        assert result.startswith("~/")
        assert result.endswith("note.md")

    def test_falls_back_to_raw_on_resolve_failure(self, tmp_path, monkeypatch):
        from pathlib import Path as _Path

        from lilbee.retrieval.query import display_source_path

        cfg.documents_dir = tmp_path / "docs"

        # Force resolve() to raise so the fallback path runs.
        def _raise(self, strict=False):
            raise OSError("simulated")

        monkeypatch.setattr(_Path, "resolve", _raise)
        assert display_source_path("anything.md") == "anything.md"

    def test_returns_absolute_when_not_under_home(self, tmp_path, monkeypatch):
        """When the resolved path is not under ``Path.home()``, fall through to str(resolved)."""
        from pathlib import Path as _Path

        from lilbee.retrieval.query import display_source_path

        cfg.documents_dir = tmp_path / "docs"
        (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(_Path, "home", classmethod(lambda cls: tmp_path / "fake-home"))

        result = display_source_path("note.md")
        assert not result.startswith("~/")
        assert result.endswith("note.md")


class TestFormatSource:
    def test_pdf_single_page(self):
        r = _make_result(source="manual.pdf", content_type="pdf", page_start=5, page_end=5)
        assert "manual.pdf" in format_source(r)
        assert "page 5" in format_source(r)

    def test_pdf_page_range(self):
        r = _make_result(source="manual.pdf", content_type="pdf", page_start=3, page_end=7)
        assert "pages 3-7" in format_source(r)

    def test_code_line_range(self):
        r = _make_result(source="app.py", content_type="code", line_start=10, line_end=25)
        assert "lines 10-25" in format_source(r)

    def test_code_single_line(self):
        r = _make_result(source="app.py", content_type="code", line_start=10, line_end=10)
        assert "line 10" in format_source(r)

    def test_text_file_no_page_or_line(self):
        r = _make_result(source="readme.md", content_type="text")
        result = format_source(r)
        assert "readme.md" in result
        # Text sources carry no page/line locator suffix (", page N" / ", line N").
        assert ", page" not in result
        assert ", line" not in result

    def test_renders_a_clickable_markdown_link(self):
        r = _make_result(source="notes/readme.md", content_type="text")
        result = format_source(r)
        # [label](file-url): readable label, file:// target so the reader can open it.
        assert result.startswith("[notes/readme.md](file://")
        assert result.endswith(".md)")

    def test_web_source_gets_a_readable_host_slug_label(self):
        r = _make_result(source="_web/www.example.com/how-to-foo/index.md", content_type="text")
        assert format_source(r).startswith("[example.com · how-to-foo](file://")

    def test_degenerate_web_path_keeps_raw_source_label(self):
        # Nothing left after dropping index.md: fall back to the stored path.
        r = _make_result(source="_web/index.md", content_type="text")
        assert format_source(r).startswith("[_web/index.md](file://")

    def test_unresolvable_path_renders_plain_label_without_link(self, monkeypatch):
        # A source that can't resolve to a file URL degrades to bare text. The
        # failure is injected directly: OS-level triggers like null bytes vary
        # by platform and Python version (Windows 3.13 percent-encodes NUL
        # instead of raising).
        class _UnresolvableDir:
            def __truediv__(self, other: str) -> "_UnresolvableDir":
                raise OSError("cannot resolve")

        monkeypatch.setattr(
            "lilbee.retrieval.query.formatting.cfg",
            mock.Mock(documents_dir=_UnresolvableDir()),
        )
        r = _make_result(source="doc.md", content_type="text")
        assert format_source(r) == "doc.md"


class TestUniqueSources:
    def test_first_chunk_per_distinct_source_in_order(self):
        results = [
            _make_result(source="a.md", chunk="a1"),
            _make_result(source="b.md", chunk="b1"),
            _make_result(source="a.md", chunk="a2"),
        ]
        uniq = unique_sources(results)
        assert [r.source for r in uniq] == ["a.md", "b.md"]
        assert uniq[0].chunk == "a1"


class TestFormatSourcesBlock:
    def test_numbers_each_unique_source_matching_inline_markers(self):
        results = [
            _make_result(source="a.md", content_type="text"),
            _make_result(source="b.md", content_type="text"),
        ]
        block = format_sources_block(results)
        assert block.startswith("\n\nSources:\n\n")
        assert "1. [a.md](file://" in block
        assert "2. [b.md](file://" in block

    def test_collapses_repeated_source_to_one_number(self):
        results = [
            _make_result(source="a.md", content_type="text"),
            _make_result(source="a.md", content_type="text"),
            _make_result(source="b.md", content_type="text"),
        ]
        block = format_sources_block(results)
        assert "1. [a.md](file://" in block
        assert "2. [b.md](file://" in block
        assert "3." not in block

    def test_empty_when_no_sources(self):
        assert format_sources_block([]) == ""


class TestSortByRelevance:
    def test_sorts_by_distance(self):
        results = [
            _make_result(source="far.pdf", distance=0.9),
            _make_result(source="close.pdf", distance=0.1),
            _make_result(source="mid.pdf", distance=0.5),
        ]
        sorted_results = sort_by_relevance(results)
        assert sorted_results[0].source == "close.pdf"
        assert sorted_results[1].source == "mid.pdf"
        assert sorted_results[2].source == "far.pdf"

    def test_missing_distance_sorts_last(self):
        results = [
            _make_result(source="no_dist.pdf", distance=None),
            _make_result(source="has_dist.pdf", distance=0.3),
        ]
        sorted_results = sort_by_relevance(results)
        assert sorted_results[0].source == "has_dist.pdf"
        assert sorted_results[1].source == "no_dist.pdf"

    def test_sorts_by_canonical_score_first(self):
        results = [
            _make_result(source="low.pdf", score=0.2),
            _make_result(source="high.pdf", score=0.9),
            _make_result(source="mid.pdf", score=0.5),
        ]
        sorted_results = sort_by_relevance(results)
        assert [r.source for r in sorted_results] == ["high.pdf", "mid.pdf", "low.pdf"]

    def test_scoreless_rows_sort_last(self):
        """Rows the store never produced (no canonical score) sort behind
        every scored row instead of resurrecting per-arm comparisons."""
        results = [
            _make_result(source="unscored.pdf", distance=None, score=None, relevance_score=0.9),
            _make_result(source="scored.pdf", score=0.1, distance=None),
        ]
        sorted_results = sort_by_relevance(results)
        assert [r.source for r in sorted_results] == ["scored.pdf", "unscored.pdf"]


class TestDiversifySources:
    def test_caps_per_source(self):
        from lilbee.retrieval.query import diversify_sources

        results = [
            _make_result(source="a.md", distance=0.1),
            _make_result(source="a.md", distance=0.2),
            _make_result(source="a.md", distance=0.3),
            _make_result(source="a.md", distance=0.4),
            _make_result(source="b.md", distance=0.5),
        ]
        diverse = diversify_sources(results, max_per_source=2)
        a_count = sum(1 for r in diverse if r.source == "a.md")
        assert a_count == 2
        assert any(r.source == "b.md" for r in diverse)

    def test_preserves_order(self):
        from lilbee.retrieval.query import diversify_sources

        results = [
            _make_result(source="a.md", distance=0.1),
            _make_result(source="b.md", distance=0.2),
            _make_result(source="a.md", distance=0.3),
        ]
        diverse = diversify_sources(results, max_per_source=1)
        assert diverse[0].source == "a.md"
        assert diverse[1].source == "b.md"
        assert len(diverse) == 2

    def test_empty_input(self):
        from lilbee.retrieval.query import diversify_sources

        assert diversify_sources([]) == []

    def test_default_cap_uses_cfg_value(self, monkeypatch):
        from lilbee.core.config import cfg
        from lilbee.retrieval.query import diversify_sources

        monkeypatch.setattr(cfg, "diversity_max_per_source", 3)
        results = [_make_result(source="a.md", distance=float(i) / 10) for i in range(5)]
        diverse = diversify_sources(results)
        assert len(diverse) == 3

    def test_wiki_and_raw_with_same_stem_are_independent_sources(self):
        """``wiki/summaries/doc.md`` and ``doc.md`` keep separate caps.

        Per-source cap keys on the exact ``source`` string. A wiki page
        paraphrasing ``doc.md`` becomes ``wiki/summaries/doc.md`` in the
        store, which is a distinct source. So the cap applies to each
        side independently. With ``max_per_source=1`` both sides still
        surface together.
        """
        from lilbee.retrieval.query import diversify_sources

        results = [
            _make_result(source="wiki/summaries/doc.md", chunk_type="wiki", distance=0.1),
            _make_result(source="doc.md", chunk_type="raw", distance=0.2),
            _make_result(source="doc.md", chunk_type="raw", distance=0.3),
        ]
        diverse = diversify_sources(results, max_per_source=1)
        assert len(diverse) == 2
        assert {r.chunk_type for r in diverse} == {"wiki", "raw"}


class TestBuildContext:
    def test_numbers_passages_by_source(self):
        results = [
            _make_result(source="a.md", chunk="chunk one"),
            _make_result(source="b.md", chunk="chunk two"),
        ]
        ctx = build_context(results)
        assert "[1] (a.md, page 1)\nchunk one" in ctx
        assert "[2] (b.md, page 1)\nchunk two" in ctx

    def test_passages_from_the_same_source_share_a_number(self):
        # Stable, streamable numbers that map 1:1 to the Sources block: two
        # passages from one file are both [1], the next distinct file is [2].
        results = [
            _make_result(source="a.md", chunk="one"),
            _make_result(source="a.md", chunk="two"),
            _make_result(source="b.md", chunk="three"),
        ]
        ctx = build_context(results)
        assert "[1] (a.md, page 1)\none" in ctx
        assert "[1] (a.md, page 1)\ntwo" in ctx
        assert "[2] (b.md, page 1)\nthree" in ctx

    def test_header_names_the_source_document(self):
        """The answering model must see which document a block came from."""
        results = [
            _make_result(
                source="survey_report.pdf", chunk="core sample B17", page_start=3, page_end=4
            )
        ]
        ctx = build_context(results)
        assert "[1] (survey_report.pdf, pages 3-4)" in ctx
        assert "core sample B17" in ctx

    def test_header_shows_lines_for_code(self):
        results = [
            _make_result(
                source="app.py", content_type="code", chunk="def x():", line_start=10, line_end=20
            )
        ]
        ctx = build_context(results)
        assert "(app.py, lines 10-20)" in ctx

    def test_header_plain_for_text(self):
        results = [_make_result(source="notes.md", content_type="text", chunk="hello")]
        ctx = build_context(results)
        assert "[1] (notes.md)\nhello" in ctx

    def test_header_omits_page_zero(self):
        """PDF chunks whose page metadata was missing are stored with page 0.
        The Sources block already suppresses that locator, so the prompt
        header must too, or the model cites a page the user's list denies."""
        results = [
            _make_result(
                source="scan.pdf", content_type="pdf", chunk="body", page_start=0, page_end=0
            )
        ]
        ctx = build_context(results)
        assert "[1] (scan.pdf)\nbody" in ctx
        assert "page 0" not in ctx

    def test_header_omits_line_zero_for_code(self):
        results = [
            _make_result(
                source="app.py", content_type="code", chunk="x = 1", line_start=0, line_end=0
            )
        ]
        ctx = build_context(results)
        assert "[1] (app.py)\nx = 1" in ctx
        assert "line 0" not in ctx


class TestCitedIndexExtraction:
    def test_single_brackets(self):
        from lilbee.retrieval.query.formatting import _extract_cited_indices

        assert _extract_cited_indices("see [1] and [3]") == {1, 3}

    def test_comma_groups(self):
        from lilbee.retrieval.query.formatting import _extract_cited_indices

        assert _extract_cited_indices("as shown [1, 2] and [4,5]") == {1, 2, 4, 5}

    def test_ranges(self):
        from lilbee.retrieval.query.formatting import _extract_cited_indices

        assert _extract_cited_indices("documented across [1-3]") == {1, 2, 3}

    def test_mixed_group(self):
        from lilbee.retrieval.query.formatting import _extract_cited_indices

        assert _extract_cited_indices("per [1, 3-5]") == {1, 3, 4, 5}

    def test_absurd_range_ignored(self):
        from lilbee.retrieval.query.formatting import _extract_cited_indices

        # A page-span artifact like [1-500] must not fan out into 500 cites.
        assert _extract_cited_indices("[1-500]") == set()


@pytest.mark.usefixtures("wiki_enabled")
class TestSearchContext:
    def test_returns_results(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        results = get_services().searcher.search("question")
        assert len(results) == 1
        mock_svc.embedder.embed_query.assert_called_once_with("question")

    def test_passes_query_text(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher.search("my question")
        mock_svc.store.search.assert_called_once()
        assert mock_svc.store.search.call_args[1]["query_text"] == "my question"

    def test_passes_chunk_type(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher.search("my question", chunk_type="wiki")
        mock_svc.store.search.assert_called_once()
        assert mock_svc.store.search.call_args[1]["chunk_type"] == "wiki"

    def test_chunk_type_defaults_to_none(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher.search("my question")
        mock_svc.store.search.assert_called_once()
        assert mock_svc.store.search.call_args[1]["chunk_type"] is None

    def test_far_vector_only_rows_filtered_at_search(self, mock_svc):
        """search() applies the max_distance rule itself, so every surface
        (CLI, HTTP, MCP, library API) inherits one filter instead of each
        re-implementing its own copy."""
        near = _make_result(source="a.md", distance=0.2)
        far = _make_result(source="b.md", distance=1.4)
        mock_svc.store.search.return_value = [near, far]
        results = get_services().searcher.search("q")
        assert [r.source for r in results] == ["a.md"]

    def test_far_row_with_lexical_support_survives_search(self, mock_svc):
        """A both-arm row keeps its standing past max_distance: dropping it
        on vector distance alone would re-bury exactly the identifier hits
        rank fusion exists to preserve."""
        far_supported = _make_result(source="b.md", distance=1.4, bm25_score=12.0)
        mock_svc.store.search.return_value = [far_supported]
        results = get_services().searcher.search("q")
        assert [r.source for r in results] == ["b.md"]

    def test_expansion_merges_results(self, mock_svc):
        original = _make_result(source="a.md", chunk_index=0)
        expanded = _make_result(source="b.md", chunk_index=0)
        mock_svc.store.search.side_effect = [[original], [expanded]]
        mock_svc.embedder.embed.return_value = [0.1] * 768
        mock_svc.provider.chat.return_value = _text_result("kubernetes deployment internals")
        results = get_services().searcher.search("kubernetes deployment internals")
        assert len(results) == 2
        sources = {r.source for r in results}
        assert "a.md" in sources
        assert "b.md" in sources

    def test_expansion_deduplicates(self, mock_svc):
        same = _make_result(source="a.md", chunk_index=0)
        mock_svc.store.search.side_effect = [[same], [same]]
        mock_svc.embedder.embed.return_value = [0.1] * 768
        mock_svc.provider.chat.return_value = _text_result("kubernetes deployment internals")
        results = get_services().searcher.search("kubernetes deployment internals")
        assert len(results) == 1


class TestExpandQuery:
    _QUESTION_VEC = [0.1] * 768  # matches mock_svc.embedder default

    def test_returns_variants(self, mock_svc):
        mock_svc.provider.chat.return_value = _text_result(
            "explain how X works in detail\nexplain the purpose of X"
        )
        variants = get_services().searcher._expand_query("explain X in detail", self._QUESTION_VEC)
        assert len(variants) == 2
        for text, vec in variants:
            assert isinstance(text, str)
            assert len(vec) == 768

    def test_caps_at_three(self, mock_svc):
        mock_svc.provider.chat.return_value = _text_result("A\nB\nC\nD\nE")
        variants = get_services().searcher._expand_query(
            "explain how kubernetes pods schedule", self._QUESTION_VEC
        )
        assert len(variants) == 3

    def test_strips_reasoning_from_expansion_output(self, mock_svc):
        """A reasoning chat model's think block must not become variants that
        get embedded and searched."""
        mock_svc.provider.chat.return_value = _text_result(
            "<think>the user wants alternatives, let me consider</think>\n"
            "how do pods get scheduled\nkubernetes pod scheduling"
        )
        variants = get_services().searcher._llm_expand("explain pod scheduling", 3)
        assert variants == ["how do pods get scheduled", "kubernetes pod scheduling"]

    def test_strips_list_markers_from_variants(self, mock_svc):
        """Models number their output despite the prompt; '1.' must not reach
        the BM25 arm of the variant search."""
        mock_svc.provider.chat.return_value = _text_result(
            "1. first phrasing\n2) second phrasing\n- third phrasing"
        )
        variants = get_services().searcher._llm_expand("anything at all", 3)
        assert variants == ["first phrasing", "second phrasing", "third phrasing"]

    def test_returns_empty_on_error(self, mock_svc):
        mock_svc.provider.chat.side_effect = RuntimeError("no provider")
        assert (
            get_services().searcher._expand_query(
                "explain how kubernetes pods schedule", self._QUESTION_VEC
            )
            == []
        )

    def test_disabled_when_count_zero(self, mock_svc):
        cfg.query_expansion_count = 0
        assert get_services().searcher._expand_query("anything", self._QUESTION_VEC) == []
        cfg.query_expansion_count = 3

    def test_count_zero_still_runs_concept_expansion_when_enabled(self, mock_svc):
        # Regression: setting query_expansion_count=0 should not
        # short-circuit concept-graph expansion. Users who want an
        # off-switch for LLM expansion while keeping concept expansion
        # need this to stay live.
        old_count = cfg.query_expansion_count
        old_concept = cfg.concept_graph
        cfg.query_expansion_count = 0
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.expand_query.return_value = ["kubernetes"]
        try:
            variants = get_services().searcher._expand_query("k8s", self._QUESTION_VEC)
            assert [text for text, _ in variants] == ["kubernetes"]
            mock_svc.provider.chat.assert_not_called()
        finally:
            cfg.query_expansion_count = old_count
            cfg.concept_graph = old_concept

    def test_count_zero_and_concepts_off_returns_empty(self, mock_svc):
        # Both off-switches should short-circuit before any LLM or
        # embedder calls.
        old = cfg.query_expansion_count
        cfg.query_expansion_count = 0
        try:
            assert get_services().searcher._expand_query("q", self._QUESTION_VEC) == []
            mock_svc.provider.chat.assert_not_called()
            mock_svc.embedder.embed_query_batch.assert_not_called()
        finally:
            cfg.query_expansion_count = old

    def test_returns_empty_on_non_string(self, mock_svc):
        mock_svc.provider.chat.return_value = iter(["stream"])
        assert (
            get_services().searcher._expand_query(
                "kubernetes scheduling internals", self._QUESTION_VEC
            )
            == []
        )

    def test_skips_llm_for_short_query(self, mock_svc):
        # 1-token query is below the default threshold of 2, so LLM expansion
        # should not fire. Concept-graph is off in this test, so the result is [].
        mock_svc.provider.chat.return_value = _text_result("should not be reached")
        assert get_services().searcher._expand_query("k8s", self._QUESTION_VEC) == []
        mock_svc.provider.chat.assert_not_called()

    def test_skips_llm_at_threshold_boundary(self, mock_svc):
        # 2-token query == threshold; ≤ short_threshold means skip.
        mock_svc.provider.chat.return_value = _text_result("should not be reached")
        assert get_services().searcher._expand_query("k8s pods", self._QUESTION_VEC) == []
        mock_svc.provider.chat.assert_not_called()

    def test_runs_llm_above_threshold(self, mock_svc):
        # 3-token query > threshold; LLM expansion runs.
        mock_svc.provider.chat.return_value = _text_result("v one\nv two")
        variants = get_services().searcher._expand_query(
            "kubernetes scheduling internals", self._QUESTION_VEC
        )
        assert len(variants) == 2
        mock_svc.provider.chat.assert_called_once()

    def test_short_threshold_zero_disables_skip(self, mock_svc):
        old = cfg.expansion_short_query_tokens
        cfg.expansion_short_query_tokens = 0
        mock_svc.provider.chat.return_value = _text_result("v one\nv two")
        try:
            variants = get_services().searcher._expand_query("k8s", self._QUESTION_VEC)
            assert len(variants) == 2
            mock_svc.provider.chat.assert_called_once()
        finally:
            cfg.expansion_short_query_tokens = old

    def test_short_query_still_runs_concept_expansion(self, mock_svc):
        # Skip-LLM path must not short-circuit concept-graph expansion.
        old_concept = cfg.concept_graph
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.expand_query.return_value = ["kubernetes"]
        try:
            variants = get_services().searcher._expand_query("k8s", self._QUESTION_VEC)
            assert [text for text, _ in variants] == ["kubernetes"]
            mock_svc.provider.chat.assert_not_called()
        finally:
            cfg.concept_graph = old_concept

    def test_batches_llm_variants_in_one_call(self, mock_svc):
        """LLM variants should embed via a single embed_query_batch call, not N embed calls."""
        mock_svc.provider.chat.return_value = _text_result(
            "variant one\nvariant two\nvariant three"
        )
        get_services().searcher._expand_query("kubernetes scheduling internals", self._QUESTION_VEC)
        assert mock_svc.embedder.embed_query_batch.call_count >= 1
        batch_call_args = mock_svc.embedder.embed_query_batch.call_args_list[0].args[0]
        assert len(batch_call_args) == 3
        # Single-shot embed must not be used for the variant loop.
        mock_svc.embedder.embed_query.assert_not_called()

    def test_batches_concept_expansion_separately(self, mock_svc):
        """Concept-graph variants embed through embed_batch too (separate call
        because they bypass guardrails and must come through after guardrails
        apply to LLM variants)."""
        old_concept = cfg.concept_graph
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.expand_query.return_value = ["kubernetes", "scheduling"]
        mock_svc.provider.chat.return_value = _text_result("restate one\nrestate two")
        try:
            get_services().searcher._expand_query(
                "kubernetes scheduling internals", self._QUESTION_VEC
            )
        finally:
            cfg.concept_graph = old_concept
        # Two sources (LLM + concepts) => exactly 2 batch calls.
        assert mock_svc.embedder.embed_query_batch.call_count == 2
        concept_batch = mock_svc.embedder.embed_query_batch.call_args_list[1].args[0]
        assert concept_batch == ["kubernetes", "scheduling"]


class TestFormattingHelpers:
    def test_cited_subset_maps_markers_in_order(self):
        sources = [_make_result(source="a.pdf"), _make_result(source="b.pdf")]
        assert [s.source for s in cited_subset("see [2], then [2] again", sources)] == ["b.pdf"]

    def test_cited_subset_empty_when_no_markers(self):
        assert cited_subset("no markers here", [_make_result()]) == []

    def test_cited_subset_ignores_out_of_range_markers(self):
        assert cited_subset("[5] does not exist", [_make_result()]) == []


class TestStreamingCitationFilter:
    """The streamed answer must never surface a model-generated Sources block,
    since lilbee appends its own authoritative one right after."""

    @staticmethod
    def _run(chunks: list[str]) -> tuple[str, StreamingCitationFilter]:
        f = StreamingCitationFilter()
        shown = "".join(f.feed(c) for c in chunks) + f.flush()
        return shown, f

    def test_passes_answer_without_a_citation_block_through_verbatim(self):
        shown, f = self._run(["The answer ", "is 42.\n\nMore ", "detail here."])
        assert shown == "The answer is 42.\n\nMore detail here."
        assert f.answer == "The answer is 42.\n\nMore detail here."

    def test_drops_a_trailing_sources_block(self):
        shown, f = self._run(["Grounded answer [1].", "\n\nSources:\n- made-up.pdf"])
        assert "made-up.pdf" not in shown
        assert "Sources:" not in shown
        assert shown.startswith("Grounded answer [1].")
        assert "made-up.pdf" not in f.answer

    def test_never_leaks_a_heading_split_across_chunks(self):
        # "References:" arrives one letter at a time, then its list.
        chunks = ["Body text.", "\n\nRef", "erences", ":", "\n- x.pdf", "\n- y.pdf"]
        shown, _ = self._run(chunks)
        assert "References" not in shown
        assert "x.pdf" not in shown
        assert shown.startswith("Body text.")

    def test_prose_after_a_heading_line_streams_through(self):
        # A heading the answer legitimately discusses is not a citation block:
        # no list follows, so the heading and its prose both reach the reader.
        chunks = ["The paper is structured simply.", "\n\nReferences:", "\nIt lists 40 works."]
        shown, f = self._run(chunks)
        assert "References:" in shown
        assert "It lists 40 works." in shown
        assert f.answer.endswith("It lists 40 works.")

    def test_dangling_heading_at_stream_end_is_dropped(self):
        # The model emitted a citation heading and stopped; showing it would put
        # a stray "Sources:" right above lilbee's authoritative block.
        shown, f = self._run(["Grounded answer [1].", "\n\nSources:\n"])
        assert "Sources" not in shown
        assert shown == "Grounded answer [1]."
        assert f.answer == "Grounded answer [1]."

    def test_heading_is_held_not_shown_while_ambiguous(self):
        # Mid-stream, a bare heading must not be emitted until the next line
        # decides list (drop) versus prose (show).
        f = StreamingCitationFilter()
        assert f.feed("Answer.\n\nSources:\n") == "Answer."
        assert f.feed("- fake.pdf\n- other.pdf") == ""
        assert f.flush() == ""

    def test_holds_only_the_trailing_partial_line(self):
        # A completed line is released promptly (its newline waits with the next
        # line so a citation heading right after it can never leak); only the
        # final line is held until flush.
        f = StreamingCitationFilter()
        assert f.feed("first line\n") == "first line"
        assert f.feed("second line") == ""
        assert f.flush() == "\nsecond line"


class TestCitedSources:
    """bb-ky3: ask_raw exposes the answer's cited subset so JSON consumers get
    the same citation truth the string method computes."""

    def test_ask_raw_cited_sources_is_the_cited_subset(self, mock_svc):
        mock_svc.store.search.return_value = [
            _make_result(source="a.pdf", chunk="alpha", distance=0.1),
            _make_result(source="b.pdf", chunk="beta", distance=0.2),
        ]
        mock_svc.provider.chat.return_value = _text_result("As shown in [2].")
        result = get_services().searcher.ask_raw("q")
        assert len(result.sources) == 2
        assert [s.source for s in result.cited_sources] == ["b.pdf"]

    def test_ask_raw_cited_sources_empty_when_uncited(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(chunk="alpha")]
        mock_svc.provider.chat.return_value = _text_result("No markers in this answer.")
        result = get_services().searcher.ask_raw("q")
        assert result.sources
        assert result.cited_sources == []

    def test_ask_numbers_sources_to_match_inline_markers(self, mock_svc):
        # The text answer lists every retrieved source numbered, so an inline [1]
        # resolves to source 1 in the block. (Which sources were actually cited
        # is exposed separately via ask_raw().cited_sources, tested above.)
        mock_svc.store.search.return_value = [
            _make_result(source="a.pdf", chunk="alpha", distance=0.1),
            _make_result(source="b.pdf", chunk="beta", distance=0.2),
        ]
        mock_svc.provider.chat.return_value = _text_result("From [1] we learn the answer.")
        answer = get_services().searcher.ask("q")
        assert "1. [a.pdf](file://" in answer
        assert "2. [b.pdf](file://" in answer


class TestContextBudget:
    """bb-6kt: RAG sources are trimmed to fit num_ctx instead of overflow-erroring."""

    @pytest.fixture(autouse=True)
    def _restore_ctx(self):
        old = (cfg.num_ctx, cfg.chat_n_ctx_target)
        yield
        cfg.num_ctx, cfg.chat_n_ctx_target = old

    def test_trims_lowest_ranked_sources_to_fit(self, mock_svc):
        cfg.num_ctx = 1400
        results = [_make_result(source=f"{i}.pdf", chunk="x" * 300) for i in range(5)]
        kept = get_services().searcher._fit_context_budget(results, "sys", "q", None)
        assert 0 < len(kept) < len(results)
        assert kept == results[: len(kept)]  # keeps the top-ranked prefix

    def test_fitted_context_survives_the_provider_that_enforces_the_window(self, mock_svc):
        """bb-e0np: retrieval must fit inside what the engine will actually accept.

        The fleet rejects a prompt above ``prompt_token_budget``; retrieval used
        to fit to a ceiling one margin higher, so a grounded turn assembled a
        prompt the engine refused with a 400 nothing downstream could fix.
        """
        from lilbee.providers.base import prompt_token_budget

        ctx = 2560
        cfg.num_ctx = ctx
        system, question = "sys " * 40, "q " * 20
        results = [_make_result(source=f"{i}.pdf", chunk="x" * 900) for i in range(5)]

        kept = get_services().searcher._fit_context_budget(results, system, question, None)

        searcher = get_services().searcher
        assembled = (
            searcher._budget_tokens(system)
            + searcher._budget_tokens(question)
            + sum(searcher._budget_tokens(r.chunk) for r in kept)
        )
        assert kept, "the top-ranked source is always kept"
        assert assembled <= prompt_token_budget(ctx), (
            f"assembled {assembled} tokens exceeds the provider ceiling "
            f"{prompt_token_budget(ctx)}; the engine would reject this turn"
        )

    def test_keeps_all_when_budget_ample(self, mock_svc):
        cfg.num_ctx = 100_000
        results = [_make_result(source=f"{i}.pdf", chunk="short") for i in range(5)]
        assert get_services().searcher._fit_context_budget(results, "sys", "q", None) == results

    def test_keeps_top_source_even_if_alone_over_budget(self, mock_svc):
        cfg.num_ctx = 1
        results = [_make_result(source="big.pdf", chunk="x" * 9000), _make_result(source="b.pdf")]
        kept = get_services().searcher._fit_context_budget(results, "sys", "q", None)
        assert len(kept) == 1

    def test_history_counts_against_the_budget(self, mock_svc):
        # Sized so both cases have room to differ: the budget now excludes the
        # provider's margin as well as its generation reserve, so a 1400-token
        # window leaves too little for anything but the always-kept top source.
        cfg.num_ctx = 3000
        results = [_make_result(source=f"{i}.pdf", chunk="x" * 1200) for i in range(5)]
        history = [{"role": "user", "content": "h" * 3000}]
        no_hist = get_services().searcher._fit_context_budget(results, "sys", "q", None)
        with_hist = get_services().searcher._fit_context_budget(results, "sys", "q", history)
        assert len(with_hist) < len(no_hist)

    def test_logs_when_trimming(self, mock_svc, caplog):
        cfg.num_ctx = 1400
        results = [_make_result(source=f"{i}.pdf", chunk="x" * 300) for i in range(5)]
        with caplog.at_level("INFO"):
            get_services().searcher._fit_context_budget(results, "sys", "q", None)
        assert "to fit the model context window" in caplog.text


class TestAskRaw:
    def test_returns_structured_result(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(chunk="oil is 5 quarts")]
        mock_svc.provider.chat.return_value = _text_result("5 quarts.")
        result = get_services().searcher.ask_raw("oil capacity?")
        assert result.answer == "5 quarts."
        assert len(result.sources) == 1
        assert result.sources[0].source == "test.pdf"

    def test_no_results_returns_grounded_refusal(self, mock_svc):
        """Zero RAG hits in RAG mode return a grounded refusal instead of
        free-wheeling on the model's parametric knowledge (bb-0i0). The model is
        never called, and sources stay empty."""
        mock_svc.store.search.return_value = []
        result = get_services().searcher.ask_raw("anything")
        assert result.answer == GROUNDED_REFUSAL
        assert result.sources == []
        assert result.cited_sources == []
        mock_svc.provider.chat.assert_not_called()

    def test_empty_library_returns_add_content_guidance(self, mock_svc):
        """With nothing indexed, ask_raw points the user at adding content
        instead of the grounded refusal (which implies a search happened). The
        store is never searched and the model is never called."""
        mock_svc.store.has_chunks.return_value = False
        result = get_services().searcher.ask_raw("say hello")
        assert result.answer == EMPTY_LIBRARY
        assert result.sources == []
        mock_svc.store.search.assert_not_called()
        mock_svc.provider.chat.assert_not_called()

    def test_ask_raw_with_history(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("answer")
        history = [{"role": "user", "content": "prev"}]
        get_services().searcher.ask_raw("new q", history=history)
        messages = mock_svc.provider.chat.call_args[0][0]
        assert len(messages) == 3  # system + history + user

    def test_ask_raw_strips_think_tags(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result(
            "<think>reasoning</think>The answer is 42."
        )
        result = get_services().searcher.ask_raw("question")
        assert "<think>" not in result.answer
        assert result.answer == "The answer is 42."

    def test_ask_raw_preserves_think_tags_when_show_reasoning(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result(
            "<think>reasoning</think>The answer is 42."
        )
        old = cfg.show_reasoning
        cfg.show_reasoning = True
        try:
            result = get_services().searcher.ask_raw("question")
            assert "<think>reasoning</think>" in result.answer
        finally:
            cfg.show_reasoning = old


class TestAsk:
    def test_returns_answer_with_citations(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(chunk="oil is 5 quarts")]
        mock_svc.provider.chat.return_value = _text_result("The oil capacity is 5 quarts.")
        answer = get_services().searcher.ask("oil capacity?")
        assert "5 quarts" in answer
        assert "Sources:" in answer
        assert "test.pdf" in answer

    def test_no_results_returns_grounded_refusal(self, mock_svc):
        """ask() surfaces the grounded refusal verbatim, with no Sources block (bb-0i0)."""
        mock_svc.store.search.return_value = []
        answer = get_services().searcher.ask("anything")
        assert answer == GROUNDED_REFUSAL
        assert "Sources:" not in answer
        mock_svc.provider.chat.assert_not_called()

    def test_empty_library_returns_add_content_guidance(self, mock_svc):
        """ask() surfaces the empty-library guidance verbatim, no Sources block."""
        mock_svc.store.has_chunks.return_value = False
        answer = get_services().searcher.ask("say hello")
        assert answer == EMPTY_LIBRARY
        assert "Sources:" not in answer
        mock_svc.provider.chat.assert_not_called()

    def test_ask_with_history(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("answer")
        history = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        get_services().searcher.ask("new q", history=history)
        messages = mock_svc.provider.chat.call_args[0][0]
        assert len(messages) == 4
        assert messages[1]["content"] == "prev q"


class TestAskStream:
    def test_yields_tokens_then_citations(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = iter(["Hello", " world"])
        stream_tokens = list(get_services().searcher.ask_stream("test"))
        combined = "".join(st.content for st in stream_tokens)
        assert "Hello world" in combined
        assert "Sources:" in combined

    def test_model_sources_block_is_suppressed_leaving_one_authoritative_list(self, mock_svc):
        """A model that appends its own Sources block must not double up with
        lilbee's: the streamed output carries exactly one Sources block and none
        of the model's invented source lines."""
        mock_svc.store.search.return_value = [_make_result(source="real.pdf", chunk="alpha")]
        mock_svc.provider.chat.return_value = iter(
            ["Grounded answer [1].", "\n\nSources:\n- invented.pdf"]
        )
        combined = "".join(st.content for st in get_services().searcher.ask_stream("test"))
        assert combined.count("Sources:") == 1
        assert "invented.pdf" not in combined
        assert "real.pdf" in combined
        assert "Grounded answer [1]." in combined

    def test_empty_results_streams_grounded_refusal(self, mock_svc):
        """Zero RAG hits stream a single grounded-refusal token, no Sources block,
        and never call the model (bb-0i0)."""
        mock_svc.store.search.return_value = []
        stream_tokens = list(get_services().searcher.ask_stream("anything"))
        combined = "".join(st.content for st in stream_tokens)
        assert combined == GROUNDED_REFUSAL
        assert "Sources:" not in combined
        mock_svc.provider.chat.assert_not_called()

    def test_empty_library_streams_add_content_guidance(self, mock_svc):
        """With nothing indexed, the stream yields the empty-library guidance as a
        single token, no Sources block, and never calls the model."""
        mock_svc.store.has_chunks.return_value = False
        stream_tokens = list(get_services().searcher.ask_stream("say hello"))
        combined = "".join(st.content for st in stream_tokens)
        assert combined == EMPTY_LIBRARY
        assert "Sources:" not in combined
        mock_svc.provider.chat.assert_not_called()

    def test_ask_stream_with_history(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = iter(["response"])
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        list(get_services().searcher.ask_stream("new question", history=history))
        messages = mock_svc.provider.chat.call_args[0][0]
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "previous question"

    def test_skips_empty_tokens(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = iter(["", "data"])
        stream_tokens = list(get_services().searcher.ask_stream("test"))
        non_source = [st for st in stream_tokens if "Sources:" not in st.content]
        assert all(st.content != "" for st in non_source)

    def test_reasoning_stripped_by_default(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = iter(["<think>reasoning</think>answer"])
        stream_tokens = list(get_services().searcher.ask_stream("test"))
        # Only inspect the model-answer tokens; the trailing "Sources:" block
        # includes absolute paths that may coincidentally contain the test name.
        body_tokens = [
            st.content
            for st in stream_tokens
            if not st.is_reasoning and "Sources:" not in st.content and "→" not in st.content
        ]
        combined = "".join(body_tokens)
        assert "reasoning" not in combined
        assert "answer" in combined


class TestGenerationOptions:
    def test_ask_raw_passes_options(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("answer")
        opts = {"temperature": 0.3, "seed": 42}
        get_services().searcher.ask_raw("q", options=opts)
        assert mock_svc.provider.chat.call_args[1]["options"] == opts

    def test_ask_raw_defaults_to_cfg_options(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("answer")
        cfg.temperature = 0.7
        cfg.seed = None
        cfg.top_p = None
        cfg.top_k_sampling = None
        cfg.repeat_penalty = None
        cfg.num_ctx = None
        cfg.max_tokens = None
        try:
            get_services().searcher.ask_raw("q")
            assert mock_svc.provider.chat.call_args[1]["options"] == {"temperature": 0.7}
        finally:
            cfg.temperature = None
            cfg.max_tokens = 4096

    def test_ask_stream_passes_options(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = iter(["token"])
        opts = {"temperature": 0.1}
        list(get_services().searcher.ask_stream("q", options=opts))
        assert mock_svc.provider.chat.call_args[1]["options"] == opts

    def test_ask_passes_options_through(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("answer")
        opts = {"num_ctx": 4096}
        get_services().searcher.ask("q", options=opts)
        assert mock_svc.provider.chat.call_args[1]["options"] == opts

    def test_ask_raw_empty_options_passes_none(self, mock_svc):
        """When cfg has no generation options set, passes None to provider."""
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("answer")
        cfg.temperature = None
        cfg.top_p = None
        cfg.top_k_sampling = None
        cfg.repeat_penalty = None
        cfg.num_ctx = None
        cfg.seed = None
        cfg.max_tokens = None
        get_services().searcher.ask_raw("q")
        assert mock_svc.provider.chat.call_args[1]["options"] is None


class TestAskStreamError:
    def test_stream_handles_disconnect(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]

        def failing_stream():
            yield "partial"
            raise ConnectionError("lost connection")

        mock_svc.provider.chat.return_value = failing_stream()
        stream_tokens = list(get_services().searcher.ask_stream("test"))
        combined = "".join(st.content for st in stream_tokens)
        assert "partial" in combined
        assert "Connection lost" in combined


@pytest.fixture
def _disable_pre_chat_llm_calls():
    """Pin ``query_expansion_count`` and ``hyde`` so RAG setup doesn't burn provider.chat calls."""
    snapshot_expand = cfg.query_expansion_count
    snapshot_hyde = cfg.hyde
    cfg.query_expansion_count = 0
    cfg.hyde = False
    yield
    cfg.query_expansion_count = snapshot_expand
    cfg.hyde = snapshot_hyde


class TestAskStreamCapNotice:
    """Cap-fire surfaces a reasoning notice + the continuation answer in both streaming paths."""

    def test_ask_stream_emits_cap_notice_then_answer(self, mock_svc, _disable_pre_chat_llm_calls):
        from lilbee.retrieval.reasoning import CAP_NOTICE_TEMPLATE

        mock_svc.store.search.return_value = [_make_result()]
        long_reasoning = "<think>" + ("x " * 400) + "</think>"
        mock_svc.provider.chat.side_effect = [
            iter([long_reasoning]),
            iter(["final ", "answer"]),
        ]
        snapshot_cap = cfg.max_reasoning_chars
        try:
            cfg.max_reasoning_chars = 512
            stream_tokens = list(get_services().searcher.ask_stream("q"))
        finally:
            cfg.max_reasoning_chars = snapshot_cap
        body = "".join(st.content for st in stream_tokens)
        assert CAP_NOTICE_TEMPLATE.format(chars=512) in body
        assert "final answer" in body
        assert mock_svc.provider.chat.call_count == 2

    def test_direct_stream_emits_cap_notice_then_answer(
        self, mock_svc, _disable_pre_chat_llm_calls
    ):
        """When the searcher takes the no-RAG branch (chat mode), cap-fire still surfaces."""
        from lilbee.core.config.enums import ChatMode
        from lilbee.retrieval.reasoning import CAP_NOTICE_TEMPLATE

        mock_svc.store.search.return_value = []
        long_reasoning = "<think>" + ("x " * 400) + "</think>"
        mock_svc.provider.chat.side_effect = [
            iter([long_reasoning]),
            iter(["direct ", "answer"]),
        ]
        snapshot_cap = cfg.max_reasoning_chars
        snapshot_mode = cfg.chat_mode
        try:
            cfg.max_reasoning_chars = 512
            cfg.chat_mode = ChatMode.CHAT.value
            stream_tokens = list(get_services().searcher.ask_stream("q"))
        finally:
            cfg.max_reasoning_chars = snapshot_cap
            cfg.chat_mode = snapshot_mode
        body = "".join(st.content for st in stream_tokens)
        assert CAP_NOTICE_TEMPLATE.format(chars=512) in body
        assert "direct answer" in body
        assert mock_svc.provider.chat.call_count == 2


class TestProviderError:
    """ProviderError from the provider should propagate."""

    def test_ask_raw_provider_error(self, mock_svc):
        from lilbee.providers.base import ProviderError

        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.side_effect = ProviderError("model 'bad' not found")
        with pytest.raises(ProviderError, match="not found"):
            get_services().searcher.ask_raw("hello")

    def test_ask_stream_provider_error(self, mock_svc):
        from lilbee.providers.base import ProviderError

        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.side_effect = ProviderError("model 'bad' not found")
        with pytest.raises(ProviderError, match="not found"):
            list(get_services().searcher.ask_stream("hello"))

    def test_ask_stream_provider_error_mid_stream(self, mock_svc):
        """ProviderError raised during iteration should propagate."""
        from lilbee.providers.base import ProviderError

        mock_svc.store.search.return_value = [_make_result()]

        def failing_mid_stream():
            yield "partial"
            raise ProviderError("model 'bad' not found")

        mock_svc.provider.chat.return_value = failing_mid_stream()
        with pytest.raises(ProviderError, match="not found"):
            list(get_services().searcher.ask_stream("hello"))


class TestApplyGuardrails:
    def test_rejects_orthogonal_variant(self, mock_svc):
        question_vec = [1.0, 0.0, 0.0]
        variants = [("related rephrase", [0.9, 0.1, 0.0]), ("drifted", [0.0, 1.0, 0.0])]
        result = get_services().searcher._apply_guardrails(variants, question_vec)
        texts = [text for text, _ in result]
        assert "drifted" not in texts
        assert "related rephrase" in texts

    def test_keeps_near_duplicate(self, mock_svc):
        question_vec = [1.0, 0.0, 0.0]
        variants = [("same topic", [0.95, 0.05, 0.0])]
        result = get_services().searcher._apply_guardrails(variants, question_vec)
        assert len(result) == 1

    def test_returns_all_when_disabled(self, mock_svc):
        cfg.expansion_guardrails = False
        try:
            variants = [("drifted", [0.0, 1.0, 0.0])]
            result = get_services().searcher._apply_guardrails(variants, [1.0, 0.0, 0.0])
            assert result == variants
        finally:
            cfg.expansion_guardrails = True

    def test_respects_configurable_threshold(self, mock_svc):
        # Cosine = 0.6; passes at default 0.5 but not at 0.8.
        question_vec = [1.0, 0.0]
        variants = [("borderline", [0.6, 0.8])]
        searcher = get_services().searcher
        cfg.expansion_similarity_threshold = 0.5
        assert len(searcher._apply_guardrails(variants, question_vec)) == 1
        cfg.expansion_similarity_threshold = 0.8
        assert searcher._apply_guardrails(variants, question_vec) == []

    def test_empty_variants(self, mock_svc):
        assert get_services().searcher._apply_guardrails([], [1.0, 0.0, 0.0]) == []


class TestSelectContext:
    def test_selects_covering_chunks(self, mock_svc):
        chunks = [
            _make_result(chunk="kubernetes deployment guide", source="a.md"),
            _make_result(chunk="kubernetes networking setup", source="b.md"),
            _make_result(chunk="deployment automation tools", source="c.md"),
        ]
        result = get_services().searcher.select_context(
            chunks, "kubernetes deployment networking", max_sources=2
        )
        assert len(result) == 2
        texts = " ".join(r.chunk for r in result)
        assert "networking" in texts  # distinctive term must be covered

    def test_passes_through_when_under_max(self, mock_svc):
        chunks = [_make_result(chunk="only one")]
        result = get_services().searcher.select_context(chunks, "anything", max_sources=5)
        assert len(result) == 1

    def test_empty_query_returns_top_n(self, mock_svc):
        chunks = [_make_result(chunk=f"chunk {i}") for i in range(10)]
        result = get_services().searcher.select_context(chunks, "", max_sources=3)
        assert len(result) == 3

    def test_all_zero_weight_falls_back_to_top_n(self, mock_svc):
        # Every chunk contains both question terms → IDF is zero for both →
        # no term adds any weight → fall back to top-N by retrieval order.
        chunks = [
            _make_result(chunk="alpha beta gamma delta", source="a.md"),
            _make_result(chunk="alpha beta gamma delta", source="b.md"),
            _make_result(chunk="alpha beta gamma delta", source="c.md"),
            _make_result(chunk="alpha beta gamma delta", source="d.md"),
            _make_result(chunk="alpha beta gamma delta", source="e.md"),
        ]
        result = get_services().searcher.select_context(
            chunks, "alpha beta gamma delta", max_sources=3
        )
        assert len(result) == 3

    def test_fills_budget_with_retrieval_order(self, mock_svc):
        # The distinctive term pulls b.md first; the remaining slot is
        # filled from the top of the retrieval order (a.md), not another
        # duplicate. Final ordering is retrieval-stable.
        chunks = [
            _make_result(chunk="kafka rebalance broker", source="a.md"),
            _make_result(chunk="kafka streams consumer group rebalance", source="b.md"),
            _make_result(chunk="kafka broker replication", source="c.md"),
        ]
        result = get_services().searcher.select_context(
            chunks, "kafka consumer group", max_sources=2
        )
        assert len(result) == 2
        sources = [r.source for r in result]
        assert "b.md" in sources  # cover pick
        assert sources == sorted(sources)

    def test_rerank_scores_make_reranked_order_authoritative(self, mock_svc):
        # The top reranked chunk shares no query terms; set cover would drop it.
        chunks = [
            _make_result(chunk="grounding upgrade big three", source="fix.md", rerank_score=0.9),
            _make_result(chunk="radio resets at idle", source="radio.md", rerank_score=0.5),
            _make_result(chunk="headlights dim at idle", source="lights.md", rerank_score=0.4),
            _make_result(chunk="battery health log", source="battery.md", rerank_score=0.2),
        ]
        result = get_services().searcher.select_context(
            chunks, "radio resets headlights dim", max_sources=2
        )
        assert [r.source for r in result] == ["fix.md", "radio.md"]

    def test_partially_scored_results_keep_reranked_order(self, mock_svc):
        # Remainder chunks beyond the rerank candidate cap carry no score.
        chunks = [
            _make_result(chunk="alpha", source="a.md", rerank_score=0.9),
            _make_result(chunk="beta", source="b.md", rerank_score=0.8),
            _make_result(chunk="gamma", source="c.md"),
        ]
        result = get_services().searcher.select_context(chunks, "alpha beta gamma", max_sources=2)
        assert [r.source for r in result] == ["a.md", "b.md"]

    def test_hyphenated_phrase_splits_into_tokens(self, mock_svc):
        # Regression: the old tokenizer would strip the hyphens and
        # collapse "state-of-the-art" into one meaningless 14-char
        # token that matched nothing. The new regex-split tokenizer
        # treats every run of non-alnum characters as a boundary.
        chunks = [
            _make_result(chunk="state of the art benchmarks", source="a.md"),
            _make_result(chunk="unrelated monitoring setup", source="b.md"),
        ]
        result = get_services().searcher.select_context(
            chunks, "state-of-the-art benchmarks", max_sources=1
        )
        assert len(result) == 1
        assert result[0].source == "a.md"


class TestShouldSkipExpansion:
    """bm25_probe rows carry a raw, unbounded BM25 score in ``bm25_score``
    (LanceDB FTS _score); the probe values here use those realistic
    magnitudes. Confidence is the saturating s/(s+5) (0.8 = raw 20), and the
    gap condition is the RELATIVE raw gap (top - second) / top >= 0.15: a
    sigmoid squash compressed any two strong scores to within 0.01, so the
    old gap test could never fire when the lexical arm was most certain."""

    def test_skips_when_confident(self, mock_svc):
        # confidence(30) = 30/35 = 0.857 >= 0.8; relative gap (30-10)/30 = 0.67.
        mock_svc.store.bm25_probe.return_value = [
            _make_result(bm25_score=30.0),
            _make_result(bm25_score=10.0),
        ]
        assert get_services().searcher._should_skip_expansion("test query") is True

    def test_skips_on_strong_scores_with_real_gap(self, mock_svc):
        """The saturation regression case: two strong raw scores with a real
        relative gap must skip; the sigmoid made this branch unreachable."""
        mock_svc.store.bm25_probe.return_value = [
            _make_result(bm25_score=40.0),
            _make_result(bm25_score=25.0),
        ]
        assert get_services().searcher._should_skip_expansion("test query") is True

    def test_does_not_skip_when_low_score(self, mock_svc):
        # confidence(2) = 2/7 = 0.29 < 0.8: a weak top BM25 hit still expands.
        mock_svc.store.bm25_probe.return_value = [
            _make_result(bm25_score=2.0),
            _make_result(bm25_score=1.0),
        ]
        assert get_services().searcher._should_skip_expansion("test query") is False

    def test_does_not_skip_when_close_gap(self, mock_svc):
        # Strong but undifferentiated: relative gap (30-28)/30 = 0.07 < 0.15.
        mock_svc.store.bm25_probe.return_value = [
            _make_result(bm25_score=30.0),
            _make_result(bm25_score=28.0),
        ]
        assert get_services().searcher._should_skip_expansion("test query") is False

    def test_skips_with_single_confident_result(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [_make_result(bm25_score=30.0)]
        assert get_services().searcher._should_skip_expansion("test") is True

    def test_does_not_skip_when_empty(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = []
        assert get_services().searcher._should_skip_expansion("test") is False

    def test_disabled_when_threshold_zero(self, mock_svc):
        old = cfg.expansion_skip_threshold
        cfg.expansion_skip_threshold = 0
        try:
            assert get_services().searcher._should_skip_expansion("test") is False
        finally:
            cfg.expansion_skip_threshold = old

    def test_forwards_scope_to_probe(self, mock_svc):
        """A scoped search must probe the same pool it searches, not the mixed pool."""
        mock_svc.store.bm25_probe.return_value = []
        get_services().searcher._should_skip_expansion("q", ChunkType.WIKI)
        assert mock_svc.store.bm25_probe.call_args.kwargs["chunk_type"] == ChunkType.WIKI

    def test_does_not_skip_when_score_missing(self, mock_svc):
        """A probe row with no FTS score reads as zero confidence, never skipping."""
        mock_svc.store.bm25_probe.return_value = [_make_result(bm25_score=None)]
        assert get_services().searcher._should_skip_expansion("test") is False


class TestSearchAppliesConceptBoostOnExpansionSkip:
    def test_concept_boost_runs_even_when_expansion_skipped(self, mock_svc, monkeypatch):
        """Skipping query expansion must not also skip the concept-graph re-rank."""
        searcher = get_services().searcher
        mock_svc.store.search.return_value = [_make_result()]
        monkeypatch.setattr(searcher, "_should_skip_expansion", lambda *a, **k: True)
        called: list[bool] = []

        def _boost(results, question):
            called.append(True)
            return results

        monkeypatch.setattr(searcher, "_apply_concept_boost", _boost)
        searcher.search("test query")
        assert called == [True]
        # Expansion was skipped: no variant searches ran.
        mock_svc.store.search.assert_called_once()


class TestBm25Confidence:
    def test_squashes_and_floors(self):
        from lilbee.retrieval.query.searcher import _bm25_confidence

        # Absent or non-positive scores floor to zero confidence.
        assert _bm25_confidence(None) == 0.0
        assert _bm25_confidence(0.0) == 0.0
        assert _bm25_confidence(-2.0) == 0.0
        # Positive raw BM25 scores squash monotonically into (0, 1) without
        # saturating: strong scores stay distinguishable from each other.
        assert 0.0 < _bm25_confidence(1.0) < 1.0
        assert _bm25_confidence(5.0) > _bm25_confidence(1.0)
        assert _bm25_confidence(40.0) - _bm25_confidence(20.0) > 0.05


class TestParseStructuredQuery:
    def test_term_prefix(self, mock_svc):
        mode, query = get_services().searcher._parse_structured_query("term: kubernetes pods")
        assert mode == "term"
        assert query == "kubernetes pods"

    def test_vec_prefix(self, mock_svc):
        mode, query = get_services().searcher._parse_structured_query("vec: how does auth work")
        assert mode == "vec"
        assert "auth" in query

    def test_hyde_prefix(self, mock_svc):
        mode, _query = get_services().searcher._parse_structured_query("hyde: explain caching")
        assert mode == "hyde"

    def test_no_prefix(self, mock_svc):
        mode, query = get_services().searcher._parse_structured_query("normal question")
        assert mode is None
        assert query == "normal question"

    def test_case_insensitive(self, mock_svc):
        mode, _ = get_services().searcher._parse_structured_query("TERM: test")
        assert mode == "term"


class TestHydeSearch:
    def test_returns_results(self, mock_svc):
        mock_svc.provider.chat.return_value = _text_result("hypothetical document about X")
        mock_svc.store.search.return_value = [_make_result()]
        results = get_services().searcher._hyde_search("explain X", top_k=5)
        assert len(results) >= 1

    def test_passage_is_embedded_as_a_query(self, mock_svc):
        """Deliberate choice: the hypothetical passage stands in for the query.

        It is embedded with embed_query (query instruction), not the document
        prefix, so HyDE vectors share the space of every other query against
        the doc-prefixed index. Flipping this is an embedding-bench experiment.
        """
        mock_svc.provider.chat.return_value = _text_result("hypothetical passage")
        mock_svc.store.search.return_value = []
        get_services().searcher._hyde_search("explain X", top_k=5)
        mock_svc.embedder.embed_query.assert_called_once_with("hypothetical passage")

    def test_returns_empty_on_error(self, mock_svc):
        mock_svc.provider.chat.side_effect = RuntimeError("fail")
        assert get_services().searcher._hyde_search("test", top_k=5) == []

    def test_returns_empty_on_non_string(self, mock_svc):
        mock_svc.provider.chat.return_value = iter(["stream"])
        assert get_services().searcher._hyde_search("test", top_k=5) == []

    def test_returns_empty_on_blank(self, mock_svc):
        mock_svc.provider.chat.return_value = _text_result("   ")
        assert get_services().searcher._hyde_search("test", top_k=5) == []

    def test_strips_reasoning_before_embedding(self, mock_svc):
        """A reasoning model's deliberation must not be embedded as the
        hypothetical passage."""
        mock_svc.provider.chat.return_value = _text_result(
            "<think>what would a real document say here</think>the actual passage"
        )
        mock_svc.store.search.return_value = []
        get_services().searcher._hyde_search("explain X", top_k=5)
        mock_svc.embedder.embed_query.assert_called_once_with("the actual passage")

    def test_returns_empty_when_output_is_all_reasoning(self, mock_svc):
        mock_svc.provider.chat.return_value = _text_result("<think>only deliberation</think>")
        assert get_services().searcher._hyde_search("test", top_k=5) == []


class TestTemporalFilter:
    def test_filters_by_date(self, mock_svc):
        from datetime import UTC, datetime, timedelta

        recent = (datetime.now(UTC) - timedelta(days=5)).isoformat()
        mock_svc.store.source_ingested_at_map.return_value = {
            "old.md": "2020-01-01T00:00:00+00:00",
            "new.md": recent,
        }
        results = [
            _make_result(source="old.md"),
            _make_result(source="new.md"),
        ]
        filtered = get_services().searcher._apply_temporal_filter(results, "recent changes")
        assert any(r.source == "new.md" for r in filtered)
        assert not any(r.source == "old.md" for r in filtered)

    def test_result_without_ingested_at_passes_through(self, mock_svc):
        """Sources absent from the ingested_at map are kept (no date info = don't filter)."""
        mock_svc.store.source_ingested_at_map.return_value = {}
        results = [_make_result(source="mystery.md")]
        filtered = get_services().searcher._apply_temporal_filter(results, "recent changes")
        assert [r.source for r in filtered] == ["mystery.md"]

    def test_no_temporal_keyword_passes_through(self, mock_svc):
        results = [_make_result()]
        searcher = get_services().searcher
        assert searcher._apply_temporal_filter(results, "how does auth work") == results

    def test_bare_search_applies_temporal_filter(self, mock_svc, monkeypatch):
        """The bare search() path runs the temporal filter, matching chat/ask."""
        searcher = get_services().searcher
        mock_svc.store.search.return_value = [_make_result()]
        monkeypatch.setattr(searcher, "_should_skip_expansion", lambda *a, **k: True)
        seen: list[str] = []

        def _temporal(results, question):
            seen.append(question)
            return results

        monkeypatch.setattr(searcher, "_apply_temporal_filter", _temporal)
        searcher.search("recent changes")
        assert seen == ["recent changes"]

    def test_structured_query_still_applies_temporal_filter(self, mock_svc, monkeypatch):
        """A ``mode:`` prefix skips expansion/boost but the date filter still runs."""
        cfg.wiki = True
        try:
            searcher = get_services().searcher
            mock_svc.store.search.return_value = [_make_result()]
            seen: list[str] = []

            def _temporal(results, question):
                seen.append(question)
                return results

            monkeypatch.setattr(searcher, "_apply_temporal_filter", _temporal)
            searcher.search("wiki: recent changes")
            assert seen == ["recent changes"]  # prefix stripped, filter applied
        finally:
            cfg.wiki = False

    def test_disabled_via_config(self, mock_svc):
        old = cfg.temporal_filtering
        cfg.temporal_filtering = False
        try:
            results = [_make_result()]
            assert get_services().searcher._apply_temporal_filter(results, "recent") == results
        finally:
            cfg.temporal_filtering = old

    def test_keeps_results_without_dates(self, mock_svc):
        mock_svc.store.get_sources.return_value = [{"filename": "a.md", "ingested_at": ""}]
        results = [_make_result(source="a.md")]
        filtered = get_services().searcher._apply_temporal_filter(results, "today's notes")
        assert len(filtered) == 1

    def test_falls_back_when_nothing_matches(self, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {"filename": "old.md", "ingested_at": "2020-01-01T00:00:00+00:00"},
        ]
        results = [_make_result(source="old.md")]
        filtered = get_services().searcher._apply_temporal_filter(results, "today's notes")
        assert len(filtered) == 1

    def test_handles_invalid_date(self, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {"filename": "a.md", "ingested_at": "not-a-date"}
        ]
        results = [_make_result(source="a.md")]
        filtered = get_services().searcher._apply_temporal_filter(results, "recent")
        assert len(filtered) == 1


def _memory_fact(text: str) -> MemoryRow:
    from lilbee.data.store import LOCAL_OWNER, MemoryKind, MemorySource

    return MemoryRow(
        id="m1",
        owner=LOCAL_OWNER,
        shared=False,
        kind=MemoryKind.FACT,
        source=MemorySource.MANUAL,
        text=text,
        vector=[0.0],
        created_at="t",
        updated_at="t",
    )


class TestMemoryGroundedFallback:
    """Memory is the user's own ground truth: when retrieval cannot serve
    (empty library, or nothing relevant) but fact recall hits, the turn
    answers from memory via the direct prompt instead of a canned message.
    Reported live: 'what's my name?' with the name in memory returned the
    empty-library guidance."""

    def _enable_memory(self, mock_svc, facts):
        cfg.memory_enabled = True
        mock_svc.store.get_memories.return_value = []
        mock_svc.store.search_memories.return_value = facts
        mock_svc.provider.chat.return_value = _text_result("Your name is Tobias.")

    def test_empty_library_with_fact_answers_from_memory(self, mock_svc):
        self._enable_memory(mock_svc, [_memory_fact("The user's name is Tobias.")])
        try:
            mock_svc.store.has_chunks.return_value = False
            mock_svc.store.search.return_value = []
            mock_svc.store.bm25_probe.return_value = []
            result = get_services().searcher.ask_raw("what's my name?")
            assert result.answer == "Your name is Tobias."
            assert result.sources == []
        finally:
            cfg.memory_enabled = False

    def test_empty_library_without_facts_keeps_guidance(self, mock_svc):
        self._enable_memory(mock_svc, [])
        try:
            mock_svc.store.has_chunks.return_value = False
            result = get_services().searcher.ask_raw("what's my name?")
            assert result.answer == EMPTY_LIBRARY
        finally:
            cfg.memory_enabled = False

    def test_preferences_alone_do_not_unlock_memory_answers(self, mock_svc):
        """Always-injected preferences say nothing about answerability; only
        distance-gated fact recall may bypass the empty-library guidance."""
        self._enable_memory(mock_svc, [])
        try:
            mock_svc.store.get_memories.return_value = [_memory_fact("terse answers")]
            mock_svc.store.has_chunks.return_value = False
            result = get_services().searcher.ask_raw("what's my name?")
            assert result.answer == EMPTY_LIBRARY
        finally:
            cfg.memory_enabled = False

    def test_empty_retrieval_with_fact_answers_from_memory(self, mock_svc):
        self._enable_memory(mock_svc, [_memory_fact("The user's name is Tobias.")])
        try:
            mock_svc.store.has_chunks.return_value = True
            mock_svc.store.search.return_value = []
            mock_svc.store.bm25_probe.return_value = []
            result = get_services().searcher.ask_raw("what's my name?")
            assert result.answer == "Your name is Tobias."
        finally:
            cfg.memory_enabled = False

    def test_empty_retrieval_without_facts_refuses(self, mock_svc):
        self._enable_memory(mock_svc, [])
        try:
            mock_svc.store.has_chunks.return_value = True
            mock_svc.store.search.return_value = []
            mock_svc.store.bm25_probe.return_value = []
            result = get_services().searcher.ask_raw("what's my name?")
            assert result.answer == GROUNDED_REFUSAL
        finally:
            cfg.memory_enabled = False


class TestLlmIntentRouting:
    """The config-gated LLM classifier: consulted only when the deterministic
    patterns find nothing answerable, never able to lose a decline, and any
    failure degrades to ordinary retrieval."""

    _CLASSIFY_JSON = '{"kind": "term_mentions", "term": "blood"}'

    def test_off_by_default_no_llm_call(self, mock_svc):
        cfg.intent_llm = False
        answer = get_services().searcher.route_direct_answer("count the books that mention blood")
        assert answer is None
        mock_svc.provider.chat.assert_not_called()

    def test_llm_routes_unmatched_phrasing_to_exact_scan(self, mock_svc):
        cfg.intent_llm = True
        try:
            mock_svc.provider.chat.return_value = _text_result(self._CLASSIFY_JSON)
            mock_svc.store.count_term_mentions.return_value = (12, 5)
            answer = get_services().searcher.route_direct_answer("how many tomes mention blood?")
            assert answer is not None
            assert "5 documents" in answer
            mock_svc.provider.chat.assert_called_once()
        finally:
            cfg.intent_llm = False

    def test_deterministic_hit_skips_llm(self, mock_svc):
        cfg.intent_llm = True
        try:
            mock_svc.store.count_term_mentions.return_value = (12, 5)
            answer = get_services().searcher.route_direct_answer(
                "how many documents mention blood?"
            )
            assert answer is not None
            mock_svc.provider.chat.assert_not_called()
        finally:
            cfg.intent_llm = False

    def test_llm_failure_degrades_to_retrieval(self, mock_svc):
        """A failed classification on a pattern-missed question routes nowhere:
        the question flows to ordinary retrieval, never an error."""
        cfg.intent_llm = True
        try:
            mock_svc.provider.chat.side_effect = RuntimeError("server busy")
            answer = get_services().searcher.route_direct_answer(
                "count the books that mention blood"
            )
            assert answer is None
        finally:
            cfg.intent_llm = False

    def test_llm_failure_keeps_deterministic_decline(self, mock_svc):
        """A count-shaped question ('how many ...') keeps its precise decline
        even when the classifier call fails."""
        cfg.intent_llm = True
        try:
            mock_svc.provider.chat.side_effect = RuntimeError("server busy")
            answer = get_services().searcher.route_direct_answer("how many tomes mention blood?")
            assert answer is not None
            assert "count" in answer.lower()
        finally:
            cfg.intent_llm = False

    def test_llm_topical_keeps_deterministic_decline(self, mock_svc):
        """UNSUPPORTED from the patterns survives an LLM 'topical' verdict: the
        patterns proved the question is count-shaped, so declining precisely
        beats hedging retrieval."""
        cfg.intent_llm = True
        try:
            mock_svc.provider.chat.return_value = _text_result('{"kind": "topical"}')
            answer = get_services().searcher.route_direct_answer("how many angels are there here?")
            assert answer is not None
            assert "count" in answer.lower()
        finally:
            cfg.intent_llm = False


class TestSearchStructured:
    def test_term_mode(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [_make_result()]
        results = get_services().searcher._search_structured(QueryMode.TERM, "test query", 5)
        assert len(results) == 1
        mock_svc.store.bm25_probe.assert_called_once_with("test query", top_k=5, chunk_type=None)

    def test_vec_mode(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        results = get_services().searcher._search_structured(QueryMode.VEC, "semantic query", 5)
        assert len(results) == 1

    def test_hyde_mode(self, mock_svc):
        mock_svc.provider.chat.return_value = _text_result("hypothetical doc")
        mock_svc.store.search.return_value = [_make_result()]
        results = get_services().searcher._search_structured(QueryMode.HYDE, "vague question", 5)
        assert len(results) == 1

    def test_term_mode_forwards_chunk_type(self, mock_svc):
        """A ``term:`` query under an explicit scope must keep the scope filter."""
        mock_svc.store.bm25_probe.return_value = [_make_result()]
        get_services().searcher._search_structured(
            QueryMode.TERM, "q", 5, chunk_type=ChunkType.WIKI
        )
        assert mock_svc.store.bm25_probe.call_args[1]["chunk_type"] == ChunkType.WIKI

    def test_vec_mode_forwards_chunk_type(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher._search_structured(QueryMode.VEC, "q", 5, chunk_type=ChunkType.WIKI)
        assert mock_svc.store.search.call_args[1]["chunk_type"] == ChunkType.WIKI

    def test_hyde_mode_forwards_chunk_type(self, mock_svc):
        mock_svc.provider.chat.return_value = _text_result("doc")
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher._search_structured(
            QueryMode.HYDE, "q", 5, chunk_type=ChunkType.WIKI
        )
        assert mock_svc.store.search.call_args[1]["chunk_type"] == ChunkType.WIKI


class TestSearchContextIntegration:
    def test_structured_term_mode(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [_make_result()]
        results = get_services().searcher.search("term: kubernetes pods")
        mock_svc.store.bm25_probe.assert_called_once()
        assert len(results) >= 1

    def test_skips_expansion_when_confident(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        # Raw BM25 magnitudes: confidence(30) = 0.857 >= 0.8 with a wide
        # relative gap ((30-10)/30 = 0.67), so expansion is skipped.
        mock_svc.store.bm25_probe.return_value = [
            _make_result(bm25_score=30.0),
            _make_result(bm25_score=10.0),
        ]
        results = get_services().searcher.search("exact match query")
        # Provider.chat should NOT be called for expansion
        mock_svc.provider.chat.assert_not_called()
        assert len(results) >= 1

    def test_hyde_merges_results(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(source="normal.md")]
        mock_svc.provider.chat.return_value = _text_result("hypothetical doc")
        old = cfg.hyde
        cfg.hyde = True
        try:
            results = get_services().searcher.search("vague question")
            sources = {r.source for r in results}
            assert "normal.md" in sources
        finally:
            cfg.hyde = old

    def test_scoped_search_threads_scope_through_hyde_merge(self, mock_svc):
        """A scoped non-structured search must scope its HyDE merge too, else
        out-of-scope chunks leak into a scoped result set."""
        mock_svc.store.search.return_value = []
        mock_svc.store.bm25_probe.return_value = []  # don't skip expansion
        captured: dict[str, object] = {}

        def fake_hyde(question, top_k, chunk_type=None):
            captured["chunk_type"] = chunk_type
            return []

        old_hyde, old_cg, old_qe = cfg.hyde, cfg.concept_graph, cfg.query_expansion_count
        cfg.hyde, cfg.concept_graph, cfg.query_expansion_count = True, False, 0
        try:
            searcher = get_services().searcher
            with mock.patch.object(searcher, "_hyde_search", side_effect=fake_hyde):
                searcher.search("question", chunk_type=ChunkType.RAW)
            assert captured["chunk_type"] == ChunkType.RAW
        finally:
            cfg.hyde, cfg.concept_graph, cfg.query_expansion_count = old_hyde, old_cg, old_qe

    def test_hyde_adds_unique_results_downweighted_in_score_space(self, mock_svc):
        """HyDE results not seen in normal search are added with their canonical
        score discounted by hyde_weight; distance provenance stays untouched."""
        normal_result = _make_result(source="normal.md", chunk_index=0)
        # Under cfg.max_distance (0.75): search() now applies the shared
        # relevance cutoff, and a far vector-only HyDE row is filtered like
        # any other unsupported far row.
        hyde_only_result = _make_result(source="hyde.md", chunk_index=0, distance=0.6)
        mock_svc.store.search.side_effect = [
            [normal_result],
            [hyde_only_result],
        ]
        mock_svc.provider.chat.return_value = _text_result("hypothetical document")
        # Disable query expansion so the HyDE path owns the second
        # store.search call.
        cfg.query_expansion_count = 0
        cfg.hyde = True
        cfg.hyde_weight = 0.5
        try:
            results = get_services().searcher.search("vague question")
            sources = {r.source for r in results}
            assert "normal.md" in sources
            assert "hyde.md" in sources
            hyde_r = next(r for r in results if r.source == "hyde.md")
            assert hyde_r.score == pytest.approx((1.0 - 0.6) * 0.5)
            assert hyde_r.distance == pytest.approx(0.6)
        finally:
            cfg.query_expansion_count = 3
            cfg.hyde = False
            cfg.hyde_weight = 0.7


class TestKnownItemTitleRoute:
    """Human titles resolve like filenames (repro: 'summarize Frankenstein'
    on a corpus of '<Title>.txt' novels ran topical top-k and pulled a stray
    chunk from another book). A known-item shape plus a token-exact stem
    match routes; anything else stays topical."""

    def _source(self, filename):
        return {"filename": filename, "file_hash": "h", "ingested_at": "", "chunk_count": 2}

    def _index(self, mock_svc, filenames):
        sources = [self._source(f) for f in filenames]

        def get_sources(search=None, limit=None, offset=0):
            if not search:
                return sources[:limit]
            return [s for s in sources if search.lower() in s["filename"].lower()][:limit]

        mock_svc.store.get_sources.side_effect = get_sources
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source=filenames[0], chunk="opening", chunk_index=0),
            _make_result(source=filenames[0], chunk="ending", chunk_index=1),
        ]

    @pytest.mark.parametrize(
        "question",
        [
            "summarize Frankenstein",
            "Summarize Frankenstein.",
            "what is Frankenstein about?",
            "give me a summary of Frankenstein",
            "describe frankenstein",
        ],
    )
    def test_title_shapes_resolve_the_document(self, mock_svc, question):
        self._index(mock_svc, ["Frankenstein.txt", "The Prince.txt"])
        results = get_services().searcher.search(question)
        assert [r.chunk for r in results] == ["opening", "ending"]
        assert all(r.score == 1.0 for r in results)
        mock_svc.store.search.assert_not_called()

    def test_leading_article_matches_stem(self, mock_svc):
        self._index(mock_svc, ["The Prince.txt", "Frankenstein.txt"])
        results = get_services().searcher.search("summarize the prince")
        assert results
        mock_svc.store.get_chunks_by_source.assert_called_once_with("The Prince.txt")

    def test_topical_question_mentioning_a_title_word_is_not_hijacked(self, mock_svc):
        """No known-item shape means no title route, even when a document
        stem appears in the question."""
        self._index(mock_svc, ["Storms.txt"])
        mock_svc.store.search.return_value = [_make_result(source="Storms.txt")]
        get_services().searcher.search("how do storms form near the coast?")
        mock_svc.store.get_chunks_by_source.assert_not_called()

    def test_partial_title_does_not_resolve(self, mock_svc):
        """'summarize the report' against 'Annual Report 2020.txt' is not a
        token-exact stem match; the turn stays topical."""
        self._index(mock_svc, ["Annual Report 2020.txt"])
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher.search("summarize the report")
        mock_svc.store.get_chunks_by_source.assert_not_called()

    def test_ambiguous_title_falls_back_to_topical(self, mock_svc):
        self._index(mock_svc, ["Notes.txt", "notes.md"])
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher.search("summarize notes")
        mock_svc.store.get_chunks_by_source.assert_not_called()


class TestKnownItemRoute:
    def _source(self, filename):
        return {"filename": filename, "file_hash": "h", "ingested_at": "", "chunk_count": 2}

    def test_bare_search_routes_named_document(self, mock_svc):
        """The route covers every retrieval surface: bare search() (HTTP
        /api/search, MCP) resolves a document-naming query to that document
        instead of similarity neighbors of its wording."""
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="survey_report.pdf", chunk="second", chunk_index=1),
            _make_result(source="survey_report.pdf", chunk="first", chunk_index=0),
        ]
        results = get_services().searcher.search("summarize survey_report.pdf")
        assert [r.chunk for r in results] == ["first", "second"]
        assert all(r.score == 1.0 for r in results)
        mock_svc.store.search.assert_not_called()
        mock_svc.embedder.embed_query.assert_not_called()

    def test_bare_search_route_caps_at_return_budget(self, mock_svc):
        """A huge named document fills the standard search return budget
        (top_k*2) with its head, not the whole document."""
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="survey_report.pdf", chunk=f"c{i}", chunk_index=i)
            for i in range(100)
        ]
        results = get_services().searcher.search("summarize survey_report.pdf", top_k=5)
        assert len(results) == 10
        assert [r.chunk_index for r in results] == list(range(10))

    def test_bare_search_skips_route_when_intent_routing_off(self, mock_svc):
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        mock_svc.store.search.return_value = [_make_result()]
        cfg.intent_routing = False
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.search("summarize survey_report.pdf")
        finally:
            cfg.intent_routing = True
            cfg.query_expansion_count = 3
        mock_svc.store.get_chunks_by_source.assert_not_called()
        mock_svc.store.search.assert_called_once()

    def test_structured_mode_prefix_bypasses_route(self, mock_svc):
        """An explicit mode: prefix is the user forcing a strategy; the
        known-item route must not override it."""
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        mock_svc.store.bm25_probe.return_value = []
        get_services().searcher.search("term: survey_report.pdf")
        mock_svc.store.get_chunks_by_source.assert_not_called()

    def test_named_document_bypasses_similarity_search(self, mock_svc):
        """A question naming one resolvable document gets that document's
        chunks in document order, not a similarity ranking."""
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="survey_report.pdf", chunk="second", chunk_index=1),
            _make_result(source="survey_report.pdf", chunk="first", chunk_index=0),
        ]
        rag = get_services().searcher.build_rag_context("summarize survey_report.pdf")
        assert rag is not None
        results, _ = rag
        assert [r.chunk for r in results] == ["first", "second"]
        assert all(r.score == 1.0 for r in results)
        mock_svc.store.search.assert_not_called()

    def test_numeric_reference_resolves_against_zero_padded_ids(self, mock_svc):
        """The private-corpus failure shape: a bare number must resolve to the
        one zero-padded id that token-matches it, not fall back to topical
        because substring search also hit a longer number."""
        mock_svc.store.get_sources.return_value = [
            self._source("ARC-REC-00000482.pdf"),
            self._source("ARC-REC-00010482.pdf"),
        ]
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="ARC-REC-00000482.pdf", chunk="body", chunk_index=0)
        ]
        rag = get_services().searcher.build_rag_context("summarize document 482")
        assert rag is not None
        mock_svc.store.get_chunks_by_source.assert_called_once_with("ARC-REC-00000482.pdf")
        mock_svc.store.search.assert_not_called()

    def test_quoted_title_resolves_via_unique_substring_match(self, mock_svc):
        """A quoted multi-word title never token-matches a hyphenated
        filename; when substring search finds exactly one source, that
        uniqueness still routes."""
        mock_svc.store.get_sources.return_value = [self._source("harbor-survey-2010.pdf")]
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="harbor-survey-2010.pdf", chunk="body", chunk_index=0)
        ]
        rag = get_services().searcher.build_rag_context('summarize "harbor survey 2010"')
        assert rag is not None
        mock_svc.store.get_chunks_by_source.assert_called_once_with("harbor-survey-2010.pdf")

    def test_routed_document_fits_the_served_context_window(self, mock_svc):
        """The campaign's overflow failure: the route pulls a whole document,
        and the budget must trim to the engine's ACTUAL serving window, not
        the configured target, or /api/ask hard-fails with a 400."""
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        # A document far larger than the serving window: 200 chunks x ~400 tokens.
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="survey_report.pdf", chunk="word " * 400, chunk_index=i)
            for i in range(200)
        ]
        mock_svc.provider.served_chat_ctx.return_value = 8192
        cfg.num_ctx = None
        rag = get_services().searcher.build_rag_context("summarize survey_report.pdf")
        assert rag is not None
        results, messages = rag
        prompt_tokens = sum(len(m["content"]) // 4 for m in messages)
        assert prompt_tokens <= 8192
        # Document order preserved after trimming: the head survives.
        assert [r.chunk_index for r in results] == list(range(len(results)))

    def test_dense_text_budgets_conservatively(self, mock_svc):
        """The private-corpus overflow: text tokenizing at ~2.5 chars/token
        passed a chars/4 budget untrimmed. Budgeting must assume dense text,
        so the kept set stays well under the window even at 3 chars/token."""
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        # Numeric-dense chunks: ~6000 chars each, real cost ~2400 tokens each.
        dense = ("4471 0482 9001 " * 400).strip()
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="survey_report.pdf", chunk=dense, chunk_index=i) for i in range(40)
        ]
        mock_svc.provider.served_chat_ctx.return_value = 24576
        cfg.num_ctx = None
        rag = get_services().searcher.build_rag_context("summarize survey_report.pdf")
        assert rag is not None
        _, messages = rag
        # At 2.56 chars/token the real prompt cost must still fit the window.
        real_tokens = sum(len(m["content"]) / 2.56 for m in messages)
        assert real_tokens <= 24576 * 1.05

    def test_overflow_from_provider_retries_once_with_tighter_fit(self, mock_svc):
        """When the engine still reports overflow, ask refits and retries
        instead of surfacing a hard failure."""
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="survey_report.pdf", chunk="word " * 400, chunk_index=i)
            for i in range(40)
        ]
        mock_svc.provider.served_chat_ctx.return_value = 8192
        mock_svc.provider.chat.side_effect = [
            ProviderError("overflow", kind=ProviderErrorKind.CONTEXT_OVERFLOW),
            _text_result("fits now [1]"),
        ]
        result = get_services().searcher.ask_raw("summarize survey_report.pdf")
        assert result.answer.startswith("fits now")
        assert mock_svc.provider.chat.call_count == 2
        first = mock_svc.provider.chat.call_args_list[0][0][0][-1]["content"]
        second = mock_svc.provider.chat.call_args_list[1][0][0][-1]["content"]
        assert len(second) < len(first)

    def test_budget_falls_back_to_config_when_served_ctx_unknown(self, mock_svc):
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="survey_report.pdf", chunk="word " * 400, chunk_index=i)
            for i in range(200)
        ]
        mock_svc.provider.served_chat_ctx.return_value = None
        cfg.num_ctx = 4096
        try:
            rag = get_services().searcher.build_rag_context("summarize survey_report.pdf")
        finally:
            cfg.num_ctx = None
        assert rag is not None
        _, messages = rag
        assert sum(len(m["content"]) // 4 for m in messages) <= 4096

    def test_docket_reference_resolves_by_content_concentration(self, mock_svc):
        """A reference that appears in a document's text but not its filename
        (a docket number) resolves when BM25 hits concentrate in one source."""
        mock_svc.store.get_sources.return_value = []
        mock_svc.store.bm25_probe.return_value = [
            _make_result(source="scan_aa17.pdf", chunk_index=i, bm25_score=20.0 - i)
            for i in range(6)
        ]
        mock_svc.store.get_chunks_by_source.return_value = [
            _make_result(source="scan_aa17.pdf", chunk="body", chunk_index=0)
        ]
        rag = get_services().searcher.build_rag_context("summarize document 482")
        assert rag is not None
        mock_svc.store.get_chunks_by_source.assert_called_once_with("scan_aa17.pdf")
        mock_svc.store.search.assert_not_called()

    def test_multiple_token_owners_stay_topical_without_probing(self, mock_svc):
        """Two files legitimately carrying the same reference token (a split
        document) are true ambiguity: no route, and no content probe either,
        since the filenames already prove the reference is shared."""
        mock_svc.store.get_sources.return_value = [
            self._source("ARC-482-part1.pdf"),
            self._source("ARC-482-part2.pdf"),
        ]
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("summarize document 482")
        finally:
            cfg.query_expansion_count = 3
        mock_svc.store.get_chunks_by_source.assert_not_called()
        mock_svc.store.search.assert_called_once()

    def test_scattered_content_hits_stay_topical(self, mock_svc):
        """A reference mentioned across many documents (a number cited in
        related filings) must not route: concentration is the signal."""
        mock_svc.store.get_sources.return_value = []
        mock_svc.store.bm25_probe.return_value = [
            _make_result(source=f"scan_{i}.pdf", chunk_index=0, bm25_score=20.0 - i)
            for i in range(6)
        ]
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("summarize document 482")
        finally:
            cfg.query_expansion_count = 3
        mock_svc.store.get_chunks_by_source.assert_not_called()
        mock_svc.store.search.assert_called_once()

    def test_numeric_ref_with_only_substring_hit_stays_topical(self, mock_svc):
        """ "12" inside "notes-2012" is a false match; a numeric reference that
        token-matches nothing must not route via the substring fallback."""
        mock_svc.store.get_sources.return_value = [self._source("notes-2012.pdf")]
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("summarize document 12")
        finally:
            cfg.query_expansion_count = 3
        mock_svc.store.get_chunks_by_source.assert_not_called()
        mock_svc.store.search.assert_called_once()

    def test_ambiguous_reference_falls_back_to_topical(self, mock_svc):
        mock_svc.store.get_sources.return_value = [
            self._source("report_12a.pdf"),
            self._source("report_12b.pdf"),
        ]
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("summarize document 12")
        finally:
            cfg.query_expansion_count = 3
        mock_svc.store.search.assert_called_once()

    def test_disabled_by_config(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        cfg.intent_routing = False
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("summarize survey_report.pdf")
        finally:
            cfg.intent_routing = True
            cfg.query_expansion_count = 3
        mock_svc.store.get_sources.assert_not_called()
        mock_svc.store.search.assert_called_once()

    def test_resolved_source_with_no_chunks_falls_back(self, mock_svc):
        mock_svc.store.get_sources.return_value = [self._source("survey_report.pdf")]
        mock_svc.store.get_chunks_by_source.return_value = []
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("summarize survey_report.pdf")
        finally:
            cfg.query_expansion_count = 3
        mock_svc.store.search.assert_called_once()


class TestAggregateRoute:
    def test_term_count_answers_without_llm(self, mock_svc):
        """A count question gets an exact scan answer; no model is called, so
        no model can hedge or invent the number."""
        mock_svc.store.count_term_mentions.return_value = (412, 57)
        result = get_services().searcher.ask_raw("how many documents mention the observatory?")
        assert "57 documents" in result.answer
        assert "412 passages" in result.answer
        mock_svc.provider.chat.assert_not_called()
        mock_svc.store.search.assert_not_called()

    def test_total_count(self, mock_svc):
        mock_svc.store.count_sources.return_value = 369
        mock_svc.store.count_chunks.return_value = 123456
        result = get_services().searcher.ask_raw("how many documents are there?")
        assert "369" in result.answer
        assert "123456" in result.answer

    def test_typed_count_declines_precisely(self, mock_svc, tmp_path):
        old_dir = cfg.data_dir
        cfg.data_dir = tmp_path / "no_schema_here"
        try:
            result = get_services().searcher.ask_raw(
                "how many shipments is each part number associated with?"
            )
        finally:
            cfg.data_dir = old_dir
        assert "aren't extracted" in result.answer
        mock_svc.provider.chat.assert_not_called()

    def test_stream_path_routes_aggregates_too(self, mock_svc):
        mock_svc.store.count_term_mentions.return_value = (10, 3)
        tokens = list(get_services().searcher.ask_stream("how many documents mention kerosene?"))
        text = "".join(t.content for t in tokens)
        assert "3 documents" in text
        mock_svc.provider.chat.assert_not_called()

    def test_topical_question_unaffected(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("an answer [1]")
        cfg.query_expansion_count = 0
        try:
            result = get_services().searcher.ask_raw("what did the keeper record in October?")
        finally:
            cfg.query_expansion_count = 3
        assert result.sources
        mock_svc.store.count_term_mentions.assert_not_called()

    def test_disabled_by_config_goes_topical(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("an answer [1]")
        cfg.intent_routing = False
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.ask_raw("how many documents mention the observatory?")
        finally:
            cfg.intent_routing = True
            cfg.query_expansion_count = 3
        mock_svc.store.count_term_mentions.assert_not_called()
        mock_svc.store.search.assert_called_once()


class TestTypedAggregates:
    @pytest.fixture()
    def part_schema(self, mock_svc):
        import json

        from lilbee.retrieval.entities import EntitySchema, EntityType, ExtractorKind

        schema = EntitySchema(
            types=[
                EntityType(
                    name="part_number",
                    kind=ExtractorKind.REGEX,
                    pattern=r"PX\d{4}",
                    synonyms=["part"],
                ),
                EntityType(name="depot", kind=ExtractorKind.SPACY, pattern="GPE"),
            ]
        )
        mock_svc.store.entity_schema_state.return_value = {
            "schema_json": json.dumps(schema.model_dump(mode="json")),
            "applied": True,
            "source_count": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        return schema

    def test_distinct_count_answers_exactly(self, mock_svc, part_schema):
        mock_svc.store.entity_value_counts.return_value = (57, 12)
        result = get_services().searcher.ask_raw("how many distinct part numbers are recorded?")
        assert "12 distinct part number values" in result.answer
        assert "57 mentions" in result.answer
        mock_svc.provider.chat.assert_not_called()

    def test_association_answers_grouped_counts(self, mock_svc, part_schema):
        mock_svc.store.entity_association_counts.return_value = {"fresno": 3, "reno": 1}
        result = get_services().searcher.ask_raw("how many parts is each depot associated with?")
        assert "fresno: 3" in result.answer
        assert "reno: 1" in result.answer
        mock_svc.store.entity_association_counts.assert_called_once_with(
            "part_number", grouped_by="depot"
        )

    def test_unresolvable_noun_declines_naming_types(self, mock_svc, part_schema):
        result = get_services().searcher.ask_raw("how many distinct vessels are recorded?")
        assert "Countable entity types" in result.answer
        assert "part number" in result.answer

    def test_empty_extraction_says_so(self, mock_svc, part_schema):
        mock_svc.store.entity_value_counts.return_value = (0, 0)
        result = get_services().searcher.ask_raw("how many distinct part numbers are there?")
        assert "extracted yet" in result.answer

    def test_synonym_proxy_count_discloses_the_measured_type(self, mock_svc, part_schema):
        """A noun resolved through a synonym gets the count of a DIFFERENT
        quantity; the answer must say which type it actually measured."""
        mock_svc.store.entity_value_counts.return_value = (57, 12)
        result = get_services().searcher.ask_raw("how many distinct parts are recorded?")
        assert "12 distinct part number values" in result.answer
        assert "closest extracted type" in result.answer

    def test_direct_type_name_carries_no_proxy_note(self, mock_svc, part_schema):
        mock_svc.store.entity_value_counts.return_value = (57, 12)
        result = get_services().searcher.ask_raw("how many distinct part numbers are recorded?")
        assert "closest extracted type" not in result.answer

    def test_association_with_unresolvable_group_noun_declines(self, mock_svc, part_schema):
        result = get_services().searcher.ask_raw("how many parts is each vessel associated with?")
        assert "Countable entity types" in result.answer

    def test_association_with_no_extracted_rows_says_so(self, mock_svc, part_schema):
        mock_svc.store.entity_association_counts.return_value = {}
        result = get_services().searcher.ask_raw("how many parts is each depot associated with?")
        assert "extracted yet" in result.answer

    def test_without_schema_keeps_the_generic_decline(self, mock_svc, tmp_path):
        old_dir = cfg.data_dir
        cfg.data_dir = tmp_path / "no_schema_here"
        try:
            result = get_services().searcher.ask_raw(
                "how many parts is each depot associated with?"
            )
        finally:
            cfg.data_dir = old_dir
        assert "aren't extracted from this corpus yet" in result.answer


class TestHistoryCondensation:
    _HISTORY: ClassVar[list[dict[str, str]]] = [
        {"role": "user", "content": "who kept the lighthouse journal at Split Rock"},
        {"role": "assistant", "content": "It was kept by E. Larsen [1]."},
    ]

    def test_follow_up_is_rewritten_for_retrieval(self, mock_svc):
        """Retrieval must see the standalone rewrite, not the pronouns; the
        answering prompt keeps the user's original wording."""
        rewritten = "when was the Split Rock lighthouse journal written"
        mock_svc.provider.chat.return_value = _text_result(rewritten)
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            rag = get_services().searcher.build_rag_context(
                "and when was it written?", history=list(self._HISTORY)
            )
        finally:
            cfg.query_expansion_count = 3
        assert rag is not None
        _, messages = rag
        assert mock_svc.store.search.call_args[1]["query_text"] == rewritten
        assert "and when was it written?" in messages[-1]["content"]
        assert rewritten not in messages[-1]["content"]

    def test_no_history_means_no_rewrite_call(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("standalone question")
        finally:
            cfg.query_expansion_count = 3
        mock_svc.provider.chat.assert_not_called()

    def test_falls_back_to_original_on_provider_error(self, mock_svc):
        mock_svc.provider.chat.side_effect = RuntimeError("no provider")
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            rag = get_services().searcher.build_rag_context(
                "and when was it written?", history=list(self._HISTORY)
            )
        finally:
            cfg.query_expansion_count = 3
        assert rag is not None
        assert mock_svc.store.search.call_args[1]["query_text"] == "and when was it written?"

    def test_disabled_by_config(self, mock_svc):
        cfg.history_rewrite = False
        cfg.query_expansion_count = 0
        mock_svc.store.search.return_value = [_make_result()]
        try:
            get_services().searcher.build_rag_context(
                "and when was it written?", history=list(self._HISTORY)
            )
        finally:
            cfg.history_rewrite = True
            cfg.query_expansion_count = 3
        mock_svc.provider.chat.assert_not_called()
        assert mock_svc.store.search.call_args[1]["query_text"] == "and when was it written?"

    def test_reasoning_stripped_from_rewrite(self, mock_svc):
        mock_svc.provider.chat.return_value = _text_result(
            "<think>the user means the journal</think>when was the Split Rock journal written"
        )
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context(
                "and when was it written?", history=list(self._HISTORY)
            )
        finally:
            cfg.query_expansion_count = 3
        expected = "when was the Split Rock journal written"
        assert mock_svc.store.search.call_args[1]["query_text"] == expected


class TestAskRawWithReranker:
    def test_reranker_called_when_configured(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = _text_result("answer")
        mock_svc.reranker.rerank.return_value = [_make_result()]
        old = cfg.reranker_model
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        try:
            result = get_services().searcher.ask_raw("question")
            mock_svc.reranker.rerank.assert_called_once()
            assert result.answer == "answer"
        finally:
            cfg.reranker_model = old


class TestRerankerPoolDepth:
    def test_reranker_configured_retrieves_candidate_depth(self, mock_svc):
        """With a cross-encoder re-scoring the pool, retrieval must fetch
        rerank_candidates deep so the reranker sees its configured count."""
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.reranker.rerank.return_value = [_make_result()]
        old = cfg.reranker_model
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("harbor light logistics")
        finally:
            cfg.reranker_model = old
            cfg.query_expansion_count = 3
        assert mock_svc.store.search.call_args[1]["top_k"] == cfg.rerank_candidates

    def test_no_reranker_retrieves_exactly_top_k(self, mock_svc):
        """Without a reranker the fused order is final; the pool stays at
        top_k so deep rank fusion cannot bury single-arm certainty."""
        mock_svc.store.search.return_value = [_make_result()]
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context("harbor light logistics")
        finally:
            cfg.query_expansion_count = 3
        assert mock_svc.store.search.call_args[1]["top_k"] == cfg.top_k

    def test_explicit_top_k_above_candidates_wins(self, mock_svc):
        """A caller asking for more rows than rerank_candidates keeps them."""
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.reranker.rerank.return_value = [_make_result()]
        old = cfg.reranker_model
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        cfg.query_expansion_count = 0
        try:
            get_services().searcher.build_rag_context(
                "harbor light logistics", top_k=cfg.rerank_candidates + 40
            )
        finally:
            cfg.reranker_model = old
            cfg.query_expansion_count = 3
        assert mock_svc.store.search.call_args[1]["top_k"] == cfg.rerank_candidates + 40


class TestAskStreamWithReranker:
    def test_reranker_called_when_configured(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = iter(["token"])
        mock_svc.reranker.rerank.return_value = [_make_result()]
        old = cfg.reranker_model
        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF/bge-Q4_K_M.gguf"
        try:
            list(get_services().searcher.ask_stream("question"))
            mock_svc.reranker.rerank.assert_called_once()
        finally:
            cfg.reranker_model = old


class TestConceptBoosting:
    def test_boost_applied_when_enabled(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(distance=0.5)]
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts.return_value = ["python"]
        mock_svc.concepts.boost_results.return_value = [_make_result(distance=0.3)]
        old = cfg.concept_graph
        cfg.concept_graph = True
        cfg.query_expansion_count = 0
        try:
            # Rebuild searcher with updated config
            searcher = Searcher(
                cfg,
                mock_svc.provider,
                mock_svc.store,
                mock_svc.embedder,
                mock_svc.reranker,
                mock_svc.concepts,
            )
            results = searcher.search("python code")
            mock_svc.concepts.boost_results.assert_called_once()
            assert results[0].distance == 0.3
        finally:
            cfg.concept_graph = old
            cfg.query_expansion_count = 3

    def test_boost_resorts_by_boosted_score(self, mock_svc):
        """Boost mutates scores in place; the list must be re-sorted so callers that
        consume search() order directly (CLI search, MCP lilbee_search) see the
        boost change the ranking, not just the scores."""
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts.return_value = ["python"]
        weak = _make_result(source="weak.md", distance=0.4)
        strong = _make_result(source="strong.md", distance=0.1)
        mock_svc.concepts.boost_results.return_value = [weak, strong]
        old = cfg.concept_graph
        cfg.concept_graph = True
        try:
            out = get_services().searcher._apply_concept_boost([weak, strong], "python question")
            assert [r.source for r in out] == ["strong.md", "weak.md"]
        finally:
            cfg.concept_graph = old

    def test_boost_resort_keeps_strong_hybrid_above_hyde(self, mock_svc):
        """The boosted list mixes RRF (hybrid) and distance (HyDE) rows; per-family
        normalization keeps a strong hybrid hit on top instead of letting a HyDE
        recall's larger 1-distance dominate the tiny RRF score."""
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts.return_value = ["python"]
        strong = _make_result(source="strong.md", distance=None, relevance_score=0.05)
        weak = _make_result(source="weak.md", distance=None, relevance_score=0.02)
        hyde = _make_result(source="hyde.md", distance=0.3, relevance_score=None)
        mock_svc.concepts.boost_results.return_value = [strong, weak, hyde]
        old = cfg.concept_graph
        cfg.concept_graph = True
        try:
            out = get_services().searcher._apply_concept_boost([strong, weak, hyde], "python q")
            assert out[0].source == "strong.md"
        finally:
            cfg.concept_graph = old

    def test_boost_skipped_when_disabled(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(distance=0.5)]
        cfg.query_expansion_count = 0
        try:
            results = get_services().searcher.search("python code")
            mock_svc.concepts.boost_results.assert_not_called()
            assert results[0].distance == 0.5
        finally:
            cfg.query_expansion_count = 3

    def test_boost_skipped_when_extract_returns_empty(self, mock_svc):
        """extract_concepts returning no concepts short-circuits before boost_results."""
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts.return_value = []
        old = cfg.concept_graph
        cfg.concept_graph = True
        try:
            results = [_make_result(source="a.md", distance=0.5)]
            out = get_services().searcher._apply_concept_boost(results, "empty question")
            assert out == results
            mock_svc.concepts.boost_results.assert_not_called()
        finally:
            cfg.concept_graph = old

    def test_boost_failure_returns_original(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(distance=0.5)]
        mock_svc.concepts.get_graph.side_effect = RuntimeError("broken")
        old = cfg.concept_graph
        cfg.concept_graph = True
        cfg.query_expansion_count = 0
        try:
            searcher = Searcher(
                cfg,
                mock_svc.provider,
                mock_svc.store,
                mock_svc.embedder,
                mock_svc.reranker,
                mock_svc.concepts,
            )
            results = searcher.search("python code")
            assert results[0].distance == 0.5
        finally:
            cfg.concept_graph = old
            cfg.query_expansion_count = 3

    def test_boost_graph_none_returns_original(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(distance=0.5)]
        mock_svc.concepts.get_graph.return_value = False
        old = cfg.concept_graph
        cfg.concept_graph = True
        cfg.query_expansion_count = 0
        try:
            searcher = Searcher(
                cfg,
                mock_svc.provider,
                mock_svc.store,
                mock_svc.embedder,
                mock_svc.reranker,
                mock_svc.concepts,
            )
            results = searcher.search("python code")
            assert results[0].distance == 0.5
        finally:
            cfg.concept_graph = old
            cfg.query_expansion_count = 3


class TestConceptQueryExpansion:
    def test_expansion_includes_concept_terms(self, mock_svc):
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.expand_query.return_value = ["python web frameworks"]
        mock_svc.provider.chat.return_value = _text_result("variant query about python")
        old = cfg.concept_graph
        cfg.concept_graph = True
        try:
            searcher = Searcher(
                cfg,
                mock_svc.provider,
                mock_svc.store,
                mock_svc.embedder,
                mock_svc.reranker,
                mock_svc.concepts,
            )
            variants = searcher._expand_query("python frameworks", [0.1] * 768)
            texts = [text for text, _ in variants]
            assert "python web frameworks" in texts
        finally:
            cfg.concept_graph = old

    def test_expansion_disabled_returns_empty(self, mock_svc):
        old = cfg.concept_graph
        cfg.concept_graph = False
        try:
            result = get_services().searcher._concept_query_expansion("test query")
            assert result == []
        finally:
            cfg.concept_graph = old

    def test_expansion_failure_returns_empty(self, mock_svc):
        mock_svc.concepts.get_graph.side_effect = RuntimeError("broken")
        old = cfg.concept_graph
        cfg.concept_graph = True
        try:
            searcher = Searcher(
                cfg,
                mock_svc.provider,
                mock_svc.store,
                mock_svc.embedder,
                mock_svc.reranker,
                mock_svc.concepts,
            )
            result = searcher._concept_query_expansion("test query")
            assert result == []
        finally:
            cfg.concept_graph = old

    def test_expansion_graph_none_returns_empty(self, mock_svc):
        mock_svc.concepts.get_graph.return_value = None
        old = cfg.concept_graph
        cfg.concept_graph = True
        try:
            searcher = Searcher(
                cfg,
                mock_svc.provider,
                mock_svc.store,
                mock_svc.embedder,
                mock_svc.reranker,
                mock_svc.concepts,
            )
            result = searcher._concept_query_expansion("test query")
            assert result == []
        finally:
            cfg.concept_graph = old


class TestSearchEdgeCases:
    def test_empty_query(self, mock_svc):
        results = get_services().searcher.search("")
        assert results == [] or isinstance(results, list)

    def test_whitespace_query(self, mock_svc):
        results = get_services().searcher.search("   ")
        assert isinstance(results, list)


class TestFormatCitation:
    def test_page_location(self):
        rec = make_citation(page_start=3, page_end=3)
        result = _format_citation(rec)
        assert "page 3" in result
        assert rec["source_filename"] in result

    def test_page_range(self):
        rec = make_citation(page_start=2, page_end=5)
        result = _format_citation(rec)
        assert "pages 2-5" in result

    def test_line_location(self):
        rec = make_citation(line_start=10, line_end=20)
        result = _format_citation(rec)
        assert "lines 10-20" in result

    def test_single_line(self):
        rec = make_citation(line_start=7, line_end=7)
        result = _format_citation(rec)
        assert "line 7" in result

    def test_no_location(self):
        rec = make_citation()
        result = _format_citation(rec)
        assert rec["source_filename"] in result
        assert "page" not in result
        assert "line" not in result


class TestFormatSourceWiki:
    def test_wiki_chunk_with_citations(self):
        r = _make_result(
            source="wiki/summaries/doc.md",
            content_type="text",
            chunk_type="wiki",
        )
        cits = [make_citation(page_start=3, page_end=3)]
        result = format_source(r, citations=cits)
        assert "wiki/summaries/doc.md" in result.replace("\\", "/")
        assert "page 3" in result

    def test_wiki_chunk_without_citations(self):
        r = _make_result(
            source="wiki/summaries/doc.md",
            content_type="text",
            chunk_type="wiki",
        )
        result = format_source(r)
        assert "wiki/summaries/doc.md" in result.replace("\\", "/")


@pytest.mark.usefixtures("wiki_enabled")
class TestChunkTypeScope:
    """build_rag_context and ask_stream forward ``chunk_type`` to the store search.

    Replaced the former implicit pool preference. Callers (CLI --scope,
    MCP scope, HTTP chunk_type, TUI toggle) always choose raw/wiki/both
    explicitly.
    """

    def test_build_rag_context_default_is_mixed_pool(self, mock_svc):
        """No ``chunk_type`` arg means no filter. Both sides survive."""
        wiki_chunk = _make_result(source="wiki/summaries/doc.md", chunk_type="wiki")
        raw_chunk = _make_result(source="doc.md", chunk_type="raw")
        mock_svc.store.search.return_value = [wiki_chunk, raw_chunk]
        result = get_services().searcher.build_rag_context("question")
        assert result is not None
        chunks, _ = result
        assert len(chunks) == 2

    def test_build_rag_context_forwards_chunk_type_to_store(self, mock_svc):
        mock_svc.store.search.return_value = []
        get_services().searcher.build_rag_context("question", chunk_type="wiki")
        assert mock_svc.store.search.called
        kwargs = mock_svc.store.search.call_args.kwargs
        assert kwargs.get("chunk_type") == "wiki"

    def test_ask_raw_forwards_chunk_type(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(source="doc.md", chunk_type="raw")]
        mock_svc.provider.chat.return_value = _text_result("answer")
        get_services().searcher.ask_raw("question", chunk_type="raw")
        kwargs = mock_svc.store.search.call_args.kwargs
        assert kwargs.get("chunk_type") == "raw"

    def test_ask_forwards_chunk_type(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(source="doc.md", chunk_type="raw")]
        mock_svc.provider.chat.return_value = _text_result("answer")
        get_services().searcher.ask("question", chunk_type="raw")
        kwargs = mock_svc.store.search.call_args.kwargs
        assert kwargs.get("chunk_type") == "raw"

    def test_ask_stream_forwards_chunk_type(self, mock_svc):
        mock_svc.store.search.return_value = [
            _make_result(source="wiki/summaries/doc.md", chunk_type="wiki")
        ]
        mock_svc.provider.chat.return_value = iter(["answer"])
        tokens = list(get_services().searcher.ask_stream("question", chunk_type="wiki"))
        assert tokens  # stream produced something
        kwargs = mock_svc.store.search.call_args.kwargs
        assert kwargs.get("chunk_type") == "wiki"


class TestNormalizeChunkType:
    """Wiki-scope requests normalize to no-filter when wiki generation is off.

    Keeps the CLI/MCP/HTTP experience honest (no silent empty results)
    and mirrors the TUI hiding the toggle altogether.
    """

    def test_wiki_scope_with_wiki_disabled_normalizes_to_none(self, mock_svc, caplog):
        cfg.wiki = False
        try:
            mock_svc.store.search.return_value = []
            get_services().searcher.search("q", chunk_type="wiki")
            kwargs = mock_svc.store.search.call_args.kwargs
            assert kwargs.get("chunk_type") is None
            assert any("wiki is disabled" in r.message for r in caplog.records)
        finally:
            cfg.wiki = True

    def test_raw_scope_with_wiki_disabled_unchanged(self, mock_svc):
        cfg.wiki = False
        try:
            mock_svc.store.search.return_value = []
            get_services().searcher.search("q", chunk_type="raw")
            kwargs = mock_svc.store.search.call_args.kwargs
            assert kwargs.get("chunk_type") == "raw"
        finally:
            cfg.wiki = True


class TestStructuredQueryScopeInteraction:
    """Explicit ``chunk_type`` beats the ``wiki:``/``raw:`` prefix shortcut."""

    def test_explicit_chunk_type_overrides_wiki_prefix(self, mock_svc):
        mock_svc.store.search.return_value = []
        get_services().searcher.search("wiki: energy", chunk_type="raw")
        kwargs = mock_svc.store.search.call_args.kwargs
        assert kwargs.get("chunk_type") == "raw"

    def test_wiki_prefix_alone_filters_to_wiki_when_enabled(self, mock_svc):
        cfg.wiki = True
        try:
            mock_svc.store.search.return_value = []
            get_services().searcher.search("wiki: energy")
            kwargs = mock_svc.store.search.call_args.kwargs
            assert kwargs.get("chunk_type") == "wiki"
        finally:
            cfg.wiki = False

    def test_wiki_prefix_falls_back_to_full_pool_when_wiki_disabled(self, mock_svc):
        """The ``wiki:`` prefix goes through the same wiki-disabled guard as the
        explicit chunk_type arg, so it can't search an empty wiki pool."""
        cfg.wiki = False
        mock_svc.store.search.return_value = []
        get_services().searcher.search("wiki: energy")
        kwargs = mock_svc.store.search.call_args.kwargs
        assert kwargs.get("chunk_type") is None


class TestStructuredQueryWikiRaw:
    def test_wiki_prefix(self, mock_svc):
        mode, query = get_services().searcher._parse_structured_query("wiki: python typing")
        assert mode == "wiki"
        assert query == "python typing"

    def test_raw_prefix(self, mock_svc):
        mode, query = get_services().searcher._parse_structured_query("raw: python typing")
        assert mode == "raw"
        assert query == "python typing"

    def test_wiki_mode_passes_chunk_type(self, mock_svc):
        cfg.wiki = True
        try:
            mock_svc.store.search.return_value = [_make_result()]
            get_services().searcher._search_structured(QueryMode.WIKI, "test", 5)
            mock_svc.store.search.assert_called_once()
            assert mock_svc.store.search.call_args[1]["chunk_type"] == "wiki"
        finally:
            cfg.wiki = False

    def test_raw_mode_passes_chunk_type(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher._search_structured(QueryMode.RAW, "test", 5)
        mock_svc.store.search.assert_called_once()
        assert mock_svc.store.search.call_args[1]["chunk_type"] == "raw"


class TestDirectMessagesNoEmbed:
    def test_builds_system_history_user(self, mock_svc):
        """direct_messages builds [system, ...history, user] when no embedding."""
        searcher = get_services().searcher
        history = [
            {"role": "user", "content": "prev"},
            {"role": "assistant", "content": "prev answer"},
        ]
        msgs = searcher.direct_messages("new question", history=history)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "prev"
        assert msgs[2]["content"] == "prev answer"
        assert msgs[3]["content"] == "new question"

    def test_no_history(self, mock_svc):
        msgs = get_services().searcher.direct_messages("q")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestAskRawNoEmbed:
    def test_search_mode_no_embedder_refuses(self, mock_svc):
        """Search mode (the default) with no embedder refuses cleanly instead of
        silently answering ungrounded -- the answer can't be grounded."""
        mock_svc.embedder.embedding_available.return_value = False

        searcher = Searcher(
            cfg,
            mock_svc.provider,
            mock_svc.store,
            mock_svc.embedder,
            mock_svc.reranker,
            mock_svc.concepts,
        )
        result = searcher.ask_raw("hello")
        assert result.answer == SEARCH_NEEDS_EMBEDDER
        assert result.sources == []
        mock_svc.provider.chat.assert_not_called()

    def test_chat_mode_no_embedder_answers_ungrounded(self, mock_svc):
        """Chat mode continues without an embedder: a direct, ungrounded answer
        under the general system prompt, with <think> tags stripped."""
        old = cfg.chat_mode
        cfg.chat_mode = "chat"
        try:
            mock_svc.embedder.embedding_available.return_value = False
            mock_svc.provider.chat.return_value = _text_result(
                "<think>inner thought</think>direct answer"
            )
            searcher = Searcher(
                cfg,
                mock_svc.provider,
                mock_svc.store,
                mock_svc.embedder,
                mock_svc.reranker,
                mock_svc.concepts,
            )
            result = searcher.ask_raw("hello")
            assert "<think>" not in result.answer
            assert "direct answer" in result.answer
            assert result.sources == []
            sent = mock_svc.provider.chat.call_args[0][0]
            assert sent[0]["content"] == cfg.general_system_prompt
        finally:
            cfg.chat_mode = old


class TestAskRawChatMode:
    """ask_raw consults cfg.chat_mode before consulting the embedder."""

    def test_chat_mode_skips_retrieval_even_when_embedding_ready(self, mock_svc):
        old = cfg.chat_mode
        cfg.chat_mode = "chat"
        try:
            mock_svc.embedder.embedding_available.return_value = True
            mock_svc.provider.chat.return_value = _text_result("no-search answer")
            result = get_services().searcher.ask_raw("any question")
            assert result.answer == "no-search answer"
            assert result.sources == []
            mock_svc.store.search.assert_not_called()
            sent = mock_svc.provider.chat.call_args[0][0]
            assert sent[0]["content"] == cfg.general_system_prompt
        finally:
            cfg.chat_mode = old

    def test_search_mode_with_results_runs_rag(self, mock_svc):
        old = cfg.chat_mode
        cfg.chat_mode = "search"
        try:
            mock_svc.store.search.return_value = [_make_result(chunk="grounded")]
            mock_svc.provider.chat.return_value = _text_result("grounded answer")
            result = get_services().searcher.ask_raw("question")
            assert result.answer == "grounded answer"
            assert len(result.sources) == 1
        finally:
            cfg.chat_mode = old

    def test_search_mode_empty_results_returns_grounded_refusal(self, mock_svc):
        """Search mode is a RAG path, so zero hits refuse rather than free-wheel (bb-0i0)."""
        old = cfg.chat_mode
        cfg.chat_mode = "search"
        try:
            mock_svc.store.search.return_value = []
            result = get_services().searcher.ask_raw("question")
            assert result.answer == GROUNDED_REFUSAL
            assert result.sources == []
            mock_svc.provider.chat.assert_not_called()
        finally:
            cfg.chat_mode = old

    def test_search_mode_without_embedding_refuses(self, mock_svc):
        """Search mode with no embedder can't ground, so it refuses cleanly rather
        than free-wheeling on the model's parametric knowledge."""
        old = cfg.chat_mode
        cfg.chat_mode = "search"
        try:
            mock_svc.embedder.embedding_available.return_value = False
            result = get_services().searcher.ask_raw("question")
            assert result.answer == SEARCH_NEEDS_EMBEDDER
            assert result.sources == []
            mock_svc.store.search.assert_not_called()
            mock_svc.provider.chat.assert_not_called()
        finally:
            cfg.chat_mode = old


class TestAskStreamNoEmbed:
    def test_search_mode_no_embedder_refuses(self, mock_svc):
        """ask_stream in search mode with no embedder yields a clean refusal token
        instead of hard-failing or streaming an ungrounded answer."""
        mock_svc.embedder.embedding_available.return_value = False

        searcher = Searcher(
            cfg,
            mock_svc.provider,
            mock_svc.store,
            mock_svc.embedder,
            mock_svc.reranker,
            mock_svc.concepts,
        )
        tokens = list(searcher.ask_stream("hello"))
        combined = "".join(st.content for st in tokens)
        assert combined == SEARCH_NEEDS_EMBEDDER
        mock_svc.provider.chat.assert_not_called()

    def test_chat_mode_stream_handles_connection_error(self, mock_svc):
        """Chat mode streams ungrounded without an embedder; a mid-stream
        ConnectionError is reported gracefully rather than propagating."""
        old = cfg.chat_mode
        cfg.chat_mode = "chat"
        try:
            mock_svc.embedder.embedding_available.return_value = False

            def failing():
                yield "partial"
                raise ConnectionError("lost")

            mock_svc.provider.chat.return_value = failing()

            searcher = Searcher(
                cfg,
                mock_svc.provider,
                mock_svc.store,
                mock_svc.embedder,
                mock_svc.reranker,
                mock_svc.concepts,
            )
            tokens = list(searcher.ask_stream("hello"))
            combined = "".join(st.content for st in tokens)
            assert "Connection lost" in combined
        finally:
            cfg.chat_mode = old


class TestFilterResults:
    def test_drops_high_distance(self):
        results = [
            _make_result(source="close.pdf", distance=0.3),
            _make_result(source="far.pdf", distance=0.95, chunk_index=1),
        ]
        filtered = filter_results(results, max_distance=0.9)
        assert len(filtered) == 1
        assert filtered[0].source == "close.pdf"

    def test_drops_high_distance_legacy_rows(self):
        results = [
            _make_result(source="close.pdf", distance=0.3, score=None),
            _make_result(source="far.pdf", distance=0.95, score=None, chunk_index=1),
        ]
        filtered = filter_results(results, max_distance=0.9)
        assert [r.source for r in filtered] == ["close.pdf"]

    def test_drops_low_canonical_score(self):
        results = [
            _make_result(source="good.pdf", distance=None, score=0.8),
            _make_result(source="bad.pdf", distance=None, score=0.01, chunk_index=1),
        ]
        filtered = filter_results(results, max_distance=0.9, min_relevance_score=0.05)
        assert len(filtered) == 1
        assert filtered[0].source == "good.pdf"

    def test_passes_results_with_neither_score(self):
        r = _make_result(distance=None, relevance_score=None)
        filtered = filter_results([r], max_distance=0.9, min_relevance_score=0.1)
        assert len(filtered) == 1

    def test_disabled_when_zero(self):
        results = [_make_result(distance=2.0)]
        filtered = filter_results(results, max_distance=0, min_relevance_score=0)
        assert len(filtered) == 1

    def test_canonical_score_gates_on_min_relevance(self):
        """min_relevance_score is a real abstention threshold against the
        canonical [0,1] score; with every row below it, retrieval comes back
        empty and ask() can refuse instead of feeding noise as context."""
        results = [
            _make_result(source="noise1.pdf", score=0.04),
            _make_result(source="noise2.pdf", score=0.02, chunk_index=1),
        ]
        assert filter_results(results, max_distance=0.9, min_relevance_score=0.1) == []

    def test_lexically_supported_row_survives_far_distance(self):
        """A row the BM25 arm matched keeps its standing regardless of vector
        distance; only unsupported far rows are dropped."""
        results = [
            _make_result(source="identifier.pdf", distance=1.4, score=0.35, bm25_score=30.0),
            _make_result(source="drift.pdf", distance=1.4, score=0.1, chunk_index=1),
        ]
        filtered = filter_results(results, max_distance=0.9)
        assert [r.source for r in filtered] == ["identifier.pdf"]

    def test_keeps_results_at_threshold(self):
        r = _make_result(distance=0.9)
        filtered = filter_results([r], max_distance=0.9)
        assert len(filtered) == 1


class TestRelevanceWeight:
    def test_distance_based(self):
        r = _make_result(distance=0.3, relevance_score=None)
        assert _relevance_weight(r) == pytest.approx(0.7)

    def test_scoreless_row_returns_default(self):
        r = _make_result(distance=None, score=None)
        assert _relevance_weight(r) == pytest.approx(0.5)

    def test_canonical_score_is_authoritative(self):
        r = _make_result(distance=0.3, score=0.6)
        assert _relevance_weight(r) == pytest.approx(0.6)

    def test_scoreless_row_ignores_other_signals(self):
        """The pre-score per-arm arithmetic is gone: a row without the
        canonical score weighs neutrally even when legacy fields are set."""
        r = _make_result(distance=0.3, relevance_score=0.9, score=None)
        assert _relevance_weight(r) == pytest.approx(0.5)

    def test_clamps_high_score(self):
        r = _make_result(distance=None, score=1.5)
        assert _relevance_weight(r) == pytest.approx(1.0)

    def test_clamps_negative_distance(self):
        r = _make_result(distance=1.5, relevance_score=None)
        assert _relevance_weight(r) == pytest.approx(0.0)


class TestCitedSubsetByName:
    def test_name_mention_counts_as_citation(self):
        from lilbee.retrieval.query.formatting import cited_subset

        sources = [
            _make_result(source="ARC-REC-00000482.pdf", chunk_index=0),
            _make_result(source="survey_report.pdf", chunk_index=1),
        ]
        answer = "According to ARC-REC-00000482.pdf, the request was approved."
        assert [c.source for c in cited_subset(answer, sources)] == ["ARC-REC-00000482.pdf"]

    def test_stem_mention_counts_when_identifier_shaped(self):
        from lilbee.retrieval.query.formatting import cited_subset

        sources = [_make_result(source="ARC-REC-00000482.pdf", chunk_index=0)]
        answer = "The memo in ARC-REC-00000482 approves the request."
        assert len(cited_subset(answer, sources)) == 1

    def test_prose_word_stem_does_not_false_positive(self):
        from lilbee.retrieval.query.formatting import cited_subset

        sources = [_make_result(source="notes.md", chunk_index=0)]
        answer = "The witness notes that the repair happened in March."
        assert cited_subset(answer, sources) == []

    def test_stem_embedded_in_a_longer_identifier_is_not_cited(self):
        """Substring containment marks 'log-1' cited by an answer that only
        discusses 'catalog-10'. cited_sources is the grounding signal JSON
        consumers read, so a false positive inflates grounding."""
        from lilbee.retrieval.query.formatting import cited_subset

        sources = [_make_result(source="log-1.md", chunk_index=0)]
        answer = "The catalog-10.pdf entry lists the part."
        assert cited_subset(answer, sources) == []

    def test_name_embedded_in_a_longer_filename_is_not_cited(self):
        from lilbee.retrieval.query.formatting import cited_subset

        sources = [_make_result(source="notes.md", chunk_index=0)]
        answer = "See footnotes.md for the caveats."
        assert cited_subset(answer, sources) == []

    def test_name_mention_still_counts_next_to_punctuation(self):
        from lilbee.retrieval.query.formatting import cited_subset

        sources = [_make_result(source="log-1.md", chunk_index=0)]
        for answer in ("As log-1.md shows,", "(log-1.md)", "see log-1.md."):
            assert len(cited_subset(answer, sources)) == 1, answer

    def test_marker_and_name_citations_combine(self):
        from lilbee.retrieval.query.formatting import cited_subset

        sources = [
            _make_result(source="alpha_report.pdf", chunk_index=0),
            _make_result(source="beta_summary.pdf", chunk_index=1),
        ]
        answer = "Per [2], and as alpha_report.pdf notes, both agree."
        assert {c.source for c in cited_subset(answer, sources)} == {
            "alpha_report.pdf",
            "beta_summary.pdf",
        }


class TestStripLlmCitations:
    def test_removes_sources_block(self):
        text = "The answer is 42.\n\nSources:\n- test.pdf, page 5"
        assert strip_llm_citations(text) == "The answer is 42."

    def test_removes_key_sources_block(self):
        text = "The answer.\n\nKey Sources:\n- test.pdf"
        assert strip_llm_citations(text) == "The answer."

    def test_removes_key_sources_lowercase(self):
        text = "The answer.\n\nKey sources:\n- [1] test.pdf"
        assert strip_llm_citations(text) == "The answer."

    def test_removes_references_block(self):
        text = "The answer.\n\nReferences:\n1. test.pdf"
        assert strip_llm_citations(text) == "The answer."

    def test_removes_markdown_heading_sources(self):
        text = "The answer.\n\n### Sources\n- test.pdf"
        assert strip_llm_citations(text) == "The answer."

    def test_preserves_answer_without_block(self):
        text = "The answer is 42."
        assert strip_llm_citations(text) == text

    def test_preserves_inline_source_mention(self):
        text = "The sources indicate that oil capacity is 5 quarts."
        assert strip_llm_citations(text) == text

    def test_preserves_heading_followed_by_prose(self):
        # An answer discussing a document's References section is not a
        # citation block; only heading-plus-list gets stripped.
        text = "The paper has three parts.\n\nReferences:\nIt lists 40 works."
        assert strip_llm_citations(text) == text

    def test_removes_a_block_that_starts_the_answer(self):
        """An answer that is nothing but a fabricated citation block must not
        stream through: lilbee stacks its authoritative list underneath, which
        is the double-list shape the filter exists to prevent."""
        text = "Sources:\n- fake.pdf\n- other.pdf"
        assert strip_llm_citations(text) == ""

    def test_removes_a_bare_heading_that_starts_the_answer(self):
        assert strip_llm_citations("Sources:") == ""

    def test_keeps_prose_that_follows_a_mid_answer_block(self):
        """The block is removed, not everything after it: a model that emits a
        citation list and then keeps answering must not lose the continuation."""
        text = "Answer.\n\nSources:\n- a.pdf\n\nAdditionally, more prose here."
        assert strip_llm_citations(text) == "Answer.\n\nAdditionally, more prose here."

    def test_removes_dangling_heading(self):
        text = "The answer is 42.\n\nSources:\n"
        assert strip_llm_citations(text) == "The answer is 42."


class TestExtractCitedIndices:
    def test_extracts_multiple(self):
        assert _extract_cited_indices("See [1] and [3].") == {1, 3}

    def test_no_citations(self):
        assert _extract_cited_indices("The answer is yes.") == set()

    def test_deduplicates(self):
        assert _extract_cited_indices("[1] and [1] again.") == {1}


class TestAskSourcesBlock:
    def test_ask_numbers_all_sources_so_markers_resolve(self, mock_svc):
        # Every retrieved source is listed and numbered so an inline [1] resolves
        # to source 1; the answer no longer drops the passages it didn't cite.
        r1 = _make_result(source="used.pdf", chunk="oil info", chunk_index=0)
        r2 = _make_result(source="unused.pdf", chunk="unrelated", chunk_index=1)
        mock_svc.store.search.return_value = [r1, r2]
        mock_svc.provider.chat.return_value = _text_result("Oil is 5 quarts [1].")
        answer = get_services().searcher.ask("oil capacity?")
        assert "1. [used.pdf](file://" in answer
        assert "2. [unused.pdf](file://" in answer

    def test_ask_falls_back_to_all_sources_when_no_refs(self, mock_svc):
        r1 = _make_result(source="a.pdf", chunk="oil info", chunk_index=0)
        mock_svc.store.search.return_value = [r1]
        mock_svc.provider.chat.return_value = _text_result("Oil is 5 quarts.")
        answer = get_services().searcher.ask("oil capacity?")
        assert "a.pdf" in answer

    def test_ask_strips_llm_citation_block(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(chunk="oil info")]
        mock_svc.provider.chat.return_value = _text_result(
            "5 quarts [1].\n\nKey sources:\n- [1] test.pdf"
        )
        answer = get_services().searcher.ask("oil capacity?")
        assert "Key sources" not in answer
        assert answer.count("Sources:") == 1


class TestAskStreamSourcesBlock:
    def test_stream_releases_held_back_final_line_before_sources(self, mock_svc):
        # The filter holds the last (possibly partial) line until the stream
        # ends; the flush tail must still reach the reader ahead of the block.
        mock_svc.store.search.return_value = [_make_result(source="a.pdf", chunk="oil")]
        mock_svc.provider.chat.return_value = iter(["First line [1].\n", "Held final line."])
        combined = "".join(st.content for st in get_services().searcher.ask_stream("q"))
        assert "First line [1].\nHeld final line." in combined
        assert combined.index("Held final line.") < combined.index("Sources:")

    def test_stream_numbers_all_sources_so_markers_resolve(self, mock_svc):
        r1 = _make_result(source="used.pdf", chunk="oil info", chunk_index=0)
        r2 = _make_result(source="unused.pdf", chunk="unrelated", chunk_index=1)
        mock_svc.store.search.return_value = [r1, r2]
        mock_svc.provider.chat.return_value = iter(["Oil is 5 quarts ", "[1]."])
        tokens = list(get_services().searcher.ask_stream("oil capacity?"))
        combined = "".join(st.content for st in tokens)
        assert "1. [used.pdf](file://" in combined
        assert "2. [unused.pdf](file://" in combined

    def test_stream_falls_back_when_no_refs(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(source="a.pdf", chunk="oil")]
        mock_svc.provider.chat.return_value = iter(["Oil is 5 quarts."])
        tokens = list(get_services().searcher.ask_stream("oil?"))
        combined = "".join(st.content for st in tokens)
        assert "a.pdf" in combined


class TestBuildRagContextFilters:
    def test_filters_high_distance_results(self, mock_svc):
        close = _make_result(source="close.pdf", distance=0.3, chunk="relevant")
        far = _make_result(source="far.pdf", distance=0.95, chunk="irrelevant", chunk_index=1)
        mock_svc.store.search.return_value = [close, far]
        result = get_services().searcher.build_rag_context("question")
        assert result is not None
        results, _ = result
        sources = [r.source for r in results]
        assert "close.pdf" in sources
        assert "far.pdf" not in sources

    def test_returns_none_when_all_filtered(self, mock_svc):
        far = _make_result(source="far.pdf", distance=0.95, chunk="irrelevant")
        mock_svc.store.search.return_value = [far]
        result = get_services().searcher.build_rag_context("question")
        assert result is None
