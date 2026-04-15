"""Tests for the RAG query pipeline (mocked — no live server needed)."""

import pytest

import lilbee.services as svc_mod
from lilbee.config import cfg
from lilbee.query import (
    Searcher,
    _extract_cited_indices,
    _format_citation,
    _relevance_weight,
    build_context,
    deduplicate_sources,
    filter_results,
    format_source,
    prefer_wiki,
    sort_by_relevance,
    strip_llm_citations,
)
from lilbee.services import get_services
from lilbee.store import SearchChunk
from tests.conftest import make_citation


@pytest.fixture(autouse=True)
def _disable_concepts():
    """Disable concept graph by default in query tests to avoid spaCy loads."""
    old = cfg.concept_graph
    cfg.concept_graph = False
    yield
    cfg.concept_graph = old


@pytest.fixture(autouse=True)
def mock_svc():
    """Inject mock Services so tests never hit real backends."""
    from tests.conftest import make_mock_services

    services = make_mock_services()
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


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
    vector=None,
) -> SearchChunk:
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
        vector=vector or [0.1],
    )


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
        assert "page" not in result
        assert "line" not in result


class TestDeduplicateSources:
    def test_removes_duplicates(self):
        results = [
            _make_result(source="a.pdf", page_start=1, page_end=1),
            _make_result(source="a.pdf", page_start=1, page_end=1),
            _make_result(source="b.pdf", page_start=2, page_end=2),
        ]
        citations = deduplicate_sources(results)
        assert len(citations) == 2

    def test_caps_at_max_citations(self):
        results = [_make_result(source=f"file{i}.pdf", page_start=i, page_end=i) for i in range(10)]
        citations = deduplicate_sources(results, max_citations=5)
        assert len(citations) == 5

    def test_custom_max_citations(self):
        results = [_make_result(source=f"file{i}.pdf", page_start=i, page_end=i) for i in range(10)]
        citations = deduplicate_sources(results, max_citations=3)
        assert len(citations) == 3


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

    def test_sorts_by_relevance_score_when_present(self):
        results = [
            _make_result(source="low.pdf", relevance_score=0.2),
            _make_result(source="high.pdf", relevance_score=0.9),
            _make_result(source="mid.pdf", relevance_score=0.5),
        ]
        sorted_results = sort_by_relevance(results)
        assert sorted_results[0].source == "high.pdf"
        assert sorted_results[1].source == "mid.pdf"
        assert sorted_results[2].source == "low.pdf"


class TestDiversifySources:
    def test_caps_per_source(self):
        from lilbee.query import diversify_sources

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
        from lilbee.query import diversify_sources

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
        from lilbee.query import diversify_sources

        assert diversify_sources([]) == []

    def test_default_cap_is_three(self):
        from lilbee.query import diversify_sources

        results = [_make_result(source="a.md", distance=float(i) / 10) for i in range(5)]
        diverse = diversify_sources(results)
        assert len(diverse) == 3


class TestBuildContext:
    def test_numbers_chunks(self):
        results = [_make_result(chunk="chunk one"), _make_result(chunk="chunk two")]
        ctx = build_context(results)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "chunk one" in ctx


class TestSearchContext:
    def test_returns_results(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        results = get_services().searcher.search("question")
        assert len(results) == 1
        mock_svc.embedder.embed.assert_called_once_with("question")

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

    def test_expansion_merges_results(self, mock_svc):
        original = _make_result(source="a.md", chunk_index=0)
        expanded = _make_result(source="b.md", chunk_index=0)
        mock_svc.store.search.side_effect = [[original], [expanded]]
        mock_svc.embedder.embed.return_value = [0.1] * 768
        mock_svc.provider.chat.return_value = "kubernetes deployment details"
        results = get_services().searcher.search("kubernetes deployment")
        assert len(results) == 2
        sources = {r.source for r in results}
        assert "a.md" in sources
        assert "b.md" in sources

    def test_expansion_deduplicates(self, mock_svc):
        same = _make_result(source="a.md", chunk_index=0)
        mock_svc.store.search.side_effect = [[same], [same]]
        mock_svc.embedder.embed.return_value = [0.1] * 768
        mock_svc.provider.chat.return_value = "kubernetes deployment details"
        results = get_services().searcher.search("kubernetes deployment")
        assert len(results) == 1


class TestExpandQuery:
    _QUESTION_VEC = [0.1] * 768  # matches mock_svc.embedder default

    def test_returns_variants(self, mock_svc):
        mock_svc.provider.chat.return_value = (
            "explain how X works in detail\nexplain the purpose of X"
        )
        variants = get_services().searcher._expand_query("explain X in detail", self._QUESTION_VEC)
        assert len(variants) == 2
        for text, vec in variants:
            assert isinstance(text, str)
            assert len(vec) == 768

    def test_caps_at_three(self, mock_svc):
        mock_svc.provider.chat.return_value = "A\nB\nC\nD\nE"
        variants = get_services().searcher._expand_query("q", self._QUESTION_VEC)
        assert len(variants) == 3

    def test_returns_empty_on_error(self, mock_svc):
        mock_svc.provider.chat.side_effect = RuntimeError("no provider")
        assert get_services().searcher._expand_query("q", self._QUESTION_VEC) == []

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
            mock_svc.embedder.embed.assert_not_called()
        finally:
            cfg.query_expansion_count = old

    def test_returns_empty_on_non_string(self, mock_svc):
        mock_svc.provider.chat.return_value = iter(["stream"])
        assert get_services().searcher._expand_query("q", self._QUESTION_VEC) == []


class TestAskRaw:
    def test_returns_structured_result(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(chunk="oil is 5 quarts")]
        mock_svc.provider.chat.return_value = "5 quarts."
        result = get_services().searcher.ask_raw("oil capacity?")
        assert result.answer == "5 quarts."
        assert len(result.sources) == 1
        assert result.sources[0].source == "test.pdf"

    def test_no_results(self, mock_svc):
        mock_svc.store.search.return_value = []
        result = get_services().searcher.ask_raw("anything")
        assert "No relevant documents" in result.answer
        assert result.sources == []

    def test_ask_raw_with_history(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = "answer"
        history = [{"role": "user", "content": "prev"}]
        get_services().searcher.ask_raw("new q", history=history)
        messages = mock_svc.provider.chat.call_args[0][0]
        assert len(messages) == 3  # system + history + user

    def test_ask_raw_strips_think_tags(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = "<think>reasoning</think>The answer is 42."
        result = get_services().searcher.ask_raw("question")
        assert "<think>" not in result.answer
        assert result.answer == "The answer is 42."

    def test_ask_raw_preserves_think_tags_when_show_reasoning(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = "<think>reasoning</think>The answer is 42."
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
        mock_svc.provider.chat.return_value = "The oil capacity is 5 quarts."
        answer = get_services().searcher.ask("oil capacity?")
        assert "5 quarts" in answer
        assert "Sources:" in answer
        assert "test.pdf" in answer

    def test_no_results_message(self, mock_svc):
        mock_svc.store.search.return_value = []
        answer = get_services().searcher.ask("anything")
        assert "No relevant documents" in answer

    def test_ask_with_history(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = "answer"
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

    def test_empty_results_yields_message(self, mock_svc):
        mock_svc.store.search.return_value = []
        stream_tokens = list(get_services().searcher.ask_stream("anything"))
        assert any("No relevant documents" in st.content for st in stream_tokens)

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
        combined = "".join(st.content for st in stream_tokens if not st.is_reasoning)
        assert "reasoning" not in combined
        assert "answer" in combined


class TestGenerationOptions:
    def test_ask_raw_passes_options(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = "answer"
        opts = {"temperature": 0.3, "seed": 42}
        get_services().searcher.ask_raw("q", options=opts)
        assert mock_svc.provider.chat.call_args[1]["options"] == opts

    def test_ask_raw_defaults_to_cfg_options(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = "answer"
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
        mock_svc.provider.chat.return_value = "answer"
        opts = {"num_ctx": 4096}
        get_services().searcher.ask("q", options=opts)
        assert mock_svc.provider.chat.call_args[1]["options"] == opts

    def test_ask_raw_empty_options_passes_none(self, mock_svc):
        """When cfg has no generation options set, passes None to provider."""
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = "answer"
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
    def test_skips_when_confident(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [
            _make_result(relevance_score=0.9),
            _make_result(relevance_score=0.5),
        ]
        assert get_services().searcher._should_skip_expansion("test query") is True

    def test_does_not_skip_when_low_score(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [
            _make_result(relevance_score=0.5),
            _make_result(relevance_score=0.4),
        ]
        assert get_services().searcher._should_skip_expansion("test query") is False

    def test_does_not_skip_when_close_gap(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [
            _make_result(relevance_score=0.85),
            _make_result(relevance_score=0.82),
        ]
        assert get_services().searcher._should_skip_expansion("test query") is False

    def test_skips_with_single_confident_result(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [_make_result(relevance_score=0.9)]
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
        mock_svc.provider.chat.return_value = "hypothetical document about X"
        mock_svc.store.search.return_value = [_make_result()]
        results = get_services().searcher._hyde_search("explain X", top_k=5)
        assert len(results) >= 1

    def test_returns_empty_on_error(self, mock_svc):
        mock_svc.provider.chat.side_effect = RuntimeError("fail")
        assert get_services().searcher._hyde_search("test", top_k=5) == []

    def test_returns_empty_on_non_string(self, mock_svc):
        mock_svc.provider.chat.return_value = iter(["stream"])
        assert get_services().searcher._hyde_search("test", top_k=5) == []

    def test_returns_empty_on_blank(self, mock_svc):
        mock_svc.provider.chat.return_value = "   "
        assert get_services().searcher._hyde_search("test", top_k=5) == []


class TestTemporalFilter:
    def test_filters_by_date(self, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {"filename": "old.md", "ingested_at": "2025-01-01T00:00:00+00:00"},
            {"filename": "new.md", "ingested_at": "2026-03-22T12:00:00+00:00"},
        ]
        results = [
            _make_result(source="old.md"),
            _make_result(source="new.md"),
        ]
        filtered = get_services().searcher._apply_temporal_filter(results, "recent changes")
        assert any(r.source == "new.md" for r in filtered)

    def test_no_temporal_keyword_passes_through(self, mock_svc):
        results = [_make_result()]
        searcher = get_services().searcher
        assert searcher._apply_temporal_filter(results, "how does auth work") == results

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


class TestSearchStructured:
    def test_term_mode(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [_make_result()]
        results = get_services().searcher._search_structured("term", "test query", 5)
        assert len(results) == 1
        mock_svc.store.bm25_probe.assert_called_once_with("test query", top_k=5)

    def test_vec_mode(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        results = get_services().searcher._search_structured("vec", "semantic query", 5)
        assert len(results) == 1

    def test_hyde_mode(self, mock_svc):
        mock_svc.provider.chat.return_value = "hypothetical doc"
        mock_svc.store.search.return_value = [_make_result()]
        results = get_services().searcher._search_structured("hyde", "vague question", 5)
        assert len(results) == 1

    def test_unknown_mode_returns_empty(self, mock_svc):
        assert get_services().searcher._search_structured("unknown", "test", 5) == []


class TestSearchContextIntegration:
    def test_structured_term_mode(self, mock_svc):
        mock_svc.store.bm25_probe.return_value = [_make_result()]
        results = get_services().searcher.search("term: kubernetes pods")
        mock_svc.store.bm25_probe.assert_called_once()
        assert len(results) >= 1

    def test_skips_expansion_when_confident(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.store.bm25_probe.return_value = [
            _make_result(relevance_score=0.9),
            _make_result(relevance_score=0.5),
        ]
        results = get_services().searcher.search("exact match query")
        # Provider.chat should NOT be called for expansion
        mock_svc.provider.chat.assert_not_called()
        assert len(results) >= 1

    def test_hyde_merges_results(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(source="normal.md")]
        mock_svc.provider.chat.return_value = "hypothetical doc"
        old = cfg.hyde
        cfg.hyde = True
        try:
            results = get_services().searcher.search("vague question")
            sources = {r.source for r in results}
            assert "normal.md" in sources
        finally:
            cfg.hyde = old

    def test_hyde_adds_unique_results_with_distance_adjustment(self, mock_svc):
        """HyDE results not seen in normal search are added with adjusted distance."""
        normal_result = _make_result(source="normal.md", chunk_index=0)
        hyde_only_result = _make_result(source="hyde.md", chunk_index=0, distance=0.8)
        mock_svc.store.search.side_effect = [
            [normal_result],
            [hyde_only_result],
        ]
        mock_svc.provider.chat.return_value = "hypothetical document"
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
            assert hyde_r.distance == pytest.approx(0.8 / 0.5)
        finally:
            cfg.query_expansion_count = 3
            cfg.hyde = False
            cfg.hyde_weight = 0.7


class TestAskRawWithReranker:
    def test_reranker_called_when_configured(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = "answer"
        mock_svc.reranker.rerank.return_value = [_make_result()]
        old = cfg.reranker_model
        cfg.reranker_model = "test-reranker"
        try:
            result = get_services().searcher.ask_raw("question")
            mock_svc.reranker.rerank.assert_called_once()
            assert result.answer == "answer"
        finally:
            cfg.reranker_model = old


class TestAskStreamWithReranker:
    def test_reranker_called_when_configured(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        mock_svc.provider.chat.return_value = iter(["token"])
        mock_svc.reranker.rerank.return_value = [_make_result()]
        old = cfg.reranker_model
        cfg.reranker_model = "test-reranker"
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

    def test_boost_skipped_when_disabled(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(distance=0.5)]
        cfg.query_expansion_count = 0
        try:
            results = get_services().searcher.search("python code")
            mock_svc.concepts.boost_results.assert_not_called()
            assert results[0].distance == 0.5
        finally:
            cfg.query_expansion_count = 3

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
        mock_svc.provider.chat.return_value = "variant query about python"
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
        assert "wiki/summaries/doc.md" in result
        assert "page 3" in result

    def test_wiki_chunk_without_citations(self):
        r = _make_result(
            source="wiki/summaries/doc.md",
            content_type="text",
            chunk_type="wiki",
        )
        result = format_source(r)
        assert "wiki/summaries/doc.md" in result


class TestPreferWiki:
    def test_prefers_wiki_over_raw(self):
        results = [
            _make_result(source="wiki/summaries/doc.md", chunk_type="wiki"),
            _make_result(source="doc.md", chunk_type="raw"),
        ]
        filtered = prefer_wiki(results)
        assert len(filtered) == 1
        assert filtered[0].chunk_type == "wiki"

    def test_keeps_raw_when_no_wiki(self):
        results = [
            _make_result(source="doc.md", chunk_type="raw"),
            _make_result(source="other.md", chunk_type="raw"),
        ]
        assert prefer_wiki(results) == results

    def test_keeps_raw_without_wiki_coverage(self):
        results = [
            _make_result(source="wiki/summaries/doc.md", chunk_type="wiki"),
            _make_result(source="other.md", chunk_type="raw"),
        ]
        filtered = prefer_wiki(results)
        assert len(filtered) == 2

    def test_empty_input(self):
        assert prefer_wiki([]) == []

    def test_nested_path_matching(self):
        """Raw source "subdir/doc.md" maps to wiki slug "subdir--doc"."""
        results = [
            _make_result(source="wiki/summaries/subdir--doc.md", chunk_type="wiki"),
            _make_result(source="subdir/doc.md", chunk_type="raw"),
        ]
        filtered = prefer_wiki(results)
        assert len(filtered) == 1
        assert filtered[0].chunk_type == "wiki"

    def test_deeply_nested_path_matching(self):
        """Raw source "a/b/c.md" maps to wiki slug "a--b--c"."""
        results = [
            _make_result(source="wiki/summaries/a--b--c.md", chunk_type="wiki"),
            _make_result(source="a/b/c.md", chunk_type="raw"),
        ]
        filtered = prefer_wiki(results)
        assert len(filtered) == 1
        assert filtered[0].chunk_type == "wiki"


class TestPreferWikiGuard:
    """prefer_wiki should only run in build_rag_context when wiki is enabled."""

    def test_prefer_wiki_skipped_when_wiki_disabled(self, mock_svc):
        wiki_chunk = _make_result(source="wiki/summaries/doc.md", chunk_type="wiki")
        raw_chunk = _make_result(source="doc.md", chunk_type="raw")
        mock_svc.store.search.return_value = [wiki_chunk, raw_chunk]
        old_wiki = cfg.wiki
        cfg.wiki = False
        try:
            result = get_services().searcher.build_rag_context("question")
            assert result is not None
            chunks, _ = result
            # Both chunks survive — prefer_wiki was NOT applied
            assert len(chunks) == 2
        finally:
            cfg.wiki = old_wiki

    def test_prefer_wiki_applied_when_wiki_enabled(self, mock_svc):
        wiki_chunk = _make_result(source="wiki/summaries/doc.md", chunk_type="wiki")
        raw_chunk = _make_result(source="doc.md", chunk_type="raw")
        mock_svc.store.search.return_value = [wiki_chunk, raw_chunk]
        old_wiki = cfg.wiki
        cfg.wiki = True
        try:
            result = get_services().searcher.build_rag_context("question")
            assert result is not None
            chunks, _ = result
            # Only wiki chunk survives — prefer_wiki removed the raw duplicate
            assert len(chunks) == 1
            assert chunks[0].chunk_type == "wiki"
        finally:
            cfg.wiki = old_wiki


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
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher._search_structured("wiki", "test", 5)
        mock_svc.store.search.assert_called_once()
        assert mock_svc.store.search.call_args[1]["chunk_type"] == "wiki"

    def test_raw_mode_passes_chunk_type(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result()]
        get_services().searcher._search_structured("raw", "test", 5)
        mock_svc.store.search.assert_called_once()
        assert mock_svc.store.search.call_args[1]["chunk_type"] == "raw"


class TestDirectMessagesNoEmbed:
    def test_builds_system_history_user(self, mock_svc):
        """_direct_messages builds [system, ...history, user] when no embedding."""
        searcher = get_services().searcher
        history = [
            {"role": "user", "content": "prev"},
            {"role": "assistant", "content": "prev answer"},
        ]
        msgs = searcher._direct_messages("new question", history=history)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "prev"
        assert msgs[2]["content"] == "prev answer"
        assert msgs[3]["content"] == "new question"

    def test_no_history(self, mock_svc):
        msgs = get_services().searcher._direct_messages("q")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestAskRawNoEmbed:
    def test_direct_llm_when_no_embedding(self, mock_svc):
        """ask_raw without embedding calls LLM directly with warning prefix."""
        mock_svc.embedder.embedding_available.return_value = False
        mock_svc.provider.chat.return_value = "direct answer"

        searcher = Searcher(
            cfg,
            mock_svc.provider,
            mock_svc.store,
            mock_svc.embedder,
            mock_svc.reranker,
            mock_svc.concepts,
        )
        result = searcher.ask_raw("hello")
        assert "Chat only" in result.answer
        assert "direct answer" in result.answer
        assert result.sources == []

    def test_no_embed_strips_think_tags(self, mock_svc):
        """ask_raw no-embed path strips <think> tags."""
        mock_svc.embedder.embedding_available.return_value = False
        mock_svc.provider.chat.return_value = "<think>inner thought</think>direct answer"

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


class TestAskStreamNoEmbed:
    def test_streams_directly_when_no_embedding(self, mock_svc):
        """ask_stream without embedding streams from LLM directly."""
        mock_svc.embedder.embedding_available.return_value = False
        mock_svc.provider.chat.return_value = iter(["chunk1", "chunk2"])

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
        assert "Chat only" in combined
        assert "chunk1" in combined
        assert "chunk2" in combined

    def test_stream_handles_connection_error(self, mock_svc):
        """ask_stream without embedding handles ConnectionError gracefully."""
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


class TestFilterResults:
    def test_drops_high_distance(self):
        results = [
            _make_result(source="close.pdf", distance=0.3),
            _make_result(source="far.pdf", distance=0.95, chunk_index=1),
        ]
        filtered = filter_results(results, max_distance=0.9)
        assert len(filtered) == 1
        assert filtered[0].source == "close.pdf"

    def test_drops_low_relevance_score(self):
        results = [
            _make_result(source="good.pdf", distance=None, relevance_score=0.8),
            _make_result(source="bad.pdf", distance=None, relevance_score=0.01, chunk_index=1),
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

    def test_keeps_results_at_threshold(self):
        r = _make_result(distance=0.9)
        filtered = filter_results([r], max_distance=0.9)
        assert len(filtered) == 1


class TestRelevanceWeight:
    def test_distance_based(self):
        r = _make_result(distance=0.3, relevance_score=None)
        assert _relevance_weight(r) == pytest.approx(0.7)

    def test_relevance_score_based(self):
        r = _make_result(distance=None, relevance_score=0.8)
        assert _relevance_weight(r) == pytest.approx(0.8)

    def test_neither_returns_default(self):
        r = _make_result(distance=None, relevance_score=None)
        assert _relevance_weight(r) == pytest.approx(0.5)

    def test_relevance_score_takes_priority(self):
        r = _make_result(distance=0.3, relevance_score=0.9)
        assert _relevance_weight(r) == pytest.approx(0.9)

    def test_clamps_high_relevance(self):
        r = _make_result(distance=None, relevance_score=1.5)
        assert _relevance_weight(r) == pytest.approx(1.0)

    def test_clamps_negative_distance(self):
        r = _make_result(distance=1.5, relevance_score=None)
        assert _relevance_weight(r) == pytest.approx(0.0)


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


class TestExtractCitedIndices:
    def test_extracts_multiple(self):
        assert _extract_cited_indices("See [1] and [3].") == {1, 3}

    def test_no_citations(self):
        assert _extract_cited_indices("The answer is yes.") == set()

    def test_deduplicates(self):
        assert _extract_cited_indices("[1] and [1] again.") == {1}


class TestAskCitesOnlyUsedSources:
    def test_ask_cites_only_referenced(self, mock_svc):
        r1 = _make_result(source="used.pdf", chunk="oil info", chunk_index=0)
        r2 = _make_result(source="unused.pdf", chunk="unrelated", chunk_index=1)
        mock_svc.store.search.return_value = [r1, r2]
        mock_svc.provider.chat.return_value = "Oil is 5 quarts [1]."
        answer = get_services().searcher.ask("oil capacity?")
        assert "used.pdf" in answer
        assert "unused.pdf" not in answer

    def test_ask_falls_back_to_all_sources_when_no_refs(self, mock_svc):
        r1 = _make_result(source="a.pdf", chunk="oil info", chunk_index=0)
        mock_svc.store.search.return_value = [r1]
        mock_svc.provider.chat.return_value = "Oil is 5 quarts."
        answer = get_services().searcher.ask("oil capacity?")
        assert "a.pdf" in answer

    def test_ask_strips_llm_citation_block(self, mock_svc):
        mock_svc.store.search.return_value = [_make_result(chunk="oil info")]
        mock_svc.provider.chat.return_value = "5 quarts [1].\n\nKey sources:\n- [1] test.pdf"
        answer = get_services().searcher.ask("oil capacity?")
        assert "Key sources" not in answer
        assert answer.count("Sources:") == 1


class TestAskStreamCitesOnlyUsedSources:
    def test_stream_cites_only_referenced(self, mock_svc):
        r1 = _make_result(source="used.pdf", chunk="oil info", chunk_index=0)
        r2 = _make_result(source="unused.pdf", chunk="unrelated", chunk_index=1)
        mock_svc.store.search.return_value = [r1, r2]
        mock_svc.provider.chat.return_value = iter(["Oil is 5 quarts ", "[1]."])
        tokens = list(get_services().searcher.ask_stream("oil capacity?"))
        combined = "".join(st.content for st in tokens)
        assert "used.pdf" in combined
        assert "unused.pdf" not in combined

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
