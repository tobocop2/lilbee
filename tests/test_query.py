"""Tests for the RAG query pipeline (mocked -- no live server needed)."""

from unittest import mock

import pytest

from lilbee.config import cfg
from lilbee.query import (
    _ask,
    _ask_raw,
    _ask_stream,
    _expand_query,
    _hyde_search,
    _search_context,
    _search_structured,
    build_context,
    deduplicate_sources,
    format_source,
    sort_by_relevance,
)
from lilbee.store import SearchChunk


@pytest.fixture(autouse=True)
def _disable_concepts():
    """Disable concept graph by default in query tests to avoid spaCy loads."""
    old = cfg.concept_graph
    cfg.concept_graph = False
    yield
    cfg.concept_graph = old


def _make_provider(chat_return=None):
    """Create a mock LLM provider."""
    p = mock.MagicMock()
    p.chat.return_value = chat_return
    p.embed.return_value = [[0.1] * 768]
    return p


def _make_result(
    source="test.pdf",
    content_type="pdf",
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
    @mock.patch("lilbee.query._expand_query", return_value=[])
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_returns_results(self, mock_embed, mock_search, mock_expand):
        provider = _make_provider()
        results = _search_context("question", config=cfg, provider=provider)
        assert len(results) == 1
        mock_embed.assert_called_once_with("question", provider=provider, config=cfg)

    @mock.patch("lilbee.query._expand_query", return_value=[])
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_passes_query_text(self, mock_embed, mock_search, mock_expand):
        provider = _make_provider()
        _search_context("my question", config=cfg, provider=provider)
        mock_search.assert_called_once()
        assert mock_search.call_args[1]["query_text"] == "my question"

    @mock.patch("lilbee.store.search")
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    @mock.patch("lilbee.query._expand_query", return_value=["alt query 1"])
    def test_expansion_merges_results(self, mock_expand, mock_embed, mock_search):
        original = _make_result(source="a.md", chunk_index=0)
        expanded = _make_result(source="b.md", chunk_index=0)
        mock_search.side_effect = [[original], [expanded]]
        provider = _make_provider()
        results = _search_context("question", config=cfg, provider=provider)
        assert len(results) == 2
        sources = {r.source for r in results}
        assert "a.md" in sources
        assert "b.md" in sources

    @mock.patch("lilbee.store.search")
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    @mock.patch("lilbee.query._expand_query", return_value=["alt"])
    def test_expansion_deduplicates(self, mock_expand, mock_embed, mock_search):
        same = _make_result(source="a.md", chunk_index=0)
        mock_search.side_effect = [[same], [same]]
        provider = _make_provider()
        results = _search_context("question", config=cfg, provider=provider)
        assert len(results) == 1


class TestExpandQuery:
    def test_returns_variants(self):
        provider = _make_provider("explain how X works in detail\nexplain the purpose of X")
        variants = _expand_query("explain X in detail", config=cfg, provider=provider)
        assert len(variants) == 2

    def test_caps_at_three(self):
        provider = _make_provider("A\nB\nC\nD\nE")
        assert len(_expand_query("q", config=cfg, provider=provider)) == 3

    def test_returns_empty_on_error(self):
        provider = _make_provider()
        provider.chat.side_effect = RuntimeError("no provider")
        assert _expand_query("q", config=cfg, provider=provider) == []

    def test_disabled_when_count_zero(self):
        old = cfg.query_expansion_count
        cfg.query_expansion_count = 0
        try:
            provider = _make_provider()
            assert _expand_query("anything", config=cfg, provider=provider) == []
        finally:
            cfg.query_expansion_count = old

    def test_returns_empty_on_non_string(self):
        provider = _make_provider(iter(["stream"]))  # not a string
        assert _expand_query("q", config=cfg, provider=provider) == []


class TestAskRaw:
    @mock.patch("lilbee.store.search", return_value=[_make_result(chunk="oil is 5 quarts")])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_returns_structured_result(self, mock_embed, mock_search):
        provider = _make_provider("5 quarts.")
        result = _ask_raw("oil capacity?", config=cfg, provider=provider)
        assert result.answer == "5 quarts."
        assert len(result.sources) == 1
        assert result.sources[0].source == "test.pdf"

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_no_results(self, mock_embed, mock_search):
        provider = _make_provider()
        result = _ask_raw("anything", config=cfg, provider=provider)
        assert "No relevant documents" in result.answer
        assert result.sources == []

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_raw_with_history(self, mock_embed, mock_search):
        provider = _make_provider("answer")
        history = [{"role": "user", "content": "prev"}]
        _ask_raw("new q", history=history, config=cfg, provider=provider)
        messages = provider.chat.call_args[0][0]
        assert len(messages) == 3  # system + history + user


class TestAsk:
    @mock.patch("lilbee.store.search", return_value=[_make_result(chunk="oil is 5 quarts")])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_returns_answer_with_citations(self, mock_embed, mock_search):
        provider = _make_provider("The oil capacity is 5 quarts.")
        answer = _ask("oil capacity?", config=cfg, provider=provider)
        assert "5 quarts" in answer
        assert "Sources:" in answer
        assert "test.pdf" in answer

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_no_results_message(self, mock_embed, mock_search):
        provider = _make_provider()
        answer = _ask("anything", config=cfg, provider=provider)
        assert "No relevant documents" in answer

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_with_history(self, mock_embed, mock_search):
        provider = _make_provider("answer")
        history = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        _ask("new q", history=history, config=cfg, provider=provider)
        messages = provider.chat.call_args[0][0]
        # System + 2 history + user = 4
        assert len(messages) == 4
        assert messages[1]["content"] == "prev q"


class TestAskStream:
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_yields_tokens_then_citations(self, mock_embed, mock_search):
        provider = _make_provider(iter(["Hello", " world"]))
        stream_tokens = list(_ask_stream("test", config=cfg, provider=provider))
        combined = "".join(st.content for st in stream_tokens)
        assert "Hello world" in combined
        assert "Sources:" in combined

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_empty_results_yields_message(self, mock_embed, mock_search):
        provider = _make_provider()
        stream_tokens = list(_ask_stream("anything", config=cfg, provider=provider))
        assert any("No relevant documents" in st.content for st in stream_tokens)

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_stream_with_history(self, mock_embed, mock_search):
        provider = _make_provider(iter(["response"]))
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        list(_ask_stream("new question", history=history, config=cfg, provider=provider))
        call_args = provider.chat.call_args
        messages = call_args[0][0]
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "previous question"

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_skips_empty_tokens(self, mock_embed, mock_search):
        provider = _make_provider(iter(["", "data"]))
        stream_tokens = list(_ask_stream("test", config=cfg, provider=provider))
        non_source = [st for st in stream_tokens if "Sources:" not in st.content]
        assert all(st.content != "" for st in non_source)

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_reasoning_stripped_by_default(self, mock_embed, mock_search):
        provider = _make_provider(iter(["<think>reasoning</think>answer"]))
        stream_tokens = list(_ask_stream("test", config=cfg, provider=provider))
        combined = "".join(st.content for st in stream_tokens if not st.is_reasoning)
        assert "reasoning" not in combined
        assert "answer" in combined


class TestGenerationOptions:
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_raw_passes_options(self, mock_embed, mock_search):
        provider = _make_provider("answer")
        opts = {"temperature": 0.3, "seed": 42}
        _ask_raw("q", options=opts, config=cfg, provider=provider)
        assert provider.chat.call_args[1]["options"] == opts

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_raw_defaults_to_cfg_options(self, mock_embed, mock_search):
        provider = _make_provider("answer")
        cfg.temperature = 0.7
        cfg.seed = None
        cfg.top_p = None
        cfg.top_k_sampling = None
        cfg.repeat_penalty = None
        cfg.num_ctx = None
        try:
            _ask_raw("q", config=cfg, provider=provider)
            assert provider.chat.call_args[1]["options"] == {"temperature": 0.7}
        finally:
            cfg.temperature = None

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_stream_passes_options(self, mock_embed, mock_search):
        provider = _make_provider(iter(["token"]))
        opts = {"temperature": 0.1}
        list(_ask_stream("q", options=opts, config=cfg, provider=provider))
        assert provider.chat.call_args[1]["options"] == opts

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_passes_options_through(self, mock_embed, mock_search):
        provider = _make_provider("answer")
        opts = {"num_ctx": 4096}
        _ask("q", options=opts, config=cfg, provider=provider)
        assert provider.chat.call_args[1]["options"] == opts

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_raw_empty_options_passes_none(self, mock_embed, mock_search):
        """When cfg has no generation options set, passes None to provider."""
        provider = _make_provider("answer")
        cfg.temperature = None
        cfg.top_p = None
        cfg.top_k_sampling = None
        cfg.repeat_penalty = None
        cfg.num_ctx = None
        cfg.seed = None
        _ask_raw("q", config=cfg, provider=provider)
        assert provider.chat.call_args[1]["options"] is None


class TestAskStreamError:
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_stream_handles_disconnect(self, mock_embed, mock_search):
        def failing_stream():
            yield "partial"
            raise ConnectionError("lost connection")

        provider = _make_provider(failing_stream())
        stream_tokens = list(_ask_stream("test", config=cfg, provider=provider))
        combined = "".join(st.content for st in stream_tokens)
        assert "partial" in combined
        assert "Connection lost" in combined


class TestProviderError:
    """ProviderError from the provider should propagate."""

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_raw_provider_error(self, mock_embed, mock_search):
        from lilbee.providers.base import ProviderError

        provider = _make_provider()
        provider.chat.side_effect = ProviderError("model 'bad' not found")
        with pytest.raises(ProviderError, match="not found"):
            _ask_raw("hello", config=cfg, provider=provider)

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_stream_provider_error(self, mock_embed, mock_search):
        from lilbee.providers.base import ProviderError

        provider = _make_provider()
        provider.chat.side_effect = ProviderError("model 'bad' not found")
        with pytest.raises(ProviderError, match="not found"):
            list(_ask_stream("hello", config=cfg, provider=provider))

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_ask_stream_provider_error_mid_stream(self, mock_embed, mock_search):
        """ProviderError raised during iteration should propagate."""
        from lilbee.providers.base import ProviderError

        def failing_mid_stream():
            yield "partial"
            raise ProviderError("model 'bad' not found")

        provider = _make_provider(failing_mid_stream())
        with pytest.raises(ProviderError, match="not found"):
            list(_ask_stream("hello", config=cfg, provider=provider))


class TestApplyGuardrails:
    def test_filters_drifted_variants(self):
        from lilbee.query import _apply_guardrails

        variants = ["completely unrelated topic", "explain kubernetes deployment"]
        result = _apply_guardrails(variants, "explain kubernetes deployment")
        assert "completely unrelated topic" not in result
        assert "explain kubernetes deployment" in result

    def test_keeps_overlapping_variants(self):
        from lilbee.query import _apply_guardrails

        variants = ["kubernetes deployment steps", "deploying kubernetes clusters"]
        result = _apply_guardrails(variants, "kubernetes deployment guide")
        assert len(result) >= 1

    def test_returns_all_when_guardrails_disabled(self):
        from lilbee.query import _apply_guardrails

        old = cfg.expansion_guardrails
        cfg.expansion_guardrails = False
        try:
            result = _apply_guardrails(["anything"], "question")
            assert result == ["anything"]
        finally:
            cfg.expansion_guardrails = old

    def test_empty_variants(self):
        from lilbee.query import _apply_guardrails

        assert _apply_guardrails([], "question") == []

    def test_empty_original_tokens(self):
        from lilbee.query import _apply_guardrails

        result = _apply_guardrails(["variant"], "a the is")
        assert result == ["variant"]

    def test_empty_variant_tokens(self):
        from lilbee.query import _apply_guardrails

        result = _apply_guardrails(["a the is", "real content here"], "real content here")
        assert len(result) == 1


class TestTokenizeQuery:
    def test_removes_stop_words(self):
        from lilbee.query import _tokenize_query

        result = _tokenize_query("how does the system work")
        assert "how" not in result
        assert "does" not in result
        assert "the" not in result
        assert "system" in result
        assert "work" in result

    def test_lowercases(self):
        from lilbee.query import _tokenize_query

        result = _tokenize_query("Kubernetes Deployment")
        assert "kubernetes" in result

    def test_removes_single_chars(self):
        from lilbee.query import _tokenize_query

        result = _tokenize_query("a b c real word")
        assert "real" in result
        assert "word" in result
        assert "b" not in result


class TestSelectContext:
    def test_selects_covering_chunks(self):
        from lilbee.query import select_context

        chunks = [
            _make_result(chunk="kubernetes deployment guide", source="a.md"),
            _make_result(chunk="kubernetes networking setup", source="b.md"),
            _make_result(chunk="deployment automation tools", source="c.md"),
        ]
        result = select_context(chunks, "kubernetes deployment networking", max_sources=2)
        assert len(result) == 2
        texts = " ".join(r.chunk for r in result)
        assert "kubernetes" in texts
        assert "networking" in texts

    def test_passes_through_when_under_max(self):
        from lilbee.query import select_context

        chunks = [_make_result(chunk="only one")]
        result = select_context(chunks, "anything", max_sources=5)
        assert len(result) == 1

    def test_empty_query_returns_top_n(self):
        from lilbee.query import select_context

        chunks = [_make_result(chunk=f"chunk {i}") for i in range(10)]
        result = select_context(chunks, "a the is", max_sources=3)
        assert len(result) == 3

    def test_stops_on_full_coverage(self):
        from lilbee.query import select_context

        chunks = [
            _make_result(chunk="alpha beta gamma delta", source="a.md"),
            _make_result(chunk="alpha beta gamma delta", source="b.md"),
            _make_result(chunk="alpha beta gamma delta", source="c.md"),
            _make_result(chunk="alpha beta gamma delta", source="d.md"),
            _make_result(chunk="alpha beta gamma delta", source="e.md"),
        ]
        result = select_context(chunks, "alpha beta gamma delta", max_sources=3)
        assert len(result) == 1  # first chunk covers everything

    def test_stops_on_zero_gain(self):
        from lilbee.query import select_context

        chunks = [
            _make_result(chunk="alpha beta unique1", source="a.md"),
            _make_result(chunk="alpha beta unique1", source="b.md"),
            _make_result(chunk="alpha beta unique1", source="c.md"),
            _make_result(chunk="alpha beta unique1", source="d.md"),
            _make_result(chunk="alpha beta unique1", source="e.md"),
            _make_result(chunk="alpha beta unique1", source="f.md"),
        ]
        # Query has "alpha beta unique1 unique2" -- first chunk covers 3/4
        # Remaining chunks add 0 new terms, so selection stops early
        result = select_context(chunks, "alpha beta unique1 unique2", max_sources=5)
        # Should stop after 1 or 2 chunks since no gain after first
        assert len(result) < 5


class TestShouldSkipExpansion:
    @mock.patch("lilbee.store.bm25_probe")
    def test_skips_when_confident(self, mock_probe):
        from lilbee.query import _should_skip_expansion

        mock_probe.return_value = [
            _make_result(relevance_score=0.9),
            _make_result(relevance_score=0.5),
        ]
        assert _should_skip_expansion("test query") is True

    @mock.patch("lilbee.store.bm25_probe")
    def test_does_not_skip_when_low_score(self, mock_probe):
        from lilbee.query import _should_skip_expansion

        mock_probe.return_value = [
            _make_result(relevance_score=0.5),
            _make_result(relevance_score=0.4),
        ]
        assert _should_skip_expansion("test query") is False

    @mock.patch("lilbee.store.bm25_probe")
    def test_does_not_skip_when_close_gap(self, mock_probe):
        from lilbee.query import _should_skip_expansion

        mock_probe.return_value = [
            _make_result(relevance_score=0.85),
            _make_result(relevance_score=0.82),
        ]
        assert _should_skip_expansion("test query") is False

    @mock.patch("lilbee.store.bm25_probe")
    def test_skips_with_single_confident_result(self, mock_probe):
        from lilbee.query import _should_skip_expansion

        mock_probe.return_value = [_make_result(relevance_score=0.9)]
        assert _should_skip_expansion("test") is True

    @mock.patch("lilbee.store.bm25_probe")
    def test_does_not_skip_when_empty(self, mock_probe):
        from lilbee.query import _should_skip_expansion

        mock_probe.return_value = []
        assert _should_skip_expansion("test") is False

    def test_disabled_when_threshold_zero(self):
        from lilbee.query import _should_skip_expansion

        old = cfg.expansion_skip_threshold
        cfg.expansion_skip_threshold = 0
        try:
            assert _should_skip_expansion("test") is False
        finally:
            cfg.expansion_skip_threshold = old


class TestParseStructuredQuery:
    def test_term_prefix(self):
        from lilbee.query import _parse_structured_query

        mode, query = _parse_structured_query("term: kubernetes pods")
        assert mode == "term"
        assert query == "kubernetes pods"

    def test_vec_prefix(self):
        from lilbee.query import _parse_structured_query

        mode, query = _parse_structured_query("vec: how does auth work")
        assert mode == "vec"
        assert "auth" in query

    def test_hyde_prefix(self):
        from lilbee.query import _parse_structured_query

        mode, _query = _parse_structured_query("hyde: explain caching")
        assert mode == "hyde"

    def test_no_prefix(self):
        from lilbee.query import _parse_structured_query

        mode, query = _parse_structured_query("normal question")
        assert mode is None
        assert query == "normal question"

    def test_case_insensitive(self):
        from lilbee.query import _parse_structured_query

        mode, _ = _parse_structured_query("TERM: test")
        assert mode == "term"


class TestHydeSearch:
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_returns_results(self, mock_embed, mock_search):
        provider = _make_provider("hypothetical document about X")
        results = _hyde_search("explain X", top_k=5, config=cfg, provider=provider)
        assert len(results) >= 1

    def test_returns_empty_on_error(self):
        provider = _make_provider()
        provider.chat.side_effect = RuntimeError("fail")
        assert _hyde_search("test", top_k=5, config=cfg, provider=provider) == []

    def test_returns_empty_on_non_string(self):
        provider = _make_provider(iter(["stream"]))
        assert _hyde_search("test", top_k=5, config=cfg, provider=provider) == []

    def test_returns_empty_on_blank(self):
        provider = _make_provider("   ")
        assert _hyde_search("test", top_k=5, config=cfg, provider=provider) == []


class TestTemporalFilter:
    @mock.patch("lilbee.store.get_sources")
    def test_filters_by_date(self, mock_sources):
        from lilbee.query import _apply_temporal_filter

        mock_sources.return_value = [
            {"filename": "old.md", "ingested_at": "2025-01-01T00:00:00+00:00"},
            {"filename": "new.md", "ingested_at": "2026-03-22T12:00:00+00:00"},
        ]
        results = [
            _make_result(source="old.md"),
            _make_result(source="new.md"),
        ]
        filtered = _apply_temporal_filter(results, "recent changes")
        assert any(r.source == "new.md" for r in filtered)

    def test_no_temporal_keyword_passes_through(self):
        from lilbee.query import _apply_temporal_filter

        results = [_make_result()]
        assert _apply_temporal_filter(results, "how does auth work") == results

    def test_disabled_via_config(self):
        from lilbee.query import _apply_temporal_filter

        old = cfg.temporal_filtering
        cfg.temporal_filtering = False
        try:
            results = [_make_result()]
            assert _apply_temporal_filter(results, "recent") == results
        finally:
            cfg.temporal_filtering = old

    @mock.patch("lilbee.store.get_sources")
    def test_keeps_results_without_dates(self, mock_sources):
        from lilbee.query import _apply_temporal_filter

        mock_sources.return_value = [{"filename": "a.md", "ingested_at": ""}]
        results = [_make_result(source="a.md")]
        filtered = _apply_temporal_filter(results, "today's notes")
        assert len(filtered) == 1

    @mock.patch("lilbee.store.get_sources")
    def test_falls_back_when_nothing_matches(self, mock_sources):
        from lilbee.query import _apply_temporal_filter

        mock_sources.return_value = [
            {"filename": "old.md", "ingested_at": "2020-01-01T00:00:00+00:00"},
        ]
        results = [_make_result(source="old.md")]
        filtered = _apply_temporal_filter(results, "today's notes")
        assert len(filtered) == 1  # falls back to unfiltered

    @mock.patch("lilbee.store.get_sources")
    def test_handles_invalid_date(self, mock_sources):
        from lilbee.query import _apply_temporal_filter

        mock_sources.return_value = [{"filename": "a.md", "ingested_at": "not-a-date"}]
        results = [_make_result(source="a.md")]
        filtered = _apply_temporal_filter(results, "recent")
        assert len(filtered) == 1


class TestSearchStructured:
    @mock.patch("lilbee.store.bm25_probe", return_value=[_make_result()])
    def test_term_mode(self, mock_probe):
        provider = _make_provider()
        results = _search_structured("term", "test query", 5, config=cfg, provider=provider)
        assert len(results) == 1
        mock_probe.assert_called_once_with("test query", top_k=5)

    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_vec_mode(self, mock_embed, mock_search):
        provider = _make_provider()
        results = _search_structured("vec", "semantic query", 5, config=cfg, provider=provider)
        assert len(results) == 1

    @mock.patch("lilbee.query._hyde_search", return_value=[_make_result()])
    def test_hyde_mode(self, mock_hyde):
        provider = _make_provider()
        results = _search_structured("hyde", "vague question", 5, config=cfg, provider=provider)
        assert len(results) == 1

    def test_unknown_mode_returns_empty(self):
        provider = _make_provider()
        assert _search_structured("unknown", "test", 5, config=cfg, provider=provider) == []


class TestSearchContextIntegration:
    @mock.patch("lilbee.query._expand_query", return_value=[])
    @mock.patch("lilbee.store.bm25_probe", return_value=[])
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_structured_term_mode(self, mock_embed, mock_search, mock_probe, mock_expand):
        mock_probe.return_value = [_make_result()]
        provider = _make_provider()
        results = _search_context("term: kubernetes pods", config=cfg, provider=provider)
        mock_probe.assert_called_once()
        assert len(results) >= 1

    @mock.patch("lilbee.query._expand_query", return_value=[])
    @mock.patch("lilbee.query._should_skip_expansion", return_value=True)
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_skips_expansion_when_confident(self, mock_embed, mock_search, mock_skip, mock_expand):
        provider = _make_provider()
        results = _search_context("exact match query", config=cfg, provider=provider)
        mock_expand.assert_not_called()
        assert len(results) >= 1

    @mock.patch("lilbee.query._hyde_search", return_value=[_make_result(source="hyde.md")])
    @mock.patch("lilbee.query._expand_query", return_value=[])
    @mock.patch("lilbee.query._should_skip_expansion", return_value=False)
    @mock.patch("lilbee.store.search", return_value=[_make_result(source="normal.md")])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_hyde_merges_results(self, mock_embed, mock_search, mock_skip, mock_expand, mock_hyde):
        old = cfg.hyde
        cfg.hyde = True
        try:
            provider = _make_provider()
            results = _search_context("vague question", config=cfg, provider=provider)
            sources = {r.source for r in results}
            assert "hyde.md" in sources
        finally:
            cfg.hyde = old


class TestAskRawWithReranker:
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_reranker_called_when_configured(self, mock_embed, mock_search):
        provider = _make_provider("answer")
        old = cfg.reranker_model
        cfg.reranker_model = "test-reranker"
        try:
            with mock.patch("lilbee.reranker.rerank", return_value=[_make_result()]) as mock_rerank:
                result = _ask_raw("question", config=cfg, provider=provider)
                mock_rerank.assert_called_once()
                assert result.answer == "answer"
        finally:
            cfg.reranker_model = old


class TestAskStreamWithReranker:
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_reranker_called_when_configured(self, mock_embed, mock_search):
        provider = _make_provider(iter(["token"]))
        old = cfg.reranker_model
        cfg.reranker_model = "test-reranker"
        try:
            with mock.patch("lilbee.reranker.rerank", return_value=[_make_result()]) as mock_rerank:
                list(_ask_stream("question", config=cfg, provider=provider))
                mock_rerank.assert_called_once()
        finally:
            cfg.reranker_model = old


class TestConceptBoosting:
    @mock.patch("lilbee.store.search", return_value=[_make_result(distance=0.5)])
    @mock.patch("lilbee.store.bm25_probe", return_value=[])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_boost_applied_when_enabled(self, mock_embed, mock_bm25, mock_search):
        old = cfg.concept_graph
        cfg.concept_graph = True
        cfg.query_expansion_count = 0
        try:
            with (
                mock.patch("lilbee.concepts.get_graph", return_value=True),
                mock.patch("lilbee.concepts.extract_concepts", return_value=["python"]),
                mock.patch(
                    "lilbee.concepts.boost_results",
                    return_value=[_make_result(distance=0.3)],
                ) as mock_boost,
            ):
                provider = _make_provider()
                results = _search_context("python code", config=cfg, provider=provider)
            mock_boost.assert_called_once()
            assert results[0].distance == 0.3
        finally:
            cfg.concept_graph = old
            cfg.query_expansion_count = 3

    @mock.patch("lilbee.store.search", return_value=[_make_result(distance=0.5)])
    @mock.patch("lilbee.store.bm25_probe", return_value=[])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_boost_skipped_when_disabled(self, mock_embed, mock_bm25, mock_search):
        old = cfg.concept_graph
        cfg.concept_graph = False
        cfg.query_expansion_count = 0
        try:
            with mock.patch("lilbee.concepts.get_graph") as m_graph:
                provider = _make_provider()
                results = _search_context("python code", config=cfg, provider=provider)
            m_graph.assert_not_called()
            assert results[0].distance == 0.5
        finally:
            cfg.concept_graph = old
            cfg.query_expansion_count = 3

    @mock.patch("lilbee.store.search", return_value=[_make_result(distance=0.5)])
    @mock.patch("lilbee.store.bm25_probe", return_value=[])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_boost_failure_returns_original(self, mock_embed, mock_bm25, mock_search):
        old = cfg.concept_graph
        cfg.concept_graph = True
        cfg.query_expansion_count = 0
        try:
            with mock.patch("lilbee.concepts.get_graph", side_effect=RuntimeError("broken")):
                provider = _make_provider()
                results = _search_context("python code", config=cfg, provider=provider)
            assert results[0].distance == 0.5
        finally:
            cfg.concept_graph = old
            cfg.query_expansion_count = 3

    @mock.patch("lilbee.store.search", return_value=[_make_result(distance=0.5)])
    @mock.patch("lilbee.store.bm25_probe", return_value=[])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_boost_graph_none_returns_original(self, mock_embed, mock_bm25, mock_search):
        old = cfg.concept_graph
        cfg.concept_graph = True
        cfg.query_expansion_count = 0
        try:
            with mock.patch("lilbee.concepts.get_graph", return_value=False):
                provider = _make_provider()
                results = _search_context("python code", config=cfg, provider=provider)
            assert results[0].distance == 0.5
        finally:
            cfg.concept_graph = old
            cfg.query_expansion_count = 3


class TestConceptQueryExpansion:
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.store.bm25_probe", return_value=[])
    @mock.patch("lilbee.embedder.embed_di", return_value=[0.1] * 768)
    def test_expansion_includes_concept_terms(self, mock_embed, mock_bm25, mock_search):
        old_graph = cfg.concept_graph
        cfg.concept_graph = True
        provider = _make_provider("variant query about python")
        try:
            with (
                mock.patch("lilbee.concepts.get_graph", return_value=True),
                mock.patch(
                    "lilbee.concepts.expand_query",
                    return_value=["python web frameworks"],
                ),
            ):
                variants = _expand_query("python frameworks", config=cfg, provider=provider)
            assert "python web frameworks" in variants
        finally:
            cfg.concept_graph = old_graph

    def test_expansion_disabled_returns_empty(self):
        from lilbee.query import _concept_query_expansion

        old = cfg.concept_graph
        cfg.concept_graph = False
        try:
            result = _concept_query_expansion("test query")
            assert result == []
        finally:
            cfg.concept_graph = old

    def test_expansion_failure_returns_empty(self):
        from lilbee.query import _concept_query_expansion

        old = cfg.concept_graph
        cfg.concept_graph = True
        try:
            with mock.patch("lilbee.concepts.get_graph", side_effect=RuntimeError("broken")):
                result = _concept_query_expansion("test query")
            assert result == []
        finally:
            cfg.concept_graph = old

    def test_expansion_graph_none_returns_empty(self):
        from lilbee.query import _concept_query_expansion

        old = cfg.concept_graph
        cfg.concept_graph = True
        try:
            with mock.patch("lilbee.concepts.get_graph", return_value=None):
                result = _concept_query_expansion("test query")
            assert result == []
        finally:
            cfg.concept_graph = old
