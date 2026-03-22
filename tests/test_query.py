"""Tests for the RAG query pipeline (mocked — no live server needed)."""

from unittest import mock

import pytest

from lilbee.query import (
    ask,
    ask_raw,
    ask_stream,
    build_context,
    deduplicate_sources,
    format_source,
    search_context,
    sort_by_relevance,
)
from lilbee.store import SearchChunk


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
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_returns_results(self, mock_embed, mock_search, mock_expand):
        results = search_context("question")
        assert len(results) == 1
        mock_embed.assert_called_once_with("question")

    @mock.patch("lilbee.query._expand_query", return_value=[])
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_passes_query_text(self, mock_embed, mock_search, mock_expand):
        search_context("my question")
        mock_search.assert_called_once()
        assert mock_search.call_args[1]["query_text"] == "my question"

    @mock.patch("lilbee.store.search")
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    @mock.patch("lilbee.query._expand_query", return_value=["alt query 1"])
    def test_expansion_merges_results(self, mock_expand, mock_embed, mock_search):
        original = _make_result(source="a.md", chunk_index=0)
        expanded = _make_result(source="b.md", chunk_index=0)
        mock_search.side_effect = [[original], [expanded]]
        results = search_context("question")
        assert len(results) == 2
        sources = {r.source for r in results}
        assert "a.md" in sources
        assert "b.md" in sources

    @mock.patch("lilbee.store.search")
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    @mock.patch("lilbee.query._expand_query", return_value=["alt"])
    def test_expansion_deduplicates(self, mock_expand, mock_embed, mock_search):
        same = _make_result(source="a.md", chunk_index=0)
        mock_search.side_effect = [[same], [same]]
        results = search_context("question")
        assert len(results) == 1


class TestExpandQuery:
    @mock.patch("lilbee.query.get_provider")
    def test_returns_variants(self, mock_get_provider):
        mock_provider = mock.MagicMock()
        mock_provider.chat.return_value = "explain how X works in detail\nexplain the purpose of X"
        mock_get_provider.return_value = mock_provider
        from lilbee.query import _expand_query

        variants = _expand_query("explain X in detail")
        assert len(variants) == 2

    @mock.patch("lilbee.query.get_provider")
    def test_caps_at_three(self, mock_get_provider):
        mock_provider = mock.MagicMock()
        mock_provider.chat.return_value = "A\nB\nC\nD\nE"
        mock_get_provider.return_value = mock_provider
        from lilbee.query import _expand_query

        assert len(_expand_query("q")) == 3

    @mock.patch("lilbee.query.get_provider")
    def test_returns_empty_on_error(self, mock_get_provider):
        mock_get_provider.side_effect = RuntimeError("no provider")
        from lilbee.query import _expand_query

        assert _expand_query("q") == []

    def test_disabled_when_count_zero(self):
        from lilbee.config import cfg
        from lilbee.query import _expand_query

        old = cfg.query_expansion_count
        cfg.query_expansion_count = 0
        try:
            assert _expand_query("anything") == []
        finally:
            cfg.query_expansion_count = old

    @mock.patch("lilbee.query.get_provider")
    def test_returns_empty_on_non_string(self, mock_get_provider):
        mock_provider = mock.MagicMock()
        mock_provider.chat.return_value = iter(["stream"])  # not a string
        mock_get_provider.return_value = mock_provider
        from lilbee.query import _expand_query

        assert _expand_query("q") == []


class TestAskRaw:
    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result(chunk="oil is 5 quarts")])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_returns_structured_result(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = "5 quarts."
        result = ask_raw("oil capacity?")
        assert result.answer == "5 quarts."
        assert len(result.sources) == 1
        assert result.sources[0].source == "test.pdf"

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_no_results(self, mock_embed, mock_search):
        result = ask_raw("anything")
        assert "No relevant documents" in result.answer
        assert result.sources == []

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_with_history(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = "answer"
        history = [{"role": "user", "content": "prev"}]
        ask_raw("new q", history=history)
        messages = mock_provider.return_value.chat.call_args[0][0]
        assert len(messages) == 3  # system + history + user


class TestAsk:
    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result(chunk="oil is 5 quarts")])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_returns_answer_with_citations(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = "The oil capacity is 5 quarts."
        answer = ask("oil capacity?")
        assert "5 quarts" in answer
        assert "Sources:" in answer
        assert "test.pdf" in answer

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_no_results_message(self, mock_embed, mock_search):
        answer = ask("anything")
        assert "No relevant documents" in answer

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_with_history(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = "answer"
        history = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        ask("new q", history=history)
        messages = mock_provider.return_value.chat.call_args[0][0]
        # System + 2 history + user = 4
        assert len(messages) == 4
        assert messages[1]["content"] == "prev q"


class TestAskStream:
    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_yields_tokens_then_citations(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = iter(["Hello", " world"])
        stream_tokens = list(ask_stream("test"))
        combined = "".join(st.content for st in stream_tokens)
        assert "Hello world" in combined
        assert "Sources:" in combined

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_empty_results_yields_message(self, mock_embed, mock_search):
        stream_tokens = list(ask_stream("anything"))
        assert any("No relevant documents" in st.content for st in stream_tokens)

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_stream_with_history(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = iter(["response"])
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        list(ask_stream("new question", history=history))
        call_args = mock_provider.return_value.chat.call_args
        messages = call_args[0][0]
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "previous question"

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_skips_empty_tokens(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = iter(["", "data"])
        stream_tokens = list(ask_stream("test"))
        non_source = [st for st in stream_tokens if "Sources:" not in st.content]
        assert all(st.content != "" for st in non_source)

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_reasoning_stripped_by_default(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = iter(["<think>reasoning</think>answer"])
        stream_tokens = list(ask_stream("test"))
        combined = "".join(st.content for st in stream_tokens if not st.is_reasoning)
        assert "reasoning" not in combined
        assert "answer" in combined


class TestGenerationOptions:
    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_passes_options(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = "answer"
        opts = {"temperature": 0.3, "seed": 42}
        ask_raw("q", options=opts)
        assert mock_provider.return_value.chat.call_args[1]["options"] == opts

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_defaults_to_cfg_options(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = "answer"
        from lilbee.config import cfg

        cfg.temperature = 0.7
        cfg.seed = None
        cfg.top_p = None
        cfg.top_k_sampling = None
        cfg.repeat_penalty = None
        cfg.num_ctx = None
        try:
            ask_raw("q")
            assert mock_provider.return_value.chat.call_args[1]["options"] == {"temperature": 0.7}
        finally:
            cfg.temperature = None

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_stream_passes_options(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = iter(["token"])
        opts = {"temperature": 0.1}
        list(ask_stream("q", options=opts))
        assert mock_provider.return_value.chat.call_args[1]["options"] == opts

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_passes_options_through(self, mock_embed, mock_search, mock_provider):
        mock_provider.return_value.chat.return_value = "answer"
        opts = {"num_ctx": 4096}
        ask("q", options=opts)
        assert mock_provider.return_value.chat.call_args[1]["options"] == opts

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_empty_options_passes_none(self, mock_embed, mock_search, mock_provider):
        """When cfg has no generation options set, passes None to provider."""
        mock_provider.return_value.chat.return_value = "answer"
        from lilbee.config import cfg

        cfg.temperature = None
        cfg.top_p = None
        cfg.top_k_sampling = None
        cfg.repeat_penalty = None
        cfg.num_ctx = None
        cfg.seed = None
        ask_raw("q")
        assert mock_provider.return_value.chat.call_args[1]["options"] is None


class TestAskStreamError:
    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_stream_handles_disconnect(self, mock_embed, mock_search, mock_provider):
        def failing_stream():
            yield "partial"
            raise ConnectionError("lost connection")

        mock_provider.return_value.chat.return_value = failing_stream()
        stream_tokens = list(ask_stream("test"))
        combined = "".join(st.content for st in stream_tokens)
        assert "partial" in combined
        assert "Connection lost" in combined


class TestProviderError:
    """ProviderError from the provider should propagate."""

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_provider_error(self, mock_embed, mock_search, mock_provider):
        from lilbee.providers.base import ProviderError

        mock_provider.return_value.chat.side_effect = ProviderError("model 'bad' not found")
        with pytest.raises(ProviderError, match="not found"):
            ask_raw("hello")

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_stream_provider_error(self, mock_embed, mock_search, mock_provider):
        from lilbee.providers.base import ProviderError

        mock_provider.return_value.chat.side_effect = ProviderError("model 'bad' not found")
        with pytest.raises(ProviderError, match="not found"):
            list(ask_stream("hello"))

    @mock.patch("lilbee.query.get_provider")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_stream_provider_error_mid_stream(self, mock_embed, mock_search, mock_provider):
        """ProviderError raised during iteration should propagate."""
        from lilbee.providers.base import ProviderError

        def failing_mid_stream():
            yield "partial"
            raise ProviderError("model 'bad' not found")

        mock_provider.return_value.chat.return_value = failing_mid_stream()
        with pytest.raises(ProviderError, match="not found"):
            list(ask_stream("hello"))


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
        from lilbee.config import cfg
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
        # Query has "alpha beta unique1 unique2" — first chunk covers 3/4
        # Remaining chunks add 0 new terms, so selection stops early
        result = select_context(chunks, "alpha beta unique1 unique2", max_sources=5)
        # Should stop after 1 or 2 chunks since no gain after first
        assert len(result) < 5
