"""Tests for the RAG query pipeline (mocked — no live server needed)."""

from unittest import mock

import ollama
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


def _make_result(
    source="test.pdf",
    content_type="pdf",
    page_start=1,
    page_end=1,
    line_start=0,
    line_end=0,
    chunk="some text",
    chunk_index=0,
    _distance=0.5,
    vector=None,
):
    return {
        "source": source,
        "content_type": content_type,
        "page_start": page_start,
        "page_end": page_end,
        "line_start": line_start,
        "line_end": line_end,
        "chunk": chunk,
        "chunk_index": chunk_index,
        "_distance": _distance,
        "vector": vector or [0.1],
    }


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
            _make_result(source="far.pdf", _distance=0.9),
            _make_result(source="close.pdf", _distance=0.1),
            _make_result(source="mid.pdf", _distance=0.5),
        ]
        sorted_results = sort_by_relevance(results)
        assert sorted_results[0]["source"] == "close.pdf"
        assert sorted_results[1]["source"] == "mid.pdf"
        assert sorted_results[2]["source"] == "far.pdf"

    def test_missing_distance_sorts_last(self):
        results = [
            {"source": "no_dist.pdf", "chunk": "text"},
            _make_result(source="has_dist.pdf", _distance=0.3),
        ]
        sorted_results = sort_by_relevance(results)
        assert sorted_results[0]["source"] == "has_dist.pdf"
        assert sorted_results[1]["source"] == "no_dist.pdf"


class TestBuildContext:
    def test_numbers_chunks(self):
        results = [_make_result(chunk="chunk one"), _make_result(chunk="chunk two")]
        ctx = build_context(results)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "chunk one" in ctx


class TestSearchContext:
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_returns_results(self, mock_embed, mock_search):
        results = search_context("question")
        assert len(results) == 1
        mock_embed.assert_called_once_with("question")


class TestAskRaw:
    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result(chunk="oil is 5 quarts")])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_returns_structured_result(self, _embed, _search, mock_chat):
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="5 quarts."))
        result = ask_raw("oil capacity?")
        assert result.answer == "5 quarts."
        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "test.pdf"

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_no_results(self, _embed, _search):
        result = ask_raw("anything")
        assert "No relevant documents" in result.answer
        assert result.sources == []

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_with_history(self, _embed, _search, mock_chat):
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="answer"))
        history = [{"role": "user", "content": "prev"}]
        ask_raw("new q", history=history)
        messages = mock_chat.call_args[1]["messages"]
        assert len(messages) == 3  # system + history + user


class TestAsk:
    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result(chunk="oil is 5 quarts")])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_returns_answer_with_citations(self, mock_embed, mock_search, mock_chat):
        mock_chat.return_value = mock.MagicMock(
            message=mock.MagicMock(content="The oil capacity is 5 quarts.")
        )
        answer = ask("oil capacity?")
        assert "5 quarts" in answer
        assert "Sources:" in answer
        assert "test.pdf" in answer

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_no_results_message(self, mock_embed, mock_search):
        answer = ask("anything")
        assert "No relevant documents" in answer

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_with_history(self, mock_embed, mock_search, mock_chat):
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="answer"))
        history = [
            {"role": "user", "content": "prev q"},
            {"role": "assistant", "content": "prev a"},
        ]
        ask("new q", history=history)
        messages = mock_chat.call_args[1]["messages"]
        # System + 2 history + user = 4
        assert len(messages) == 4
        assert messages[1]["content"] == "prev q"


class TestAskStream:
    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_yields_tokens_then_citations(self, mock_embed, mock_search, mock_chat):
        mock_chat.return_value = iter(
            [
                mock.MagicMock(message=mock.MagicMock(content="Hello")),
                mock.MagicMock(message=mock.MagicMock(content=" world")),
            ]
        )
        tokens = list(ask_stream("test"))
        combined = "".join(tokens)
        assert "Hello world" in combined
        assert "Sources:" in combined

    @mock.patch("lilbee.store.search", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_empty_results_yields_message(self, mock_embed, mock_search):
        tokens = list(ask_stream("anything"))
        assert any("No relevant documents" in t for t in tokens)

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_stream_with_history(self, mock_embed, mock_search, mock_chat):
        mock_chat.return_value = iter([mock.MagicMock(message=mock.MagicMock(content="response"))])
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        list(ask_stream("new question", history=history))
        call_args = mock_chat.call_args
        messages = call_args[1]["messages"]
        # System + 2 history + user = 4 messages
        assert len(messages) == 4
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "previous question"

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_skips_empty_tokens(self, mock_embed, mock_search, mock_chat):
        mock_chat.return_value = iter(
            [
                mock.MagicMock(message=mock.MagicMock(content="")),
                mock.MagicMock(message=mock.MagicMock(content="data")),
            ]
        )
        tokens = list(ask_stream("test"))
        # Empty string token should not appear as a separate yield
        non_source_tokens = [t for t in tokens if "Sources:" not in t]
        assert all(t != "" for t in non_source_tokens if t.strip())


class TestGenerationOptions:
    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_passes_options(self, _embed, _search, mock_chat):
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="answer"))
        opts = {"temperature": 0.3, "seed": 42}
        ask_raw("q", options=opts)
        assert mock_chat.call_args[1]["options"] == opts

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_defaults_to_cfg_options(self, _embed, _search, mock_chat):
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="answer"))
        from lilbee.config import cfg

        cfg.temperature = 0.7
        cfg.seed = None
        cfg.top_p = None
        cfg.top_k_sampling = None
        cfg.repeat_penalty = None
        cfg.num_ctx = None
        try:
            ask_raw("q")
            assert mock_chat.call_args[1]["options"] == {"temperature": 0.7}
        finally:
            cfg.temperature = None

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_stream_passes_options(self, _embed, _search, mock_chat):
        mock_chat.return_value = iter([mock.MagicMock(message=mock.MagicMock(content="token"))])
        opts = {"temperature": 0.1}
        list(ask_stream("q", options=opts))
        assert mock_chat.call_args[1]["options"] == opts

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_passes_options_through(self, _embed, _search, mock_chat):
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="answer"))
        opts = {"num_ctx": 4096}
        ask("q", options=opts)
        assert mock_chat.call_args[1]["options"] == opts

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_empty_options_passes_none(self, _embed, _search, mock_chat):
        """When cfg has no generation options set, passes None to ollama."""
        mock_chat.return_value = mock.MagicMock(message=mock.MagicMock(content="answer"))
        from lilbee.config import cfg

        cfg.temperature = None
        cfg.top_p = None
        cfg.top_k_sampling = None
        cfg.repeat_penalty = None
        cfg.num_ctx = None
        cfg.seed = None
        ask_raw("q")
        assert mock_chat.call_args[1]["options"] is None


class TestAskStreamError:
    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_stream_handles_disconnect(self, mock_embed, mock_search, mock_chat):
        def failing_stream():
            yield mock.MagicMock(message=mock.MagicMock(content="partial"))
            raise ConnectionError("lost connection")

        mock_chat.return_value = failing_stream()
        tokens = list(ask_stream("test"))
        combined = "".join(tokens)
        assert "partial" in combined
        assert "Connection lost" in combined


class TestModelNotFound:
    """ResponseError from Ollama (e.g. model not found) should raise RuntimeError."""

    @mock.patch("ollama.chat", side_effect=ollama.ResponseError("model 'bad' not found", 404))
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_raw_model_not_found(self, _embed, _search, _chat):
        with pytest.raises(RuntimeError, match="not found"):
            ask_raw("hello")

    @mock.patch("ollama.chat", side_effect=ollama.ResponseError("model 'bad' not found", 404))
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_stream_model_not_found(self, _embed, _search, _chat):
        with pytest.raises(RuntimeError, match="not found"):
            list(ask_stream("hello"))

    @mock.patch("ollama.chat")
    @mock.patch("lilbee.store.search", return_value=[_make_result()])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_ask_stream_model_not_found_mid_stream(self, _embed, _search, mock_chat):
        """ResponseError raised during iteration should also become RuntimeError."""

        def failing_mid_stream():
            yield mock.MagicMock(message=mock.MagicMock(content="partial"))
            raise ollama.ResponseError("model 'bad' not found", 404)

        mock_chat.return_value = failing_mid_stream()
        with pytest.raises(RuntimeError, match="not found"):
            list(ask_stream("hello"))
