"""The retrieval prompt is budgeted with the chat engine's tokenizer when it has one."""

import math

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind, prompt_token_budget
from tests.conftest import make_mock_services
from tests.test_query import _make_result

_WINDOW = 8192
_DENSE_CHUNK = "4471 0482 " * 100
_CHUNKS = 12


@pytest.fixture(autouse=True)
def _services():
    services = make_mock_services()
    set_services(services)
    old = (cfg.num_ctx, cfg.neighbor_expansion, cfg.filter_structural_chunks)
    cfg.num_ctx = _WINDOW
    cfg.neighbor_expansion = 0
    yield services
    cfg.num_ctx, cfg.neighbor_expansion, cfg.filter_structural_chunks = old
    set_services(None)


def _build(chunk: str = _DENSE_CHUNK, count: int = _CHUNKS):
    results = [_make_result(source=f"{i}.pdf", chunk=chunk, chunk_index=i) for i in range(count)]
    return get_services().searcher._finalize_context(results, "what is in them?", None)


def _mixed_tokens(text: str) -> int:
    """A tokenizer that spends one token per CJK character and one per four other characters."""
    wide = sum(1 for c in text if ord(c) > 0x2E80)
    return wide + math.ceil((len(text) - wide) / 4)


def _prompt_chars(rag) -> int:
    return sum(len(m["content"]) for m in rag.messages)


class TestEngineTokenizerBudget:
    def test_dense_text_is_trimmed_by_the_engine_count(self, _services):
        """An engine that counts one token per character keeps the prompt inside the window.

        The character rule costs this text at a third of its real size and keeps
        every chunk, which overflows the window.
        """
        _services.provider.count_chat_text_tokens.side_effect = len
        rag = _build()
        assert _prompt_chars(rag) <= prompt_token_budget(_WINDOW)
        assert 0 < len(rag.results) < _CHUNKS

    def test_sparse_text_never_costs_less_than_the_provider_window_check(self, _services):
        """The provider refuses a prompt its own 3-character estimate puts over the window.

        An engine that counts six characters per token must not make retrieval
        keep more than the character rule keeps, or the provider rejects the prompt.
        """
        _services.provider.count_chat_text_tokens.return_value = None
        fallback_kept = len(_build(count=30).results)
        assert 0 < fallback_kept < 30
        _services.provider.count_chat_text_tokens.side_effect = lambda text: len(text) // 6
        assert len(_build(count=30).results) == fallback_kept

    def test_a_mixed_dense_and_sparse_list_fits_the_window(self, _services):
        """Dense chunks ranked first do not borrow the cheaper cost of the sparse ones after them.

        One average over the list costs a dense chunk at a third of its tokens, so
        the list overflows the window the engine count says is full.
        """
        dense = "\u4e00\u4e8c\u4e09\u56db" * 250
        sparse = "plain english words " * 60
        _services.provider.count_chat_text_tokens.side_effect = _mixed_tokens
        results = [
            _make_result(source=f"d{i}.pdf", chunk=dense, chunk_index=i) for i in range(5)
        ] + [_make_result(source=f"s{i}.pdf", chunk=sparse, chunk_index=i) for i in range(20)]
        rag = get_services().searcher._finalize_context(results, "what is in them?", None)
        kept = [r.chunk for r in rag.results]
        assert dense in kept
        assert sparse in kept
        assert sum(_mixed_tokens(m["content"]) for m in rag.messages) <= prompt_token_budget(
            _WINDOW
        )

    def test_each_chunk_is_counted_on_its_own_and_once(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = len
        cost = get_services().searcher._token_cost()
        assert cost("abcdef") == 6
        assert cost("abcdef") == 6
        assert cost("x") == 1
        counted = [c.args[0] for c in _services.provider.count_chat_text_tokens.call_args_list]
        assert counted == ["abcdef", "x"]

    def test_a_failing_count_costs_every_later_text_by_the_character_rule(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = ProviderError(
            "engine down", kind=ProviderErrorKind.CONNECTION
        )
        cost = get_services().searcher._token_cost()
        assert cost("a" * 9) == 3
        assert cost("b" * 30) == 10
        assert _services.provider.count_chat_text_tokens.call_count == 1

    def test_counted_text_holds_the_system_prompt_and_every_candidate(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = len
        _build(chunk="marker chunk", count=3)
        counted = [c.args[0] for c in _services.provider.count_chat_text_tokens.call_args_list]
        assert cfg.rag_system_prompt in counted
        assert "marker chunk" in counted

    def test_counted_text_holds_every_history_message(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = len
        history = [
            {"role": "user", "content": "earlier question about marigolds"},
            {"role": "assistant", "content": "earlier answer about marigolds"},
        ]
        results = [_make_result(source="0.pdf", chunk="marker chunk", chunk_index=0)]
        get_services().searcher._finalize_context(results, "what is in them?", history)
        counted = [c.args[0] for c in _services.provider.count_chat_text_tokens.call_args_list]
        assert "earlier question about marigolds" in counted
        assert "earlier answer about marigolds" in counted

    def test_neighbor_widening_spends_the_engine_measured_leftover(self, _services):
        """Widening stops where the engine's count says the window is full."""
        cfg.neighbor_expansion = 3
        rows = [
            _make_result(source="0.pdf", chunk=f"{i}471 0482 " * 100, chunk_index=i)
            for i in range(1, 8)
        ]
        _services.store.get_chunks_by_indices.return_value = rows
        _services.provider.count_chat_text_tokens.side_effect = len
        centre = [rows[3]]
        rag = get_services().searcher._finalize_context(centre, "what is in them?", None)
        assert len(rag.results[0].chunk) > len(rows[3].chunk)
        assert _prompt_chars(rag) <= prompt_token_budget(_WINDOW)

    def test_history_takes_window_from_the_sources(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = len
        results = [
            _make_result(source=f"{i}.pdf", chunk=_DENSE_CHUNK, chunk_index=i)
            for i in range(_CHUNKS)
        ]
        history = [{"role": "user", "content": "earlier question " * 150}]
        searcher = get_services().searcher
        bare = searcher._finalize_context(list(results), "what is in them?", None)
        with_history = searcher._finalize_context(list(results), "what is in them?", history)
        assert 0 < len(with_history.results) < len(bare.results)

    def test_widening_leaves_out_a_neighbor_that_is_a_table_of_contents(self, _services):
        cfg.neighbor_expansion = 1
        cfg.filter_structural_chunks = True
        toc = "A. Summary ......... 1\nB. Intro ......... 3\nC. Trends ......... 9\n"
        rows = [
            _make_result(source="0.pdf", chunk=toc, chunk_index=1),
            _make_result(source="0.pdf", chunk="the centre passage", chunk_index=2),
            _make_result(source="0.pdf", chunk="the next prose passage", chunk_index=3),
        ]
        _services.store.get_chunks_by_indices.return_value = rows
        _services.provider.count_chat_text_tokens.side_effect = len
        rag = get_services().searcher._finalize_context([rows[1]], "what is in them?", None)
        assert "the next prose passage" in rag.results[0].chunk
        assert "Summary" not in rag.results[0].chunk

    def test_an_engine_without_a_route_keeps_the_character_rule(self, _services):
        _services.provider.count_chat_text_tokens.return_value = None
        rag = _build()
        # 12 chunks of 1000 characters cost 334 + 24 each at 3 characters per token
        assert len(rag.results) == _CHUNKS
        assert _prompt_chars(rag) > prompt_token_budget(_WINDOW)

    def test_a_failed_count_keeps_the_character_rule(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = ProviderError(
            "engine down", kind=ProviderErrorKind.CONNECTION
        )
        rag = _build()
        assert len(rag.results) == _CHUNKS

    def test_a_zero_count_keeps_the_character_rule(self, _services):
        _services.provider.count_chat_text_tokens.return_value = 0
        assert len(_build().results) == _CHUNKS
