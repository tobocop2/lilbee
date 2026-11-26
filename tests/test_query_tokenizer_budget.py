"""The retrieval prompt is measured with the chat engine's tokenizer when it has one."""

import math

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.core.config import cfg
from lilbee.providers.base import (
    CHAT_MESSAGE_OVERHEAD_TOKENS,
    ProviderError,
    ProviderErrorKind,
    prompt_token_budget,
)
from tests.conftest import make_mock_services
from tests.test_query import _make_result

_WINDOW = 8192
_DENSE_CHUNK = "4471 0482 " * 100
_CHUNKS = 12
# A fixed cost on every request that a per-character ratio cannot capture.
_FIXED_TOKENS = 6000


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


def _fixed_cost_tokens(text: str) -> int:
    return len(text) // 3 + _FIXED_TOKENS


def _sent(rag) -> str:
    return "\n".join(m["content"] for m in rag.messages)


def _measured(rag, tokens) -> int:
    """Tokens of the prompt *rag* sends, counted the way the budget counts them."""
    return tokens(_sent(rag)) + CHAT_MESSAGE_OVERHEAD_TOKENS * len(rag.messages)


def _counted(services) -> list[str]:
    return [c.args[0] for c in services.provider.count_chat_text_tokens.call_args_list]


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
        assert _measured(rag, len) <= prompt_token_budget(_WINDOW)
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

    def test_a_prompt_that_fits_is_measured_once_as_it_is_sent(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = lambda text: len(text) // 6
        rag = _build()
        assert _counted(_services) == [_sent(rag)]

    def test_a_first_estimate_that_overflows_fits_after_the_second_measurement(self, _services):
        """The second assembly uses the ratio the first measurement found."""
        _services.provider.count_chat_text_tokens.side_effect = len
        rag = _build()
        counted = _counted(_services)
        assert len(counted) == 2
        assert counted[-1] == _sent(rag)
        assert len(counted[0]) > len(counted[1])
        assert _measured(rag, len) <= prompt_token_budget(_WINDOW)

    def test_a_mixed_dense_and_sparse_list_fits_the_window(self, _services):
        """Dense chunks ranked first do not borrow the cheaper cost of the sparse ones after them.

        One average over the list costs a dense chunk at a third of its tokens, so
        the list overflows the window the engine count says is full.
        """
        dense = "一二三四" * 250
        sparse = "plain english words " * 60
        _services.provider.count_chat_text_tokens.side_effect = _mixed_tokens
        results = [
            _make_result(source=f"d{i}.pdf", chunk=dense, chunk_index=i) for i in range(5)
        ] + [_make_result(source=f"s{i}.pdf", chunk=sparse, chunk_index=i) for i in range(20)]
        rag = get_services().searcher._finalize_context(results, "what is in them?", None)
        kept = [r.chunk for r in rag.results]
        assert dense in kept
        assert sparse in kept
        assert _measured(rag, _mixed_tokens) <= prompt_token_budget(_WINDOW)
        assert _sent(rag) in _counted(_services)

    def test_the_refit_margin_saves_a_measurement_on_a_mixed_list(self, _services):
        """The ratio is an average over the prompt.

        A refit packed to the brim by the ratio alone overflows again on this
        list, so it takes a third measurement.
        """
        dense = "一二三四" * 400
        sparse = "plain english words " * 120
        _services.provider.count_chat_text_tokens.side_effect = _mixed_tokens
        results = [
            _make_result(source=f"d{i}.pdf", chunk=dense, chunk_index=i) for i in range(3)
        ] + [_make_result(source=f"s{i}.pdf", chunk=sparse, chunk_index=i) for i in range(10)]
        rag = get_services().searcher._finalize_context(results, "what is in them?", None)
        counted = _counted(_services)
        assert len(counted) == 2
        assert counted[-1] == _sent(rag)
        assert _measured(rag, _mixed_tokens) <= prompt_token_budget(_WINDOW)

    def test_long_source_paths_do_not_make_the_refit_cheaper_than_the_character_rule(
        self, _services
    ):
        """A source path longer than the per-source allowance costs more than the fit charges.

        The first overflow then measures a ratio below 1. A refit at that ratio
        costs text lower than the character rule, packs past the window and
        takes four more measurements to recover.
        """

        def tokens(text: str) -> int:
            return math.ceil(len(text) * 0.95 / 3)

        _services.provider.count_chat_text_tokens.side_effect = tokens
        sparse = "plain english words " * 30
        results = [
            _make_result(
                source=f"docs/{i:02d}/".ljust(116, "p") + ".pdf",
                chunk=f"{i:02d} {sparse}"[:600],
                chunk_index=i,
            )
            for i in range(80)
        ]
        rag = get_services().searcher._finalize_context(results, "what is in them?", None)
        counted = _counted(_services)
        assert len(counted) == 2
        assert counted[-1] == _sent(rag)
        assert _measured(rag, tokens) <= prompt_token_budget(_WINDOW)

    def test_a_cost_the_ratio_misses_ends_in_a_search_that_measures_a_fit(self, _services):
        """Three scaled measurements still overflow, so the kept count is searched."""
        _services.provider.count_chat_text_tokens.side_effect = _fixed_cost_tokens
        rag = _build(count=30)
        counted = _counted(_services)
        assert len(counted) > 3
        assert all(_fixed_cost_tokens(text) > prompt_token_budget(_WINDOW) for text in counted[:3])
        assert _sent(rag) in counted
        assert _measured(rag, _fixed_cost_tokens) <= prompt_token_budget(_WINDOW)
        assert len(rag.results) + len(rag.dropped) == 30

    def test_the_search_keeps_the_most_sources_that_fit(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = _fixed_cost_tokens
        rag = _build(count=30)
        one_more_source = len(_DENSE_CHUNK) // 3
        assert _measured(rag, _fixed_cost_tokens) + one_more_source > prompt_token_budget(_WINDOW)

    def test_a_failed_measurement_uses_the_character_rule_for_that_request_only(self, _services):
        failed = []

        def flaky(text: str) -> int:
            if not failed:
                failed.append(text)
                raise ProviderError("engine down", kind=ProviderErrorKind.CONNECTION)
            return len(text)

        _services.provider.count_chat_text_tokens.side_effect = flaky
        assert len(_build().results) == _CHUNKS
        rag = _build()
        assert _counted(_services)[-1] == _sent(rag)
        assert _measured(rag, len) <= prompt_token_budget(_WINDOW)

    def test_a_later_failed_measurement_sends_the_scaled_estimate(self, _services):
        counts = iter([len])

        def second_fails(text: str) -> int:
            tokens = next(counts, None)
            if tokens is None:
                raise ProviderError("engine down", kind=ProviderErrorKind.CONNECTION)
            return tokens(text)

        _services.provider.count_chat_text_tokens.side_effect = second_fails
        rag = _build()
        assert len(_counted(_services)) == 2
        assert 0 < len(rag.results) < _CHUNKS

    def test_a_failed_measurement_in_the_search_sends_the_probed_count(self, _services):
        def fourth_fails(text: str) -> int:
            if _services.provider.count_chat_text_tokens.call_count > 3:
                raise ProviderError("engine down", kind=ProviderErrorKind.CONNECTION)
            return _fixed_cost_tokens(text)

        _services.provider.count_chat_text_tokens.side_effect = fourth_fails
        rag = _build(count=30)
        assert _counted(_services)[-1] == _sent(rag)

    def test_a_failed_measurement_in_the_search_sends_the_best_measured_fit(self, _services):
        """The probe that the failed request was for is not sent; the last measured fit is."""
        target = prompt_token_budget(_WINDOW)

        def fit_then_fail(text: str) -> int:
            call = _services.provider.count_chat_text_tokens.call_count
            if call <= 3:
                return target + 20
            if call == 4:
                return 1
            raise ProviderError("engine down", kind=ProviderErrorKind.CONNECTION)

        _services.provider.count_chat_text_tokens.side_effect = fit_then_fail
        rag = _build()
        counted = _counted(_services)
        assert len(counted) == 5
        assert _sent(rag) == counted[3]

    def test_the_top_source_alone_is_sent_when_every_probe_overflows(self, _services):
        """The assembly that overflowed holds more sources than the top one; it is not sent."""
        target = prompt_token_budget(_WINDOW)

        def near_then_huge(text: str) -> int:
            near = _services.provider.count_chat_text_tokens.call_count <= 3
            return target + 20 if near else 10**6

        _services.provider.count_chat_text_tokens.side_effect = near_then_huge
        rag = _build()
        counted = _counted(_services)
        assert len(rag.results) == 1
        assert len(counted[2]) > len(_sent(rag))

    def test_a_count_within_the_message_overhead_of_the_target_overflows(self, _services):
        """The chat template adds tokens per message, so the measurement charges them."""
        target = prompt_token_budget(_WINDOW)

        def just_under(text: str) -> int:
            if _services.provider.count_chat_text_tokens.call_count == 1:
                return target - 5
            return len(text) // 6

        _services.provider.count_chat_text_tokens.side_effect = just_under
        rag = _build()
        counted = _counted(_services)
        assert len(counted) == 2
        assert counted[1] == _sent(rag)

    def test_the_top_source_stays_when_it_alone_overflows(self, _services):
        _services.provider.count_chat_text_tokens.return_value = 10**6
        rag = _build()
        assert len(rag.results) == 1
        assert len(rag.dropped) == _CHUNKS - 1

    def test_a_refit_that_repeats_the_overflowed_prompt_is_not_measured(self, _services):
        """The top source alone overflows, so the refit keeps the same prompt."""
        _services.provider.count_chat_text_tokens.side_effect = lambda text: len(text) // 4
        rag = _build(chunk="sparse words " * 3000, count=2)
        assert len(rag.results) == 1
        assert _services.provider.count_chat_text_tokens.call_count == 1

    def test_the_search_does_not_measure_the_assembly_that_overflowed_again(self, _services):
        _services.provider.count_chat_text_tokens.return_value = 10**6
        _build()
        counted = _counted(_services)
        assert len(counted) == len(set(counted))

    def test_measured_text_holds_every_history_message(self, _services):
        _services.provider.count_chat_text_tokens.side_effect = len
        history = [
            {"role": "user", "content": "earlier question about marigolds"},
            {"role": "assistant", "content": "earlier answer about marigolds"},
        ]
        results = [_make_result(source="0.pdf", chunk="marker chunk", chunk_index=0)]
        get_services().searcher._finalize_context(results, "what is in them?", history)
        (text,) = _counted(_services)
        assert "earlier question about marigolds" in text
        assert "earlier answer about marigolds" in text

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
        assert _measured(rag, len) <= prompt_token_budget(_WINDOW)

    def test_a_refit_reads_the_neighbor_rows_of_the_request_once(self, _services):
        cfg.neighbor_expansion = 3
        rows = [
            _make_result(source="0.pdf", chunk=f"{i}471 0482 " * 100, chunk_index=i)
            for i in range(1, 8)
        ]
        _services.store.get_chunks_by_indices.return_value = rows
        _services.provider.count_chat_text_tokens.side_effect = len
        get_services().searcher._finalize_context([rows[3]], "what is in them?", None)
        assert _services.provider.count_chat_text_tokens.call_count == 2
        _services.store.get_chunks_by_indices.assert_called_once()

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
