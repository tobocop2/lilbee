"""Tests for reasoning token filter and cap-aware chat orchestrator."""

from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.model_defaults import ModelDefaults
from lilbee.retrieval.reasoning import (
    CAP_CONTINUATION_PROMPT,
    CapNotice,
    StreamToken,
    effective_reasoning_cap,
    filter_reasoning,
    stream_chat_with_cap,
    strip_reasoning,
)

_NO_CAP = 1_000_000


def _collect(
    tokens: list[str],
    *,
    show: bool,
    cap_chars: int = _NO_CAP,
) -> list[StreamToken]:
    return list(filter_reasoning(iter(tokens), show=show, cap_chars=cap_chars))


class TestFilterReasoningShowFalse:
    def test_clean_text_passes_through(self):
        result = _collect(["Hello ", "world"], show=False)
        assert len(result) == 2
        assert all(not st.is_reasoning for st in result)
        assert "".join(st.content for st in result) == "Hello world"

    def test_strips_thinking_block(self):
        result = _collect(["<think>reasoning</think>answer"], show=False)
        content = "".join(st.content for st in result)
        assert "<think>" not in content
        assert "reasoning" not in content
        assert "answer" in content

    def test_strips_thinking_across_tokens(self):
        result = _collect(["<thi", "nk>deep thought</thi", "nk>final"], show=False)
        content = "".join(st.content for st in result)
        assert "deep thought" not in content
        assert "final" in content

    def test_content_before_and_after(self):
        result = _collect(["before<think>middle</think>after"], show=False)
        content = "".join(st.content for st in result)
        assert content == "beforeafter"

    def test_empty_thinking_block(self):
        result = _collect(["<think></think>answer"], show=False)
        content = "".join(st.content for st in result)
        assert content == "answer"

    def test_no_tokens(self):
        result = _collect([], show=False)
        assert result == []


class TestFilterReasoningShowTrue:
    def test_clean_text_not_reasoning(self):
        result = _collect(["Hello"], show=True)
        assert len(result) == 1
        assert result[0].content == "Hello"
        assert result[0].is_reasoning is False

    def test_thinking_yielded_as_reasoning(self):
        result = _collect(["<think>reasoning</think>answer"], show=True)
        reasoning = [st for st in result if st.is_reasoning]
        response = [st for st in result if not st.is_reasoning]
        assert len(reasoning) >= 1
        assert "reasoning" in "".join(st.content for st in reasoning)
        assert "answer" in "".join(st.content for st in response)

    def test_thinking_across_token_boundaries(self):
        tokens = ["<th", "ink>", "deep ", "thought", "</th", "ink>", "answer"]
        result = _collect(tokens, show=True)
        reasoning_text = "".join(st.content for st in result if st.is_reasoning)
        response_text = "".join(st.content for st in result if not st.is_reasoning)
        assert "deep thought" in reasoning_text
        assert "answer" in response_text

    def test_content_before_thinking(self):
        result = _collect(["before<think>thinking</think>after"], show=True)
        parts = [(st.content, st.is_reasoning) for st in result]
        before = [c for c, r in parts if not r and "before" in c]
        assert len(before) >= 1

    def test_empty_thinking_block(self):
        result = _collect(["<think></think>answer"], show=True)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "answer" in response

    def test_multiple_thinking_blocks(self):
        result = _collect(["<think>first</think>mid<think>second</think>end"], show=True)
        reasoning = "".join(st.content for st in result if st.is_reasoning)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "first" in reasoning
        assert "second" in reasoning
        assert "mid" in response
        assert "end" in response


class TestCouldBePartial:
    def test_partial_open_tag(self):
        result = _collect(["text<thi", "nk>reasoning</think>done"], show=False)
        content = "".join(st.content for st in result)
        assert "reasoning" not in content
        assert "text" in content
        assert "done" in content

    def test_partial_close_tag(self):
        result = _collect(["<think>thought</thi", "nk>done"], show=True)
        reasoning = "".join(st.content for st in result if st.is_reasoning)
        assert "thought" in reasoning

    def test_false_partial_not_tag(self):
        result = _collect(["text<", "b>not a tag"], show=False)
        content = "".join(st.content for st in result)
        assert "text" in content

    def test_unterminated_thinking_flushed_when_show(self):
        result = _collect(["<think>unterminated"], show=True)
        reasoning = [st for st in result if st.is_reasoning]
        assert len(reasoning) >= 1
        assert "unterminated" in "".join(st.content for st in reasoning)

    def test_unterminated_thinking_with_partial_close(self):
        result = _collect(["<think>deep thought</thi"], show=True)
        reasoning = "".join(st.content for st in result if st.is_reasoning)
        assert "deep thought" in reasoning

    def test_unterminated_thinking_stripped_when_hidden(self):
        result = _collect(["<think>unterminated"], show=False)
        content = "".join(st.content for st in result)
        assert content == ""

    def test_trailing_text_after_thinking(self):
        result = _collect(["<think>thought</think>trailing"], show=True)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "trailing" in response

    def test_normal_text_ending_with_partial_tag(self):
        result = _collect(["hello<t"], show=False)
        content = "".join(st.content for st in result)
        assert "hello<t" in content


class TestReasoningCap:
    def test_cap_fires_on_long_reasoning(self):
        """on_cap is invoked when reasoning exceeds cap_chars."""
        fired = [False]

        def _on_cap() -> None:
            fired[0] = True

        long_think = "x" * 1500
        tokens = list(f"<think>{long_think}</think>answer")
        list(filter_reasoning(iter(tokens), show=True, cap_chars=1024, on_cap=_on_cap))
        assert fired[0]

    def test_cap_yields_no_response_after_fire(self):
        """When the cap fires, iteration stops; later tokens don't reach the consumer."""
        long_think = "x" * 1500
        tokens = list(f"<think>{long_think}</think>answer")
        result = list(
            filter_reasoning(iter(tokens), show=True, cap_chars=1024, on_cap=lambda: None)
        )
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert "answer" not in response

    def test_no_cap_when_zero(self):
        """cap_chars=0 disables the cap entirely; on_cap never fires."""
        fired = [False]

        def _on_cap() -> None:
            fired[0] = True

        long_think = "x" * 5000
        tokens = list(f"<think>{long_think}</think>answer")
        list(filter_reasoning(iter(tokens), show=False, cap_chars=0, on_cap=_on_cap))
        assert fired == [False]

    def test_on_cap_optional(self):
        """Cap firing without an on_cap callback still terminates cleanly."""
        long_think = "x" * 1500
        tokens = list(f"<think>{long_think}</think>answer")
        result = list(filter_reasoning(iter(tokens), show=False, cap_chars=512))
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert response == ""

    def test_on_progress_fires_during_reasoning(self):
        """on_progress receives running reasoning-chars counts as content arrives."""
        progress: list[int] = []
        chunks = ["<think>"] + ["x" * 100 for _ in range(10)] + ["</think>", "answer"]
        list(
            filter_reasoning(
                iter(chunks), show=False, cap_chars=_NO_CAP, on_progress=progress.append
            )
        )
        assert progress
        assert progress == sorted(progress)
        assert progress[-1] >= 1000

    def test_on_progress_optional(self):
        """Stream completes cleanly when on_progress is omitted."""
        result = _collect(["<think>x" * 200 + "</think>answer"], show=False)
        response = "".join(st.content for st in result if not st.is_reasoning)
        assert response == "answer"

    def test_unclosed_think_terminates_show_false(self):
        """An unclosed <think> with show=False still hits the cap, no hang."""
        fired = [False]

        def _on_cap() -> None:
            fired[0] = True

        tokens = ["<think>"] + ["x"] * 2000
        list(filter_reasoning(iter(tokens), show=False, cap_chars=512, on_cap=_on_cap))
        assert fired[0]

    def test_upstream_generator_closed_on_cap(self):
        """Cap firing closes the upstream iterator, releasing llama.cpp's chat lock."""
        closed = {"value": False}

        def runaway_tokens():
            try:
                yield "<think>"
                while True:
                    yield "x"
            except GeneratorExit:
                closed["value"] = True
                raise

        list(filter_reasoning(runaway_tokens(), show=True, cap_chars=512, on_cap=lambda: None))
        assert closed["value"]

    def test_close_called_on_stream_wrapper_early_exit(self):
        """A stream wrapper exposing close() has it called when the cap fires."""

        class FakeStream:
            def __init__(self) -> None:
                self.tokens = iter(["<think>"] + ["x"] * 2000)
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.tokens)

            def close(self) -> None:
                self.closed = True

        stream = FakeStream()
        list(filter_reasoning(stream, show=True, cap_chars=512, on_cap=lambda: None))
        assert stream.closed


class TestEffectiveReasoningCap:
    """The single source of truth for which cap value applies right now."""

    @pytest.fixture(autouse=True)
    def _isolated(self):
        snapshot_cap = cfg.max_reasoning_chars
        snapshot_defaults = cfg.model_defaults
        cfg.clear_model_defaults()
        yield
        cfg.max_reasoning_chars = snapshot_cap
        cfg.apply_model_defaults(snapshot_defaults)

    def test_uses_cfg_when_no_per_model_override(self):
        cfg.max_reasoning_chars = 8000
        assert effective_reasoning_cap() == 8000

    def test_per_model_override_wins(self):
        cfg.max_reasoning_chars = 8000
        cfg.apply_model_defaults(ModelDefaults(max_reasoning_chars=20_000))
        assert effective_reasoning_cap() == 20_000

    def test_per_model_zero_means_unlimited_for_that_model(self):
        """A per-model 0 is an explicit opt-out, not 'fall through to global'."""
        cfg.max_reasoning_chars = 8000
        cfg.apply_model_defaults(ModelDefaults(max_reasoning_chars=0))
        assert effective_reasoning_cap() == 0

    def test_no_override_field_falls_through(self):
        cfg.max_reasoning_chars = 8000
        cfg.apply_model_defaults(ModelDefaults(temperature=0.7))
        assert effective_reasoning_cap() == 8000

    def test_global_zero_means_unlimited(self):
        cfg.max_reasoning_chars = 0
        assert effective_reasoning_cap() == 0


class TestStreamChatWithCap:
    """End-to-end orchestrator: filter + cap-fire + continuation re-issue."""

    def _make_provider(self, *responses: object) -> MagicMock:
        provider = MagicMock()
        provider.chat.side_effect = [iter(r) if not callable(r) else r() for r in responses]
        return provider

    def test_no_cap_fire_yields_only_stream_tokens(self):
        provider = self._make_provider(["<think>brief</think>", "the answer"])
        events = list(
            stream_chat_with_cap(
                provider,
                [{"role": "user", "content": "hi"}],
                options=None,
                model="test-model",
                show_reasoning=True,
                cap_chars=64_000,
            )
        )
        assert not any(isinstance(e, CapNotice) for e in events)
        assert provider.chat.call_count == 1
        response = "".join(
            e.content for e in events if isinstance(e, StreamToken) and not e.is_reasoning
        )
        assert "the answer" in response

    def test_cap_fire_emits_notice_then_continuation_tokens(self):
        long_think = "<think>" + ("x " * 400) + "</think>not reached"
        provider = self._make_provider([long_think], ["final ", "answer."])
        events = list(
            stream_chat_with_cap(
                provider,
                [{"role": "user", "content": "explain X"}],
                options=None,
                model="test-model",
                show_reasoning=True,
                cap_chars=512,
            )
        )
        notices = [e for e in events if isinstance(e, CapNotice)]
        assert len(notices) == 1
        assert notices[0].cap_chars == 512
        response = "".join(
            e.content for e in events if isinstance(e, StreamToken) and not e.is_reasoning
        )
        assert "final " in response and "answer." in response
        assert provider.chat.call_count == 2

    def test_cap_fire_closes_the_first_stream_through_text_only(self):
        # bb-ziks.17: on cap-fire the first stream is closed via _text_only's
        # forwarded close, or its HTTP connection / in_flight slot leaks until GC.
        class ClosableStream:
            def __init__(self, tokens) -> None:
                self._tokens = iter(tokens)
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._tokens)

            def close(self) -> None:
                self.closed = True

        first = ClosableStream(["<think>" + ("x " * 400) + "</think>not reached"])
        provider = MagicMock()
        provider.chat.side_effect = [first, iter(["answer"])]
        list(
            stream_chat_with_cap(
                provider,
                [{"role": "user", "content": "q"}],
                options=None,
                model="test-model",
                show_reasoning=True,
                cap_chars=512,
            )
        )
        assert first.closed  # the capped first stream was closed, not leaked

    def test_continuation_call_appends_user_nudge(self):
        long_think = "<think>" + ("x " * 400) + "</think>"
        provider = self._make_provider([long_think], ["done"])
        list(
            stream_chat_with_cap(
                provider,
                [{"role": "user", "content": "q"}],
                options=None,
                model="test-model",
                show_reasoning=False,
                cap_chars=512,
            )
        )
        nudged = provider.chat.call_args_list[1].args[0]
        assert nudged[-1] == {"role": "user", "content": CAP_CONTINUATION_PROMPT}
        assert nudged[0] == {"role": "user", "content": "q"}

    def test_unlimited_cap_skips_continuation_even_for_long_reasoning(self):
        """cap_chars=0 disables the cap; the orchestrator never re-issues."""
        very_long = "<think>" + ("x " * 5000) + "</think>real answer"
        provider = self._make_provider([very_long])
        events = list(
            stream_chat_with_cap(
                provider,
                [{"role": "user", "content": "go deep"}],
                options=None,
                model="test-model",
                show_reasoning=False,
                cap_chars=0,
            )
        )
        assert not any(isinstance(e, CapNotice) for e in events)
        assert provider.chat.call_count == 1
        response = "".join(
            e.content for e in events if isinstance(e, StreamToken) and not e.is_reasoning
        )
        assert "real answer" in response

    def test_consumer_close_propagates_to_continuation_stream(self):
        """If the consumer stops iterating, the continuation stream is closed too."""
        long_think = "<think>" + ("x " * 400) + "</think>"
        second_closed = {"value": False}

        def second_pass():
            try:
                yield "first "
                yield "second"
            except GeneratorExit:
                second_closed["value"] = True
                raise

        provider = MagicMock()
        provider.chat.side_effect = [iter([long_think]), second_pass()]
        gen = stream_chat_with_cap(
            provider,
            [{"role": "user", "content": "q"}],
            options=None,
            model="test-model",
            show_reasoning=False,
            cap_chars=512,
        )
        produced: list[object] = []
        for event in gen:
            produced.append(event)
            if isinstance(event, StreamToken) and event.content == "first ":
                gen.close()
                break
        assert second_closed["value"]


class TestStripReasoning:
    def test_strips_think_block(self):
        assert strip_reasoning("<think>internal</think>answer") == "answer"

    def test_no_think_block(self):
        assert strip_reasoning("plain text") == "plain text"

    def test_multiple_blocks(self):
        text = "<think>a</think>one<think>b</think>two"
        assert strip_reasoning(text) == "onetwo"

    def test_multiline_think(self):
        text = "<think>\nline1\nline2\n</think>\nresult"
        assert strip_reasoning(text) == "result"

    def test_empty_string(self):
        assert strip_reasoning("") == ""

    def test_trailing_whitespace_after_tag_stripped(self):
        assert strip_reasoning("<think>x</think>  answer") == "answer"

    def test_unclosed_think_tag_stripped(self):
        assert strip_reasoning("answer<think>truncated reasoning") == "answer"

    def test_only_unclosed_think_tag(self):
        assert strip_reasoning("<think>all reasoning no answer") == ""
