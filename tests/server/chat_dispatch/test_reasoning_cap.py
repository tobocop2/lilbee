"""The reasoning cap on the canonical chat surfaces."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lilbee.retrieval.reasoning import CAP_CONTINUATION_PROMPT, CAP_NOTICE_TEMPLATE
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    CanonicalResponse,
    CanonicalUsage,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    MessageStop,
    StopReason,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    ToolUseDelta,
)
from lilbee.server.chat_dispatch.reasoning_cap import (
    budget_capped_chars,
    cap_aware_chat,
    cap_aware_chat_stream,
    nudged_request,
)

_MODEL = "vendor/Model-GGUF/model-Q4.gguf"


def _request(**overrides) -> CanonicalChatRequest:
    body = {
        "model": _MODEL,
        "messages": [CanonicalMessage.from_string(role="user", text="hi")],
    }
    body.update(overrides)
    return CanonicalChatRequest(**body)


def _response(text: str, *, tool_calls=(), output_tokens: int = 5) -> CanonicalResponse:
    content: list = [TextBlock(text=text)] if text else []
    content.extend(tool_calls)
    return CanonicalResponse(
        id="x",
        model=_MODEL,
        content=content,
        stop_reason=StopReason.TOOL_USE if tool_calls else StopReason.END_TURN,
        usage=CanonicalUsage(input_tokens=10, output_tokens=output_tokens),
    )


def _text_of(resp: CanonicalResponse) -> str:
    return "".join(b.text for b in resp.content if isinstance(b, TextBlock))


async def _collect(events, req, *, cap_chars):
    """Drain the cap wrapper over a canned first stream."""
    return [
        event
        async for event in cap_aware_chat_stream(
            events, req, canonical_model=_MODEL, cap_chars=cap_chars
        )
    ]


def _stream_of(*texts):
    """An async canonical stream that emits one text block of *texts*."""

    async def _gen():
        yield MessageStart(id="m", model=_MODEL)
        yield ContentBlockStart(index=0, block=TextBlock(text=""))
        for text in texts:
            yield ContentBlockDelta(index=0, delta=TextDelta(text=text))
        yield ContentBlockStop(index=0)
        yield MessageDelta(
            stop_reason=StopReason.END_TURN, usage=CanonicalUsage(input_tokens=1, output_tokens=2)
        )
        yield MessageStop()

    return _gen()


def _text_deltas(events) -> str:
    return "".join(
        e.delta.text
        for e in events
        if isinstance(e, ContentBlockDelta) and isinstance(e.delta, TextDelta)
    )


class TestBudgetCappedChars:
    """``budget_tokens`` may tighten a configured cap, never loosen it."""

    def test_no_budget_keeps_the_configured_cap(self):
        assert budget_capped_chars(64_000, None) == 64_000

    def test_budget_tightens_a_larger_cap(self):
        assert budget_capped_chars(64_000, 1_000) == 4_000

    def test_budget_cannot_loosen_a_smaller_cap(self):
        assert budget_capped_chars(4_000, 1_000_000) == 4_000

    def test_budget_bounds_an_unlimited_cap(self):
        """0 means unlimited, so any budget is a tightening."""
        assert budget_capped_chars(0, 1_000) == 4_000

    def test_no_budget_leaves_unlimited_unlimited(self):
        assert budget_capped_chars(0, None) == 0

    def test_negative_budget_is_ignored(self):
        assert budget_capped_chars(64_000, -1) == 64_000

    def test_zero_budget_does_not_erase_the_cap(self):
        """0 chars would read as unlimited downstream, inverting the rule."""
        assert budget_capped_chars(64_000, 0) == 64_000

    def test_zero_budget_leaves_unlimited_unlimited(self):
        assert budget_capped_chars(0, 0) == 0


class TestNudgedRequest:
    def test_appends_the_continuation_prompt_and_keeps_tools(self):
        tools = [MagicMock()]
        req = _request(tools=tools, temperature=0.5)
        nudged = nudged_request(req)
        assert len(nudged.messages) == len(req.messages) + 1
        last = nudged.messages[-1]
        assert last.role == "user"
        assert last.content[0].text == CAP_CONTINUATION_PROMPT
        # A capped turn must still be able to call tools and keep its sampling.
        assert nudged.tools is tools
        assert nudged.temperature == 0.5


class TestCapAwareChatStream:
    """The streaming cap: stop the reasoning, then force an answer."""

    @pytest.mark.asyncio
    async def test_uncapped_stream_passes_through_untouched(self):
        req = _request()
        events = await _collect(_stream_of("<think>a</think>", "answer"), req, cap_chars=0)
        assert _text_deltas(events) == "<think>a</think>answer"
        assert isinstance(events[-1], MessageStop)

    @pytest.mark.asyncio
    async def test_capped_stream_reports_both_calls_usage(self):
        """Streaming must count the capped generation, like the non-streaming path.

        Pairs with ``test_re_issued_response_sums_both_calls_usage``: the same
        40 + 7 tokens over the streaming arm. Without this the client is billed
        for a reasoning chain it is never told about.
        """

        async def _first():
            yield MessageStart(id="m", model=_MODEL)
            yield ContentBlockStart(index=0, block=TextBlock(text=""))
            # Providers that report usage as they go put it here; one that only
            # reports at the end has none to add, because the cap closes the
            # stream first.
            yield MessageDelta(usage=CanonicalUsage(input_tokens=3, output_tokens=40))
            yield ContentBlockDelta(index=0, delta=TextDelta(text="<think>" + "x" * 100))

        def _continuation(*_a, **_k):
            async def _gen():
                yield MessageStart(id="m2", model=_MODEL)
                yield ContentBlockStart(index=0, block=TextBlock(text=""))
                yield ContentBlockDelta(index=0, delta=TextDelta(text="answer"))
                yield ContentBlockStop(index=0)
                yield MessageDelta(
                    stop_reason=StopReason.END_TURN,
                    usage=CanonicalUsage(input_tokens=1, output_tokens=7),
                )
                yield MessageStop()

            return _gen()

        with patch(
            "lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat_stream",
            side_effect=_continuation,
        ):
            events = await _collect(_first(), _request(), cap_chars=10)
        usage = [e.usage for e in events if isinstance(e, MessageDelta) and e.usage][-1]
        assert usage.output_tokens == 47
        assert usage.input_tokens == 4

    @pytest.mark.asyncio
    async def test_reasoning_under_the_cap_is_not_interrupted(self):
        req = _request()
        with patch("lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat_stream") as dispatch:
            events = await _collect(_stream_of("<think>ab</think>", "answer"), req, cap_chars=100)
        dispatch.assert_not_called()
        assert _text_deltas(events) == "<think>ab</think>answer"

    @pytest.mark.asyncio
    async def test_cap_stops_the_reasoning_and_splices_the_answer(self):
        """The client sees the truncated reasoning, the notice, then a real answer."""
        req = _request()
        with patch(
            "lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat_stream",
            return_value=_stream_of("forced answer"),
        ) as dispatch:
            events = await _collect(
                _stream_of("<think>", "x" * 50, "y" * 50, "</think>", "never"),
                req,
                cap_chars=10,
            )
        text = _text_deltas(events)
        # The reasoning is cut off mid-block, so the rest of the first stream
        # (including its own answer) never arrives.
        assert "never" not in text
        assert CAP_NOTICE_TEMPLATE.format(chars=10) in text
        assert text.endswith("</think>forced answer")
        # The notice lands inside the thinking block, so every reasoning mode
        # keeps it out of the answer.
        assert text.index(CAP_NOTICE_TEMPLATE.format(chars=10)) < text.index("</think>")
        dispatch.assert_called_once()
        nudged = dispatch.call_args.args[0]
        assert nudged.messages[-1].content[0].text == CAP_CONTINUATION_PROMPT

    @pytest.mark.asyncio
    async def test_cap_closes_the_upstream_stream(self):
        """The point of the cap is to stop paying for the reasoning."""
        closed = False

        async def _endless():
            nonlocal closed
            try:
                yield ContentBlockStart(index=0, block=TextBlock(text=""))
                yield ContentBlockDelta(index=0, delta=TextDelta(text="<think>"))
                while True:
                    yield ContentBlockDelta(index=0, delta=TextDelta(text="z" * 20))
            finally:
                closed = True

        with patch(
            "lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat_stream",
            return_value=_stream_of("done"),
        ):
            await _collect(_endless(), _request(), cap_chars=10)
        assert closed is True

    @pytest.mark.asyncio
    async def test_cap_tolerates_a_stream_that_cannot_be_closed(self):
        """A plain async iterator has no aclose; the cap must still fire."""

        class _Iterator:
            def __init__(self) -> None:
                self._events = [
                    ContentBlockStart(index=0, block=TextBlock(text="")),
                    ContentBlockDelta(index=0, delta=TextDelta(text="<think>" + "x" * 50)),
                ]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._events:
                    raise StopAsyncIteration
                return self._events.pop(0)

        with patch(
            "lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat_stream",
            return_value=_stream_of("forced answer"),
        ):
            events = await _collect(_Iterator(), _request(), cap_chars=10)
        assert "forced answer" in _text_deltas(events)

    @pytest.mark.asyncio
    async def test_continuation_is_not_capped_again(self):
        """Re-capping the continuation would cut the forced answer off."""
        req = _request()
        continuation = _stream_of("<think>" + "q" * 500 + "</think>", "late answer")
        with patch(
            "lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat_stream",
            return_value=continuation,
        ) as dispatch:
            events = await _collect(_stream_of("<think>", "x" * 50), req, cap_chars=10)
        assert dispatch.call_count == 1
        assert "late answer" in _text_deltas(events)

    @pytest.mark.asyncio
    async def test_tool_calls_survive_the_cap_wrapper(self):
        """The chat surfaces stream tool calls; the wrapper must not eat them."""

        async def _tool_stream():
            yield ContentBlockStart(index=0, block=ToolUseBlock(id="t1", name="search", input={}))
            yield ContentBlockDelta(index=0, delta=ToolUseDelta(partial_json='{"q":1}'))
            yield ContentBlockStop(index=0)
            yield MessageDelta(stop_reason=StopReason.TOOL_USE)
            yield MessageStop()

        events = await _collect(_tool_stream(), _request(), cap_chars=10)
        assert any(
            isinstance(e, ContentBlockDelta) and isinstance(e.delta, ToolUseDelta) for e in events
        )
        assert any(isinstance(e, ContentBlockStart) for e in events)

    @pytest.mark.asyncio
    async def test_continuation_blocks_do_not_reuse_a_first_stream_index(self):
        """Colliding indices would merge two blocks in the surface translators."""

        async def _continuation():
            yield MessageStart(id="m2", model=_MODEL)
            yield ContentBlockStart(index=0, block=TextBlock(text=""))
            yield ContentBlockDelta(index=0, delta=TextDelta(text="answer"))
            yield ContentBlockStop(index=0)
            yield MessageStop()

        with patch(
            "lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat_stream",
            return_value=_continuation(),
        ):
            events = await _collect(_stream_of("<think>", "x" * 50), _request(), cap_chars=10)
        indices = {e.index for e in events if isinstance(e, ContentBlockStart)}
        assert indices == {0, 1}
        # The continuation's own prelude must not restart the message.
        assert sum(isinstance(e, MessageStart) for e in events) == 1

    @pytest.mark.asyncio
    async def test_cap_does_not_fire_once_the_answer_started(self):
        """Long reasoning that already finished needs no continuation."""
        req = _request()
        with patch("lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat_stream") as dispatch:
            events = await _collect(
                _stream_of("<think>" + "x" * 100 + "</think>answer"), req, cap_chars=10
            )
        dispatch.assert_not_called()
        assert "answer" in _text_deltas(events)


class TestCapAwareChat:
    """The non-streaming cap: force an answer when reasoning produced none."""

    def _dispatch(self, *responses):
        return patch(
            "lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat",
            side_effect=list(responses),
        )

    def test_uncapped_response_passes_through(self):
        resp = _response("<think>" + "x" * 100 + "</think>")
        with self._dispatch(resp) as dispatch:
            out = cap_aware_chat(_request(), canonical_model=_MODEL, cap_chars=0)
        assert out is resp
        assert dispatch.call_count == 1

    def test_reasoning_under_the_cap_passes_through(self):
        resp = _response("<think>ab</think>answer")
        with self._dispatch(resp):
            out = cap_aware_chat(_request(), canonical_model=_MODEL, cap_chars=100)
        assert out is resp

    def test_long_reasoning_with_an_answer_is_kept(self):
        """The cap exists to get an answer; this turn already has one."""
        resp = _response("<think>" + "x" * 100 + "</think>answer")
        with self._dispatch(resp) as dispatch:
            out = cap_aware_chat(_request(), canonical_model=_MODEL, cap_chars=10)
        assert dispatch.call_count == 1
        assert out is resp

    def test_long_reasoning_with_a_tool_call_is_kept(self):
        resp = _response(
            "<think>" + "x" * 100 + "</think>",
            tool_calls=(ToolUseBlock(id="t1", name="search", input={}),),
        )
        with self._dispatch(resp) as dispatch:
            out = cap_aware_chat(_request(), canonical_model=_MODEL, cap_chars=10)
        assert dispatch.call_count == 1
        assert out is resp

    def test_reasoning_only_response_is_re_issued_with_the_nudge(self):
        first = _response("<think>" + "x" * 100 + "</think>")
        second = _response("forced answer", output_tokens=7)
        with self._dispatch(first, second) as dispatch:
            out = cap_aware_chat(_request(), canonical_model=_MODEL, cap_chars=10)
        assert dispatch.call_count == 2
        nudged = dispatch.call_args_list[1].args[0]
        assert nudged.messages[-1].content[0].text == CAP_CONTINUATION_PROMPT
        text = _text_of(out)
        assert text.endswith("forced answer")
        assert CAP_NOTICE_TEMPLATE.format(chars=10) in text

    def test_re_issued_response_truncates_the_reasoning_to_the_cap(self):
        first = _response("<think>" + "x" * 100 + "</think>")
        with self._dispatch(first, _response("answer")):
            out = cap_aware_chat(_request(), canonical_model=_MODEL, cap_chars=10)
        assert "x" * 100 not in _text_of(out)
        assert "x" * 10 in _text_of(out)

    def test_re_issued_response_sums_both_calls_usage(self):
        """The caller paid for the capped turn too; the count must say so."""
        first = _response("<think>" + "x" * 100 + "</think>", output_tokens=40)
        second = _response("answer", output_tokens=7)
        with self._dispatch(first, second):
            out = cap_aware_chat(_request(), canonical_model=_MODEL, cap_chars=10)
        assert out.usage.output_tokens == 47

    def test_continuation_tool_calls_are_kept(self):
        first = _response("<think>" + "x" * 100 + "</think>")
        second = _response("", tool_calls=(ToolUseBlock(id="t1", name="search", input={}),))
        with self._dispatch(first, second):
            out = cap_aware_chat(_request(), canonical_model=_MODEL, cap_chars=10)
        assert any(isinstance(b, ToolUseBlock) for b in out.content)
        assert out.stop_reason is StopReason.TOOL_USE
