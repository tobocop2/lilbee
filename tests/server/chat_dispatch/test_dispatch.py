"""Tests for the canonical chat-dispatch layer."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.app.services import set_services
from lilbee.providers.base import (
    ChatResult,
    FinishReason,
    StreamFinish,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolChoice,
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
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
    dispatch_chat,
    dispatch_chat_stream,
)


@pytest.fixture
def services_with_model(monkeypatch):
    """Install a mock services container with one installed model."""
    from lilbee.core.config import cfg
    from tests.conftest import make_mock_services

    monkeypatch.setattr(cfg, "chat_model", "vendor/model::Q4")
    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text="hello", tool_calls=(), finish_reason=FinishReason.STOP
    )
    provider.supports_tools.return_value = False

    services = make_mock_services(provider=provider)
    installed = MagicMock()
    installed.ref = "vendor/model::Q4"
    services.registry.list_installed = MagicMock(return_value=[installed])

    # KnownModelCache normally walks the registry + Ollama tags + frontier
    # APIs; here we hand it a fixed set so dispatch tests don't run real
    # discovery. The resolve() helper mirrors the production semantics:
    # canonical match wins, then the ``ollama/<bare:tag>`` probe.
    known = {"vendor/model::Q4", "ollama/gemma4:26b"}
    services.known_models.refs = MagicMock(return_value=known)

    def _resolve(model: str) -> str | None:
        if model in known:
            return model
        if "/" not in model and ":" in model:
            prefixed = f"ollama/{model}"
            if prefixed in known:
                return prefixed
        return None

    services.known_models.resolve = MagicMock(side_effect=_resolve)

    set_services(services)
    yield services
    set_services(None)


def _req(**overrides: Any) -> CanonicalChatRequest:
    base = {
        "model": "vendor/model::Q4",
        "messages": [CanonicalMessage(role="user", content=[TextBlock(text="hi")])],
    }
    base.update(overrides)
    return CanonicalChatRequest(**base)


class TestDispatchChat:
    def test_text_only_response(self, services_with_model) -> None:
        services_with_model.provider.chat.return_value = ChatResult(
            text="hello", tool_calls=(), finish_reason=FinishReason.STOP
        )
        resp = dispatch_chat(_req())
        assert resp.model == "vendor/model::Q4"
        assert resp.content == [TextBlock(text="hello")]
        assert resp.stop_reason == StopReason.END_TURN
        assert resp.id.startswith("msg_")

    def test_tool_calls_force_tool_use_even_when_the_provider_says_stop(
        self, services_with_model
    ) -> None:
        """A provider that returns tool calls without saying so still means tool_use.

        FinishReason.coerce falls back to STOP for a missing or unknown value,
        so the response carried tool_use content under end_turn and a client
        reading stop_reason would not run the tools. The streaming path already
        refuses the same downgrade.
        """
        from lilbee.providers.base import ToolCall

        services_with_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="call_1", name="search", arguments='{"q": "x"}'),),
            finish_reason=FinishReason.STOP,
        )
        resp = dispatch_chat(_req())
        assert resp.stop_reason == StopReason.TOOL_USE
        assert any(getattr(block, "name", None) == "search" for block in resp.content)

    def test_empty_text_returns_empty_content(self, services_with_model) -> None:
        services_with_model.provider.chat.return_value = ChatResult(
            text="", tool_calls=(), finish_reason=FinishReason.STOP
        )
        resp = dispatch_chat(_req())
        assert resp.content == []

    def test_usage_threaded_from_result(self, services_with_model) -> None:
        """ChatResult.usage flows into the canonical response usage. (F4)"""
        services_with_model.provider.chat.return_value = ChatResult(
            text="hello",
            tool_calls=(),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=11, completion_tokens=4),
        )
        resp = dispatch_chat(_req())
        assert resp.usage == CanonicalUsage(input_tokens=11, output_tokens=4)

    def test_max_tokens_finish_reason_maps_to_max_tokens(self, services_with_model) -> None:
        services_with_model.provider.chat.return_value = ChatResult(
            text="cut", tool_calls=(), finish_reason=FinishReason.LENGTH
        )
        resp = dispatch_chat(_req())
        assert resp.stop_reason == StopReason.MAX_TOKENS

    def test_content_filter_finish_reason_maps_to_end_turn(self, services_with_model) -> None:
        services_with_model.provider.chat.return_value = ChatResult(
            text="", tool_calls=(), finish_reason=FinishReason.CONTENT_FILTER
        )
        resp = dispatch_chat(_req())
        assert resp.stop_reason == StopReason.END_TURN

    def test_tool_call_becomes_tool_use_block(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        services_with_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="c1", name="search", arguments='{"q": "foo"}'),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        req = _req(
            tools=[
                CanonicalTool(
                    name="search",
                    description="Search",
                    input_schema={"type": "object"},
                )
            ],
        )
        resp = dispatch_chat(req)
        assert resp.stop_reason == StopReason.TOOL_USE
        tool_blocks = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].id == "c1"
        assert tool_blocks[0].name == "search"
        assert tool_blocks[0].input == {"q": "foo"}

    def test_tool_call_with_blank_id_gets_generated_id(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        services_with_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="", name="search", arguments="{}"),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        resp = dispatch_chat(
            _req(tools=[CanonicalTool(name="search", description="", input_schema={})])
        )
        tool_blocks = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks[0].id.startswith("call_")

    def test_malformed_tool_arguments_fall_back_to_raw(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        services_with_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="c1", name="search", arguments="not json{"),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        resp = dispatch_chat(
            _req(tools=[CanonicalTool(name="search", description="", input_schema={})])
        )
        tool_blocks = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks[0].input == {"_raw": "not json{"}

    def test_empty_tool_arguments_parse_to_empty_dict(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        services_with_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="c1", name="search", arguments=""),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        resp = dispatch_chat(
            _req(tools=[CanonicalTool(name="search", description="", input_schema={})])
        )
        tool_blocks = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks[0].input == {}

    def test_unknown_model_raises_model_not_found(self, services_with_model) -> None:
        with pytest.raises(ModelNotFoundError) as exc_info:
            dispatch_chat(_req(model="missing/model::Q4"))
        assert exc_info.value.model == "missing/model::Q4"

    def test_remote_prefixed_ref_resolves_via_cache(self, services_with_model) -> None:
        """A canonical ``ollama/...`` ref lands directly in the discovered set;
        the route hands it to the provider verbatim (bb-zsnf).
        """
        services_with_model.provider.chat.return_value = ChatResult(
            text="ok", tool_calls=(), finish_reason=FinishReason.STOP
        )
        resp = dispatch_chat(_req(model="ollama/gemma4:26b"))
        assert resp.model == "ollama/gemma4:26b"
        # Provider sees the canonical form, matching the cached ref.
        assert services_with_model.provider.chat.call_args.kwargs["model"] == "ollama/gemma4:26b"

    def test_bare_ollama_name_canonicalizes_via_cache(self, services_with_model) -> None:
        """A bare ``name:tag`` resolves to the cache's ``ollama/<name:tag>`` entry
        and the canonical form is what reaches the provider (bb-zsnf).
        """
        services_with_model.provider.chat.return_value = ChatResult(
            text="ok", tool_calls=(), finish_reason=FinishReason.STOP
        )
        resp = dispatch_chat(_req(model="gemma4:26b"))
        assert resp.model == "ollama/gemma4:26b"
        assert services_with_model.provider.chat.call_args.kwargs["model"] == "ollama/gemma4:26b"

    def test_bare_name_without_cache_entry_raises(self, services_with_model) -> None:
        """A bare ``name:tag`` with no matching ``ollama/<name:tag>`` in the
        discovered set surfaces ``ModelNotFoundError`` instead of silently
        forwarding a guess to Ollama.
        """
        with pytest.raises(ModelNotFoundError) as exc_info:
            dispatch_chat(_req(model="nonexistent:99b"))
        assert exc_info.value.model == "nonexistent:99b"

    def test_tools_against_unsupported_model_raises(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = False
        with pytest.raises(ModelDoesNotSupportToolsError) as exc_info:
            dispatch_chat(_req(tools=[CanonicalTool(name="x", description="", input_schema={})]))
        assert exc_info.value.model == "vendor/model::Q4"

    def test_system_prompt_is_forwarded_as_system_message(self, services_with_model) -> None:
        dispatch_chat(_req(system="be terse"))
        sent_messages = services_with_model.provider.chat.call_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "system", "content": "be terse"}

    def test_options_passed_to_provider(self, services_with_model) -> None:
        dispatch_chat(
            _req(
                temperature=0.2,
                top_p=0.8,
                top_k=20,
                max_tokens=64,
                stop=["</s>"],
            )
        )
        opts = services_with_model.provider.chat.call_args.kwargs["options"]
        assert opts == {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 20,
            "num_predict": 64,
            "stop": ["</s>"],
        }

    def test_seed_and_penalties_passed_to_provider(self, services_with_model) -> None:
        dispatch_chat(
            _req(seed=7, frequency_penalty=0.5, presence_penalty=-0.5),
        )
        opts = services_with_model.provider.chat.call_args.kwargs["options"]
        assert opts == {
            "seed": 7,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
        }

    def test_seed_and_penalties_omitted_when_absent(self, services_with_model) -> None:
        """Unset sampler fields never reach the provider options."""
        dispatch_chat(_req(temperature=0.2))
        opts = services_with_model.provider.chat.call_args.kwargs["options"]
        assert opts == {"temperature": 0.2}

    def test_think_false_passed_to_provider(self, services_with_model) -> None:
        dispatch_chat(_req(think=False))
        opts = services_with_model.provider.chat.call_args.kwargs["options"]
        assert opts == {"think": False}

    def test_think_omitted_when_unset(self, services_with_model) -> None:
        dispatch_chat(_req(temperature=0.2))
        opts = services_with_model.provider.chat.call_args.kwargs["options"]
        assert "think" not in opts

    def test_tool_use_message_round_trips_as_assistant_tool_calls(
        self, services_with_model
    ) -> None:
        history = [
            CanonicalMessage(role="user", content=[TextBlock(text="search foo")]),
            CanonicalMessage(
                role="assistant",
                content=[
                    TextBlock(text="ok"),
                    ToolUseBlock(id="c1", name="search", input={"q": "foo"}),
                ],
            ),
            CanonicalMessage(
                role="tool",
                content=[ToolResultBlock(tool_use_id="c1", content=[TextBlock(text="42")])],
            ),
        ]
        dispatch_chat(_req(messages=history))
        sent = services_with_model.provider.chat.call_args.kwargs["messages"]
        # First message: plain user text
        assert sent[0] == {"role": "user", "content": "search foo"}
        # Second: assistant with tool_calls
        assert sent[1]["role"] == "assistant"
        assert sent[1]["content"] == "ok"
        assert sent[1]["tool_calls"][0]["id"] == "c1"
        assert sent[1]["tool_calls"][0]["type"] == "function"
        assert sent[1]["tool_calls"][0]["function"]["name"] == "search"
        assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"q": "foo"}'
        # Third: tool role with tool_call_id
        assert sent[2] == {"role": "tool", "tool_call_id": "c1", "content": "42"}

    def test_provider_tools_translation(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        dispatch_chat(
            _req(
                tools=[
                    CanonicalTool(
                        name="search",
                        description="Find docs",
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
            )
        )
        tools = services_with_model.provider.chat.call_args.kwargs["tools"]
        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Find docs",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    @pytest.mark.parametrize(
        "choice,expected",
        [
            (CanonicalToolChoice(mode="auto"), "auto"),
            (CanonicalToolChoice(mode="any"), "required"),
            (CanonicalToolChoice(mode="none"), "none"),
            (
                CanonicalToolChoice(mode="tool", tool_name="search"),
                {"type": "function", "function": {"name": "search"}},
            ),
        ],
    )
    def test_provider_tool_choice_translation(self, services_with_model, choice, expected) -> None:
        services_with_model.provider.supports_tools.return_value = True
        dispatch_chat(
            _req(
                tools=[CanonicalTool(name="search", description="", input_schema={})],
                tool_choice=choice,
            )
        )
        sent = services_with_model.provider.chat.call_args.kwargs["tool_choice"]
        assert sent == expected

    def test_assistant_text_only_message_translates_to_plain_content(
        self, services_with_model
    ) -> None:
        history = [
            CanonicalMessage(role="assistant", content=[TextBlock(text="prior")]),
            CanonicalMessage(role="user", content=[TextBlock(text="next")]),
        ]
        dispatch_chat(_req(messages=history))
        sent = services_with_model.provider.chat.call_args.kwargs["messages"]
        assert sent[0] == {"role": "assistant", "content": "prior"}

    def test_tool_role_with_multiple_results_emits_one_per_block(self, services_with_model) -> None:
        history = [
            CanonicalMessage(
                role="tool",
                content=[
                    ToolResultBlock(tool_use_id="a", content=[TextBlock(text="A")]),
                    ToolResultBlock(tool_use_id="b", content=[TextBlock(text="B")]),
                ],
            )
        ]
        dispatch_chat(_req(messages=history))
        sent = services_with_model.provider.chat.call_args.kwargs["messages"]
        # Two tool entries, one per result block
        assert sent == [
            {"role": "tool", "tool_call_id": "a", "content": "A"},
            {"role": "tool", "tool_call_id": "b", "content": "B"},
        ]

    def test_tool_results_with_text_keep_the_text_message(self, services_with_model) -> None:
        """A message carrying tool results AND text emits both, results first."""
        history = [
            CanonicalMessage(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="a", content=[TextBlock(text="result A")]),
                    TextBlock(text="and here is my follow-up"),
                ],
            )
        ]
        dispatch_chat(_req(messages=history))
        sent = services_with_model.provider.chat.call_args.kwargs["messages"]
        assert sent == [
            {"role": "tool", "tool_call_id": "a", "content": "result A"},
            {"role": "user", "content": "and here is my follow-up"},
        ]

    def test_tool_results_without_text_emit_no_empty_text_message(
        self, services_with_model
    ) -> None:
        history = [
            CanonicalMessage(
                role="tool",
                content=[ToolResultBlock(tool_use_id="a", content=[TextBlock(text="A")])],
            )
        ]
        dispatch_chat(_req(messages=history))
        sent = services_with_model.provider.chat.call_args.kwargs["messages"]
        assert sent == [{"role": "tool", "tool_call_id": "a", "content": "A"}]


class _FakeStream:
    """Test-only ``ClosableIterator[ChatStreamItem]``.

    This used to be an *async* iterator, despite the docstring, so every
    streaming test drove the async-native branch of the dispatch iterator
    rather than the sync path every real provider returns.
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self._frames:
            raise StopIteration
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


class TestConfiguredModelPreflight:
    """An installed-but-not-configured local model is rejected before dispatch."""

    def test_local_model_not_configured_rejected_before_provider_call(
        self, services_with_model, monkeypatch
    ) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        monkeypatch.setattr(cfg, "chat_model", "vendor/other::Q4")
        with pytest.raises(ProviderError) as exc_info:
            dispatch_chat(_req())
        assert exc_info.value.kind is ProviderErrorKind.BAD_REQUEST
        assert "set it as the chat model in lilbee settings" in str(exc_info.value)
        assert "vendor/other::Q4" in str(exc_info.value)
        services_with_model.provider.chat.assert_not_called()

    def test_configured_local_model_passes_preflight(self, services_with_model) -> None:
        from lilbee.server.chat_dispatch.dispatch import preflight_chat_request

        assert preflight_chat_request(_req()) == "vendor/model::Q4"

    def test_remote_ref_differing_from_configured_passes_preflight(
        self, services_with_model
    ) -> None:
        from lilbee.server.chat_dispatch.dispatch import preflight_chat_request

        assert preflight_chat_request(_req(model="ollama/gemma4:26b")) == "ollama/gemma4:26b"

    async def test_stream_rejects_not_configured_model_before_yielding(
        self, services_with_model, monkeypatch
    ) -> None:
        from lilbee.core.config import cfg
        from lilbee.providers.base import ProviderError

        monkeypatch.setattr(cfg, "chat_model", "vendor/other::Q4")
        gen = dispatch_chat_stream(_req())
        with pytest.raises(ProviderError, match="set it as the chat model in lilbee settings"):
            await gen.__anext__()
        services_with_model.provider.chat.assert_not_called()


class TestDispatchChatStream:
    async def _drain(self, gen):
        return [event async for event in gen]

    async def test_text_only_stream_emits_open_delta_close_terminator(
        self, services_with_model
    ) -> None:
        stream = _FakeStream(["he", "llo"])
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(dispatch_chat_stream(_req()))

        assert isinstance(events[0], MessageStart)
        assert isinstance(events[1], ContentBlockStart)
        assert events[1].index == 0
        assert isinstance(events[1].block, TextBlock)
        assert isinstance(events[2], ContentBlockDelta)
        assert events[2].delta == TextDelta(text="he")
        assert isinstance(events[3], ContentBlockDelta)
        assert events[3].delta == TextDelta(text="llo")
        assert isinstance(events[4], ContentBlockStop)
        assert events[4].index == 0
        assert isinstance(events[5], MessageDelta)
        assert events[5].stop_reason == StopReason.END_TURN
        assert isinstance(events[-1], MessageStop)
        assert stream.closed is True

    async def test_stream_usage_frame_attaches_to_message_delta(self, services_with_model) -> None:
        """A trailing TokenUsage frame becomes the closing MessageDelta usage. (F4)"""
        stream = _FakeStream(["hi", TokenUsage(prompt_tokens=8, completion_tokens=2)])
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(dispatch_chat_stream(_req()))
        msg_delta = next(e for e in events if isinstance(e, MessageDelta))
        assert msg_delta.usage == CanonicalUsage(input_tokens=8, output_tokens=2)

    async def test_stream_without_usage_frame_has_no_usage(self, services_with_model) -> None:
        stream = _FakeStream(["hi"])
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(dispatch_chat_stream(_req()))
        msg_delta = next(e for e in events if isinstance(e, MessageDelta))
        assert msg_delta.usage is None

    async def test_stream_length_finish_maps_to_max_tokens(self, services_with_model) -> None:
        """A StreamFinish(LENGTH) terminator closes the stream as MAX_TOKENS. (bb-ziks.46)"""
        stream = _FakeStream(["hi", StreamFinish(reason=FinishReason.LENGTH)])
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(dispatch_chat_stream(_req()))
        msg_delta = next(e for e in events if isinstance(e, MessageDelta))
        assert msg_delta.stop_reason == StopReason.MAX_TOKENS

    async def test_stream_stop_finish_keeps_end_turn(self, services_with_model) -> None:
        stream = _FakeStream(["hi", StreamFinish(reason=FinishReason.STOP)])
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(dispatch_chat_stream(_req()))
        msg_delta = next(e for e in events if isinstance(e, MessageDelta))
        assert msg_delta.stop_reason == StopReason.END_TURN

    async def test_stream_finish_does_not_downgrade_tool_use(self, services_with_model) -> None:
        """A trailing finish frame must not override the TOOL_USE a tool stream set."""
        services_with_model.provider.supports_tools.return_value = True
        stream = _FakeStream(
            [
                ToolCallDelta(index=0, id="c1", name="search", arguments_delta='{"q":"x"}'),
                StreamFinish(reason=FinishReason.STOP),
            ]
        )
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(dispatch_chat_stream(_req()))
        msg_delta = next(e for e in events if isinstance(e, MessageDelta))
        assert msg_delta.stop_reason == StopReason.TOOL_USE

    async def test_empty_stream_yields_message_envelope_only(self, services_with_model) -> None:
        stream = _FakeStream([])
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(dispatch_chat_stream(_req()))
        kinds = [type(e).__name__ for e in events]
        # No content blocks were ever opened; just MessageStart + MessageDelta + MessageStop.
        assert kinds == ["MessageStart", "MessageDelta", "MessageStop"]

    async def test_stream_runs_preflight_off_the_event_loop(
        self, services_with_model, monkeypatch
    ) -> None:
        """The preflight can do blocking model discovery; it must not run on the loop."""
        import threading

        from lilbee.server.chat_dispatch import dispatch as dispatch_mod

        loop_thread = threading.current_thread()
        seen_threads: list[threading.Thread] = []
        real_preflight = dispatch_mod.preflight_chat_request

        def _recording_preflight(req):
            seen_threads.append(threading.current_thread())
            return real_preflight(req)

        monkeypatch.setattr(dispatch_mod, "preflight_chat_request", _recording_preflight)
        services_with_model.provider.chat.return_value = _FakeStream(["hi"])
        await self._drain(dispatch_chat_stream(_req()))
        assert seen_threads and seen_threads[0] is not loop_thread

    async def test_tool_call_stream_opens_and_closes_tool_use_block(
        self, services_with_model
    ) -> None:
        services_with_model.provider.supports_tools.return_value = True
        frames = [
            ToolCallDelta(index=0, id="c1", name="search", arguments_delta=None),
            ToolCallDelta(index=0, id=None, name=None, arguments_delta='{"q":'),
            ToolCallDelta(index=0, id=None, name=None, arguments_delta='"foo"}'),
        ]
        stream = _FakeStream(frames)
        services_with_model.provider.chat.return_value = stream

        events = await self._drain(
            dispatch_chat_stream(
                _req(tools=[CanonicalTool(name="search", description="", input_schema={})])
            )
        )

        starts = [e for e in events if isinstance(e, ContentBlockStart)]
        deltas = [e for e in events if isinstance(e, ContentBlockDelta)]
        stops = [e for e in events if isinstance(e, ContentBlockStop)]
        assert len(starts) == 1
        assert isinstance(starts[0].block, ToolUseBlock)
        assert starts[0].block.id == "c1"
        assert starts[0].block.name == "search"
        # Two argument-fragment deltas
        assert [d.delta for d in deltas] == [
            ToolUseDelta(partial_json='{"q":'),
            ToolUseDelta(partial_json='"foo"}'),
        ]
        assert len(stops) == 1
        msg_delta = next(e for e in events if isinstance(e, MessageDelta))
        assert msg_delta.stop_reason == StopReason.TOOL_USE

    async def test_text_then_tool_call_closes_text_block_first(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        frames = [
            "thinking",
            ToolCallDelta(index=0, id="c1", name="search", arguments_delta=None),
            ToolCallDelta(index=0, id=None, name=None, arguments_delta="{}"),
        ]
        stream = _FakeStream(frames)
        services_with_model.provider.chat.return_value = stream

        events = await self._drain(
            dispatch_chat_stream(
                _req(tools=[CanonicalTool(name="search", description="", input_schema={})])
            )
        )

        # Sequence: MessageStart, BlockStart(text), Delta(text), BlockStop(0),
        # BlockStart(tool@1), Delta(args), BlockStop(1), MessageDelta(TOOL_USE), MessageStop
        kinds = [type(e).__name__ for e in events]
        assert kinds == [
            "MessageStart",
            "ContentBlockStart",
            "ContentBlockDelta",
            "ContentBlockStop",
            "ContentBlockStart",
            "ContentBlockDelta",
            "ContentBlockStop",
            "MessageDelta",
            "MessageStop",
        ]
        block_starts = [e for e in events if isinstance(e, ContentBlockStart)]
        assert isinstance(block_starts[0].block, TextBlock)
        assert isinstance(block_starts[1].block, ToolUseBlock)
        assert block_starts[1].index == 1

    async def test_tool_then_text_opens_new_text_block(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        frames = [
            ToolCallDelta(index=0, id="c1", name="search", arguments_delta=None),
            ToolCallDelta(index=0, id=None, name=None, arguments_delta="{}"),
            "after",
        ]
        stream = _FakeStream(frames)
        services_with_model.provider.chat.return_value = stream

        events = await self._drain(
            dispatch_chat_stream(
                _req(tools=[CanonicalTool(name="search", description="", input_schema={})])
            )
        )
        block_starts = [e for e in events if isinstance(e, ContentBlockStart)]
        assert len(block_starts) == 2
        assert isinstance(block_starts[0].block, ToolUseBlock)
        assert isinstance(block_starts[1].block, TextBlock)

    async def test_stream_closes_iterator_on_generator_exit(self, services_with_model) -> None:
        stream = _FakeStream(["a", "b", "c", "d"])
        services_with_model.provider.chat.return_value = stream

        gen = dispatch_chat_stream(_req())
        # Pull two events then close early.
        first = await gen.__anext__()
        assert isinstance(first, MessageStart)
        await gen.aclose()
        assert stream.closed is True

    async def test_stream_unknown_model_raises_before_yielding(self, services_with_model) -> None:
        gen = dispatch_chat_stream(_req(model="missing/x"))
        with pytest.raises(ModelNotFoundError):
            await gen.__anext__()

    async def test_stream_tools_without_capability_raises_before_yielding(
        self, services_with_model
    ) -> None:
        services_with_model.provider.supports_tools.return_value = False
        gen = dispatch_chat_stream(
            _req(tools=[CanonicalTool(name="x", description="", input_schema={})])
        )
        with pytest.raises(ModelDoesNotSupportToolsError):
            await gen.__anext__()

    async def test_tool_call_with_blank_id_gets_generated_call_id(
        self, services_with_model
    ) -> None:
        services_with_model.provider.supports_tools.return_value = True
        frames = [
            ToolCallDelta(index=0, id=None, name="search", arguments_delta="{}"),
        ]
        stream = _FakeStream(frames)
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(
            dispatch_chat_stream(
                _req(tools=[CanonicalTool(name="search", description="", input_schema={})])
            )
        )
        starts = [e for e in events if isinstance(e, ContentBlockStart)]
        assert isinstance(starts[0].block, ToolUseBlock)
        assert starts[0].block.id.startswith("call_")
        assert starts[0].block.name == "search"

    async def test_new_tool_call_index_closes_prior_tool_block(self, services_with_model) -> None:
        services_with_model.provider.supports_tools.return_value = True
        frames = [
            ToolCallDelta(index=0, id="c1", name="search", arguments_delta="{}"),
            ToolCallDelta(index=1, id="c2", name="open", arguments_delta="{}"),
        ]
        stream = _FakeStream(frames)
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(
            dispatch_chat_stream(
                _req(
                    tools=[
                        CanonicalTool(name="search", description="", input_schema={}),
                        CanonicalTool(name="open", description="", input_schema={}),
                    ]
                )
            )
        )
        starts = [e for e in events if isinstance(e, ContentBlockStart)]
        stops = [e for e in events if isinstance(e, ContentBlockStop)]
        assert len(starts) == 2
        assert all(isinstance(s.block, ToolUseBlock) for s in starts)
        # Two content blocks opened, two closed.
        assert len(stops) == 2

    async def test_text_between_argument_deltas_keeps_one_call_identity(
        self, services_with_model
    ) -> None:
        """A text frame closes the open tool block, so the next delta for the
        same call opens a second one. Continuation deltas carry no id or name,
        so it used to get a synthetic id and an empty name, splitting one call
        across two blocks."""
        services_with_model.provider.supports_tools.return_value = True
        frames = [
            ToolCallDelta(index=0, id="c1", name="search", arguments_delta='{"q":'),
            "thinking out loud",
            ToolCallDelta(index=0, id=None, name=None, arguments_delta='"x"}'),
        ]
        services_with_model.provider.chat.return_value = _FakeStream(frames)
        events = await self._drain(
            dispatch_chat_stream(
                _req(tools=[CanonicalTool(name="search", description="", input_schema={})])
            )
        )
        tool_starts = [
            e
            for e in events
            if isinstance(e, ContentBlockStart) and isinstance(e.block, ToolUseBlock)
        ]
        assert len(tool_starts) == 2
        assert {s.block.id for s in tool_starts} == {"c1"}
        assert {s.block.name for s in tool_starts} == {"search"}


class _SyncFakeStream:
    """Sync-only iterator that mimics the SDK provider's chat-stream return.

    The SDK path (``providers/sdk_llm_provider.py::_chat_stream``) is a plain
    sync generator with no ``__aiter__``. Used to verify ``dispatch_chat_stream``
    can drive a non-async-iterable provider stream without raising
    ``TypeError: object is not an async iterator``.
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self._frames:
            raise StopIteration
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


class TestDispatchChatStreamSyncProvider:
    """``dispatch_chat_stream`` must work for providers whose stream is sync-only."""

    async def _drain(self, gen):
        return [event async for event in gen]

    async def test_drains_sync_iterator_without_blocking_event_loop(
        self, services_with_model
    ) -> None:
        stream = _SyncFakeStream(["hi", " there"])
        services_with_model.provider.chat.return_value = stream

        events = await self._drain(dispatch_chat_stream(_req()))

        text_deltas = [e.delta.text for e in events if isinstance(e, ContentBlockDelta)]
        assert text_deltas == ["hi", " there"]
        assert isinstance(events[0], MessageStart)
        assert isinstance(events[-1], MessageStop)
        # Stream must be closed in dispatch_chat_stream's finally block.
        assert stream.closed is True

    async def test_drains_empty_sync_iterator(self, services_with_model) -> None:
        stream = _SyncFakeStream([])
        services_with_model.provider.chat.return_value = stream

        events = await self._drain(dispatch_chat_stream(_req()))

        kinds = [type(e).__name__ for e in events]
        assert kinds == ["MessageStart", "MessageDelta", "MessageStop"]
        assert stream.closed is True


# One turn of a Claude Code conversation: a system prompt, three messages, two
# tool schemas. Each arm's number is llama-server's own ``usage.prompt_tokens``
# for the rendered prompt, measured on Qwen3-0.6B-Q8_0, and pins the direction
# of the estimate rather than its value.
_ENGINE_SYSTEM = (
    "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
    "You are an interactive CLI tool that helps users with software engineering tasks.\n"
    "Use the instructions below and the tools available to you.\n\n"
    "IMPORTANT: Refuse to write code that may be used maliciously.\n"
)
_ENGINE_MESSAGES = [
    CanonicalMessage(
        role="user",
        content=[
            TextBlock(
                text="Read the config file and tell me what the chunk size is.\nThen explain why."
            )
        ],
    ),
    CanonicalMessage(role="assistant", content=[TextBlock(text="I'll read the file.")]),
    CanonicalMessage(role="user", content=[TextBlock(text="go ahead")]),
]
_ENGINE_TOOLS = [
    CanonicalTool(
        name="Read",
        description="Reads a file from the local filesystem.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The absolute path to the file"},
                "limit": {"type": "integer", "description": "The number of lines to read"},
            },
            "required": ["file_path"],
        },
    ),
    CanonicalTool(
        name="Bash",
        description="Executes a bash command and returns its output.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to execute"},
                "timeout": {"type": "number", "description": "Timeout in ms"},
            },
            "required": ["command"],
        },
    ),
]
_MEASURED_PROMPT_TOKENS = [
    ({}, 43),
    ({"system": _ENGINE_SYSTEM}, 100),
    ({"tools": _ENGINE_TOOLS}, 302),
    ({"system": _ENGINE_SYSTEM, "tools": _ENGINE_TOOLS}, 354),
]
_ONE_TURN = [CanonicalMessage(role="user", content=[TextBlock(text="hi")])]
# The same measurement for the cases the template's fixed tool preamble
# dominates: one short turn and one schema, with and without a forced call.
_SMALL_TOOL = CanonicalTool(
    name="Read",
    description="Reads a file.",
    input_schema={"type": "object", "properties": {"p": {"type": "string"}}, "required": ["p"]},
)
_MEASURED_TOOL_TOKENS = [
    pytest.param(_ENGINE_TOOLS[0], None, 182, id="one-schema"),
    pytest.param(_SMALL_TOOL, CanonicalToolChoice(mode="tool", tool_name="Read"), 145, id="forced"),
]
# Input that tokenizes at roughly one token per character or per UTF-8 byte. A
# chars-per-token average reads these short by up to three times, so they pin
# the direction of the fallback against any return to a ratio. Same instrument.
_DENSE_DIGITS = " ".join("0123456789" * 40)
_DENSE_IDS = "aG3kZpQ9xVb2Lm7TyRc0" * 40
_DENSE_CODE = "\n".join(f"    x{i} = {i}" for i in range(200))
_DENSE_JSON = '{ "a" : 1 , "b" : 2 , "c" : 3 }\n' * 40
_DENSE_CJK = "𠀀𠀁𠀂𠀃𠀄" * 40
_DENSE_EMOJI = "😀🧑‍🔬🚀🦄🌍" * 40
_MEASURED_DENSE_TOKENS = [
    pytest.param(_DENSE_DIGITS, 807, id="spaced-digits"),
    pytest.param(_DENSE_IDS, 768, id="random-ids"),
    pytest.param(_DENSE_CODE, 1987, id="newline-dense-code"),
    pytest.param(_DENSE_JSON, 888, id="pretty-printed-json"),
    pytest.param(_DENSE_CJK, 608, id="rare-cjk"),
    pytest.param(_DENSE_EMOJI, 328, id="emoji"),
]
# The same measurement on SmolLM3-3B-Q4_K_M, whose template substitutes a fixed
# system block of about 215 tokens when the request carries none. Every arm here
# holds the request text near empty, so the template rather than the input
# decides the count, which is the case a byte count of the request cannot see.
_MINIMAL_TOOL = CanonicalTool(name="a", description="", input_schema={})
_BIG_TOOLS = [
    CanonicalTool(
        name=f"Tool{i}",
        description="Reads a file from the local filesystem and returns its text.",
        input_schema={
            "type": "object",
            "properties": {
                f"field_{n}": {
                    "type": "string",
                    "description": (
                        "A parameter documented at the length a real tool documents one."
                    ),
                }
                for n in range(8)
            },
            "required": ["field_0"],
        },
    )
    for i in range(4)
]
_MANY_TURNS = [
    CanonicalMessage(role="user" if i % 2 == 0 else "assistant", content=[TextBlock(text=str(i))])
    for i in range(32)
]
_MEASURED_TEMPLATE_TOKENS = [
    pytest.param({"messages": [CanonicalMessage(role="user", content=[])]}, 248, id="empty-turn"),
    pytest.param({"messages": _ONE_TURN}, 249, id="one-short-turn"),
    pytest.param({"messages": _ONE_TURN, "tools": [_MINIMAL_TOOL]}, 354, id="minimal-tool"),
    pytest.param({"messages": _ONE_TURN, "tools": [_MINIMAL_TOOL] * 8}, 529, id="eight-tools"),
    pytest.param({"messages": _ONE_TURN, "tools": _BIG_TOOLS}, 1445, id="four-big-schemas"),
    pytest.param({"messages": _MANY_TURNS}, 432, id="thirty-two-turns"),
]
# Ordinary prose, where the estimate is at its loosest: one UTF-8 byte per token
# is the safe direction but English runs four to five bytes to the token. Same
# instrument, Qwen3-0.6B-Q8_0.
_PROSE_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. Retrieval augmented generation "
    "systems combine a vector index with a language model so answers can cite the "
    "documents they came from. This paragraph exists to measure how an estimate "
    "behaves on ordinary English prose rather than on dense identifiers. "
) * 4
_MEASURED_PROSE_TOKENS = 217
_MEASURED_PROSE_CEILING = 7.0
# One tool call and the log it returned, measured at 2656 tokens on Qwen3-0.6B-Q8_0.
# The call arguments and the result body are most of the prompt here, so the arm
# fails if either stops being counted.
_TOOL_LOG = (
    "2026-09-18T10:00:00Z INFO indexed chunk 41 of 900 from survey_report.pdf\n"
    "2026-09-18T10:00:01Z INFO indexed chunk 42 of 900 from survey_report.pdf\n"
) * 30
_TOOL_EXCHANGE = [
    CanonicalMessage(role="user", content=[TextBlock(text="read it")]),
    CanonicalMessage(
        role="assistant",
        content=[
            ToolUseBlock(
                id="t1",
                name="Read",
                input={"file_path": "/var/log/lilbee/sync.log", "pattern": "indexed chunk " * 40},
            )
        ],
    ),
    CanonicalMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id="t1", content=[TextBlock(text=_TOOL_LOG)])],
    ),
]
_MEASURED_TOOL_EXCHANGE_TOKENS = 2656


class TestCountRequestTokens:
    """``count_request_tokens`` asks the engine for the prompt a chat call sends."""

    def test_counts_the_arguments_the_chat_call_would_send(self, services_with_model) -> None:
        from lilbee.server.chat_dispatch.dispatch import _provider_chat_kwargs, count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.return_value = 4242
        req = _req(tools=[_ENGINE_TOOLS[0]], system="be brief", think=False)

        assert count_request_tokens(req, canonical_model="vendor/model::Q4") == 4242
        assert services_with_model.provider.count_chat_prompt_tokens.call_args.kwargs == (
            _provider_chat_kwargs(req, "vendor/model::Q4")
        )

    def test_a_model_without_tool_support_is_still_counted(self, services_with_model) -> None:
        """Counting runs no tool, so the chat call's capability gate does not apply."""
        from lilbee.server.chat_dispatch.dispatch import (
            preflight_chat_request,
            resolve_served_model,
        )

        req = _req(tools=[_ENGINE_TOOLS[0]])
        services_with_model.provider.supports_tools.return_value = False

        assert resolve_served_model(req) == "vendor/model::Q4"
        with pytest.raises(ModelDoesNotSupportToolsError):
            preflight_chat_request(req)

    def test_an_unknown_model_is_still_rejected(self, services_with_model) -> None:
        from lilbee.server.chat_dispatch.dispatch import resolve_served_model

        with pytest.raises(ModelNotFoundError):
            resolve_served_model(_req(model="nope/missing"))

    def test_a_backend_without_a_tokenizer_is_estimated(self, services_with_model) -> None:
        from lilbee.server.chat_dispatch.dispatch import count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.side_effect = NotImplementedError
        assert count_request_tokens(_req(), canonical_model="vendor/model::Q4") > 0

    @pytest.mark.parametrize(("extra", "measured"), _MEASURED_PROMPT_TOKENS)
    def test_the_estimate_never_falls_below_the_engines_own_count(
        self, services_with_model, extra: dict[str, Any], measured: int
    ) -> None:
        """Under-counting makes a client overflow its window, so the estimate stays above."""
        from lilbee.server.chat_dispatch.dispatch import count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.side_effect = NotImplementedError
        req = _req(messages=_ENGINE_MESSAGES, **extra)

        assert count_request_tokens(req, canonical_model="vendor/model::Q4") >= measured

    @pytest.mark.parametrize(("tool", "tool_choice", "measured"), _MEASURED_TOOL_TOKENS)
    def test_the_estimate_covers_the_templates_tool_preamble(
        self,
        services_with_model,
        tool: CanonicalTool,
        tool_choice: CanonicalToolChoice | None,
        measured: int,
    ) -> None:
        """One short turn with one schema is almost entirely template preamble."""
        from lilbee.server.chat_dispatch.dispatch import count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.side_effect = NotImplementedError
        req = _req(messages=_ONE_TURN, tools=[tool], tool_choice=tool_choice)

        assert count_request_tokens(req, canonical_model="vendor/model::Q4") >= measured

    @pytest.mark.parametrize(("text", "measured"), _MEASURED_DENSE_TOKENS)
    def test_the_estimate_holds_on_input_that_tokenizes_densely(
        self, services_with_model, text: str, measured: int
    ) -> None:
        """Input a chars-per-token average reads short, so a ratio fails this arm."""
        from lilbee.server.chat_dispatch.dispatch import count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.side_effect = NotImplementedError
        req = _req(messages=[CanonicalMessage(role="user", content=[TextBlock(text=text)])])

        assert count_request_tokens(req, canonical_model="vendor/model::Q4") >= measured

    @pytest.mark.parametrize(("extra", "measured"), _MEASURED_TEMPLATE_TOKENS)
    def test_the_estimate_covers_a_template_that_supplies_its_own_system_block(
        self, services_with_model, extra: dict[str, Any], measured: int
    ) -> None:
        """A near-empty request whose cost is almost all template, not input."""
        from lilbee.server.chat_dispatch.dispatch import count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.side_effect = NotImplementedError
        req = _req(**extra)

        assert count_request_tokens(req, canonical_model="vendor/model::Q4") >= measured

    def test_one_more_tool_costs_more_than_that_tools_own_text(self, services_with_model) -> None:
        """The template wraps each schema, so the allowance grows with the tool count."""
        from lilbee.server.chat_dispatch.dispatch import count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.side_effect = NotImplementedError
        one = count_request_tokens(_req(tools=[_MINIMAL_TOOL]), canonical_model="vendor/model::Q4")
        two = count_request_tokens(
            _req(tools=[_MINIMAL_TOOL] * 2), canonical_model="vendor/model::Q4"
        )

        own_text = _MINIMAL_TOOL.name + _MINIMAL_TOOL.description
        own_text += json.dumps(_MINIMAL_TOOL.input_schema)
        assert two - one > len(own_text.encode("utf-8"))

    def test_the_estimate_stays_within_seven_times_the_count_on_prose(
        self, services_with_model
    ) -> None:
        """Prose is where a byte count is loosest, so the arm pins how loose."""
        from lilbee.server.chat_dispatch.dispatch import count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.side_effect = NotImplementedError
        req = _req(
            messages=[CanonicalMessage(role="user", content=[TextBlock(text=_PROSE_PARAGRAPH)])]
        )

        estimate = count_request_tokens(req, canonical_model="vendor/model::Q4")
        assert (
            _MEASURED_PROSE_TOKENS <= estimate <= _MEASURED_PROSE_TOKENS * _MEASURED_PROSE_CEILING
        )

    def test_the_estimate_counts_a_tool_call_and_its_result(self, services_with_model) -> None:
        """Tool arguments and tool output reach the prompt, so they are counted."""
        from lilbee.server.chat_dispatch.dispatch import count_request_tokens

        services_with_model.provider.count_chat_prompt_tokens.side_effect = NotImplementedError
        req = _req(messages=_TOOL_EXCHANGE, tools=[_BIG_TOOLS[0]])

        estimate = count_request_tokens(req, canonical_model="vendor/model::Q4")
        assert estimate >= _MEASURED_TOOL_EXCHANGE_TOKENS
