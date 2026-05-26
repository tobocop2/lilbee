"""Tests for the canonical chat-dispatch layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.app.services import set_services
from lilbee.providers.base import (
    ChatResult,
    FinishReason,
    ToolCall,
    ToolCallDelta,
)
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolChoice,
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
    from tests.conftest import make_mock_services

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

    def test_empty_text_returns_empty_content(self, services_with_model) -> None:
        services_with_model.provider.chat.return_value = ChatResult(
            text="", tool_calls=(), finish_reason=FinishReason.STOP
        )
        resp = dispatch_chat(_req())
        assert resp.content == []

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


class _FakeStream:
    """Test-only async iterator that mimics ``ClosableIterator[ChatStreamItem]``."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


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

    async def test_empty_stream_yields_message_envelope_only(self, services_with_model) -> None:
        stream = _FakeStream([])
        services_with_model.provider.chat.return_value = stream
        events = await self._drain(dispatch_chat_stream(_req()))
        kinds = [type(e).__name__ for e in events]
        # No content blocks were ever opened; just MessageStart + MessageDelta + MessageStop.
        assert kinds == ["MessageStart", "MessageDelta", "MessageStop"]

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
