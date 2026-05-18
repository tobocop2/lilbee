"""Chat worker passes tool definitions through and returns ``ChatResult``."""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.providers.worker.chat_worker import _handle_chat
from lilbee.providers.worker.transport import (
    ChatRequest,
    ChatResult,
    FinishReason,
    ToolCall,
    ToolCallDelta,
)
from lilbee.providers.worker.worker_runtime import Reply, WorkerLoopState


class _RecordingConn:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)


def _make_reply() -> tuple[Reply, _RecordingConn]:
    conn = _RecordingConn()
    return Reply(conn), conn


class _FlagStub:
    def __init__(self) -> None:
        self.value = 0


class _StubSession:
    """Captures chat() kwargs and returns a canned llama-cpp response."""

    def __init__(self, *, response: Any) -> None:
        self._response = response
        self._abort_flag = _FlagStub()
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool,
        options: dict[str, Any] | None,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> Any:
        self.calls.append(
            {
                "messages": messages,
                "stream": stream,
                "options": options,
                "model": model,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return self._response


def test_non_streaming_returns_chat_result_with_text_only() -> None:
    reply, conn = _make_reply()
    session = _StubSession(
        response={
            "choices": [
                {
                    "message": {"content": "hello world", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=False)
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    assert len(conn.sent) == 1
    kind, value = conn.sent[0]
    assert kind == "result"
    assert isinstance(value, ChatResult)
    assert value.text == "hello world"
    assert value.tool_calls == ()
    assert value.finish_reason == FinishReason.STOP


def test_non_streaming_returns_chat_result_with_tool_calls() -> None:
    reply, conn = _make_reply()
    session = _StubSession(
        response={
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "search", "arguments": '{"q":"foo"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    payload = ChatRequest(
        messages=[{"role": "user", "content": "search foo"}],
        stream=False,
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
        tool_choice="auto",
    )
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    kind, value = conn.sent[0]
    assert kind == "result"
    assert isinstance(value, ChatResult)
    assert value.text == ""
    assert value.tool_calls == (ToolCall(id="c1", name="search", arguments='{"q":"foo"}'),)
    assert value.finish_reason == FinishReason.TOOL_CALLS


def test_tools_and_tool_choice_forwarded_to_session() -> None:
    reply, _conn = _make_reply()
    session = _StubSession(
        response={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    )
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    payload = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        tools=tools,
        tool_choice="auto",
    )
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    assert session.calls[0]["tools"] == tools
    assert session.calls[0]["tool_choice"] == "auto"


def test_session_chat_drops_none_tool_fields_from_kwargs(monkeypatch, tmp_path) -> None:
    """`_ChatSession.chat` forwards tools/tool_choice but skips them when None."""
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    captured: dict[str, Any] = {}

    class _Stub:
        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, *, messages: Any, stream: bool, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Stub())
    session.chat(
        messages=[],
        stream=False,
        options=None,
        model=None,
        tools=None,
        tool_choice=None,
    )
    assert "tools" not in captured
    assert "tool_choice" not in captured


def test_session_chat_passes_tools_when_present(monkeypatch, tmp_path) -> None:
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    captured: dict[str, Any] = {}

    class _Stub:
        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, *, messages: Any, stream: bool, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Stub())
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    session.chat(
        messages=[],
        stream=False,
        options=None,
        model=None,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "f"}},
    )
    assert captured["tools"] == tools
    assert captured["tool_choice"] == {"type": "function", "function": {"name": "f"}}


def test_streaming_yields_text_deltas_and_tool_call_deltas() -> None:
    reply, conn = _make_reply()
    stream = iter(
        [
            {"choices": [{"delta": {"content": "hi "}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "search", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"q":'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"foo"}'},
                                }
                            ]
                        }
                    }
                ]
            },
        ]
    )
    session = _StubSession(response=stream)
    payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    chunks = [v for kind, v in conn.sent if kind == "stream_chunk"]
    end_frames = [kind for kind, _v in conn.sent if kind == "stream_end"]
    assert end_frames == ["stream_end"]
    text_chunks = [c for c in chunks if isinstance(c, str)]
    tool_deltas = [c for c in chunks if isinstance(c, ToolCallDelta)]
    assert "".join(text_chunks) == "hi "
    assert tool_deltas[0] == ToolCallDelta(index=0, id="c1", name="search", arguments_delta=None)
    assert tool_deltas[1] == ToolCallDelta(index=0, id=None, name=None, arguments_delta='{"q":')
    assert tool_deltas[2] == ToolCallDelta(index=0, id=None, name=None, arguments_delta='"foo"}')


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("content_filter", FinishReason.CONTENT_FILTER),
        (None, FinishReason.STOP),
        ("weird-unknown", FinishReason.STOP),
    ],
)
def test_finish_reason_mapping(raw: str | None, expected: FinishReason) -> None:
    from lilbee.providers.worker.chat_worker import _coerce_finish_reason

    assert _coerce_finish_reason(raw) == expected


@pytest.mark.parametrize(
    "chunk",
    [
        {"choices": [{"delta": None}]},  # delta absent
        {"choices": [{"delta": {"tool_calls": "not-a-list"}}]},
        {"choices": [{"delta": {"tool_calls": ["not-a-dict"]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": "no-dict"}]}}]},
    ],
)
def test_extract_tool_call_deltas_skips_malformed_shapes(chunk: Any) -> None:
    """Defensive isinstance checks fall through to empty / no-arg deltas."""
    from lilbee.providers.worker.chat_worker import _extract_tool_call_deltas

    deltas = _extract_tool_call_deltas(chunk)
    # Either no deltas (top-level shape rejected) or one delta with no name/args
    # (function dict replaced with empty {}).
    assert all(d.arguments_delta is None for d in deltas)
    assert all(d.name is None for d in deltas)


@pytest.mark.parametrize(
    "raw_calls",
    [
        "not-a-list",
        ["not-a-dict"],
        [{"id": "c1", "function": "no-dict"}],
        [{"id": "c1", "function": {"name": 42}}],
        [{"id": "c1", "function": {"name": ""}}],
    ],
)
def test_coerce_tool_calls_drops_malformed_entries(raw_calls: Any) -> None:
    """``_coerce_tool_calls`` returns ``()`` when no entry has a valid name."""
    from lilbee.providers.worker.chat_worker import _coerce_tool_calls

    assert _coerce_tool_calls(raw_calls) == ()


def test_session_chat_raises_context_window_exceeded_on_unfittable_prompt(
    monkeypatch, tmp_path
) -> None:
    """A prompt whose un-droppable subset exceeds the budget raises the typed
    ContextWindowExceededError instead of falling through to llama-cpp's
    own ValueError.
    """
    from lilbee.providers.base import ContextWindowExceededError
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _TinyCtx:
        def n_ctx(self) -> int:
            return 256  # tight: 256 - 1024 reserve - 64 margin < 0

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            raise AssertionError("inference must not run when windowing rejects the prompt")

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _TinyCtx())
    big_user_text = "x" * 1000
    with pytest.raises(ContextWindowExceededError) as excinfo:
        session.chat(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": big_user_text},
            ],
            stream=False,
            options=None,
            model=None,
            tools=None,
            tool_choice=None,
        )
    assert "exceeds" in str(excinfo.value)


def test_session_chat_zero_n_ctx_clamps_budget_and_raises_overflow(monkeypatch, tmp_path) -> None:
    """A misbehaving ``llm.n_ctx() == 0`` clamps the budget to 0 (no negative
    arithmetic) and surfaces as ``ContextWindowExceededError`` since the
    un-droppable subset cannot fit in zero tokens.
    """
    from lilbee.providers.base import ContextWindowExceededError
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _ZeroCtx:
        def n_ctx(self) -> int:
            return 0

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            raise AssertionError("inference must not run with a zero-budget prompt")

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _ZeroCtx())
    with pytest.raises(ContextWindowExceededError):
        session.chat(
            messages=[{"role": "user", "content": "anything"}],
            stream=False,
            options=None,
            model=None,
            tools=None,
            tool_choice=None,
        )


def test_session_chat_logs_when_messages_dropped(monkeypatch, tmp_path, caplog) -> None:
    """When windowing drops messages to fit budget, an INFO log records the count."""
    import logging

    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _Llama:
        def n_ctx(self) -> int:
            return 2048

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(
            self, *, messages: list[dict[str, str]], stream: bool, **_kwargs: Any
        ) -> Any:
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Llama())
    with caplog.at_level(logging.INFO, logger="lilbee.providers.worker.chat_worker"):
        session.chat(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "y" * 1800},
                {"role": "user", "content": "ask"},
            ],
            stream=False,
            options={"num_predict": 256},
            model=None,
            tools=None,
            tool_choice=None,
        )
    assert any("Chat windowing dropped" in rec.message for rec in caplog.records)


def test_session_chat_drops_oldest_tool_pair_when_prompt_overflows(monkeypatch, tmp_path) -> None:
    """When trimming makes the prompt fit, the worker forwards the trimmed
    list to ``create_chat_completion`` and inference runs normally.
    """
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    seen_messages: list[Any] = []

    class _Llama:
        def n_ctx(self) -> int:
            return 2048

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(
            self, *, messages: list[dict[str, str]], stream: bool, **_kwargs: Any
        ) -> Any:
            seen_messages.append(messages)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Llama())
    huge_tool_result = "x" * 1800  # bigger than the remaining budget
    session.chat(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "s", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": huge_tool_result},
            {"role": "user", "content": "now answer"},
        ],
        stream=False,
        options={"num_predict": 256},
        model=None,
        tools=None,
        tool_choice=None,
    )
    forwarded = seen_messages[0]
    # The tool pair was dropped; system and the in-flight user message survive.
    assert all(m.get("role") != "tool" for m in forwarded)
    assert forwarded[0]["role"] == "system"
    assert forwarded[-1]["content"] == "now answer"
