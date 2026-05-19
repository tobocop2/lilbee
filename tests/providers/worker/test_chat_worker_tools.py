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
        self.response_schema = None

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


def test_session_chat_warns_once_when_tools_requested_without_schema(
    monkeypatch, tmp_path, caplog
) -> None:
    """Loading a model whose chat template has no known marker, then sending
    a tools= request, must log the unsupported-extraction warning exactly once
    across repeated calls.

    The whole path runs: the real ``_ensure_loaded`` reads the (stubbed) GGUF
    chat template, ``detect_family`` returns UNKNOWN, ``_response_schema`` stays
    None, and the per-model warning de-dupe kicks in on the second call.
    """
    import logging

    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    fake_path = tmp_path / "fake-unknown-family.gguf"
    fake_path.write_bytes(b"")
    role_config = RoleConfig(role="chat", model_path=fake_path, mode="chat")
    session = _ChatSession(role_config, _FlagStub())

    class _Llama:
        def create_chat_completion(self, **_kwargs: Any) -> Any:
            return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

    def _fake_load_llama(path, *, mode, abort_callback_override=None):
        return _Llama()

    def _fake_resolve(model_override):
        return fake_path

    def _fake_metadata(path):
        return {"chat_template": "no recognized markers here"}

    monkeypatch.setattr("lilbee.providers.llama_cpp.provider.load_llama", _fake_load_llama)
    monkeypatch.setattr("lilbee.providers.llama_cpp.provider.resolve_model_path", _fake_resolve)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.safe_read_gguf_metadata", _fake_metadata
    )

    tools = [{"type": "function", "function": {"name": "f"}}]
    with caplog.at_level(logging.WARNING, logger="lilbee.providers.worker.chat_worker"):
        for _ in range(2):
            session.chat(
                messages=[],
                stream=False,
                options=None,
                model="Some/Model/path.gguf",
                tools=tools,
                tool_choice=None,
            )

    assert session.response_schema is None
    warnings = [r for r in caplog.records if "Tool-call extraction not available" in r.message]
    assert len(warnings) == 1
    assert "Some/Model/path.gguf" in warnings[0].message


def test_session_chat_does_not_warn_when_schema_available(monkeypatch, tmp_path, caplog) -> None:
    """When a schema applies, the unsupported-tools warning stays silent."""
    import logging

    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily
    from lilbee.providers.worker.transport import RoleConfig

    class _Stub:
        def create_chat_completion(self, **_kwargs: Any) -> Any:
            return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Stub())
    session._response_schema = SCHEMAS[ModelFamily.QWEN3]
    with caplog.at_level(logging.WARNING, logger="lilbee.providers.worker.chat_worker"):
        session.chat(
            messages=[],
            stream=False,
            options=None,
            model="Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice=None,
        )
    assert not any("Tool-call extraction not available" in r.message for r in caplog.records)


def test_close_model_resets_response_schema(tmp_path) -> None:
    """``_close_model`` drops the cached response schema so the next load reclassifies."""
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily
    from lilbee.providers.worker.transport import RoleConfig

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    session._response_schema = SCHEMAS[ModelFamily.QWEN3]
    session._close_model()
    assert session._response_schema is None


def test_session_chat_passes_tools_when_present(monkeypatch, tmp_path) -> None:
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    captured: dict[str, Any] = {}

    class _Stub:
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


def test_non_streaming_schema_extraction_promotes_text_to_tool_calls() -> None:
    """When llama-cpp returns Qwen-style text containing a ``<tool_call>``,
    schema-driven extraction promotes it to structured ``tool_calls`` and
    remaps ``finish_reason`` from ``stop`` to ``tool_calls``.
    """
    from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily

    reply, conn = _make_reply()
    qwen_tool_call = '<tool_call>{"name": "search", "arguments": {"q": "foo"}}</tool_call>'
    session = _StubSession(
        response={
            "choices": [
                {
                    "message": {"content": qwen_tool_call, "tool_calls": None},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    session.response_schema = SCHEMAS[ModelFamily.QWEN3]
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
    assert len(value.tool_calls) == 1
    assert value.tool_calls[0].name == "search"
    assert value.finish_reason == FinishReason.TOOL_CALLS


def test_non_streaming_schema_extraction_skipped_when_tools_absent() -> None:
    """A text response that happens to contain ``<tool_call>`` is left alone
    when no tools were requested. The schema-extraction path only runs when
    the request itself carried tools.
    """
    from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily

    reply, conn = _make_reply()
    session = _StubSession(
        response={
            "choices": [
                {
                    "message": {
                        "content": '<tool_call>{"name":"x","arguments":{}}</tool_call>',
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )
    session.response_schema = SCHEMAS[ModelFamily.QWEN3]
    payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=False)
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    kind, value = conn.sent[0]
    assert kind == "result"
    assert isinstance(value, ChatResult)
    assert value.tool_calls == ()
    assert value.finish_reason == FinishReason.STOP


def test_chat_session_caches_schema_from_template_metadata(monkeypatch, tmp_path) -> None:
    """``_ChatSession._ensure_loaded`` reads the GGUF chat_template, classifies
    it via ``detect_family``, and caches the matching schema on the session.
    Tests the wiring that other tests bypass by assigning ``response_schema``
    directly.
    """
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily
    from lilbee.providers.worker.transport import RoleConfig

    fake_path = tmp_path / "fake-qwen.gguf"
    fake_path.write_bytes(b"")
    role_config = RoleConfig(role="chat", model_path=fake_path, mode="chat")
    session = _ChatSession(role_config, _FlagStub())

    qwen3_template = "<tool_call>{...}</tool_call>"

    def _fake_load_llama(path, *, mode, abort_callback_override=None):
        return object()

    def _fake_resolve(model_override):
        return fake_path

    def _fake_metadata(path):
        return {"chat_template": qwen3_template}

    monkeypatch.setattr("lilbee.providers.llama_cpp.provider.load_llama", _fake_load_llama)
    monkeypatch.setattr("lilbee.providers.llama_cpp.provider.resolve_model_path", _fake_resolve)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.safe_read_gguf_metadata", _fake_metadata
    )

    session._ensure_loaded(None)

    assert session.response_schema is SCHEMAS[ModelFamily.QWEN3]


def test_chat_session_caches_none_schema_for_unrecognised_template(monkeypatch, tmp_path) -> None:
    """An unrecognised template caches no schema so extraction is skipped."""
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    fake_path = tmp_path / "fake-unknown.gguf"
    fake_path.write_bytes(b"")
    role_config = RoleConfig(role="chat", model_path=fake_path, mode="chat")
    session = _ChatSession(role_config, _FlagStub())

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.load_llama",
        lambda path, *, mode, abort_callback_override=None: object(),
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path", lambda _m: fake_path
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.safe_read_gguf_metadata",
        lambda _p: {"chat_template": "no recognised markers here"},
    )

    session._ensure_loaded(None)

    assert session.response_schema is None


def test_non_streaming_schema_extraction_skipped_when_model_emits_plain_text() -> None:
    """When schema is cached but the model just emits text (no tool markup),
    the response passes through unchanged with finish_reason left at STOP.
    """
    from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily

    reply, conn = _make_reply()
    session = _StubSession(
        response={
            "choices": [
                {
                    "message": {"content": "plain prose", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    session.response_schema = SCHEMAS[ModelFamily.QWEN3]
    payload = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
    )
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    kind, value = conn.sent[0]
    assert kind == "result"
    assert isinstance(value, ChatResult)
    assert value.text == "plain prose"
    assert value.tool_calls == ()
    assert value.finish_reason == FinishReason.STOP


def test_streaming_flush_releases_held_content_via_drain() -> None:
    """A short text-only stream gets held in the safety margin until flush.

    Exercises ``_drain_schema_parser_flush``: when the buffer ends mid-stream
    with content held back by the marker-opener safety margin (here, the
    trailing ``<``), the end-of-stream drain releases it as a final text
    delta. Without the drain those bytes would never reach the client.
    """
    from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily

    reply, conn = _make_reply()
    # Chunk ends with '<' which the safety margin holds back; no </tool_call>
    # ever arrives, so the drain on stream end must release it.
    stream = iter([{"choices": [{"delta": {"content": "hi <"}}]}])
    session = _StubSession(response=stream)
    session.response_schema = SCHEMAS[ModelFamily.QWEN3]
    payload = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    text_chunks = [v for kind, v in conn.sent if kind == "stream_chunk" and isinstance(v, str)]
    assert "hi <" in "".join(text_chunks)


def test_streaming_schema_extraction_emits_tool_delta_from_text_chunks() -> None:
    """A streamed sequence of text deltas containing ``<tool_call>...{json}...</tool_call>``
    surfaces a ``ToolCallDelta`` via schema extraction, with prefix text flushed first.
    """
    from lilbee.providers.worker.response_parser import SCHEMAS, ModelFamily

    reply, conn = _make_reply()
    stream = iter(
        [
            {"choices": [{"delta": {"content": "Looking up "}}]},
            {"choices": [{"delta": {"content": "the weather."}}]},
            {"choices": [{"delta": {"content": '<tool_call>{"name": "weather"'}}]},
            {"choices": [{"delta": {"content": ', "arguments": {"city": "Paris"}}</tool_call>'}}]},
        ]
    )
    session = _StubSession(response=stream)
    session.response_schema = SCHEMAS[ModelFamily.QWEN3]
    payload = ChatRequest(
        messages=[{"role": "user", "content": "weather"}],
        stream=True,
        tools=[{"type": "function", "function": {"name": "weather", "parameters": {}}}],
    )
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    sent_chunks = [v for kind, v in conn.sent if kind == "stream_chunk"]
    text_chunks = [c for c in sent_chunks if isinstance(c, str)]
    tool_deltas = [c for c in sent_chunks if isinstance(c, ToolCallDelta)]
    assert "Looking up the weather." in "".join(text_chunks)
    assert len(tool_deltas) == 1
    assert tool_deltas[0].name == "weather"
