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
        from lilbee.providers.families.profile import FamilyProfile

        self._response = response
        self._abort_flag = _FlagStub()
        self.calls: list[dict[str, Any]] = []
        self.response_schema = None
        self._profile: FamilyProfile | None = None

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
        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

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
    monkeypatch.setattr("lilbee.providers.llama_cpp.gguf_meta.read_gguf_metadata", _fake_metadata)

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
    from lilbee.providers.worker.response_parser import TemplateFamily, get_schemas
    from lilbee.providers.worker.transport import RoleConfig

    class _Stub:
        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            return {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Stub())
    session._response_schema = get_schemas()[TemplateFamily.QWEN3]
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
    from lilbee.providers.worker.response_parser import TemplateFamily, get_schemas
    from lilbee.providers.worker.transport import RoleConfig

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    session._response_schema = get_schemas()[TemplateFamily.QWEN3]
    session._close_model()
    assert session._response_schema is None


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


def test_non_streaming_schema_extraction_promotes_text_to_tool_calls() -> None:
    """When llama-cpp returns Qwen-style text containing a ``<tool_call>``,
    schema-driven extraction promotes it to structured ``tool_calls`` and
    remaps ``finish_reason`` from ``stop`` to ``tool_calls``.
    """
    from lilbee.providers.worker.response_parser import TemplateFamily, get_schemas

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
    session.response_schema = get_schemas()[TemplateFamily.QWEN3]
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
    from lilbee.providers.worker.response_parser import TemplateFamily, get_schemas

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
    session.response_schema = get_schemas()[TemplateFamily.QWEN3]
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
    from lilbee.providers.worker.response_parser import TemplateFamily, get_schemas
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
    monkeypatch.setattr("lilbee.providers.llama_cpp.gguf_meta.read_gguf_metadata", _fake_metadata)

    session._ensure_loaded(None)

    assert session.response_schema is get_schemas()[TemplateFamily.QWEN3]


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
        "lilbee.providers.llama_cpp.gguf_meta.read_gguf_metadata",
        lambda _p: {"chat_template": "no recognised markers here"},
    )

    session._ensure_loaded(None)

    assert session.response_schema is None


def test_non_streaming_schema_extraction_skipped_when_model_emits_plain_text() -> None:
    """When schema is cached but the model just emits text (no tool markup),
    the response passes through unchanged with finish_reason left at STOP.
    """
    from lilbee.providers.worker.response_parser import TemplateFamily, get_schemas

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
    session.response_schema = get_schemas()[TemplateFamily.QWEN3]
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
    from lilbee.providers.worker.response_parser import TemplateFamily, get_schemas

    reply, conn = _make_reply()
    # Chunk ends with '<' which the safety margin holds back; no </tool_call>
    # ever arrives, so the drain on stream end must release it.
    stream = iter([{"choices": [{"delta": {"content": "hi <"}}]}])
    session = _StubSession(response=stream)
    session.response_schema = get_schemas()[TemplateFamily.QWEN3]
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
    from lilbee.providers.worker.response_parser import TemplateFamily, get_schemas

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
    session.response_schema = get_schemas()[TemplateFamily.QWEN3]
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


def test_session_chat_overflow_error_includes_usable_budget_and_breakdown(
    monkeypatch, tmp_path
) -> None:
    """The error must surface the usable budget AND the breakdown so the
    user can see which lever to pull. With n_ctx=40960, response=1024,
    tools=25000, safety=64 the budget is 14872, and a 19737-token prompt
    must show "exceeds the usable budget of 14872 tokens" (not the
    nonsense "exceeds the 40960-token context window").
    """
    from lilbee.providers.base import ContextWindowExceededError

    exc = ContextWindowExceededError.from_breakdown(
        requested=19_737,
        n_ctx=40_960,
        response_budget=1024,
        tools_overhead=25_000,
        safety_margin=64,
        model="Qwen/Qwen3-8B",
    )
    message = str(exc)
    assert "19737" in message
    assert "usable budget of 14872" in message
    assert "n_ctx=40960" in message
    assert "response_budget=1024" in message
    assert "tools_schema=25000" in message
    assert "safety_margin=64" in message
    assert "top_k" in message  # remediation hint present
    assert exc.requested == 19_737
    assert exc.usable_budget == 14_872
    assert exc.n_ctx == 40_960


def test_session_chat_overflow_error_reports_runtime_n_ctx_not_residual_budget(
    monkeypatch, tmp_path
) -> None:
    """The error must name the model's actual ``llm.n_ctx()`` value, never the
    residual budget after reserving for the response + tools. A model loaded
    with n_ctx=7168 that overflows must say "7168-token context window", not
    "0-token context window".
    """
    from lilbee.providers.base import ContextWindowExceededError
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _SmallCtx:
        def n_ctx(self) -> int:
            return 7168

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            raise AssertionError("must not reach inference on a windowed-out prompt")

    role_config = RoleConfig(role="chat", model_path=tmp_path / "gemma.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _SmallCtx())
    # Huge tools schema + a fat system message blow the budget to zero, but the
    # error message must still report n_ctx=7168, not "0".
    big_tools = [{"type": "function", "function": {"name": "x" * 4000}}]
    with pytest.raises(ContextWindowExceededError) as excinfo:
        session.chat(
            messages=[
                {"role": "system", "content": "y" * 6000},
                {"role": "user", "content": "z" * 1000},
            ],
            stream=False,
            options=None,
            model=None,
            tools=big_tools,
            tool_choice=None,
        )
    assert "7168" in str(excinfo.value)
    assert "0-token" not in str(excinfo.value)


@pytest.mark.parametrize("num_predict", [0, -1, -100, "not-an-int"])
def test_session_chat_invalid_num_predict_uses_default_reserve(
    monkeypatch, tmp_path, num_predict: Any
) -> None:
    """Bogus ``num_predict`` (0, negative, non-int) falls back to the default reserve."""
    from lilbee.providers.worker.chat_worker import _DEFAULT_RESPONSE_BUDGET, _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    captured: dict[str, int] = {}

    class _Llama:
        def n_ctx(self) -> int:
            return 4096

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, *, messages: list[Any], stream: bool, **_kw: Any) -> Any:
            captured["len"] = len(messages)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Llama())
    session.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        options={"num_predict": num_predict},
        model=None,
        tools=None,
        tool_choice=None,
    )
    # If default reserve wasn't used, a negative num_predict would have inflated
    # the budget and never triggered the safety path. We assert the default
    # was applied indirectly: the prompt fits naturally so messages survive.
    assert captured["len"] == 1
    assert _DEFAULT_RESPONSE_BUDGET == 1024  # contract anchor


def test_session_chat_catches_llama_cpp_overflow_and_reraises_typed(monkeypatch, tmp_path) -> None:
    """When pre-flight estimates undercount and llama-cpp raises its own
    ``Requested tokens (N) exceed context window`` ValueError mid-render,
    the worker translates that into our ``ContextWindowExceededError`` so
    the route layer surfaces a 400 ``context_length_exceeded`` instead of
    a generic 500.
    """
    from lilbee.providers.base import ContextWindowExceededError
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _UndercountingLlama:
        def n_ctx(self) -> int:
            return 7168

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            # Underestimate: return one token per 10 bytes, so the pre-flight
            # thinks even a huge system message fits the budget.
            return list(range(len(data) // 10 + 1))

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            raise ValueError("Requested tokens (18690) exceed context window of 7168")

    role_config = RoleConfig(role="chat", model_path=tmp_path / "gemma.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _UndercountingLlama())
    with pytest.raises(ContextWindowExceededError) as excinfo:
        session.chat(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            options=None,
            model="some/Gemma-4-26B-A4B-it-GGUF",
            tools=None,
            tool_choice=None,
        )
    assert "18690" in str(excinfo.value)
    assert "7168" in str(excinfo.value)


class TestParseRequestedTokens:
    """Direct unit tests for ``_parse_requested_tokens`` so regex breakage on
    an upstream llama-cpp wording change surfaces here, not in production.
    """

    def test_extracts_count_from_canonical_overflow_message(self) -> None:
        from lilbee.providers.worker.chat_worker import _parse_requested_tokens

        assert (
            _parse_requested_tokens("Requested tokens (18690) exceed context window of 7168")
            == 18690
        )

    def test_returns_none_for_unrelated_value_error(self) -> None:
        from lilbee.providers.worker.chat_worker import _parse_requested_tokens

        assert _parse_requested_tokens("something else entirely") is None

    def test_returns_none_for_empty_string(self) -> None:
        from lilbee.providers.worker.chat_worker import _parse_requested_tokens

        assert _parse_requested_tokens("") is None

    def test_does_not_match_batch_size_message(self) -> None:
        """llama-cpp also raises ``ValueError: Requested tokens (N) exceed
        batch size of M`` for a different upstream condition; that ValueError
        should NOT be misclassified as context overflow.
        """
        from lilbee.providers.worker.chat_worker import _parse_requested_tokens

        assert _parse_requested_tokens("Requested tokens (4096) exceed batch size of 512") is None

    def test_finds_match_when_embedded_in_a_longer_traceback(self) -> None:
        """Worker error wrappers may prepend module path / type names; the
        regex must still find the canonical phrasing inside.
        """
        from lilbee.providers.worker.chat_worker import _parse_requested_tokens

        message = (
            "ValueError: Requested tokens (12345) exceed context window of 4096\n"
            "  at create_chat_completion"
        )
        assert _parse_requested_tokens(message) == 12345


def test_session_chat_streaming_catches_deferred_overflow_and_reraises_typed(
    monkeypatch, tmp_path
) -> None:
    """llama-cpp returns the streaming generator unadvanced; the overflow
    ValueError fires on the FIRST ``next()`` after the parent starts iterating.
    The encoder must translate it into ``ContextWindowExceededError`` so the
    streaming route surfaces a 400 ``context_length_exceeded`` rather than a
    generic worker error.
    """
    from lilbee.providers.base import ContextWindowExceededError
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _DeferredOverflowLlama:
        def n_ctx(self) -> int:
            return 7168

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            # Undercount so pre-flight thinks everything fits.
            return list(range(len(data) // 10 + 1))

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            def _generator():
                # llama-cpp raises here on the first __next__, exactly the upstream-deferred
                # path.
                raise ValueError("Requested tokens (18690) exceed context window of 7168")
                yield  # unreachable: keeps this function a generator
                raise AssertionError("unreachable")

            return _generator()

    role_config = RoleConfig(role="chat", model_path=tmp_path / "gemma.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _DeferredOverflowLlama())
    iterator = session.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        options=None,
        model="some/Gemma",
        tools=None,
        tool_choice=None,
    )
    with pytest.raises(ContextWindowExceededError) as excinfo:
        next(iter(iterator))
    assert "18690" in str(excinfo.value)
    assert "7168" in str(excinfo.value)


def test_session_chat_streaming_does_not_swallow_unrelated_value_errors(
    monkeypatch, tmp_path
) -> None:
    """A non-overflow ValueError from the streaming generator propagates unchanged."""
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _BrokenStreamLlama:
        def n_ctx(self) -> int:
            return 4096

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            def _generator():
                raise ValueError("some other streaming failure")
                yield  # unreachable: keeps this function a generator
                raise AssertionError("unreachable")

            return _generator()

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _BrokenStreamLlama())
    iterator = session.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        options=None,
        model=None,
        tools=None,
        tool_choice=None,
    )
    with pytest.raises(ValueError, match="some other streaming failure"):
        next(iter(iterator))


def test_session_chat_does_not_swallow_unrelated_value_errors(monkeypatch, tmp_path) -> None:
    """A non-overflow ``ValueError`` from llama-cpp propagates unchanged."""
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _BrokenLlama:
        def n_ctx(self) -> int:
            return 4096

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            raise ValueError("something else entirely")

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _BrokenLlama())
    with pytest.raises(ValueError, match="something else entirely"):
        session.chat(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            options=None,
            model=None,
            tools=None,
            tool_choice=None,
        )


def test_session_chat_subtracts_tools_overhead_from_budget(monkeypatch, tmp_path) -> None:
    """Tools schema cost is counted against the prompt budget."""
    from lilbee.providers.base import ContextWindowExceededError
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    class _Llama:
        def n_ctx(self) -> int:
            # n_ctx - default reserve (1024) - margin (64) = 100 token budget.
            # A small user message fits; a 500-byte tools schema does not.
            return 1024 + 64 + 100

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            raise AssertionError("inference must not run when tools overhead overflows")

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Llama())
    # A tools schema that JSON-encodes to >100 bytes: blows the 100-token budget.
    big_tools = [{"type": "function", "function": {"name": "x" * 500}}]
    with pytest.raises(ContextWindowExceededError):
        session.chat(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            stream=False,
            options=None,
            model=None,
            tools=big_tools,
            tool_choice=None,
        )


def test_session_chat_without_tools_skips_tools_overhead(monkeypatch, tmp_path) -> None:
    """No ``tools`` means no overhead deducted; otherwise-fitting prompts run."""
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    inference_ran = False

    class _Llama:
        def n_ctx(self) -> int:
            return 1024 + 64 + 100

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, **_kwargs: Any) -> Any:
            nonlocal inference_ran
            inference_ran = True
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Llama())
    session.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        options=None,
        model=None,
        tools=None,
        tool_choice=None,
    )
    assert inference_ran is True


def test_session_chat_logs_when_messages_dropped(monkeypatch, tmp_path, caplog) -> None:
    """When windowing drops messages to fit budget, a DEBUG log records the count."""
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
    with caplog.at_level(logging.DEBUG, logger="lilbee.providers.worker.chat_worker"):
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


def test_session_chat_downgrades_stream_for_chatml_function_calling(monkeypatch, tmp_path) -> None:
    """When chat_format=chatml-function-calling is active and the caller asks
    for stream=True with tool_choice=auto, llama-cpp-python's preset raises
    'Automatic streaming tool choice is not supported'. The session swaps
    stream=False under the hood and synthesizes a single-chunk stream so the
    downstream pipeline keeps working.
    """
    from lilbee.providers.families import registry
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.response_parser.families import TemplateFamily
    from lilbee.providers.worker.transport import RoleConfig

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    # Pin the session profile to Hermes-3, whose StreamingPolicy is
    # DOWNGRADE_AUTO_TOOL_CHOICE; that's the contract the downgrade
    # decision is now based on.
    session._profile = registry().by_family(TemplateFamily.HERMES)
    captured: dict[str, Any] = {}

    class _Stub:
        chat_format = "chatml-function-calling"

        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, *, messages: Any, stream: bool, **kwargs: Any) -> Any:
            captured["stream"] = stream
            captured["kwargs"] = kwargs
            return {
                "id": "x1",
                "created": 1,
                "model": "hermes",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "lilbee_search",
                                        "arguments": '{"q": "x"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }

    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Stub())
    result = session.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        options=None,
        model=None,
        tools=[{"type": "function", "function": {"name": "lilbee_search"}}],
        tool_choice=None,
    )
    # The downstream pipeline expects an iterator of chat.completion.chunk dicts.
    chunks = list(result)
    assert captured["stream"] is False  # internal call was downgraded
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "lilbee_search"
    assert chunk["choices"][0]["finish_reason"] == "tool_calls"


def test_session_chat_does_not_downgrade_when_chat_format_unchanged(monkeypatch, tmp_path) -> None:
    """Embedded-template (no override) GGUFs keep streaming as-is."""
    from lilbee.providers.worker.chat_worker import _ChatSession
    from lilbee.providers.worker.transport import RoleConfig

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    session = _ChatSession(role_config, _FlagStub())
    captured: dict[str, Any] = {}

    class _Stub:
        # No chat_format attribute = embedded-template path.

        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, *, messages: Any, stream: bool, **kwargs: Any) -> Any:
            captured["stream"] = stream
            return iter([])  # empty stream

    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Stub())
    session.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        options=None,
        model=None,
        tools=[{"type": "function", "function": {"name": "lilbee_search"}}],
        tool_choice=None,
    )
    assert captured["stream"] is True


def test_wrap_single_completion_as_stream_handles_malformed_completions() -> None:
    """Defensive guards on the downgrade synthesiser: skip non-dict + empty-choices payloads."""
    from lilbee.providers.worker.chat_worker import _wrap_single_completion_as_stream

    assert list(_wrap_single_completion_as_stream("not a dict")) == []
    assert list(_wrap_single_completion_as_stream({"choices": []})) == []


def test_normalize_tool_call_arguments_parses_json_string_to_dict() -> None:
    """OpenAI-format string arguments become dicts so HF jinja templates render.

    Qwen3-Coder's GGUF template iterates ``tool_call.arguments|items`` and
    raises ``TypeError: Can only get item pairs from a mapping`` when handed
    the OpenAI wire-format JSON string. The normalizer parses it to a dict.
    """
    from lilbee.providers.worker.chat_worker import _normalize_tool_call_arguments

    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lilbee_search", "arguments": '{"query": "chat worker"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    out = _normalize_tool_call_arguments(messages)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"query": "chat worker"}
    # User and tool messages pass through untouched.
    assert out[0] is messages[0]
    assert out[2] is messages[2]


def test_normalize_tool_call_arguments_leaves_dicts_and_bad_json_untouched() -> None:
    """Already-dict arguments and non-JSON strings are not mangled."""
    from lilbee.providers.worker.chat_worker import _normalize_tool_call_arguments

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "a", "arguments": {"already": "dict"}}},
                {"function": {"name": "b", "arguments": "not json{"}},
            ],
        },
    ]
    out = _normalize_tool_call_arguments(messages)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"already": "dict"}
    assert out[0]["tool_calls"][1]["function"]["arguments"] == "not json{"
