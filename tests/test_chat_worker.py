"""Tests for the persistent chat worker subprocess.

Streaming protocol coverage end-to-end via real spawn-context
subprocesses, plus pure-function tests for the dispatch table and the
in-process loop. The Llama loader is patched at the worker side so the
tests do not need a real GGUF.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any

import pytest

from lilbee.providers.worker.chat_worker import (
    _ChatSession,
    _extract_stream_content,
    chat_worker_main,
)
from lilbee.providers.worker.transport import (
    ChatRequest,
    ChatResult,
    FinishReason,
    RoleConfig,
)
from lilbee.providers.worker.transport_pipe import (
    PipeSpawner,
    WorkerError,
)
from lilbee.providers.worker.worker_runtime import Reply

pytestmark = pytest.mark.xdist_group("worker_pool_chat")


_TEST_CALL_TIMEOUT_S = 10.0
_TEST_SHUTDOWN_TIMEOUT_S = 2.0


# Module-level worker entrypoints so spawn pickling succeeds.


def _stub_load_streaming(_self: _ChatSession) -> Any:
    class _StubLlama:
        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(
            self, *, messages: list[dict[str, str]], stream: bool, **kwargs: Any
        ) -> Any:
            tokens = ["hello", " ", "world"]
            if stream:
                return iter({"choices": [{"delta": {"content": tok}}]} for tok in tokens)
            return {"choices": [{"message": {"content": "".join(tokens)}}]}

    return _StubLlama()


def _stub_load_aborts_mid_stream(_self: _ChatSession) -> Any:
    """Stub that emits one chunk, then checks the abort flag before more.

    The abort_callback is bound at load time (mirroring real llama-cpp,
    which only accepts ``abort_callback`` on ``Llama(...)``, not on
    ``create_chat_completion``).
    """

    class _StubLlama:
        def __init__(self, abort_flag: Any) -> None:
            self._abort_flag = abort_flag

        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(
            self,
            *,
            messages: list[dict[str, str]],
            stream: bool,
            **kwargs: Any,
        ) -> Any:
            abort_flag = self._abort_flag

            def _gen():
                yield {"choices": [{"delta": {"content": "first"}}]}
                # Wait until the parent flips the flag (max 5s for safety).
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if bool(abort_flag.value):
                        return
                    time.sleep(0.01)
                # If the flag was never flipped, emit a sentinel for the test
                # to assert against.
                yield {"choices": [{"delta": {"content": "TIMEOUT"}}]}

            return _gen()

    return _StubLlama(_self._abort_flag)


def _patched_chat_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    from lilbee.providers.worker import chat_worker

    chat_worker._ChatSession._ensure_loaded = lambda self, _override: _stub_load_streaming(self)  # type: ignore[method-assign]
    chat_worker_main(data_conn, health_conn, abort_flag, role_config)


def _aborting_chat_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    from lilbee.providers.worker import chat_worker

    chat_worker._ChatSession._ensure_loaded = lambda self, _override: _stub_load_aborts_mid_stream(
        self
    )  # type: ignore[method-assign]
    chat_worker_main(data_conn, health_conn, abort_flag, role_config)


def _stub_load_paced_stream(_self: _ChatSession) -> Any:
    """Stub whose stream paces itself so a mid-stream ping has time to land."""

    class _StubLlama:
        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(
            self, *, messages: list[dict[str, str]], stream: bool, **kwargs: Any
        ) -> Any:
            tokens = ["alpha", "beta", "gamma", "delta"]
            if not stream:
                return {"choices": [{"message": {"content": "".join(tokens)}}]}

            def _gen():
                for tok in tokens:
                    yield {"choices": [{"delta": {"content": tok}}]}
                    time.sleep(0.05)

            return _gen()

    return _StubLlama()


def _paced_chat_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    from lilbee.providers.worker import chat_worker

    chat_worker._ChatSession._ensure_loaded = lambda self, _override: _stub_load_paced_stream(self)  # type: ignore[method-assign]
    chat_worker_main(data_conn, health_conn, abort_flag, role_config)


@pytest.fixture()
def role_config(tmp_path) -> RoleConfig:
    return RoleConfig(role="chat", model_path=tmp_path / "chat.gguf", mode="chat")


@pytest.fixture()
def spawner() -> PipeSpawner:
    return PipeSpawner()


# End-to-end streaming and non-streaming.


@pytest.mark.asyncio
async def test_chat_worker_streams_chunks(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_chat_worker_main, role_config)
    try:
        chunks: list[str] = []
        payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
        async for chunk in channel.stream("chat", payload):
            chunks.append(chunk)
        # Token batching may coalesce these into one or more frames; only
        # the joined content matters at the wire level.
        assert "".join(chunks) == "hello world"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_chat_worker_non_streaming_returns_chat_result(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_chat_worker_main, role_config)
    try:
        payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=False)
        result = await channel.call("chat", payload, timeout=_TEST_CALL_TIMEOUT_S)
        assert isinstance(result, ChatResult)
        assert result.text == "hello world"
        assert result.tool_calls == ()
        assert result.finish_reason == FinishReason.STOP
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_chat_worker_rejects_non_chatrequest_payload(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_chat_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("chat", "not-a-chatrequest", timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "TypeError"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_chat_worker_unknown_kind_returns_error(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_chat_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("not_real", None, timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "ValueError"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_chat_worker_pongs_pings(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_chat_worker_main, role_config)
    try:
        await channel.ping(timeout=_TEST_CALL_TIMEOUT_S)
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


# Cross-boundary cancel via the abort flag.


@pytest.mark.asyncio
async def test_chat_worker_honors_abort_flag_mid_stream(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    """Parent flips the abort flag; worker stops emitting before the timeout marker."""
    channel, _ = spawner.spawn(_aborting_chat_worker_main, role_config)
    try:
        chunks: list[str] = []
        payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
        import asyncio as _asyncio

        async def _flip_abort_after_first_chunk() -> None:
            # Wait for the first chunk to land, then flip.
            for _ in range(500):
                if chunks:
                    channel.cancel()
                    return
                await _asyncio.sleep(0.01)

        flipper = _asyncio.create_task(_flip_abort_after_first_chunk())
        async for chunk in channel.stream("chat", payload):
            chunks.append(chunk)
        await flipper
        # Token batching: "first" arrives as a single batch (the 50 ms
        # interval flushes it well before the second token would land);
        # the abort prevents any further tokens.
        assert "".join(chunks) == "first", f"Expected only first token before abort, got {chunks!r}"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_chat_worker_back_to_back_streams_with_concurrent_pings(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    """Concurrent health pings during a stream do not break the next stream.

    Pings ride a separate pipe from chat traffic, so a stream and a ping
    can interleave in time without sharing wire frames. The next stream
    after the first completes cleanly with no orphan-pong recovery
    needed in PipeChannel.
    """
    import asyncio as _asyncio

    channel, _ = spawner.spawn(_paced_chat_worker_main, role_config)
    try:
        chunks: list[str] = []
        payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)

        async def _ping_during_stream() -> None:
            for _ in range(500):
                if chunks:
                    await channel.ping(timeout=_TEST_CALL_TIMEOUT_S)
                    return
                await _asyncio.sleep(0.01)

        ping_task = _asyncio.create_task(_ping_during_stream())
        async for chunk in channel.stream("chat", payload):
            chunks.append(chunk)
        await ping_task
        # Token batching may coalesce some chunks; assert on joined text.
        assert "".join(chunks) == "alphabetagammadelta"

        # Second stream must succeed; with separate pipes, no orphan pongs
        # can have leaked into the data pipe from the previous ping.
        chunks_two: list[str] = []
        async for chunk in channel.stream("chat", payload):
            chunks_two.append(chunk)
        assert "".join(chunks_two) == "alphabetagammadelta"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


# Pure-function helpers.


def test_extract_stream_content_returns_string() -> None:
    chunk = {"choices": [{"delta": {"content": "abc"}}]}
    assert _extract_stream_content(chunk) == "abc"


def test_extract_stream_content_returns_none_for_empty_content() -> None:
    chunk = {"choices": [{"delta": {"content": ""}}]}
    assert _extract_stream_content(chunk) is None


def test_extract_stream_content_returns_none_for_missing_delta() -> None:
    assert _extract_stream_content({"choices": [{}]}) is None


def test_extract_stream_content_returns_none_for_missing_choices() -> None:
    assert _extract_stream_content({}) is None


def test_extract_stream_content_handles_non_dict() -> None:
    assert _extract_stream_content("garbage") is None


# Pure-function dispatch tests.


class _RecordingConn:
    """Captures ``(kind, payload)`` frames sent through Reply."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)


def _make_reply() -> tuple[Reply, _RecordingConn]:
    """Build a Reply bound to a recording conn so tests can inspect emitted frames."""
    conn = _RecordingConn()
    return Reply(conn), conn


def _kinds_payloads(conn: _RecordingConn) -> list[tuple[str, Any]]:
    """Surface the captured ``(kind, payload)`` frames for assertion."""
    return list(conn.sent)


class _FlagStub:
    """Mimics the .value attribute on an mp.Value bool flag."""

    def __init__(self, value: int = 0) -> None:
        self.value = value


class _StubSession:
    def __init__(self, *, response: Any = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc
        self._abort_flag = _FlagStub()
        self.response_schema = None

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool,
        options: dict[str, Any] | None,
        model: str | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._response


def test_handle_chat_streaming() -> None:
    """First token flushes immediately; the rest land in the final flush.

    The worker holds subsequent chunks in a buffer until 16 tokens or
    50 ms accumulate. Two fast tokens stay together after the first
    eager flush. Exact framing is determined by the batching policy.
    """
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(
        response=iter(
            [
                {"choices": [{"delta": {"content": "a"}}]},
                {"choices": [{"delta": {"content": "b"}}]},
            ]
        )
    )
    payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
    state = WorkerLoopState(session=session)
    _handle_chat(reply, payload, state)
    # First "a" goes out immediately, "b" sits in buffer and flushes on
    # stream end, then the stream_end frame.
    assert _kinds_payloads(conn) == [
        ("stream_chunk", "a"),
        ("stream_chunk", "b"),
        ("stream_end", None),
    ]


def test_handle_chat_streaming_flushes_when_batch_full() -> None:
    """Once 16 chunks queue (after the first eager flush), the buffer flushes."""
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    tokens = [f"t{i}" for i in range(20)]
    session = _StubSession(
        response=iter({"choices": [{"delta": {"content": tok}}]} for tok in tokens)
    )
    payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
    state = WorkerLoopState(session=session)
    _handle_chat(reply, payload, state)
    frames = _kinds_payloads(conn)
    chunk_payloads = [p for kind, p in frames if kind == "stream_chunk"]
    end_frames = [kind for kind, _p in frames if kind == "stream_end"]
    assert end_frames == ["stream_end"]
    assert "".join(chunk_payloads) == "".join(tokens)
    # First eager flush ("t0"), then 16-batch flush at index 17, then
    # remaining tail ("t17","t18","t19") at end-of-stream.
    assert chunk_payloads[0] == "t0"
    assert chunk_payloads[1] == "".join(tokens[1:17])
    assert chunk_payloads[2] == "".join(tokens[17:])


def test_handle_chat_streaming_skips_chunks_with_no_content() -> None:
    """Chunks with empty/None content are dropped, not flushed as empty strings."""
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(
        response=iter(
            [
                {"choices": [{"delta": {"content": None}}]},  # skipped
                {"choices": [{"delta": {"content": "hi"}}]},
            ]
        )
    )
    payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
    state = WorkerLoopState(session=session)
    _handle_chat(reply, payload, state)
    # Only "hi" surfaces; the None-content chunk is dropped.
    assert _kinds_payloads(conn) == [
        ("stream_chunk", "hi"),
        ("stream_end", None),
    ]


def test_handle_chat_streaming_aborts_on_flag() -> None:
    """Cancel raised mid-stream stops the iterator at the next token boundary."""
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession()

    closed: list[bool] = []

    class _Iter:
        def __init__(self) -> None:
            self._chunks = iter(
                [
                    {"choices": [{"delta": {"content": "a"}}]},
                    {"choices": [{"delta": {"content": "b"}}]},
                    {"choices": [{"delta": {"content": "c"}}]},
                ]
            )
            self._calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            chunk = next(self._chunks)
            self._calls += 1
            if self._calls == 2:
                session._abort_flag.value = 1
            return chunk

        def close(self) -> None:
            closed.append(True)

    session._response = _Iter()
    payload = ChatRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    # First token flushes eagerly, abort flips before the second token
    # appends, then stream_end fires.
    assert _kinds_payloads(conn) == [
        ("stream_chunk", "a"),
        ("stream_end", None),
    ]
    assert closed == [True]


def test_handle_chat_non_streaming() -> None:
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(
        response={
            "choices": [
                {"message": {"content": "joined"}, "finish_reason": "stop"},
            ]
        }
    )
    payload = ChatRequest(messages=[], stream=False)
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert len(frames) == 1
    kind, value = frames[0]
    assert kind == "result"
    assert isinstance(value, ChatResult)
    assert value.text == "joined"
    assert value.tool_calls == ()
    assert value.finish_reason == FinishReason.STOP


def test_handle_chat_emits_error_on_setup_exception() -> None:
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(exc=RuntimeError("setup boom"))
    payload = ChatRequest(messages=[], stream=False)
    _handle_chat(reply, payload, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "RuntimeError"


def test_handle_chat_emits_error_on_stream_failure() -> None:
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()

    def _generator():
        yield {"choices": [{"delta": {"content": "a"}}]}
        raise RuntimeError("stream broke")

    session = _StubSession(response=_generator())
    payload = ChatRequest(messages=[], stream=True)
    state = WorkerLoopState(session=session)
    _handle_chat(reply, payload, state)
    # Buffered "a" flushes via finally-clause before the error frame so
    # the user still sees partial output.
    frames = _kinds_payloads(conn)
    assert frames[0] == ("stream_chunk", "a")
    assert frames[1][0] == "error"


def test_handle_chat_rejects_non_chatrequest_payload() -> None:
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession()
    _handle_chat(reply, "not-a-chatrequest", WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "TypeError"


def test_handle_chat_rejects_dict_payload() -> None:
    """Bare dicts no longer accepted; only ChatRequest."""
    from lilbee.providers.worker.chat_worker import _handle_chat
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession()
    _handle_chat(reply, {"messages": [], "stream": False}, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "TypeError"


def test_extract_non_streaming_result_walks_defensively() -> None:
    from lilbee.providers.worker.chat_worker import _extract_non_streaming_result

    # Happy path.
    happy = _extract_non_streaming_result(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
        tools_requested=False,
        schema=None,
    )
    assert happy.text == "hi"
    assert happy.tool_calls == ()
    assert happy.finish_reason == FinishReason.STOP
    # None content -> empty string.
    none_content = _extract_non_streaming_result(
        {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        tools_requested=False,
        schema=None,
    )
    assert none_content.text == ""
    # Malformed shapes raise typed errors.
    with pytest.raises(TypeError):
        _extract_non_streaming_result("not a dict", tools_requested=False, schema=None)
    with pytest.raises(TypeError):
        _extract_non_streaming_result({}, tools_requested=False, schema=None)
    with pytest.raises(TypeError):
        _extract_non_streaming_result({"choices": []}, tools_requested=False, schema=None)
    with pytest.raises(TypeError):
        _extract_non_streaming_result(
            {"choices": ["not a dict"]}, tools_requested=False, schema=None
        )
    with pytest.raises(TypeError):
        _extract_non_streaming_result(
            {"choices": [{"message": "not a dict"}]}, tools_requested=False, schema=None
        )


def test_chat_session_close_idempotent_and_swallows() -> None:
    role_config = RoleConfig(
        role="chat",
        model_path=__import__("pathlib").Path("/nope"),
        mode="chat",
    )
    flag = multiprocessing.Value("b", 0)
    session = _ChatSession(role_config, flag)
    session.close()  # no-op when llm is None

    class _BadLlama:
        def close(self) -> None:
            raise RuntimeError("close blew up")

    session._llm = _BadLlama()
    session.close()
    assert session._llm is None


def test_chat_session_ensure_loaded_routes_through_real_loader(monkeypatch, tmp_path) -> None:
    """Default _ensure_loaded reaches load_llama with the role config's path.

    No abort_callback_override is passed for chat: routing the cancel
    signal through ggml's mid-token abort path crashed the worker on
    macOS Metal. Cancel is enforced one token boundary later by the
    Python-side polling loop in _handle_chat_streaming.
    """
    role_config = RoleConfig(role="chat", model_path=tmp_path / "stub.gguf", mode="chat")
    flag = multiprocessing.Value("b", 0)
    session = _ChatSession(role_config, flag)
    sentinel = object()
    captured: dict[str, Any] = {}

    def fake_load_llama(path: Any, *, mode: str) -> Any:
        captured["path"] = path
        captured["mode"] = mode
        return sentinel

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.load_llama",
        fake_load_llama,
    )
    result = session._ensure_loaded(None)
    assert result is sentinel
    assert captured["path"] == tmp_path / "stub.gguf"
    assert captured["mode"] == "chat"


def test_chat_session_chat_passes_options_to_llama(monkeypatch, tmp_path) -> None:
    """Options dict is forwarded into create_chat_completion kwargs."""
    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    flag = multiprocessing.Value("b", 0)
    session = _ChatSession(role_config, flag)
    captured: dict[str, Any] = {}

    class _Stub:
        def n_ctx(self) -> int:
            return 8192

        def tokenize(
            self, data: bytes, *, add_bos: bool = False, special: bool = False
        ) -> list[int]:
            return list(data)

        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Stub())
    session.chat(
        messages=[],
        stream=False,
        options={"temperature": 0.42, "max_tokens": 32},
        model=None,
        tools=None,
        tool_choice=None,
    )
    assert captured["temperature"] == 0.42
    assert captured["max_tokens"] == 32
    # llama-cpp's create_chat_completion does not accept abort_callback;
    # the worker binds it at load time instead. See _ensure_loaded.
    assert "abort_callback" not in captured


def test_chat_session_ensure_loaded_swaps_on_per_call_model(monkeypatch, tmp_path) -> None:
    """A different per-call model path triggers a transparent reload."""
    role_config = RoleConfig(role="chat", model_path=tmp_path / "default.gguf", mode="chat")
    flag = multiprocessing.Value("b", 0)
    session = _ChatSession(role_config, flag)
    load_calls: list[Any] = []

    def fake_load_llama(path: Any, *, mode: str, abort_callback_override: Any = None) -> Any:
        class _LlmStub:
            closed = False

            def close(self) -> None:
                self.closed = True

        load_calls.append(path)
        return _LlmStub()

    def fake_resolve(model: str) -> Any:
        return tmp_path / f"{model}.gguf"

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.load_llama",
        fake_load_llama,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        fake_resolve,
    )
    first = session._ensure_loaded(None)
    # Same model: no reload.
    again = session._ensure_loaded(None)
    assert first is again
    # New model: reload.
    swapped = session._ensure_loaded("override")
    assert swapped is not first
    assert load_calls == [tmp_path / "default.gguf", tmp_path / "override.gguf"]


def test_abort_bridge_forwards_parent_flag_to_request_abort(monkeypatch) -> None:
    """The bridge thread polls the parent's mp.Value and calls request_abort.

    Real chat-worker tests stub the loaded Llama and check the flag inline,
    which bypasses the bridge thread; this exercises the bridge directly so
    the poll loop's flag-detection branch is covered.
    """
    from lilbee.providers.worker import chat_worker

    abort_flag = multiprocessing.Value("i", 0)
    aborted: list[int] = []

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.abort_signal.request_abort",
        lambda: aborted.append(1),
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.abort_signal.clear_abort",
        lambda: None,
    )

    with chat_worker._AbortBridge(abort_flag):
        abort_flag.value = 1
        deadline = time.monotonic() + 1.0
        while not aborted and time.monotonic() < deadline:
            time.sleep(0.01)

    assert aborted, "abort bridge poll thread did not forward the parent flag"


def test_chat_worker_main_routes_through_run_worker(monkeypatch) -> None:
    """``chat_worker_main`` passes both pipes + the chat handler to run_worker."""
    from lilbee.providers.worker import chat_worker

    captured: dict[str, Any] = {}

    def _fake_run_worker(data_conn, health_conn, abort_flag, role_config, **kwargs):
        captured["data"] = data_conn
        captured["health"] = health_conn
        captured["kwargs"] = kwargs

    monkeypatch.setattr(chat_worker, "run_worker", _fake_run_worker)
    role_config = RoleConfig(
        role="chat", model_path=__import__("pathlib").Path("/nope"), mode="chat"
    )
    chat_worker.chat_worker_main("DATA", "HEALTH", "ABORT", role_config)
    assert captured["data"] == "DATA"
    assert captured["health"] == "HEALTH"
    assert "chat" in captured["kwargs"]["kind_handlers"]


def test_abort_bridge_polls_flag_and_calls_request_abort(monkeypatch):
    """_AbortBridge's poll thread observes the abort flag and calls request_abort.

    The polling lives on a daemon thread polling at _ABORT_BRIDGE_POLL_S
    intervals, so without an explicit test for the abort branch the
    coverage hit is timing-sensitive (passes on some runners, fails on
    others). This test deterministically: enters the bridge, sets the
    flag from outside, waits long enough for the next poll iteration to
    fire request_abort, then exits.
    """
    from lilbee.providers.worker import chat_worker as cw_mod
    from lilbee.providers.worker.chat_worker import _AbortBridge

    calls: list[None] = []
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.abort_signal.request_abort",
        lambda: calls.append(None),
    )
    monkeypatch.setattr(cw_mod, "_ABORT_BRIDGE_POLL_S", 0.005)

    class _Flag:
        value = 0

    flag = _Flag()
    bridge = _AbortBridge(flag)
    with bridge:
        flag.value = 1
        # Wait long enough for the poll to observe the flag (>> 0.005s).
        for _ in range(50):
            if calls:
                break
            time.sleep(0.01)
    assert calls, "request_abort was not invoked by the poll thread"
