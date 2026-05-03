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
    _dispatch,
    _extract_stream_content,
    _make_abort_callback,
    chat_worker_main,
)
from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import (
    PipeSpawner,
    WorkerError,
)

pytestmark = pytest.mark.xdist_group("worker_pool_chat")


_TEST_CALL_TIMEOUT_S = 10.0
_TEST_SHUTDOWN_TIMEOUT_S = 2.0


# Module-level worker entrypoints so spawn pickling succeeds.


def _stub_load_streaming(_self: _ChatSession) -> Any:
    class _StubLlama:
        def create_chat_completion(
            self, *, messages: list[dict[str, str]], stream: bool, **kwargs: Any
        ) -> Any:
            tokens = ["hello", " ", "world"]
            if stream:
                return iter({"choices": [{"delta": {"content": tok}}]} for tok in tokens)
            return {"choices": [{"message": {"content": "".join(tokens)}}]}

    return _StubLlama()


def _stub_load_aborts_mid_stream(_self: _ChatSession) -> Any:
    """Stub that emits one chunk, then checks the abort flag before more."""

    class _StubLlama:
        def __init__(self, abort_flag: Any) -> None:
            self._abort_flag = abort_flag

        def create_chat_completion(
            self,
            *,
            messages: list[dict[str, str]],
            stream: bool,
            abort_callback: Any,
            **kwargs: Any,
        ) -> Any:
            def _gen():
                yield {"choices": [{"delta": {"content": "first"}}]}
                # Wait until the parent flips the flag (max 5s for safety).
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if abort_callback():
                        return
                    time.sleep(0.01)
                # If the flag was never flipped, emit a sentinel for the test
                # to assert against.
                yield {"choices": [{"delta": {"content": "TIMEOUT"}}]}

            return _gen()

    return _StubLlama(_self._abort_flag)


def _patched_chat_worker_main(conn: Any, abort_flag: Any, role_config: RoleConfig) -> None:
    from lilbee.providers.worker import chat_worker

    chat_worker._ChatSession._ensure_loaded = lambda self, _override: _stub_load_streaming(self)  # type: ignore[method-assign]
    chat_worker_main(conn, abort_flag, role_config)


def _aborting_chat_worker_main(conn: Any, abort_flag: Any, role_config: RoleConfig) -> None:
    from lilbee.providers.worker import chat_worker

    chat_worker._ChatSession._ensure_loaded = lambda self, _override: _stub_load_aborts_mid_stream(
        self
    )  # type: ignore[method-assign]
    chat_worker_main(conn, abort_flag, role_config)


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
        payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        async for chunk in channel.stream("chat", payload):
            chunks.append(chunk)
        assert chunks == ["hello", " ", "world"]
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_chat_worker_non_streaming_returns_joined_text(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_chat_worker_main, role_config)
    try:
        payload = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
        result = await channel.call("chat", payload, timeout=_TEST_CALL_TIMEOUT_S)
        assert result == "hello world"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_chat_worker_rejects_non_dict_payload(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_chat_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("chat", "not-a-dict", timeout=_TEST_CALL_TIMEOUT_S)
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
        payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
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
        assert chunks == ["first"], f"Expected only first chunk before abort, got {chunks!r}"
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


def test_make_abort_callback_reads_flag_value() -> None:
    flag = multiprocessing.Value("b", 0)
    cb = _make_abort_callback(flag)
    assert cb() is False
    flag.value = 1
    assert cb() is True


# Pure-function dispatch tests.


class _RecordingConn:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)


class _StubSession:
    def __init__(self, *, response: Any = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        stream: bool,
        options: dict[str, Any] | None,
        model: str | None,
    ) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._response


def test_dispatch_handles_shutdown_returns_false() -> None:
    conn = _RecordingConn()
    session = _StubSession()
    assert _dispatch(conn, "shutdown", None, session) is False  # type: ignore[arg-type]
    assert conn.sent == [("ack", None)]


def test_dispatch_handles_ping() -> None:
    conn = _RecordingConn()
    session = _StubSession()
    assert _dispatch(conn, "ping", None, session) is True  # type: ignore[arg-type]
    assert conn.sent == [("pong", None)]


def test_dispatch_handles_chat_streaming() -> None:
    conn = _RecordingConn()
    session = _StubSession(
        response=iter(
            [
                {"choices": [{"delta": {"content": "a"}}]},
                {"choices": [{"delta": {"content": "b"}}]},
            ]
        )
    )
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    assert _dispatch(conn, "chat", payload, session) is True  # type: ignore[arg-type]
    assert conn.sent == [
        ("stream_chunk", "a"),
        ("stream_chunk", "b"),
        ("stream_end", None),
    ]


def test_dispatch_handles_chat_non_streaming() -> None:
    conn = _RecordingConn()
    session = _StubSession(response={"choices": [{"message": {"content": "joined"}}]})
    payload = {"messages": [], "stream": False}
    assert _dispatch(conn, "chat", payload, session) is True  # type: ignore[arg-type]
    assert conn.sent == [("result", "joined")]


def test_dispatch_handles_chat_emits_error_on_setup_exception() -> None:
    conn = _RecordingConn()
    session = _StubSession(exc=RuntimeError("setup boom"))
    payload = {"messages": [], "stream": False}
    assert _dispatch(conn, "chat", payload, session) is True  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"
    assert conn.sent[0][1].type_name == "RuntimeError"


def test_dispatch_handles_chat_emits_error_on_stream_failure() -> None:
    conn = _RecordingConn()

    def _generator():
        yield {"choices": [{"delta": {"content": "a"}}]}
        raise RuntimeError("stream broke")

    session = _StubSession(response=_generator())
    payload = {"messages": [], "stream": True}
    assert _dispatch(conn, "chat", payload, session) is True  # type: ignore[arg-type]
    # First chunk arrived, then error.
    kinds = [m[0] for m in conn.sent]
    assert kinds == ["stream_chunk", "error"]


def test_dispatch_handles_unknown_kind_emits_error() -> None:
    conn = _RecordingConn()
    session = _StubSession()
    assert _dispatch(conn, "totally_unknown", None, session) is True  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"


def test_handle_chat_rejects_non_dict_payload() -> None:
    from lilbee.providers.worker.chat_worker import _handle_chat

    conn = _RecordingConn()
    session = _StubSession()
    _handle_chat(conn, "not-a-dict", session)  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"
    assert conn.sent[0][1].type_name == "TypeError"


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
    """Default _ensure_loaded reaches load_llama with the role config's path."""
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
    assert captured == {"path": tmp_path / "stub.gguf", "mode": "chat"}


def test_chat_session_chat_passes_options_to_llama(monkeypatch, tmp_path) -> None:
    """Options dict is forwarded into create_chat_completion kwargs."""
    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    flag = multiprocessing.Value("b", 0)
    session = _ChatSession(role_config, flag)
    captured: dict[str, Any] = {}

    class _Stub:
        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(_ChatSession, "_ensure_loaded", lambda self, _o: _Stub())
    session.chat(
        messages=[],
        stream=False,
        options={"temperature": 0.42, "max_tokens": 32},
        model=None,
    )
    assert captured["temperature"] == 0.42
    assert captured["max_tokens"] == 32
    assert "abort_callback" in captured


def test_chat_session_ensure_loaded_swaps_on_per_call_model(monkeypatch, tmp_path) -> None:
    """A different per-call model path triggers a transparent reload."""
    role_config = RoleConfig(role="chat", model_path=tmp_path / "default.gguf", mode="chat")
    flag = multiprocessing.Value("b", 0)
    session = _ChatSession(role_config, flag)
    load_calls: list[Any] = []

    def fake_load_llama(path: Any, *, mode: str) -> Any:
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


# In-process loop coverage.


class _FakeConn:
    def __init__(self, inbound: list[tuple[str, Any]]) -> None:
        from collections import deque

        self._inbound = deque(inbound)
        self.sent: list[tuple[str, Any]] = []
        self.closed = False

    def poll(self, timeout: float) -> bool:
        return bool(self._inbound)

    def recv(self) -> tuple[str, Any]:
        return self._inbound.popleft()

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)

    def close(self) -> None:
        self.closed = True


def _stub_load_for_in_process(_self: _ChatSession) -> Any:
    class _Stub:
        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            return {"choices": [{"message": {"content": "fixed"}}]}

    return _Stub()


def test_chat_worker_main_serves_then_exits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.chat_worker.redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.chat_worker.configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(
        _ChatSession,
        "_ensure_loaded",
        lambda self, _o: _stub_load_for_in_process(self),
    )

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    conn = _FakeConn(
        inbound=[
            ("ping", None),
            ("chat", {"messages": [], "stream": False}),
            ("shutdown", None),
        ]
    )
    chat_worker_main(conn, multiprocessing.Value("b", 0), role_config)
    assert conn.sent == [("pong", None), ("result", "fixed"), ("ack", None)]
    assert conn.closed is True


def test_chat_worker_main_returns_on_eof(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.chat_worker.redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.chat_worker.configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(
        _ChatSession,
        "_ensure_loaded",
        lambda self, _o: _stub_load_for_in_process(self),
    )

    class _EofConn(_FakeConn):
        def recv(self) -> tuple[str, Any]:
            raise EOFError

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    conn = _EofConn(inbound=[("ignored", None)])
    chat_worker_main(conn, multiprocessing.Value("b", 0), role_config)
    assert conn.sent == []
    assert conn.closed is True


def test_chat_worker_main_skips_idle_polls(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.chat_worker.redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.chat_worker.configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(
        _ChatSession,
        "_ensure_loaded",
        lambda self, _o: _stub_load_for_in_process(self),
    )

    class _IdleThenWorkConn(_FakeConn):
        def __init__(self) -> None:
            super().__init__(inbound=[("shutdown", None)])
            self._poll_calls = 0

        def poll(self, timeout: float) -> bool:
            self._poll_calls += 1
            if self._poll_calls == 1:
                return False
            return super().poll(timeout)

    role_config = RoleConfig(role="chat", model_path=tmp_path / "x.gguf", mode="chat")
    conn = _IdleThenWorkConn()
    chat_worker_main(conn, multiprocessing.Value("b", 0), role_config)
    assert conn._poll_calls >= 2
    assert conn.sent == [("ack", None)]
