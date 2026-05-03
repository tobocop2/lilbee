"""Tests for the persistent vision-OCR worker subprocess.

End-to-end pickle round trip via real spawn-context subprocesses, plus
pure-function tests for the dispatch table and the in-process loop.
"""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import (
    PipeSpawner,
    WorkerError,
)
from lilbee.providers.worker.vision_worker import (
    _dispatch,
    _VisionSession,
    vision_worker_main,
)

pytestmark = pytest.mark.xdist_group("worker_pool_vision")


_TEST_CALL_TIMEOUT_S = 10.0
_TEST_SHUTDOWN_TIMEOUT_S = 2.0


# Stub vision loader, applied at the worker side via monkey-patching the
# private _ensure_loaded seam. Returns the input prompt back so tests can
# assert the worker forwarded the prompt + image into the model call.


def _stub_load(_self: _VisionSession) -> Any:
    class _StubLlama:
        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            text = messages[0]["content"][1]["text"]
            return {
                "choices": [{"message": {"content": f"OCR<{text}>"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

    return _StubLlama()


def _patched_vision_worker_main(conn: Any, abort_flag: Any, role_config: RoleConfig) -> None:
    from lilbee.providers.worker import vision_worker

    vision_worker._VisionSession._ensure_loaded = lambda self, _o: _stub_load(self)  # type: ignore[method-assign]
    vision_worker_main(conn, abort_flag, role_config)


@pytest.fixture()
def role_config(tmp_path) -> RoleConfig:
    return RoleConfig(role="vision", model_path=tmp_path / "vision.gguf", mode="vision")


@pytest.fixture()
def spawner() -> PipeSpawner:
    return PipeSpawner()


@pytest.mark.asyncio
async def test_vision_worker_returns_text(spawner: PipeSpawner, role_config: RoleConfig) -> None:
    channel, _ = spawner.spawn(_patched_vision_worker_main, role_config)
    try:
        payload = {"png_bytes": b"\x89PNG fake", "model": None, "prompt": "describe"}
        result = await channel.call("vision_ocr", payload, timeout=_TEST_CALL_TIMEOUT_S)
        assert result == "OCR<describe>"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_vision_worker_rejects_non_dict_payload(
    spawner: PipeSpawner, role_config: RoleConfig
) -> None:
    channel, _ = spawner.spawn(_patched_vision_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("vision_ocr", "not-a-dict", timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "TypeError"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_vision_worker_rejects_non_bytes_png(
    spawner: PipeSpawner, role_config: RoleConfig
) -> None:
    channel, _ = spawner.spawn(_patched_vision_worker_main, role_config)
    try:
        payload = {"png_bytes": "not-bytes", "model": None, "prompt": ""}
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("vision_ocr", payload, timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "TypeError"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_vision_worker_pongs_pings(spawner: PipeSpawner, role_config: RoleConfig) -> None:
    channel, _ = spawner.spawn(_patched_vision_worker_main, role_config)
    try:
        await channel.ping(timeout=_TEST_CALL_TIMEOUT_S)
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_vision_worker_unknown_kind_returns_error(
    spawner: PipeSpawner, role_config: RoleConfig
) -> None:
    channel, _ = spawner.spawn(_patched_vision_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("not_real", None, timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "ValueError"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


# Pure-function tests.


class _RecordingConn:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)


class _StubSession:
    def __init__(self, *, text: str = "ocr-result", exc: Exception | None = None) -> None:
        self._text = text
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def ocr(self, *, png_bytes: bytes, prompt: str, model: str | None) -> str:
        self.calls.append({"png_bytes": png_bytes, "prompt": prompt, "model": model})
        if self._exc is not None:
            raise self._exc
        return self._text


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


def test_dispatch_handles_vision_emits_result() -> None:
    conn = _RecordingConn()
    session = _StubSession(text="hello")
    payload = {"png_bytes": b"x", "prompt": "p", "model": None}
    assert _dispatch(conn, "vision_ocr", payload, session) is True  # type: ignore[arg-type]
    assert conn.sent == [("result", "hello")]
    assert session.calls == [{"png_bytes": b"x", "prompt": "p", "model": None}]


def test_dispatch_handles_vision_emits_error_on_exception() -> None:
    conn = _RecordingConn()
    session = _StubSession(exc=RuntimeError("boom"))
    payload = {"png_bytes": b"x", "prompt": "", "model": None}
    assert _dispatch(conn, "vision_ocr", payload, session) is True  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"
    assert conn.sent[0][1].type_name == "RuntimeError"


def test_dispatch_handles_unknown_kind_emits_error() -> None:
    conn = _RecordingConn()
    session = _StubSession()
    assert _dispatch(conn, "totally_unknown", None, session) is True  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"


def test_handle_vision_rejects_non_dict_payload() -> None:
    from lilbee.providers.worker.vision_worker import _handle_vision

    conn = _RecordingConn()
    session = _StubSession()
    _handle_vision(conn, "garbage", session)  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"
    assert conn.sent[0][1].type_name == "TypeError"


def test_handle_vision_rejects_non_bytes_png() -> None:
    from lilbee.providers.worker.vision_worker import _handle_vision

    conn = _RecordingConn()
    session = _StubSession()
    _handle_vision(conn, {"png_bytes": "string-not-bytes"}, session)  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"
    assert conn.sent[0][1].type_name == "TypeError"


def test_session_ensure_loaded_routes_through_real_loader(monkeypatch, tmp_path) -> None:
    role_config = RoleConfig(role="vision", model_path=tmp_path / "stub.gguf", mode="vision")
    session = _VisionSession(role_config)
    sentinel = object()
    captured: dict[str, Any] = {}

    def fake_load(path: Any) -> Any:
        captured["path"] = path
        return sentinel

    monkeypatch.setattr(
        "lilbee.providers.mtmd_backend.load_vision_llama",
        fake_load,
    )
    result = session._ensure_loaded(None)
    assert result is sentinel
    assert captured == {"path": tmp_path / "stub.gguf"}


def test_session_ensure_loaded_swaps_on_per_call_model(monkeypatch, tmp_path) -> None:
    role_config = RoleConfig(role="vision", model_path=tmp_path / "default.gguf", mode="vision")
    session = _VisionSession(role_config)
    load_calls: list[Any] = []

    def fake_load(path: Any) -> Any:
        class _LlmStub:
            def close(self) -> None:
                pass

        load_calls.append(path)
        return _LlmStub()

    def fake_resolve(model: str) -> Any:
        return tmp_path / f"{model}.gguf"

    monkeypatch.setattr(
        "lilbee.providers.mtmd_backend.load_vision_llama",
        fake_load,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        fake_resolve,
    )
    first = session._ensure_loaded(None)
    again = session._ensure_loaded(None)
    assert first is again
    swapped = session._ensure_loaded("override")
    assert swapped is not first
    assert load_calls == [tmp_path / "default.gguf", tmp_path / "override.gguf"]


def test_session_close_idempotent_and_swallows() -> None:
    role_config = RoleConfig(
        role="vision",
        model_path=__import__("pathlib").Path("/nope"),
        mode="vision",
    )
    session = _VisionSession(role_config)
    session.close()  # no-op when llm is None

    class _BadLlama:
        def close(self) -> None:
            raise RuntimeError("close blew up")

    session._llm = _BadLlama()
    session.close()
    assert session._llm is None


def test_session_ocr_uses_default_prompt_when_empty(monkeypatch, tmp_path) -> None:
    """Empty prompt falls back to the global OCR_PROMPT."""
    role_config = RoleConfig(role="vision", model_path=tmp_path / "x.gguf", mode="vision")
    session = _VisionSession(role_config)
    captured: dict[str, Any] = {}

    class _Stub:
        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            captured["messages"] = messages
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(_VisionSession, "_ensure_loaded", lambda self, _o: _Stub())
    session.ocr(png_bytes=b"\x89PNG fake", prompt="", model=None)
    # The default OCR_PROMPT was passed through, not the empty string.
    text_msg = captured["messages"][0]["content"][1]["text"]
    from lilbee.vision import OCR_PROMPT

    assert text_msg == OCR_PROMPT


# In-process loop.


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


def _stub_load_for_in_process(_self: _VisionSession) -> Any:
    class _Stub:
        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            return {"choices": [{"message": {"content": "ok"}}]}

    return _Stub()


def test_vision_worker_main_serves_then_exits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.vision_worker._redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.vision_worker._configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(
        _VisionSession, "_ensure_loaded", lambda self, _o: _stub_load_for_in_process(self)
    )

    role_config = RoleConfig(role="vision", model_path=tmp_path / "x.gguf", mode="vision")
    conn = _FakeConn(
        inbound=[
            ("ping", None),
            ("vision_ocr", {"png_bytes": b"x", "prompt": "p", "model": None}),
            ("shutdown", None),
        ]
    )
    vision_worker_main(conn, _abort_flag=None, role_config=role_config)
    assert conn.sent == [("pong", None), ("result", "ok"), ("ack", None)]
    assert conn.closed is True


def test_vision_worker_main_returns_on_eof(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.vision_worker._redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.vision_worker._configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(
        _VisionSession, "_ensure_loaded", lambda self, _o: _stub_load_for_in_process(self)
    )

    class _EofConn(_FakeConn):
        def recv(self) -> tuple[str, Any]:
            raise EOFError

    role_config = RoleConfig(role="vision", model_path=tmp_path / "x.gguf", mode="vision")
    conn = _EofConn(inbound=[("ignored", None)])
    vision_worker_main(conn, _abort_flag=None, role_config=role_config)
    assert conn.sent == []
    assert conn.closed is True


def test_vision_worker_main_skips_idle_polls(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.vision_worker._redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.vision_worker._configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(
        _VisionSession, "_ensure_loaded", lambda self, _o: _stub_load_for_in_process(self)
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

    role_config = RoleConfig(role="vision", model_path=tmp_path / "x.gguf", mode="vision")
    conn = _IdleThenWorkConn()
    vision_worker_main(conn, _abort_flag=None, role_config=role_config)
    assert conn._poll_calls >= 2
    assert conn.sent == [("ack", None)]
