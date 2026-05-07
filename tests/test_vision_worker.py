"""Tests for the persistent vision-OCR worker subprocess.

End-to-end pickle round trip via real spawn-context subprocesses, plus
pure-function tests for the dispatch table and the in-process loop.
"""

from __future__ import annotations

import multiprocessing
from typing import Any

import pytest

from lilbee.providers.worker.transport import RoleConfig, VisionRequest
from lilbee.providers.worker.transport_pipe import (
    PipeSpawner,
    WorkerError,
)
from lilbee.providers.worker.vision_worker import (
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


def _patched_vision_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    from lilbee.providers.worker import vision_worker

    vision_worker._VisionSession._ensure_loaded = lambda self, _o: _stub_load(self)  # type: ignore[method-assign]
    vision_worker_main(data_conn, health_conn, abort_flag, role_config)


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
        payload = VisionRequest(png_bytes=b"\x89PNG fake", model=None, prompt="describe")
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
        payload = VisionRequest(png_bytes="not-bytes", model=None, prompt="")
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
    """Captures raw ``(call_id, kind, payload)`` 3-tuples sent through Reply."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, str, Any]] = []

    def send(self, message: tuple[int, str, Any]) -> None:
        self.sent.append(message)


def _make_reply(call_id: int = 1):
    from lilbee.providers.worker.worker_runtime import Reply

    conn = _RecordingConn()
    return Reply(conn, call_id), conn


def _kinds_payloads(conn: _RecordingConn) -> list[tuple[str, Any]]:
    return [(kind, payload) for _call_id, kind, payload in conn.sent]


class _StubSession:
    def __init__(self, *, text: str = "ocr-result", exc: Exception | None = None) -> None:
        self._text = text
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def ocr(self, *, png_bytes: bytes, prompt: str, model: str | None) -> str:
        self.calls.append(VisionRequest(png_bytes=png_bytes, prompt=prompt, model=model))
        if self._exc is not None:
            raise self._exc
        return self._text


def test_handle_vision_emits_result() -> None:
    from lilbee.providers.worker.vision_worker import _handle_vision
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(text="hello")
    payload = VisionRequest(png_bytes=b"x", prompt="p", model=None)
    _handle_vision(reply, payload, WorkerLoopState(session=session))
    assert _kinds_payloads(conn) == [("result", "hello")]
    assert session.calls == [VisionRequest(png_bytes=b"x", prompt="p", model=None)]


def test_handle_vision_emits_error_on_exception() -> None:
    from lilbee.providers.worker.vision_worker import _handle_vision
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(exc=RuntimeError("boom"))
    payload = VisionRequest(png_bytes=b"x", prompt="", model=None)
    _handle_vision(reply, payload, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "RuntimeError"


def test_handle_vision_rejects_non_visionrequest_payload() -> None:
    from lilbee.providers.worker.vision_worker import _handle_vision
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession()
    _handle_vision(reply, "garbage", WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "TypeError"


def test_handle_vision_rejects_dict_payload() -> None:
    """Bare dicts no longer accepted; only VisionRequest."""
    from lilbee.providers.worker.vision_worker import _handle_vision
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession()
    _handle_vision(
        reply, {"png_bytes": b"x", "prompt": "p", "model": None}, WorkerLoopState(session=session)
    )
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "TypeError"


def test_handle_vision_rejects_non_bytes_png() -> None:
    from lilbee.providers.worker.vision_worker import _handle_vision
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession()
    payload = VisionRequest(png_bytes="string-not-bytes", prompt="", model=None)  # type: ignore[arg-type]
    _handle_vision(reply, payload, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"


def test_extract_vision_content_walks_defensively() -> None:
    from lilbee.providers.worker.vision_worker import _extract_vision_content

    assert _extract_vision_content({"choices": [{"message": {"content": "x"}}]}) == "x"
    assert _extract_vision_content({"choices": [{"message": {"content": None}}]}) == ""
    with pytest.raises(TypeError):
        _extract_vision_content("not a dict")
    with pytest.raises(TypeError):
        _extract_vision_content({"choices": []})
    with pytest.raises(TypeError, match="choices\\[0\\] must be dict"):
        _extract_vision_content({"choices": ["not a dict"]})
    with pytest.raises(TypeError):
        _extract_vision_content({"choices": [{"message": "not a dict"}]})


def test_session_ocr_treats_non_dict_usage_as_empty(tmp_path, monkeypatch) -> None:
    """Defensive guard: if llama-cpp returns ``usage`` as something other than a dict,
    we coerce to ``{}`` so the log line still renders without raising."""
    from lilbee.providers.worker.vision_worker import _VisionSession

    role_config = RoleConfig(role="vision", model_path=tmp_path / "v.gguf", mode="vision")
    flag = multiprocessing.Value("b", 0)
    session = _VisionSession(role_config, flag)

    class _StubLlama:
        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": "not a dict",
            }

    monkeypatch.setattr(_VisionSession, "_ensure_loaded", lambda self, _o: _StubLlama())
    text = session.ocr(png_bytes=b"\x89PNG", prompt="p", model=None)
    assert text == "ok"


def test_session_ensure_loaded_routes_through_real_loader(monkeypatch, tmp_path) -> None:
    """Default _ensure_loaded reaches load_vision_llama with the role config's
    path and binds the abort_callback at load time (mp.Value-backed)."""
    role_config = RoleConfig(role="vision", model_path=tmp_path / "stub.gguf", mode="vision")
    flag = multiprocessing.Value("b", 0)
    session = _VisionSession(role_config, flag)
    sentinel = object()
    captured: dict[str, Any] = {}

    def fake_load(path: Any, *, abort_callback_override: Any = None) -> Any:
        captured["path"] = path
        captured["abort_callback_override"] = abort_callback_override
        return sentinel

    monkeypatch.setattr(
        "lilbee.providers.mtmd_backend.load_vision_llama",
        fake_load,
    )
    result = session._ensure_loaded(None)
    assert result is sentinel
    assert captured["path"] == tmp_path / "stub.gguf"
    cb = captured["abort_callback_override"]
    assert callable(cb)
    flag.value = 0
    assert cb() is False
    flag.value = 1
    assert cb() is True


def test_session_ensure_loaded_swaps_on_per_call_model(monkeypatch, tmp_path) -> None:
    role_config = RoleConfig(role="vision", model_path=tmp_path / "default.gguf", mode="vision")
    flag = multiprocessing.Value("b", 0)
    session = _VisionSession(role_config, flag)
    load_calls: list[Any] = []

    def fake_load(path: Any, *, abort_callback_override: Any = None) -> Any:
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
    flag = multiprocessing.Value("b", 0)
    session = _VisionSession(role_config, flag)
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
    flag = multiprocessing.Value("b", 0)
    session = _VisionSession(role_config, flag)
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


def test_vision_worker_main_routes_through_run_worker(monkeypatch) -> None:
    """``vision_worker_main`` passes both pipes + the vision handler to run_worker."""
    from lilbee.providers.worker import vision_worker

    captured: dict[str, Any] = {}

    def _fake_run_worker(data_conn, health_conn, abort_flag, role_config, **kwargs):
        captured["data"] = data_conn
        captured["health"] = health_conn
        captured["kwargs"] = kwargs

    monkeypatch.setattr(vision_worker, "run_worker", _fake_run_worker)
    role_config = RoleConfig(
        role="vision", model_path=__import__("pathlib").Path("/nope"), mode="vision"
    )
    vision_worker.vision_worker_main("DATA", "HEALTH", "ABORT", role_config)
    assert captured["data"] == "DATA"
    assert captured["health"] == "HEALTH"
    assert "vision_ocr" in captured["kwargs"]["kind_handlers"]
    assert "pdf_ocr" in captured["kwargs"]["kind_handlers"]


def test_handle_pdf_ocr_rejects_non_pdf_request_payload() -> None:
    from lilbee.providers.worker.vision_worker import _handle_pdf_ocr
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession()
    _handle_pdf_ocr(reply, "garbage", WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "TypeError"


def test_handle_pdf_ocr_rejects_non_vision_backend() -> None:
    from lilbee.providers.worker.transport import PdfOcrRequest
    from lilbee.providers.worker.vision_worker import _handle_pdf_ocr
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession()
    payload = PdfOcrRequest(path="/nope.pdf", backend="tesseract")
    _handle_pdf_ocr(reply, payload, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "ValueError"
    assert "Unsupported PDF OCR backend" in frames[0][1].message
    assert "tesseract" in frames[0][1].message


def test_handle_pdf_ocr_vision_streams_one_chunk_per_page(monkeypatch) -> None:
    """Vision backend rasterises pages, calls session.ocr per page, streams chunks."""
    from lilbee.providers.worker import vision_worker as vw
    from lilbee.providers.worker.transport import PdfOcrRequest
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    fake_pages = [(0, b"png0"), (1, b"png1"), (2, b"png2")]
    monkeypatch.setattr(vw, "rasterize_pdf", lambda _path: iter(fake_pages))
    monkeypatch.setattr(vw, "pdf_page_count", lambda _path: 3)
    reply, conn = _make_reply()
    session = _StubSession(text="OCR")
    payload = PdfOcrRequest(path="/fake.pdf", backend="vision", model="m")
    vw._handle_pdf_ocr(reply, payload, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    # Three streamed chunks (1-based page, total=3, text) followed by stream_end.
    # Frame must be a typed PdfOcrChunk: NamedTuple compares equal to a bare
    # tuple, so a regression to a positional 3-tuple would not surface
    # without the explicit type assertion below.
    from lilbee.vision import PdfOcrChunk

    assert isinstance(frames[0][1], PdfOcrChunk)
    assert frames[0] == ("stream_chunk", PdfOcrChunk(page=1, total=3, text="OCR"))
    assert frames[1] == ("stream_chunk", PdfOcrChunk(page=2, total=3, text="OCR"))
    assert frames[2] == ("stream_chunk", PdfOcrChunk(page=3, total=3, text="OCR"))
    assert frames[3] == ("stream_end", None)
    # session.ocr was called once per page with the model override.
    assert len(session.calls) == 3
    assert all(c.model == "m" for c in session.calls)


def test_handle_pdf_ocr_emits_error_on_session_exception(monkeypatch) -> None:
    from lilbee.providers.worker import vision_worker as vw
    from lilbee.providers.worker.transport import PdfOcrRequest
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    # Stub the page-count + rasterize lookups so the worker reaches the
    # session.ocr() call and the stub session's exception is what surfaces.
    monkeypatch.setattr(vw, "pdf_page_count", lambda _path: 1)
    monkeypatch.setattr(vw, "rasterize_pdf", lambda _path: iter([(0, b"png")]))
    reply, conn = _make_reply()
    session = _StubSession(exc=RuntimeError("session boom"))
    payload = PdfOcrRequest(path="/fake.pdf", backend="vision")
    vw._handle_pdf_ocr(reply, payload, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    # No stream_end emitted on error; just the error frame from session.ocr.
    assert frames[-1][0] == "error"
    assert frames[-1][1].type_name == "RuntimeError"
    assert "session boom" in frames[-1][1].message
