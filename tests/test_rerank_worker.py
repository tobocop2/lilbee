"""Tests for the persistent rerank worker subprocess.

Mirrors test_embed_worker.py: real spawn-context subprocesses for the
end-to-end pickle round trip plus pure-function tests for the dispatch
helpers.
"""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.providers.worker.rerank_worker import (
    _dispatch,
    _RerankSession,
    rerank_worker_main,
)
from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import (
    PipeSpawner,
    WorkerError,
)

pytestmark = pytest.mark.xdist_group("worker_pool_rerank")


_TEST_CALL_TIMEOUT_S = 10.0
_TEST_SHUTDOWN_TIMEOUT_S = 2.0


# Module-level so spawn pickling succeeds.


def _stub_load(_self: _RerankSession) -> Any:
    class _StubLlama:
        def create_embedding(self, *, input: str) -> dict[str, Any]:
            # Score = length of the candidate part (after the </s></s> sep).
            sep = "</s></s>"
            candidate = input.split(sep, 1)[-1] if sep in input else input
            return {"data": [{"embedding": [float(len(candidate))]}]}

    return _StubLlama()


def _patched_rerank_worker_main(conn: Any, abort_flag: Any, role_config: RoleConfig) -> None:
    from lilbee.providers.worker import rerank_worker

    rerank_worker._RerankSession._load = _stub_load  # type: ignore[method-assign]
    rerank_worker_main(conn, abort_flag, role_config)


@pytest.fixture()
def role_config(tmp_path) -> RoleConfig:
    return RoleConfig(role="rerank", model_path=tmp_path / "rerank.gguf", mode="rerank")


@pytest.fixture()
def spawner() -> PipeSpawner:
    return PipeSpawner()


@pytest.mark.asyncio
async def test_rerank_worker_returns_scores(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_rerank_worker_main, role_config)
    try:
        scores = await channel.call(
            "rerank",
            ("query", ["aa", "bbbb", "c"]),
            timeout=_TEST_CALL_TIMEOUT_S,
        )
        assert scores == [2.0, 4.0, 1.0]
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_rerank_worker_rejects_malformed_payload(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_rerank_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("rerank", "not-a-tuple", timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "TypeError"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_rerank_worker_unknown_kind_returns_error(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_patched_rerank_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("not_real", None, timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "ValueError"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


# Pure-function dispatch tests.


class _RecordingConn:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)


class _StubSession:
    def __init__(
        self,
        *,
        scores: list[float] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._scores = scores or []
        self._exc = exc
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, candidates: list[str]) -> list[float]:
        self.calls.append((query, list(candidates)))
        if self._exc is not None:
            raise self._exc
        return self._scores


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


def test_dispatch_handles_rerank_emits_result() -> None:
    conn = _RecordingConn()
    session = _StubSession(scores=[0.7, 0.2])
    payload = ("q", ["a", "b"])
    assert _dispatch(conn, "rerank", payload, session) is True  # type: ignore[arg-type]
    assert conn.sent == [("result", [0.7, 0.2])]
    assert session.calls == [("q", ["a", "b"])]


def test_dispatch_handles_rerank_emits_error_on_exception() -> None:
    conn = _RecordingConn()
    session = _StubSession(exc=RuntimeError("boom"))
    assert _dispatch(conn, "rerank", ("q", ["a"]), session) is True  # type: ignore[arg-type]
    assert len(conn.sent) == 1
    kind, payload = conn.sent[0]
    assert kind == "error"
    assert payload.type_name == "RuntimeError"


def test_dispatch_handles_unknown_kind_emits_error() -> None:
    conn = _RecordingConn()
    session = _StubSession()
    assert _dispatch(conn, "totally_unknown", None, session) is True  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"


def test_handle_rerank_rejects_non_tuple_payload() -> None:
    """Cover the malformed-payload guard with an in-process dispatch."""
    from lilbee.providers.worker.rerank_worker import _handle_rerank

    conn = _RecordingConn()
    role_config = RoleConfig(
        role="rerank",
        model_path=__import__("pathlib").Path("/nope"),
        mode="rerank",
    )
    session = _RerankSession(role_config)
    _handle_rerank(conn, "not-a-tuple", session)  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"
    assert conn.sent[0][1].type_name == "TypeError"


def test_handle_rerank_rejects_wrong_arity() -> None:
    from lilbee.providers.worker.rerank_worker import _handle_rerank

    conn = _RecordingConn()
    role_config = RoleConfig(
        role="rerank",
        model_path=__import__("pathlib").Path("/nope"),
        mode="rerank",
    )
    session = _RerankSession(role_config)
    _handle_rerank(conn, ("only-one",), session)  # type: ignore[arg-type]
    assert conn.sent[0][0] == "error"


def test_session_load_routes_through_real_loader(monkeypatch, tmp_path) -> None:
    role_config = RoleConfig(role="rerank", model_path=tmp_path / "stub.gguf", mode="rerank")
    session = _RerankSession(role_config)
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
    result = session._load()
    assert result is sentinel
    assert captured == {"path": role_config.model_path, "mode": "rerank"}


def test_session_close_idempotent_and_swallows_close_errors() -> None:
    role_config = RoleConfig(
        role="rerank",
        model_path=__import__("pathlib").Path("/nope"),
        mode="rerank",
    )
    session = _RerankSession(role_config)
    session.close()  # no-op when llm is None

    class _BadLlama:
        def close(self) -> None:
            raise RuntimeError("close blew up")

    session._llm = _BadLlama()
    session.close()
    assert session._llm is None


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


def _stub_load_for_in_process(_self: _RerankSession) -> Any:
    class _Stub:
        def create_embedding(self, *, input: str) -> dict[str, Any]:
            return {"data": [{"embedding": [float(len(input))]}]}

    return _Stub()


def test_rerank_worker_main_serves_then_shuts_down(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.rerank_worker.redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.rerank_worker.configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(_RerankSession, "_load", _stub_load_for_in_process)

    role_config = RoleConfig(role="rerank", model_path=tmp_path / "x.gguf", mode="rerank")
    conn = _FakeConn(
        inbound=[
            ("ping", None),
            ("rerank", ("q", ["aa"])),
            ("shutdown", None),
        ]
    )
    rerank_worker_main(conn, _abort_flag=None, role_config=role_config)
    assert conn.sent[0] == ("pong", None)
    assert conn.sent[-1] == ("ack", None)
    assert conn.closed is True


def test_rerank_worker_main_returns_on_eof(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.rerank_worker.redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.rerank_worker.configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(_RerankSession, "_load", _stub_load_for_in_process)

    class _EofConn(_FakeConn):
        def recv(self) -> tuple[str, Any]:
            raise EOFError

    role_config = RoleConfig(role="rerank", model_path=tmp_path / "x.gguf", mode="rerank")
    conn = _EofConn(inbound=[("ignored", None)])
    rerank_worker_main(conn, _abort_flag=None, role_config=role_config)
    assert conn.sent == []
    assert conn.closed is True


def test_rerank_worker_main_skips_idle_polls(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "lilbee.providers.worker.rerank_worker.redirect_stdio_to_devnull",
        lambda: None,
    )
    monkeypatch.setattr(
        "lilbee.providers.worker.rerank_worker.configure_worker_logging",
        lambda _role: None,
    )
    monkeypatch.setattr(_RerankSession, "_load", _stub_load_for_in_process)

    class _IdleThenWorkConn(_FakeConn):
        def __init__(self) -> None:
            super().__init__(inbound=[("shutdown", None)])
            self._poll_calls = 0

        def poll(self, timeout: float) -> bool:
            self._poll_calls += 1
            if self._poll_calls == 1:
                return False
            return super().poll(timeout)

    role_config = RoleConfig(role="rerank", model_path=tmp_path / "x.gguf", mode="rerank")
    conn = _IdleThenWorkConn()
    rerank_worker_main(conn, _abort_flag=None, role_config=role_config)
    assert conn._poll_calls >= 2
    assert conn.sent == [("ack", None)]
