"""Tests for the persistent embedding worker subprocess.

The worker entrypoint runs in a real spawn-context subprocess (no mocks)
so the pickle round-trip, dispatch table, and lazy model load are
exercised end to end. Llama loading itself is patched in the child via
the ``embed_worker._EmbedSession._load`` seam so the tests do not need
a real GGUF.
"""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.providers.worker.embed_worker import (
    _EmbedSession,
    embed_worker_main,
)
from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import (
    PipeSpawner,
    WorkerError,
)

pytestmark = pytest.mark.xdist_group("worker_pool_embed")


_TEST_CALL_TIMEOUT_S = 10.0
_TEST_SHUTDOWN_TIMEOUT_S = 2.0


# =====================================================================
# Worker entrypoint that swaps the model loader for a deterministic stub.
# Module-level so spawn pickling succeeds.
# =====================================================================


def _stub_embed_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Run the embed worker with the Llama load swapped for a deterministic stub.

    Each input text becomes a 4-element float vector built from its
    length and ord(first char). Lets the pool tests assert on exact
    values without needing a real model file.
    """
    from lilbee.providers.worker import embed_worker

    class _StubLlama:
        n_batch = 8192

        def tokenize(
            self, text: bytes, *, add_bos: bool = True, special: bool = False
        ) -> list[int]:
            return [0] * max(1, len(text))

        def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
            return {
                "data": [
                    {"embedding": [float(len(t)), float(ord(t[0]) if t else 0), 1.0, 2.0]}
                    for t in input
                ]
            }

    embed_worker._EmbedSession._load = lambda self: _StubLlama()  # type: ignore[method-assign]
    embed_worker_main(data_conn, health_conn, abort_flag, role_config)


def _crash_on_load_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Worker that raises in _load so the parent observes a clean error reply."""
    from lilbee.providers.worker import embed_worker

    def _raise(self: _EmbedSession) -> Any:
        raise RuntimeError("simulated load failure")

    embed_worker._EmbedSession._load = _raise  # type: ignore[method-assign]
    embed_worker_main(data_conn, health_conn, abort_flag, role_config)


# =====================================================================
# End-to-end pipe round trip with a real spawn-context subprocess.
# =====================================================================


@pytest.fixture()
def role_config(tmp_path) -> RoleConfig:
    return RoleConfig(role="embed", model_path=tmp_path / "embed.gguf", mode="embed")


@pytest.fixture()
def spawner() -> PipeSpawner:
    return PipeSpawner()


@pytest.mark.asyncio
async def test_embed_worker_returns_vectors(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_stub_embed_worker_main, role_config)
    try:
        vectors = await channel.call("embed", ["hi", "hello"], timeout=_TEST_CALL_TIMEOUT_S)
        assert vectors == [
            [2.0, float(ord("h")), 1.0, 2.0],
            [5.0, float(ord("h")), 1.0, 2.0],
        ]
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_embed_worker_rejects_non_list_payload(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_stub_embed_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("embed", "not-a-list", timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "TypeError"
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_embed_worker_pongs_pings(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_stub_embed_worker_main, role_config)
    try:
        await channel.ping(timeout=_TEST_CALL_TIMEOUT_S)
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_embed_worker_unknown_kind_returns_error(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_stub_embed_worker_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("not_a_real_kind", None, timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "ValueError"
        assert "not_a_real_kind" in str(excinfo.value)
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


@pytest.mark.asyncio
async def test_embed_worker_surfaces_load_failure(
    spawner: PipeSpawner,
    role_config: RoleConfig,
) -> None:
    channel, _ = spawner.spawn(_crash_on_load_main, role_config)
    try:
        with pytest.raises(WorkerError) as excinfo:
            await channel.call("embed", ["x"], timeout=_TEST_CALL_TIMEOUT_S)
        assert excinfo.value.original_type == "RuntimeError"
        assert "simulated load failure" in str(excinfo.value)
    finally:
        await channel.close(timeout=_TEST_SHUTDOWN_TIMEOUT_S)


# =====================================================================
# Pure-function tests for the embed handler (no subprocess).
# =====================================================================


class _RecordingConn:
    """Captures ``(kind, payload)`` frames sent through Reply."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []

    def send(self, message: tuple[str, Any]) -> None:
        self.sent.append(message)


def _make_reply():
    from lilbee.providers.worker.worker_runtime import Reply

    conn = _RecordingConn()
    return Reply(conn), conn


def _kinds_payloads(conn: _RecordingConn) -> list[tuple[str, Any]]:
    return list(conn.sent)


class _StubSession:
    def __init__(self, *, vectors: list[list[float]] | None = None, exc: Exception | None = None):
        self._vectors = vectors or []
        self._exc = exc
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self._exc is not None:
            raise self._exc
        return self._vectors


def test_handle_embed_emits_result() -> None:
    from lilbee.providers.worker.embed_worker import _handle_embed
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(vectors=[[1.0, 2.0]])
    _handle_embed(reply, ["hello"], WorkerLoopState(session=session))
    assert _kinds_payloads(conn) == [("result", [[1.0, 2.0]])]
    assert session.calls == [["hello"]]


def test_handle_embed_emits_error_on_exception() -> None:
    from lilbee.providers.worker.embed_worker import _handle_embed
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(exc=RuntimeError("boom"))
    _handle_embed(reply, ["hello"], WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert len(frames) == 1
    kind, payload = frames[0]
    assert kind == "error"
    assert payload.type_name == "RuntimeError"
    assert payload.message == "boom"


def test_session_embed_lazy_loads_then_reuses(monkeypatch) -> None:
    """First embed call triggers _load; the second reuses the cached llm."""
    role_config = RoleConfig(
        role="embed", model_path=__import__("pathlib").Path("/nope"), mode="embed"
    )
    session = _EmbedSession(role_config)
    load_calls = 0

    def fake_load(_self: _EmbedSession) -> Any:
        nonlocal load_calls
        load_calls += 1

        class _Stub:
            n_batch = 8192

            def tokenize(
                self, text: bytes, *, add_bos: bool = True, special: bool = False
            ) -> list[int]:
                return [0] * max(1, len(text))

            def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
                return {"data": [{"embedding": [float(len(t))]} for t in input]}

        return _Stub()

    monkeypatch.setattr(_EmbedSession, "_load", fake_load)
    first = session.embed(["abc"])
    second = session.embed(["de"])
    assert first == [[3.0]]
    assert second == [[2.0]]
    assert load_calls == 1
    session.close()
    # close() drops the cached llm; another embed reloads.
    session.embed(["x"])
    assert load_calls == 2


def test_session_close_is_idempotent() -> None:
    role_config = RoleConfig(
        role="embed", model_path=__import__("pathlib").Path("/nope"), mode="embed"
    )
    session = _EmbedSession(role_config)
    session.close()
    session.close()


def test_session_close_swallows_llm_close_errors() -> None:
    """Shutdown must never crash on a misbehaving model close()."""
    role_config = RoleConfig(
        role="embed", model_path=__import__("pathlib").Path("/nope"), mode="embed"
    )
    session = _EmbedSession(role_config)

    class _BadLlama:
        def close(self) -> None:
            raise RuntimeError("close blew up")

    session._llm = _BadLlama()
    session.close()
    assert session._llm is None


def test_session_load_routes_through_real_loader(monkeypatch, tmp_path) -> None:
    """The default _load reaches load_llama with the role config's model path."""
    role_config = RoleConfig(role="embed", model_path=tmp_path / "stub.gguf", mode="embed")
    session = _EmbedSession(role_config)
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
    assert captured == {"path": role_config.model_path, "mode": "embed"}


def test_configure_worker_logging_writes_to_logs_dir(tmp_path, monkeypatch) -> None:
    """configure_worker_logging routes logs to $LILBEE_DATA/logs/worker-<role>.log."""
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    import logging as _logging

    from lilbee.providers.worker.worker_runtime import configure_worker_logging

    root = _logging.getLogger()
    handlers_before = list(root.handlers)
    try:
        configure_worker_logging("embed")
        log_path = tmp_path / "logs" / "worker-embed.log"
        assert log_path.parent.is_dir()
        # Emit a record and verify the handler picked it up.
        _logging.getLogger("lilbee.test").info("hello-from-test")
        for handler in root.handlers:
            handler.flush()
        assert "hello-from-test" in log_path.read_text()
    finally:
        for handler in list(root.handlers):
            if handler not in handlers_before:
                root.removeHandler(handler)
                handler.close()


def test_configure_worker_logging_noop_when_lilbee_data_unset(monkeypatch) -> None:
    """Without ``LILBEE_DATA`` the worker installs no log handler.

    Production never sees this state because ``_build_cfg`` canonicalizes
    the env at cfg construction; the test pins the contract for tests that
    explicitly clear the variable.
    """
    import logging as _logging

    from lilbee.providers.worker.worker_runtime import configure_worker_logging

    monkeypatch.delenv("LILBEE_DATA", raising=False)
    root = _logging.getLogger()
    handlers_before = len(root.handlers)
    configure_worker_logging("embed")
    assert len(root.handlers) == handlers_before


def test_handle_embed_rejects_non_list_payload_in_process() -> None:
    """In-process coverage for the non-list branch (subprocess test confirms wire shape)."""
    from lilbee.providers.worker.embed_worker import _handle_embed
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    role_config = RoleConfig(
        role="embed", model_path=__import__("pathlib").Path("/nope"), mode="embed"
    )
    session = _EmbedSession(role_config)
    _handle_embed(reply, "not-a-list", WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert len(frames) == 1
    kind, payload = frames[0]
    assert kind == "error"
    assert payload.type_name == "TypeError"
    assert "list[str]" in payload.message


def test_embed_worker_main_routes_through_run_worker(monkeypatch) -> None:
    """``embed_worker_main`` passes both pipes + the embed handler to run_worker."""
    from lilbee.providers.worker import embed_worker

    captured: dict[str, Any] = {}

    def _fake_run_worker(data_conn, health_conn, abort_flag, role_config, **kwargs):
        captured["data"] = data_conn
        captured["health"] = health_conn
        captured["kwargs"] = kwargs

    monkeypatch.setattr(embed_worker, "run_worker", _fake_run_worker)
    role_config = RoleConfig(
        role="embed", model_path=__import__("pathlib").Path("/nope"), mode="embed"
    )
    embed_worker.embed_worker_main("DATA", "HEALTH", "ABORT", role_config)
    assert captured["data"] == "DATA"
    assert captured["health"] == "HEALTH"
    assert "embed" in captured["kwargs"]["kind_handlers"]
