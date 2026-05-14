"""Tests for the persistent rerank worker subprocess.

Mirrors test_embed_worker.py: real spawn-context subprocesses for the
end-to-end pickle round trip plus pure-function tests for the dispatch
helpers.
"""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.providers.worker.rerank_worker import (
    _RerankSession,
    rerank_worker_main,
)
from lilbee.providers.worker.transport import RerankPayload, RoleConfig
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
        n_batch = 8192

        def tokenize(
            self, text: bytes, *, add_bos: bool = True, special: bool = False
        ) -> list[int]:
            return [0] * max(1, len(text))

        def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
            # Score = length of the candidate part (after the </s></s> sep).
            sep = "</s></s>"
            data = []
            for pair in input:
                candidate = pair.split(sep, 1)[-1] if sep in pair else pair
                data.append({"embedding": [float(len(candidate))]})
            return {"data": data}

    return _StubLlama()


def _patched_rerank_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    from lilbee.providers.worker import rerank_worker

    rerank_worker._RerankSession._load = _stub_load  # type: ignore[method-assign]
    rerank_worker_main(data_conn, health_conn, abort_flag, role_config)


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
            RerankPayload(query="query", candidates=["aa", "bbbb", "c"]),
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
            await channel.call("rerank", "not-a-rerankpayload", timeout=_TEST_CALL_TIMEOUT_S)
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


def test_handle_rerank_emits_result() -> None:
    from lilbee.providers.worker.rerank_worker import _handle_rerank
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(scores=[0.7, 0.2])
    payload = RerankPayload(query="q", candidates=["a", "b"])
    _handle_rerank(reply, payload, WorkerLoopState(session=session))
    assert _kinds_payloads(conn) == [("result", [0.7, 0.2])]
    assert session.calls == [("q", ["a", "b"])]


def test_handle_rerank_emits_error_on_exception() -> None:
    from lilbee.providers.worker.rerank_worker import _handle_rerank
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    session = _StubSession(exc=RuntimeError("boom"))
    payload = RerankPayload(query="q", candidates=["a"])
    _handle_rerank(reply, payload, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert len(frames) == 1
    kind, payload = frames[0]
    assert kind == "error"
    assert payload.type_name == "RuntimeError"


def test_handle_rerank_rejects_non_rerankpayload() -> None:
    """Cover the malformed-payload guard with an in-process dispatch."""
    from lilbee.providers.worker.rerank_worker import _handle_rerank
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    role_config = RoleConfig(
        role="rerank",
        model_path=__import__("pathlib").Path("/nope"),
        mode="rerank",
    )
    session = _RerankSession(role_config)
    _handle_rerank(reply, "not-a-rerankpayload", WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "TypeError"


def test_handle_rerank_rejects_dict_payload() -> None:
    """Bare dicts no longer accepted; only RerankPayload."""
    from lilbee.providers.worker.rerank_worker import _handle_rerank
    from lilbee.providers.worker.worker_runtime import WorkerLoopState

    reply, conn = _make_reply()
    role_config = RoleConfig(
        role="rerank",
        model_path=__import__("pathlib").Path("/nope"),
        mode="rerank",
    )
    session = _RerankSession(role_config)
    _handle_rerank(reply, {"query": "q", "candidates": ["a"]}, WorkerLoopState(session=session))
    frames = _kinds_payloads(conn)
    assert frames[0][0] == "error"
    assert frames[0][1].type_name == "TypeError"


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


def test_session_score_loads_and_runs_compute_rerank_scores(monkeypatch, tmp_path) -> None:
    """``_RerankSession.score`` loads on first call and feeds compute_rerank_scores."""
    received_inputs: list[list[str]] = []

    class _StubLlama:
        n_batch = 8192

        def tokenize(
            self, text: bytes, *, add_bos: bool = True, special: bool = False
        ) -> list[int]:
            return [0] * max(1, len(text))

        def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
            received_inputs.append(list(input))
            sep = "</s></s>"
            data = []
            for pair in input:
                candidate = pair.split(sep, 1)[-1] if sep in pair else pair
                data.append({"embedding": [float(len(candidate))]})
            return {"data": data}

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.load_llama",
        lambda path, *, mode: _StubLlama(),
    )
    role_config = RoleConfig(role="rerank", model_path=tmp_path / "stub.gguf", mode="rerank")
    session = _RerankSession(role_config)
    scores = session.score("query", ["aaaa", "bb"])
    assert scores == [4.0, 2.0]
    # Pair-formatted inputs reached compute_rerank_scores in a single batched call.
    assert received_inputs == [["query</s></s>aaaa", "query</s></s>bb"]]


def test_rerank_worker_main_routes_through_run_worker(monkeypatch) -> None:
    """``rerank_worker_main`` passes both pipes + the rerank handler to run_worker."""
    from lilbee.providers.worker import rerank_worker

    captured: dict[str, Any] = {}

    def _fake_run_worker(data_conn, health_conn, abort_flag, role_config, **kwargs):
        captured["data"] = data_conn
        captured["health"] = health_conn
        captured["abort"] = abort_flag
        captured["role"] = role_config
        captured["kwargs"] = kwargs

    monkeypatch.setattr(rerank_worker, "run_worker", _fake_run_worker)
    role_config = RoleConfig(
        role="rerank",
        model_path=__import__("pathlib").Path("/nope"),
        mode="rerank",
    )
    rerank_worker.rerank_worker_main("DATA", "HEALTH", "ABORT", role_config)
    assert captured["data"] == "DATA"
    assert captured["health"] == "HEALTH"
    assert captured["abort"] == "ABORT"
    assert captured["role"] is role_config
    assert "rerank" in captured["kwargs"]["kind_handlers"]
