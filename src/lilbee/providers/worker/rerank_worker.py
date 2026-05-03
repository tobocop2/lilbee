"""Persistent reranker worker subprocess entrypoint.

Runs in a child process spawned by :class:`PipeSpawner`. Loads the
reranker GGUF on first request and serves rerank requests for the TUI's
lifetime. Wire protocol mirrors the embed worker:

* ``("ping", None)`` -> ``("pong", None)``
* ``("shutdown", None)`` -> ``("ack", None)`` then exit
* ``("rerank", (query: str, candidates: list[str]))`` -> ``("result", list[float])``
  or ``("error", _SerializedException)``
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

from lilbee.providers.worker.embed_worker import (
    _configure_worker_logging,
    _redirect_stdio_to_devnull,
)
from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception

log = logging.getLogger(__name__)


_POLL_TIMEOUT_S = 0.5
_RERANK_KIND = "rerank"
_RERANK_PAYLOAD_LEN = 2
_PING_KIND = "ping"
_SHUTDOWN_KIND = "shutdown"
_RESULT_KIND = "result"
_ERROR_KIND = "error"
_PONG_KIND = "pong"
_ACK_KIND = "ack"


class _RerankSession:
    """Lazy-loaded Llama reranker, kept alive for the worker's lifetime."""

    def __init__(self, role_config: RoleConfig) -> None:
        self._role_config = role_config
        self._llm: Any = None

    def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score *candidates* against *query*, loading the model on first call."""
        if self._llm is None:
            self._llm = self._load()
        return self._compute(self._llm, query, candidates)

    def _load(self) -> Any:
        from lilbee.providers.llama_cpp.provider import load_llama
        from lilbee.providers.model_cache import MODE_RERANK

        return load_llama(self._role_config.model_path, mode=MODE_RERANK)

    @staticmethod
    def _compute(llm: Any, query: str, candidates: list[str]) -> list[float]:
        from lilbee.providers.llama_cpp.batching import compute_rerank_scores

        return compute_rerank_scores(llm, query, candidates)

    def close(self) -> None:
        """Release the loaded model, if any. Idempotent."""
        if self._llm is None:
            return
        with contextlib.suppress(Exception):
            self._llm.close()
        self._llm = None


def _handle_rerank(conn: Any, payload: Any, session: _RerankSession) -> None:
    """Run one rerank request and send the typed reply (or error)."""
    if (
        not isinstance(payload, tuple)
        or len(payload) != _RERANK_PAYLOAD_LEN
        or not isinstance(payload[0], str)
        or not isinstance(payload[1], list)
    ):
        try:
            raise TypeError(
                f"rerank payload must be (query: str, candidates: list[str]); "
                f"got {type(payload).__name__}"
            )
        except TypeError as exc:
            conn.send((_ERROR_KIND, _serialize_exception(exc)))
        return
    query, candidates = payload
    try:
        scores = session.score(query, candidates)
    except Exception as exc:
        conn.send((_ERROR_KIND, _serialize_exception(exc)))
        return
    conn.send((_RESULT_KIND, scores))


def _dispatch(conn: Any, kind: str, payload: Any, session: _RerankSession) -> bool:
    """Handle one request. Return False to stop the worker loop."""
    if kind == _SHUTDOWN_KIND:
        conn.send((_ACK_KIND, None))
        return False
    if kind == _PING_KIND:
        conn.send((_PONG_KIND, None))
        return True
    if kind == _RERANK_KIND:
        _handle_rerank(conn, payload, session)
        return True
    try:
        raise ValueError(f"Rerank worker received unknown kind {kind!r}")
    except ValueError as exc:
        conn.send((_ERROR_KIND, _serialize_exception(exc)))
    return True


def rerank_worker_main(conn: Any, _abort_flag: Any, role_config: RoleConfig) -> None:
    """Rerank worker entrypoint: load llama-cpp lazily, serve until shutdown."""
    _redirect_stdio_to_devnull()
    _configure_worker_logging(role_config.role)
    log.info("rerank worker online (pid=%s, model=%s)", os.getpid(), role_config.model_path)
    session = _RerankSession(role_config)
    try:
        while True:
            if not conn.poll(timeout=_POLL_TIMEOUT_S):
                continue
            try:
                kind, payload = conn.recv()
            except EOFError:
                return
            if not _dispatch(conn, kind, payload, session):
                return
    finally:
        session.close()
        with contextlib.suppress(Exception):
            conn.close()


__all__ = ["rerank_worker_main"]
