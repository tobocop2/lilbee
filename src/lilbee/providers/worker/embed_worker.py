"""Persistent embedding worker subprocess entrypoint.

Runs in a child process spawned by :class:`PipeSpawner`. Loads the
embedding GGUF once at startup (or lazily on first ``embed`` request)
and serves embed requests for the TUI's lifetime. The pipe transport's
in-flight counter keeps the idle reaper from racing a long batch.

Wire protocol (each message is a tuple ``(kind, payload)``):

* ``("ping", None)`` -> ``("pong", None)``
* ``("shutdown", None)`` -> ``("ack", None)`` then exit
* ``("embed", list[str])`` -> ``("result", list[list[float]])`` or
  ``("error", _SerializedException)``

The worker uses :func:`Connection.poll` with a small timeout instead of
bare :meth:`recv` so SIGTERM and shutdown checks fire promptly
(transport_pipe discipline rule 3).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import (
    ACK_KIND,
    EMBED_KIND,
    ERROR_KIND,
    PING_KIND,
    PONG_KIND,
    RESULT_KIND,
    SHUTDOWN_KIND,
)
from lilbee.providers.worker.worker_runtime import (
    configure_worker_logging,
    redirect_stdio_to_devnull,
)

log = logging.getLogger(__name__)


_POLL_TIMEOUT_S = 0.5
"""Discipline rule 3: bounded poll so the worker can react to shutdown
within a tick instead of blocking forever on bare recv."""


class _EmbedSession:
    """Lazy-loaded Llama embedder, kept alive for the worker's lifetime.

    The model loads on the first ``embed`` request rather than at spawn
    time so the parent's lazy-spawn cost stays bounded by spawn itself
    plus pickle of the role config; the heavy llama-cpp init is paid on
    first real use.
    """

    def __init__(self, role_config: RoleConfig) -> None:
        self._role_config = role_config
        self._llm: Any = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts*, loading the model on first call."""
        if self._llm is None:
            self._llm = self._load()
        return self._embed_batch(self._llm, texts)

    def _load(self) -> Any:
        from lilbee.providers.llama_cpp.provider import load_llama
        from lilbee.providers.model_cache import MODE_EMBED

        return load_llama(self._role_config.model_path, mode=MODE_EMBED)

    @staticmethod
    def _embed_batch(llm: Any, texts: list[str]) -> list[list[float]]:
        from lilbee.providers.llama_cpp.batching import embed_one

        return [embed_one(llm, text) for text in texts]

    def close(self) -> None:
        """Release the loaded model, if any. Idempotent."""
        if self._llm is None:
            return
        with contextlib.suppress(Exception):
            self._llm.close()
        self._llm = None


def _handle_embed(conn: Any, payload: Any, session: _EmbedSession) -> None:
    """Run one embed request and send the typed reply (or error)."""
    if not isinstance(payload, list):
        try:
            raise TypeError(f"embed payload must be list[str], got {type(payload).__name__}")
        except TypeError as exc:
            conn.send((ERROR_KIND, _serialize_exception(exc)))
        return
    try:
        vectors = session.embed(payload)
    except Exception as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))
        return
    conn.send((RESULT_KIND, vectors))


def _dispatch(conn: Any, kind: str, payload: Any, session: _EmbedSession) -> bool:
    """Handle one request. Return False to stop the worker loop."""
    if kind == SHUTDOWN_KIND:
        conn.send((ACK_KIND, None))
        return False
    if kind == PING_KIND:
        conn.send((PONG_KIND, None))
        return True
    if kind == EMBED_KIND:
        _handle_embed(conn, payload, session)
        return True
    try:
        raise ValueError(f"Embed worker received unknown kind {kind!r}")
    except ValueError as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))
    return True


def embed_worker_main(conn: Any, _abort_flag: Any, role_config: RoleConfig) -> None:
    """Embed worker entrypoint: load llama-cpp lazily, serve requests until shutdown."""
    redirect_stdio_to_devnull()
    configure_worker_logging(role_config.role)
    log.info("embed worker online (pid=%s, model=%s)", os.getpid(), role_config.model_path)
    session = _EmbedSession(role_config)
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


__all__ = ["embed_worker_main"]
