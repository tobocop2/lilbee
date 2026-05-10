"""Long-lived embed worker subprocess body."""

from __future__ import annotations

import contextlib
from typing import Any

from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import WireKind
from lilbee.providers.worker.worker_runtime import Reply, WorkerLoopState, run_worker


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
        from lilbee.providers.model_cache import LoaderMode

        return load_llama(self._role_config.model_path, mode=LoaderMode.EMBED)

    @staticmethod
    def _embed_batch(llm: Any, texts: list[str]) -> list[list[float]]:
        # circular: lilbee.providers.llama_cpp.__init__ eagerly imports
        # provider.py, which imports this worker module. Function-local
        # import keeps that cycle from firing at module-load time.
        from lilbee.providers.llama_cpp.batching import embed_batch

        return embed_batch(llm, texts)

    def close(self) -> None:
        """Release the loaded model, if any. Idempotent."""
        if self._llm is None:
            return
        with contextlib.suppress(Exception):
            self._llm.close()
        self._llm = None


def _handle_embed(reply: Reply, payload: Any, state: WorkerLoopState) -> None:
    """Run one embed request and send the typed reply (or error)."""
    if not isinstance(payload, list):
        try:
            raise TypeError(f"embed payload must be list[str], got {type(payload).__name__}")
        except TypeError as exc:
            reply.send(WireKind.ERROR, _serialize_exception(exc))
        return
    session: _EmbedSession = state.session
    try:
        vectors = session.embed(payload)
    except Exception as exc:
        reply.send(WireKind.ERROR, _serialize_exception(exc))
        return
    reply.send(WireKind.RESULT, vectors)


def embed_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Embed worker entrypoint: load llama-cpp lazily, serve requests until shutdown."""
    run_worker(
        data_conn,
        health_conn,
        abort_flag,
        role_config,
        session_factory=lambda role_cfg, _abort: _EmbedSession(role_cfg),
        kind_handlers={WireKind.EMBED: _handle_embed},
    )


__all__ = ["embed_worker_main"]
