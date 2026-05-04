"""Long-lived rerank worker subprocess body."""

from __future__ import annotations

import contextlib
from typing import Any

from lilbee.providers.worker.transport import RerankPayload, RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import ERROR_KIND, RERANK_KIND, RESULT_KIND
from lilbee.providers.worker.worker_runtime import WorkerLoopState, run_worker


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


def _handle_rerank(conn: Any, payload: Any, state: WorkerLoopState) -> None:
    """Run one rerank request and send the typed reply (or error)."""
    if not isinstance(payload, RerankPayload):
        try:
            raise TypeError(f"rerank payload must be RerankPayload, got {type(payload).__name__}")
        except TypeError as exc:
            conn.send((ERROR_KIND, _serialize_exception(exc)))
        return
    session: _RerankSession = state.session
    try:
        scores = session.score(payload.query, payload.candidates)
    except Exception as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))
        return
    conn.send((RESULT_KIND, scores))


def rerank_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Rerank worker entrypoint: load llama-cpp lazily, serve until shutdown."""
    run_worker(
        data_conn,
        health_conn,
        abort_flag,
        role_config,
        session_factory=lambda role_cfg, _abort: _RerankSession(role_cfg),
        kind_handlers={RERANK_KIND: _handle_rerank},
    )


__all__ = ["rerank_worker_main"]
