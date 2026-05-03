"""Tests for LlamaCppProvider.embed routing through the persistent pool.

When ``cfg.worker_pool_enabled = True`` (the default in production),
``embed`` must reach a real subprocess via the pool. When disabled, the
in-process / legacy paths must continue to work unchanged.

These tests stand in their own file because the project-wide conftest
defaults ``worker_pool_enabled = False`` for unit tests; this file opts
back into the production default with ``@pytest.mark.worker_pool``.
"""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.core.config import cfg
from lilbee.providers.worker.embed_worker import _EmbedSession
from lilbee.providers.worker.transport import RoleConfig

pytestmark = [
    pytest.mark.worker_pool,
    pytest.mark.xdist_group("worker_pool_routing"),
]


# =====================================================================
# Stub model loader applied at the worker side via monkey-patching the
# private _load seam. Each text becomes a deterministic 4-element vector.
# =====================================================================


def _stub_load(_self: _EmbedSession) -> Any:
    class _StubLlama:
        def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
            return {"data": [{"embedding": [float(len(t)), 0.5, 0.5, 0.5]} for t in input]}

    return _StubLlama()


def _patched_embed_worker_main(conn: Any, abort_flag: Any, role_config: RoleConfig) -> None:
    """Real embed worker entrypoint with the load step swapped for a stub."""
    from lilbee.providers.worker import embed_worker

    embed_worker._EmbedSession._load = _stub_load  # type: ignore[method-assign]
    embed_worker.embed_worker_main(conn, abort_flag, role_config)


def _bad_protocol_worker_main(conn: Any, _abort: Any, _role_config: RoleConfig) -> None:
    """Worker that always replies to embed with a non-list payload (protocol violation)."""
    while True:
        if not conn.poll(timeout=0.1):
            continue
        try:
            kind, _ = conn.recv()
        except EOFError:
            return
        if kind == "shutdown":
            conn.send(("ack", None))
            return
        if kind == "embed":
            conn.send(("result", "not-a-list"))
            continue
        conn.send(("result", "ignored"))


# =====================================================================
# Provider fixture: build a real LlamaCppProvider, swap the embed
# worker entrypoint with our stubbed version, ensure shutdown.
# =====================================================================


@pytest.fixture()
def pool_provider(monkeypatch, tmp_path):
    cfg.worker_pool_enabled = True
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    fake_path = tmp_path / "models" / "stub.gguf"
    fake_path.write_bytes(b"")

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: fake_path,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.embed_worker_main",
        _patched_embed_worker_main,
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    try:
        yield provider
    finally:
        provider.shutdown()


def test_embed_routes_through_pool_when_enabled(pool_provider) -> None:
    vectors = pool_provider.embed(["abc", "de"])
    assert vectors == [
        [3.0, 0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5, 0.5],
    ]


def test_pool_is_lazy_no_spawn_until_first_call(pool_provider) -> None:
    # Before any call, _pool is None.
    assert pool_provider._pool is None
    pool_provider.embed(["x"])
    # After the first call, the pool is up.
    assert pool_provider._pool is not None
    assert pool_provider._pool_runtime is not None


def test_repeated_embed_calls_reuse_one_worker(pool_provider) -> None:
    pool_provider.embed(["a"])
    first_pool = pool_provider._pool
    pool_provider.embed(["b"])
    pool_provider.embed(["c"])
    # Same pool object across calls (no respawn).
    assert pool_provider._pool is first_pool


def test_invalidate_load_cache_drops_pool(pool_provider) -> None:
    pool_provider.embed(["a"])
    assert pool_provider._pool is not None
    pool_provider.invalidate_load_cache()
    assert pool_provider._pool is None
    # Next call rebuilds.
    pool_provider.embed(["b"])
    assert pool_provider._pool is not None


def test_invalidate_load_cache_with_path_drops_pool(pool_provider, tmp_path) -> None:
    pool_provider.embed(["a"])
    assert pool_provider._pool is not None
    pool_provider.invalidate_load_cache(tmp_path / "anything.gguf")
    assert pool_provider._pool is None


def test_shutdown_idempotent_after_pool_use(pool_provider) -> None:
    pool_provider.embed(["a"])
    pool_provider.shutdown()
    pool_provider.shutdown()
    assert pool_provider._pool is None
    assert pool_provider._pool_runtime is None


def test_embed_falls_back_to_inproc_when_pool_raises(monkeypatch, tmp_path) -> None:
    """Pool failure must not break embed; the in-process queue path takes over."""
    cfg.worker_pool_enabled = True
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()

    def _boom(_texts):
        raise RuntimeError("simulated pool failure")

    monkeypatch.setattr(provider, "_embed_via_pool", _boom)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.embed_one",
        lambda _llm, text: [float(len(text))],
    )
    monkeypatch.setattr(provider, "_get_embed_llm", lambda: object())
    try:
        result = provider.embed(["hello"])
        assert result == [[5.0]]
    finally:
        provider.shutdown()


def test_embed_pool_protocol_error_when_worker_returns_non_list(monkeypatch, tmp_path) -> None:
    """Worker returning a non-list payload surfaces as a clear ProtocolError."""
    cfg.worker_pool_enabled = True
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.embed_worker_main",
        _bad_protocol_worker_main,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.embed_one",
        lambda _llm, text: [float(len(text))],
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    monkeypatch.setattr(provider, "_get_embed_llm", lambda: object())
    try:
        result = provider.embed(["abc"])
        assert result == [[3.0]]
    finally:
        provider.shutdown()


def test_make_embed_role_config_factory_resolves_current_model(monkeypatch, tmp_path) -> None:
    cfg.embedding_model = "stub/model"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    from lilbee.providers.llama_cpp.provider import _make_embed_role_config_factory

    factory = _make_embed_role_config_factory()
    role_config = factory()
    assert role_config.role == "embed"
    assert role_config.mode == "embed"
    assert role_config.model_path == tmp_path / "models" / "stub.gguf"


def _patched_rerank_worker_main(conn: Any, abort_flag: Any, role_config: RoleConfig) -> None:
    """Real rerank worker entrypoint with the load step swapped for a stub."""
    from lilbee.providers.worker import rerank_worker

    def _load(_self) -> Any:
        class _StubLlama:
            def create_embedding(self, *, input: str) -> dict[str, Any]:
                # Length of the candidate substring after the </s></s> sep.
                sep = "</s></s>"
                candidate = input.split(sep, 1)[-1] if sep in input else input
                return {"data": [{"embedding": [float(len(candidate))]}]}

        return _StubLlama()

    rerank_worker._RerankSession._load = _load  # type: ignore[method-assign]
    rerank_worker.rerank_worker_main(conn, abort_flag, role_config)


@pytest.fixture()
def rerank_pool_provider(monkeypatch, tmp_path):
    cfg.worker_pool_enabled = True
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.reranker_model = "stub/reranker"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    fake_path = tmp_path / "models" / "stub.gguf"
    fake_path.write_bytes(b"")

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: fake_path,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.embed_worker_main",
        _patched_embed_worker_main,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.rerank_worker_main",
        _patched_rerank_worker_main,
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    try:
        yield provider
    finally:
        provider.shutdown()


def test_rerank_routes_through_pool_when_enabled(rerank_pool_provider) -> None:
    scores = rerank_pool_provider.rerank("query", ["aa", "bbbb", "c"])
    assert scores == [2.0, 4.0, 1.0]


def test_repeated_rerank_calls_reuse_one_accessor(rerank_pool_provider) -> None:
    rerank_pool_provider.rerank("q", ["a"])
    first = rerank_pool_provider._pool_rerank_accessor
    rerank_pool_provider.rerank("q", ["b"])
    rerank_pool_provider.rerank("q", ["c"])
    # Same accessor across calls (no re-register).
    assert rerank_pool_provider._pool_rerank_accessor is first


def test_rerank_with_empty_candidates_short_circuits(rerank_pool_provider) -> None:
    assert rerank_pool_provider.rerank("query", []) == []
    # Empty case must not spawn a pool worker.
    assert rerank_pool_provider._pool_rerank_accessor is None


def test_rerank_falls_back_to_inproc_when_pool_raises(monkeypatch, tmp_path) -> None:
    cfg.worker_pool_enabled = True
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.reranker_model = "stub/reranker"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()

    def _boom(_query, _candidates):
        raise RuntimeError("simulated pool failure")

    monkeypatch.setattr(provider, "_rerank_via_pool", _boom)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.compute_rerank_scores",
        lambda _llm, _q, candidates: [float(len(c)) for c in candidates],
    )
    monkeypatch.setattr(provider, "_get_rerank_llm", lambda: object())
    try:
        scores = provider.rerank("q", ["abc", "de"])
        assert scores == [3.0, 2.0]
    finally:
        provider.shutdown()


def _bad_rerank_protocol_worker_main(conn: Any, _abort: Any, _role_config: RoleConfig) -> None:
    """Worker that always replies to rerank with a non-list payload."""
    while True:
        if not conn.poll(timeout=0.1):
            continue
        try:
            kind, _ = conn.recv()
        except EOFError:
            return
        if kind == "shutdown":
            conn.send(("ack", None))
            return
        if kind == "rerank":
            conn.send(("result", "not-a-list"))
            continue
        conn.send(("result", "ignored"))


def test_rerank_pool_protocol_error_when_worker_returns_non_list(monkeypatch, tmp_path) -> None:
    cfg.worker_pool_enabled = True
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.reranker_model = "stub/reranker"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.rerank_worker_main",
        _bad_rerank_protocol_worker_main,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.compute_rerank_scores",
        lambda _llm, _q, candidates: [float(len(c)) for c in candidates],
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    monkeypatch.setattr(provider, "_get_rerank_llm", lambda: object())
    try:
        scores = provider.rerank("q", ["abc"])
        assert scores == [3.0]
    finally:
        provider.shutdown()


def test_make_rerank_role_config_factory_resolves_current_model(monkeypatch, tmp_path) -> None:
    cfg.reranker_model = "stub/reranker"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    from lilbee.providers.llama_cpp.provider import _make_rerank_role_config_factory

    factory = _make_rerank_role_config_factory()
    role_config = factory()
    assert role_config.role == "rerank"
    assert role_config.mode == "rerank"


def test_make_rerank_role_config_factory_raises_when_unset(monkeypatch) -> None:
    cfg.reranker_model = ""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.llama_cpp.provider import _make_rerank_role_config_factory

    factory = _make_rerank_role_config_factory()
    with pytest.raises(ProviderError, match="No reranker model configured"):
        factory()


def test_shutdown_handles_pool_shutdown_failure(monkeypatch, tmp_path) -> None:
    """A pool that raises during shutdown still tears down the runtime cleanly."""
    cfg.worker_pool_enabled = True
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.embed_worker_main",
        _patched_embed_worker_main,
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    provider.embed(["x"])
    real_pool = provider._pool
    assert real_pool is not None

    async def _boom() -> None:
        raise RuntimeError("simulated shutdown failure")

    monkeypatch.setattr(real_pool, "shutdown", _boom)
    provider.shutdown()
    assert provider._pool is None
    assert provider._pool_runtime is None
