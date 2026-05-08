"""End-to-end tests for LlamaCppProvider.embed routing through the persistent pool."""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.core.config import cfg
from lilbee.providers.worker.embed_worker import _EmbedSession
from lilbee.providers.worker.transport import RoleConfig

pytestmark = [pytest.mark.xdist_group("worker_pool_routing")]


# =====================================================================
# Stub model loader applied at the worker side via monkey-patching the
# private _load seam. Each text becomes a deterministic 4-element vector.
# =====================================================================


def _stub_load(_self: _EmbedSession) -> Any:
    class _StubLlama:
        n_batch = 8192

        def tokenize(
            self, text: bytes, *, add_bos: bool = True, special: bool = False
        ) -> list[int]:
            return [0] * max(1, len(text))

        def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
            return {"data": [{"embedding": [float(len(t)), 0.5, 0.5, 0.5]} for t in input]}

    return _StubLlama()


def _patched_embed_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Real embed worker entrypoint with the load step swapped for a stub."""
    from lilbee.providers.worker import embed_worker

    embed_worker._EmbedSession._load = _stub_load  # type: ignore[method-assign]
    embed_worker.embed_worker_main(data_conn, health_conn, abort_flag, role_config)


def _bad_protocol_worker_main(
    conn: Any, _health_conn: Any, _abort: Any, _role_config: RoleConfig
) -> None:
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


def _install_mock_services_with_provider(provider):
    """Install a mock Services container holding *provider* and a real pool."""
    from lilbee.app.services import set_services
    from tests.conftest import make_mock_services

    services = make_mock_services(provider=provider)
    set_services(services)
    return services


@pytest.fixture()
def pool_provider(monkeypatch, tmp_path):
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
    _install_mock_services_with_provider(provider)
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
    from lilbee.app.services import get_services

    pool = get_services().worker_pool
    # Before any call, no roles registered yet.
    assert pool.registered_roles == ()
    assert pool_provider._registered_roles == set()
    pool_provider.embed(["x"])
    # After the first call, the embed role is registered.
    assert "embed" in pool.registered_roles
    assert "embed" in pool_provider._registered_roles


def test_repeated_embed_calls_reuse_one_worker(pool_provider) -> None:
    from lilbee.app.services import get_services

    pool_provider.embed(["a"])
    first_accessor = get_services().worker_pool.accessor("embed")
    pool_provider.embed(["b"])
    pool_provider.embed(["c"])
    # Same accessor across calls (registration not repeated).
    assert pool_provider._registered_roles == {"embed"}
    second_accessor = get_services().worker_pool.accessor("embed")
    assert second_accessor._role == first_accessor._role


def test_invalidate_load_cache_drops_pool(pool_provider) -> None:
    pool_provider.embed(["a"])
    assert "embed" in pool_provider._registered_roles
    pool_provider.invalidate_load_cache()
    assert pool_provider._registered_roles == set()
    # Next call rebuilds.
    pool_provider.embed(["b"])
    assert "embed" in pool_provider._registered_roles


def test_invalidate_load_cache_with_path_drops_pool(pool_provider, tmp_path) -> None:
    pool_provider.embed(["a"])
    assert "embed" in pool_provider._registered_roles
    pool_provider.invalidate_load_cache(tmp_path / "anything.gguf")
    assert pool_provider._registered_roles == set()


def test_shutdown_idempotent_after_pool_use(pool_provider) -> None:
    pool_provider.embed(["a"])
    pool_provider.shutdown()
    pool_provider.shutdown()
    assert pool_provider._registered_roles == set()


def _patch_runtime_run_sync_to_raise(monkeypatch, exc: Exception) -> None:
    """Patch PoolRuntime.run_sync to raise *exc* without leaving coroutines hanging."""
    from lilbee.providers.worker.pool import PoolRuntime

    def _run_sync(self, coro, *, timeout):
        if hasattr(coro, "close"):
            coro.close()
        raise exc

    monkeypatch.setattr(PoolRuntime, "run_sync", _run_sync)


def _setup_provider_for_error_test(monkeypatch, tmp_path):
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.chat_model = "stub/chat"
    cfg.reranker_model = "stub/rerank"
    cfg.vision_model = "stub/vision"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    return provider


def test_embed_pool_worker_error_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """Pool worker errors must propagate as ProviderError, not silently fall back."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.worker.transport_pipe import WorkerError

    provider = _setup_provider_for_error_test(monkeypatch, tmp_path)
    _patch_runtime_run_sync_to_raise(
        monkeypatch, WorkerError("RuntimeError", "simulated pool failure", "")
    )
    try:
        with pytest.raises(ProviderError, match=r"Embedding worker (exited|reported)"):
            provider.embed(["hello"])
    finally:
        provider.shutdown()


def test_embed_pool_timeout_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """Pool ``TimeoutError`` must surface as ProviderError instead of leaking raw."""
    from lilbee.providers.base import ProviderError

    provider = _setup_provider_for_error_test(monkeypatch, tmp_path)
    _patch_runtime_run_sync_to_raise(monkeypatch, TimeoutError("simulated pool timeout"))
    try:
        with pytest.raises(ProviderError, match="Embedding worker timed out"):
            provider.embed(["hello"])
    finally:
        provider.shutdown()


def test_embed_pool_protocol_error_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """A worker returning a non-list payload trips a protocol-shaped WorkerError,
    which surfaces to the caller as ProviderError instead of being silently
    swapped for the in-process path."""
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

    from lilbee.providers.base import ProviderError
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    try:
        with pytest.raises(ProviderError, match=r"Embedding worker (exited|reported)"):
            provider.embed(["abc"])
    finally:
        provider.shutdown()


@pytest.mark.parametrize(
    ("role", "cfg_attr", "expected_mode"),
    [
        ("embed", "embedding_model", "embed"),
        ("rerank", "reranker_model", "rerank"),
        ("chat", "chat_model", "chat"),
        ("vision", "vision_model", "vision"),
    ],
)
def test_make_role_config_factory_resolves_current_model(
    monkeypatch, tmp_path, role, cfg_attr, expected_mode
) -> None:
    """Each role's factory pulls its model name from cfg and stamps the right mode."""
    setattr(cfg, cfg_attr, "stub/model")
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    from lilbee.providers.llama_cpp.provider import _make_role_config_factory

    factory = _make_role_config_factory(role)
    role_config = factory()
    assert role_config.role == role
    assert role_config.mode == expected_mode
    assert role_config.model_path == tmp_path / "models" / "stub.gguf"


@pytest.mark.parametrize(
    ("role", "cfg_attr", "match"),
    [
        ("rerank", "reranker_model", "No rerank model configured"),
        ("chat", "chat_model", "No chat model configured"),
        ("vision", "vision_model", "No vision model configured"),
    ],
)
def test_make_role_config_factory_raises_when_unset(monkeypatch, role, cfg_attr, match) -> None:
    """An empty configured model triggers ProviderError before resolve_model_path runs."""
    # Bypass pydantic min_length for fields that enforce non-empty; we are
    # testing the factory's defensive check, not the schema.
    object.__setattr__(cfg, cfg_attr, "")
    from lilbee.providers.base import ProviderError
    from lilbee.providers.llama_cpp.provider import _make_role_config_factory

    factory = _make_role_config_factory(role)
    with pytest.raises(ProviderError, match=match):
        factory()


def test_role_specs_cover_every_pool_role() -> None:
    """The data table maps every role the provider routes through the pool."""
    from lilbee.providers.llama_cpp.provider import _ROLE_SPECS, WorkerRole

    assert set(_ROLE_SPECS) == set(WorkerRole)


def _patched_rerank_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Real rerank worker entrypoint with the load step swapped for a stub."""
    from lilbee.providers.worker import rerank_worker

    def _load(_self) -> Any:
        class _StubLlama:
            n_batch = 8192

            def tokenize(
                self, text: bytes, *, add_bos: bool = True, special: bool = False
            ) -> list[int]:
                return [0] * max(1, len(text))

            def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
                # Pair-batched rerank: llama-cpp returns one embedding per pair.
                sep = "</s></s>"
                data = []
                for pair in input:
                    candidate = pair.split(sep, 1)[-1] if sep in pair else pair
                    data.append({"embedding": [float(len(candidate))]})
                return {"data": data}

        return _StubLlama()

    rerank_worker._RerankSession._load = _load  # type: ignore[method-assign]
    rerank_worker.rerank_worker_main(data_conn, health_conn, abort_flag, role_config)


@pytest.fixture()
def rerank_pool_provider(monkeypatch, tmp_path):
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
    _install_mock_services_with_provider(provider)
    try:
        yield provider
    finally:
        provider.shutdown()


def test_rerank_routes_through_pool_when_enabled(rerank_pool_provider) -> None:
    scores = rerank_pool_provider.rerank("query", ["aa", "bbbb", "c"])
    assert scores == [2.0, 4.0, 1.0]


def test_repeated_rerank_calls_reuse_one_accessor(rerank_pool_provider) -> None:
    rerank_pool_provider.rerank("q", ["a"])
    assert "rerank" in rerank_pool_provider._registered_roles
    rerank_pool_provider.rerank("q", ["b"])
    rerank_pool_provider.rerank("q", ["c"])
    # Re-register would have raised; presence of the role at the end means
    # the registration is idempotent across repeat calls.
    assert rerank_pool_provider._registered_roles == {"rerank"}


def test_rerank_with_empty_candidates_short_circuits(rerank_pool_provider) -> None:
    assert rerank_pool_provider.rerank("query", []) == []
    # Empty case must not register the rerank role.
    assert "rerank" not in rerank_pool_provider._registered_roles


def test_rerank_pool_worker_error_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """Pool worker errors must propagate as ProviderError."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.worker.transport_pipe import WorkerError

    provider = _setup_provider_for_error_test(monkeypatch, tmp_path)
    _patch_runtime_run_sync_to_raise(
        monkeypatch, WorkerError("RuntimeError", "simulated pool failure", "")
    )
    try:
        with pytest.raises(ProviderError, match=r"Rerank worker (exited|reported)"):
            provider.rerank("q", ["abc", "de"])
    finally:
        provider.shutdown()


def test_rerank_pool_timeout_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """Pool TimeoutError must surface as ProviderError."""
    from lilbee.providers.base import ProviderError

    provider = _setup_provider_for_error_test(monkeypatch, tmp_path)
    _patch_runtime_run_sync_to_raise(monkeypatch, TimeoutError("simulated pool timeout"))
    try:
        with pytest.raises(ProviderError, match="Rerank worker timed out"):
            provider.rerank("q", ["abc", "de"])
    finally:
        provider.shutdown()


def _bad_rerank_protocol_worker_main(
    conn: Any, _health_conn: Any, _abort: Any, _role_config: RoleConfig
) -> None:
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


def test_rerank_pool_protocol_error_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """A worker returning a non-list payload trips a protocol-shaped WorkerError,
    which surfaces to the caller as ProviderError instead of being silently
    swapped for the in-process path."""
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

    from lilbee.providers.base import ProviderError
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    try:
        with pytest.raises(ProviderError, match=r"Rerank worker (exited|reported)"):
            provider.rerank("q", ["abc"])
    finally:
        provider.shutdown()


def _stub_chat_load(_self) -> Any:
    class _StubLlama:
        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            tokens = ["hi", " ", "there"]
            if stream:
                return iter({"choices": [{"delta": {"content": tok}}]} for tok in tokens)
            return {"choices": [{"message": {"content": "".join(tokens)}}]}

    return _StubLlama()


def _patched_chat_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    from lilbee.providers.worker import chat_worker

    chat_worker._ChatSession._ensure_loaded = lambda self, _o: _stub_chat_load(self)  # type: ignore[method-assign]
    chat_worker.chat_worker_main(data_conn, health_conn, abort_flag, role_config)


@pytest.fixture()
def chat_pool_provider(monkeypatch, tmp_path):
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/embed"
    cfg.chat_model = "stub/chat"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    fake_path = tmp_path / "models" / "stub.gguf"
    fake_path.write_bytes(b"")

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: fake_path,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.chat_worker_main",
        _patched_chat_worker_main,
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    try:
        yield provider
    finally:
        provider.shutdown()


def test_chat_routes_through_pool_non_streaming(chat_pool_provider) -> None:
    result = chat_pool_provider.chat([{"role": "user", "content": "hi"}])
    assert result == "hi there"


def test_repeated_chat_calls_reuse_one_accessor(chat_pool_provider) -> None:
    chat_pool_provider.chat([{"role": "user", "content": "a"}])
    assert chat_pool_provider._registered_roles == {"chat"}
    chat_pool_provider.chat([{"role": "user", "content": "b"}])
    assert chat_pool_provider._registered_roles == {"chat"}


def test_chat_streaming_iterator_stops_after_exhaustion(chat_pool_provider) -> None:
    iterator = chat_pool_provider.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    list(iterator)  # exhaust
    # Subsequent next() must raise StopIteration cleanly, not WorkerError.
    with pytest.raises(StopIteration):
        next(iter(iterator))


def test_chat_streaming_close_drains_remaining_chunks(chat_pool_provider) -> None:
    """close() iterates through queued chunks (the drained += 1 path)."""
    iterator = chat_pool_provider.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    next(iter(iterator))
    iterator.close()
    assert iterator._exhausted is True


def test_chat_streaming_close_swallows_drain_exceptions(chat_pool_provider) -> None:
    """Mid-stream close handles exceptions raised by next() during drain.

    Forces __next__ to raise something other than StopIteration so the
    drain loop's exception-break branch is hit. Wraps the real async
    iterator in a stand-in whose __anext__ raises after the first call.
    """
    iterator = chat_pool_provider.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    # Pull one chunk so the iterator is mid-stream.
    next(iter(iterator))

    class _AnextRaises:
        def __anext__(self):
            raise RuntimeError("simulated drain failure")

    iterator._async_iter = _AnextRaises()
    iterator.close()
    assert iterator._exhausted is True


def test_chat_routes_through_pool_streaming_yields_chunks(chat_pool_provider) -> None:
    iterator = chat_pool_provider.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = list(iterator)
    # Token batching may coalesce subsequent tokens after the first eager flush;
    # only the joined text matters at the wire level.
    assert "".join(chunks) == "hi there"


def test_chat_streaming_close_is_idempotent(chat_pool_provider) -> None:
    iterator = chat_pool_provider.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    list(iterator)  # exhaust
    iterator.close()
    iterator.close()


def test_chat_streaming_close_before_exhaustion_releases(chat_pool_provider) -> None:
    iterator = chat_pool_provider.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    # Pull one chunk only, then close mid-stream.
    next(iter(iterator))
    iterator.close()


def test_chat_streaming_mid_stream_worker_error_raises_provider_error(
    chat_pool_provider,
) -> None:
    """A WorkerError raised by ``__anext__`` mid-stream surfaces to the caller
    as ``ProviderError`` so the streaming path matches the non-streaming
    contract. Without the translation the raw RuntimeError-shaped WorkerError
    leaks past the provider boundary.
    """
    from lilbee.providers.base import ProviderError
    from lilbee.providers.worker.transport_pipe import WorkerError

    iterator = chat_pool_provider.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )

    class _AnextWorkerError:
        async def __anext__(self):
            raise WorkerError("RuntimeError", "worker died mid-stream", "")

    iterator._async_iter = _AnextWorkerError()
    with pytest.raises(ProviderError, match=r"Chat worker (exited|reported)"):
        next(iter(iterator))
    # Iterator must mark itself exhausted so subsequent next() returns
    # StopIteration instead of repeatedly raising the same crash error.
    assert iterator._exhausted is True
    with pytest.raises(StopIteration):
        next(iter(iterator))


def test_chat_streaming_mid_stream_timeout_raises_provider_error(
    chat_pool_provider,
) -> None:
    """A TimeoutError raised by ``__anext__`` mid-stream surfaces as
    ``ProviderError`` instead of leaking the raw OSError-shaped TimeoutError."""
    from lilbee.providers.base import ProviderError

    iterator = chat_pool_provider.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )

    class _AnextTimeout:
        async def __anext__(self):
            raise TimeoutError("simulated mid-stream timeout")

    iterator._async_iter = _AnextTimeout()
    with pytest.raises(ProviderError, match="Chat worker timed out"):
        next(iter(iterator))
    assert iterator._exhausted is True
    with pytest.raises(StopIteration):
        next(iter(iterator))


def test_chat_pool_worker_error_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """Pool worker errors must propagate as ProviderError."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.worker.transport_pipe import WorkerError

    provider = _setup_provider_for_error_test(monkeypatch, tmp_path)
    _patch_runtime_run_sync_to_raise(
        monkeypatch, WorkerError("RuntimeError", "simulated pool failure", "")
    )
    try:
        with pytest.raises(ProviderError, match=r"Chat worker (exited|reported)"):
            provider.chat([{"role": "user", "content": "hi"}])
    finally:
        provider.shutdown()


def test_chat_pool_timeout_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """Pool TimeoutError must surface as ProviderError."""
    from lilbee.providers.base import ProviderError

    provider = _setup_provider_for_error_test(monkeypatch, tmp_path)
    _patch_runtime_run_sync_to_raise(monkeypatch, TimeoutError("simulated pool timeout"))
    try:
        with pytest.raises(ProviderError, match="Chat worker timed out"):
            provider.chat([{"role": "user", "content": "hi"}])
    finally:
        provider.shutdown()


def _bad_chat_protocol_worker_main(
    conn: Any, _health_conn: Any, _abort: Any, _role_config: RoleConfig
) -> None:
    """Worker that always replies to non-streaming chat with a non-str payload."""
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
        if kind == "chat":
            conn.send(("result", 12345))  # non-string protocol violation
            continue
        conn.send(("result", "ignored"))


def test_chat_pool_protocol_error_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """A worker returning a non-string payload trips a protocol-shaped WorkerError,
    which surfaces to the caller as ProviderError instead of being silently
    swapped for the in-process path."""
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/embed"
    cfg.chat_model = "stub/chat"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.chat_worker_main",
        _bad_chat_protocol_worker_main,
    )

    from lilbee.providers.base import ProviderError
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    try:
        with pytest.raises(ProviderError, match=r"Chat worker (exited|reported)"):
            provider.chat([{"role": "user", "content": "hi"}])
    finally:
        provider.shutdown()


def test_chat_kwargs_filter_translates_options_correctly() -> None:
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    kwargs = LlamaCppProvider._chat_kwargs_from_options(
        {"num_predict": 50, "num_ctx": 1024, "temperature": 0.7}
    )
    # num_predict becomes max_tokens; num_ctx is dropped.
    assert kwargs == {"max_tokens": 50, "temperature": 0.7}


def test_chat_kwargs_filter_handles_empty_options() -> None:
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    assert LlamaCppProvider._chat_kwargs_from_options(None) == {}
    assert LlamaCppProvider._chat_kwargs_from_options({}) == {}


def _stub_vision_load(_self) -> Any:
    class _StubLlama:
        def create_chat_completion(self, *, messages, stream, **kwargs) -> Any:
            return {
                "choices": [{"message": {"content": "vision-result"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    return _StubLlama()


def _patched_vision_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    from lilbee.providers.worker import vision_worker

    vision_worker._VisionSession._ensure_loaded = lambda self, _o: _stub_vision_load(self)  # type: ignore[method-assign]
    vision_worker.vision_worker_main(data_conn, health_conn, abort_flag, role_config)


@pytest.fixture()
def vision_pool_provider(monkeypatch, tmp_path):
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/embed"
    cfg.vision_model = "stub/vision"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    fake_path = tmp_path / "models" / "stub.gguf"
    fake_path.write_bytes(b"")

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: fake_path,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.vision_worker_main",
        _patched_vision_worker_main,
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    try:
        yield provider
    finally:
        provider.shutdown()


def test_vision_ocr_routes_through_pool(vision_pool_provider) -> None:
    result = vision_pool_provider.vision_ocr(b"\x89PNG", "stub/vision", "describe")
    assert result == "vision-result"


def test_vision_ocr_repeated_calls_reuse_one_accessor(vision_pool_provider) -> None:
    vision_pool_provider.vision_ocr(b"\x89PNG", "stub/vision", "p")
    assert vision_pool_provider._registered_roles == {"vision"}
    vision_pool_provider.vision_ocr(b"\x89PNG", "stub/vision", "p")
    assert vision_pool_provider._registered_roles == {"vision"}


def test_vision_ocr_pool_worker_error_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """Pool worker errors must propagate as ProviderError."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.worker.transport_pipe import WorkerError

    provider = _setup_provider_for_error_test(monkeypatch, tmp_path)
    _patch_runtime_run_sync_to_raise(
        monkeypatch, WorkerError("RuntimeError", "simulated pool failure", "")
    )
    try:
        with pytest.raises(ProviderError, match=r"Vision worker (exited|reported)"):
            provider.vision_ocr(b"\x89PNG", "stub/vision", "p")
    finally:
        provider.shutdown()


def test_vision_ocr_pool_timeout_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """Pool TimeoutError must surface as ProviderError."""
    from lilbee.providers.base import ProviderError

    provider = _setup_provider_for_error_test(monkeypatch, tmp_path)
    _patch_runtime_run_sync_to_raise(monkeypatch, TimeoutError("simulated pool timeout"))
    try:
        with pytest.raises(ProviderError, match="Vision worker timed out"):
            provider.vision_ocr(b"\x89PNG", "stub/vision", "p")
    finally:
        provider.shutdown()


def _bad_vision_protocol_worker_main(
    conn: Any, _health_conn: Any, _abort: Any, _role_config: RoleConfig
) -> None:
    """Worker that always replies to vision_ocr with a non-str payload."""
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
        if kind == "vision_ocr":
            conn.send(("result", 12345))
            continue
        conn.send(("result", "ignored"))


def test_vision_ocr_pool_protocol_error_propagates_as_provider_error(monkeypatch, tmp_path) -> None:
    """A worker returning a non-string payload trips a protocol-shaped WorkerError,
    which surfaces to the caller as ProviderError instead of being silently
    swapped for the legacy subprocess path."""
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/embed"
    cfg.vision_model = "stub/vision"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.vision_worker_main",
        _bad_vision_protocol_worker_main,
    )

    from lilbee.providers.base import ProviderError
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    try:
        with pytest.raises(ProviderError, match=r"Vision worker (exited|reported)"):
            provider.vision_ocr(b"\x89PNG", "stub/vision", "p")
    finally:
        provider.shutdown()


def test_vision_call_budget_uses_per_call_timeout_when_set() -> None:
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    assert LlamaCppProvider._vision_call_budget(45.0) == 45.0


def test_vision_call_budget_falls_back_to_cfg_ocr_timeout(monkeypatch) -> None:
    cfg.ocr_timeout = 17.5
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    assert LlamaCppProvider._vision_call_budget(None) == 17.5


def test_vision_call_budget_uses_no_cap_when_zero() -> None:
    from lilbee.providers.llama_cpp.provider import _VISION_NO_CAP_TIMEOUT_S, LlamaCppProvider

    assert LlamaCppProvider._vision_call_budget(0) == _VISION_NO_CAP_TIMEOUT_S


def test_shutdown_handles_pool_release_failure(monkeypatch, tmp_path) -> None:
    """A pool that raises during release still tears down the provider cleanly."""
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

    from lilbee.app.services import get_services
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    provider.embed(["x"])
    pool = get_services().worker_pool
    assert "embed" in pool.registered_roles

    async def _boom(_role: str) -> None:
        raise RuntimeError("simulated release failure")

    monkeypatch.setattr(pool, "release", _boom)
    provider.shutdown()
    # Release failure does not leave registrations behind.
    assert provider._registered_roles == set()


def test_warm_up_pool_registers_only_configured_roles(monkeypatch, tmp_path) -> None:
    """``warm_up_pool`` registers roles whose model is set; skips empty roles.

    Verifies the eager-start path lazily picks up a partial setup (chat +
    embed) without forcing the user to also configure rerank or vision.
    """
    cfg.embedding_model = "stub/embed"
    cfg.chat_model = "stub/chat"
    cfg.reranker_model = ""
    cfg.vision_model = ""
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    try:
        provider.warm_up_pool()
        assert provider._registered_roles == {"chat", "embed"}
    finally:
        provider.shutdown()


def test_warm_up_pool_is_idempotent(monkeypatch, tmp_path) -> None:
    """Calling warm_up_pool twice does not re-register roles or raise."""
    cfg.embedding_model = "stub/embed"
    cfg.chat_model = "stub/chat"
    cfg.reranker_model = ""
    cfg.vision_model = ""
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: tmp_path / "models" / "stub.gguf",
    )

    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    _install_mock_services_with_provider(provider)
    try:
        provider.warm_up_pool()
        provider.warm_up_pool()
        assert provider._registered_roles == {"chat", "embed"}
    finally:
        provider.shutdown()
