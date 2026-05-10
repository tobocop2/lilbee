"""Tests for the services container."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def isolated_cfg():
    snapshot = cfg.model_copy()
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestServicesDataclass:
    def test_fields_are_immutable(self):
        from lilbee.app.services import CrawlerSyncState, Services
        from lilbee.providers.worker.health_ticker import HealthTickerHandle

        services = Services(
            provider=MagicMock(),
            store=MagicMock(),
            embedder=MagicMock(),
            reranker=MagicMock(),
            concepts=MagicMock(),
            clusterer=MagicMock(),
            searcher=MagicMock(),
            registry=MagicMock(),
            hf_client=MagicMock(),
            ingest_lock_registry=MagicMock(),
            model_manager=MagicMock(),
            crawler_semaphore=None,
            crawler_sync_state=CrawlerSyncState(),
            worker_pool=MagicMock(),
            pool_runtime=MagicMock(),
            pool_health_ticker=HealthTickerHandle(),
        )
        with pytest.raises(AttributeError):
            services.clusterer = MagicMock()  # type: ignore[misc]


class TestCancelInference:
    def test_flips_abort_flag_on_every_registered_role(self):
        """``cancel_inference`` reaches every registered role's accessor."""
        called: list[str] = []

        class _FakeAccessor:
            def __init__(self, role: str) -> None:
                self.role = role

            def cancel(self) -> None:
                called.append(self.role)

        class _FakePool:
            registered_roles = ("embed", "chat")

            def accessor(self, role: str) -> _FakeAccessor:
                return _FakeAccessor(role)

        from tests.conftest import make_mock_services

        services = make_mock_services(worker_pool=_FakePool())
        services.cancel_inference()
        assert called == ["embed", "chat"]


class TestEagerStartBranch:
    """``get_services`` triggers ``pool_runtime.start`` + ``start_eager`` when
    ``cfg.worker_pool_eager_start`` is True."""

    def test_eager_start_runs_when_flag_set(self, monkeypatch):
        """Flag set: pool_runtime.start + start_eager run; suppress catches errors."""
        cfg.worker_pool_eager_start = True

        from lilbee.app import services as services_mod

        services_mod.set_services(None)
        # Stub the heavy collaborators so get_services builds without spawning anything.
        # Imports inside get_services() resolve to the source modules; patch at those
        # source paths so the lazy bindings inside the function pick up the stubs.
        monkeypatch.setattr("lilbee.providers.factory.create_provider", lambda _cfg: MagicMock())
        monkeypatch.setattr("lilbee.providers.worker.transport.default_spawner", MagicMock)

        called: list[str] = []

        async def _start_eager_records():
            called.append("start_eager")

        class _RecordingRuntime:
            def __init__(self):
                self._started = False

            def start(self):
                called.append("start")
                self._started = True

            def run_sync(self, coro, *, timeout):
                called.append("run_sync")
                # Close the coroutine so pytest does not warn "never awaited".
                coro.close()

            def shutdown(self, *, timeout=5.0):
                pass

        # Patch PoolRuntime where get_services imports it. The function does
        # ``from lilbee.providers.worker.pool import PoolRuntime`` so patch on
        # that source module.
        monkeypatch.setattr(
            "lilbee.providers.worker.pool.PoolRuntime",
            lambda: _RecordingRuntime(),
        )
        from lilbee.providers.worker.health_ticker import HealthTickerHandle

        monkeypatch.setattr(
            "lilbee.providers.worker.health_ticker.start_health_ticker",
            lambda *_args, **_kw: HealthTickerHandle(),
        )

        # Patch WorkerPool.start_eager to a recording awaitable so run_sync sees a coro.
        from lilbee.providers.worker.pool import WorkerPool

        monkeypatch.setattr(WorkerPool, "start_eager", lambda self: _start_eager_records())

        try:
            services_mod.get_services()
        finally:
            services_mod.set_services(None)
        assert "start" in called
        assert "run_sync" in called

    def test_eager_start_swallows_runtime_failure(self, monkeypatch):
        """Suppress(Exception) keeps get_services() resilient if eager start raises."""
        cfg.worker_pool_eager_start = True

        from lilbee.app import services as services_mod

        services_mod.set_services(None)
        monkeypatch.setattr("lilbee.providers.factory.create_provider", lambda _cfg: MagicMock())
        monkeypatch.setattr("lilbee.providers.worker.transport.default_spawner", MagicMock)

        class _BoomRuntime:
            def start(self):
                raise RuntimeError("simulated start failure")

            def shutdown(self, *, timeout=5.0):
                pass

        monkeypatch.setattr("lilbee.providers.worker.pool.PoolRuntime", lambda: _BoomRuntime())
        from lilbee.providers.worker.health_ticker import HealthTickerHandle

        monkeypatch.setattr(
            "lilbee.providers.worker.health_ticker.start_health_ticker",
            lambda *_args, **_kw: HealthTickerHandle(),
        )

        try:
            # Must not raise even though start() blew up.
            svc = services_mod.get_services()
            assert svc is not None
        finally:
            services_mod.set_services(None)


class TestResetStore:
    def test_keeps_provider_and_embedder_replaces_store(self, tmp_path):
        """``reset_store`` rebuilds Store-bound services without unloading the provider."""
        from lilbee.app import services as services_mod
        from lilbee.app.services import get_services, reset_services, reset_store

        cfg.data_root = tmp_path
        cfg.documents_dir = tmp_path / "documents"
        cfg.data_dir = tmp_path / "data"
        cfg.lancedb_dir = tmp_path / "data" / "lancedb"
        cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            reset_services()
            before = get_services()
            old_store = before.store
            old_concepts = before.concepts
            old_searcher = before.searcher
            old_provider = before.provider
            old_embedder = before.embedder
            old_reranker = before.reranker
            old_registry = before.registry
            old_model_manager = before.model_manager

            reset_store()

            after = services_mod.peek_services()
            assert after is not None
            assert after.store is not old_store
            assert after.concepts is not old_concepts
            assert after.searcher is not old_searcher
            # Heavy singletons stay loaded.
            assert after.provider is old_provider
            assert after.embedder is old_embedder
            assert after.reranker is old_reranker
            assert after.registry is old_registry
            assert after.model_manager is old_model_manager
        finally:
            reset_services()

    def test_no_op_when_services_uncached(self):
        """``reset_store`` is a no-op if Services has not been built yet."""
        from lilbee.app import services as services_mod
        from lilbee.app.services import reset_services, reset_store

        reset_services()
        assert services_mod.peek_services() is None
        reset_store()
        assert services_mod.peek_services() is None


def test_reset_services_dependencies_load_eagerly():
    """``reset_services`` is registered with atexit; its imports must be eager.

    Lazy-importing ``shutdown_pool_runtime`` from inside ``reset_services``
    used to first-load ``concurrent.futures.thread`` during interpreter
    shutdown, which fails with ``RuntimeError: can't register atexit
    after shutdown``. Hoisting the import to module top forces the
    stdlib atexit registration to happen at app start. This test pins
    that contract so a future contributor cannot re-introduce the lazy
    import without breaking the build.
    """
    import sys

    import lilbee.app.services  # noqa: F401

    assert "lilbee.providers.worker.pool" in sys.modules
    assert "concurrent.futures.thread" in sys.modules
