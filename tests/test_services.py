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
        from lilbee.core.services import CrawlerSyncState, Services
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


class TestPoolSpawnerSelection:
    def test_pipe_backend_returns_pipe_spawner(self):
        from lilbee.core.services import _make_pool_spawner
        from lilbee.providers.worker.transport_pipe import PipeSpawner

        cfg.worker_pool_backend = "pipe"
        assert isinstance(_make_pool_spawner(cfg), PipeSpawner)

    def test_unknown_backend_raises(self):
        from lilbee.core.services import _make_pool_spawner

        cfg.worker_pool_backend = "imaginary"
        with pytest.raises(ValueError, match="worker_pool_backend"):
            _make_pool_spawner(cfg)


class TestCancelInference:
    """Services.cancel_inference must reach pool-mode AND fallback Event."""

    def test_pool_mode_flips_per_role_abort_flag(self, monkeypatch):
        """Pool mode (default): cancel reaches every registered role."""
        cfg.worker_pool_enabled = True
        from lilbee.providers.llama_cpp import abort_signal

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
        # Make sure we reset the in-process abort flag too.
        abort_signal.clear_abort()
        services.cancel_inference()
        assert called == ["embed", "chat"]
        # In-process Event also flipped (fallback path coexists).
        assert abort_signal.is_abort_set()
        abort_signal.clear_abort()

    def test_fallback_mode_only_sets_inprocess_event(self):
        """Fallback mode: pool roles untouched, in-process Event set."""
        cfg.worker_pool_enabled = False
        from lilbee.providers.llama_cpp import abort_signal

        called: list[str] = []

        class _FakeAccessor:
            def cancel(self) -> None:
                called.append("should-not-be-called")

        class _FakePool:
            registered_roles = ("chat",)

            def accessor(self, _role: str) -> _FakeAccessor:
                return _FakeAccessor()

        from tests.conftest import make_mock_services

        services = make_mock_services(worker_pool=_FakePool())
        abort_signal.clear_abort()
        services.cancel_inference()
        assert called == []
        assert abort_signal.is_abort_set()
        abort_signal.clear_abort()


class TestEagerStartBranch:
    """``get_services`` triggers ``pool_runtime.start`` + ``start_eager`` only when
    both ``worker_pool_eager_start`` and ``worker_pool_enabled`` are True."""

    def test_eager_start_runs_when_both_flags_set(self, monkeypatch):
        """Both flags set: pool_runtime.start + start_eager run; suppress catches errors."""
        cfg.worker_pool_enabled = True
        cfg.worker_pool_eager_start = True

        from lilbee.core import services as services_mod

        services_mod.set_services(None)
        # Stub the heavy collaborators so get_services builds without spawning anything.
        # Imports inside get_services() resolve to the source modules; patch at those
        # source paths so the lazy bindings inside the function pick up the stubs.
        monkeypatch.setattr("lilbee.providers.factory.create_provider", lambda _cfg: MagicMock())
        monkeypatch.setattr(services_mod, "_make_pool_spawner", lambda _cfg: MagicMock())

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
        cfg.worker_pool_enabled = True
        cfg.worker_pool_eager_start = True

        from lilbee.core import services as services_mod

        services_mod.set_services(None)
        monkeypatch.setattr("lilbee.providers.factory.create_provider", lambda _cfg: MagicMock())
        monkeypatch.setattr(services_mod, "_make_pool_spawner", lambda _cfg: MagicMock())

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


class TestShutdownPoolErrorHandling:
    """``_shutdown_pool`` warns and force-stops when the pool drain raises."""

    def test_warns_and_forces_stop_when_run_sync_raises(self, caplog):
        from lilbee.core.services import CrawlerSyncState, Services, _shutdown_pool
        from lilbee.providers.worker.health_ticker import HealthTickerHandle

        runtime_calls: list[str] = []

        class _FailingRuntime:
            def run_sync(self, coro, *, timeout):
                runtime_calls.append("run_sync")
                # Close the coro so pytest does not warn about "never awaited".
                coro.close()
                raise RuntimeError("simulated drain failure")

            def shutdown(self, *, timeout=5.0):
                runtime_calls.append("shutdown")

        class _FakePool:
            async def shutdown(self):
                return None

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
            worker_pool=_FakePool(),
            pool_runtime=_FailingRuntime(),
            pool_health_ticker=HealthTickerHandle(),
        )
        with caplog.at_level("WARNING", logger="lilbee.core.services"):
            _shutdown_pool(services)
        assert runtime_calls == ["run_sync", "shutdown"]
        assert any("forcing runtime stop" in r.message for r in caplog.records)
