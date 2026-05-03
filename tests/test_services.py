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
        )
        with pytest.raises(AttributeError):
            services.clusterer = MagicMock()  # type: ignore[misc]


class TestSubprocessEmbedDeprecationLog:
    def test_logs_when_subprocess_embed_true(self, caplog):
        from lilbee.core import services as services_mod

        services_mod._DEPRECATED_SUBPROCESS_EMBED_LOGGED = False

        class _Cfg:
            subprocess_embed = True

        with caplog.at_level("WARNING", logger="lilbee.core.services"):
            services_mod._log_subprocess_embed_deprecation_once(_Cfg())
        assert any("subprocess_embed" in r.message for r in caplog.records)

    def test_logs_only_once(self, caplog):
        from lilbee.core import services as services_mod

        services_mod._DEPRECATED_SUBPROCESS_EMBED_LOGGED = False

        class _Cfg:
            subprocess_embed = True

        with caplog.at_level("WARNING", logger="lilbee.core.services"):
            services_mod._log_subprocess_embed_deprecation_once(_Cfg())
            services_mod._log_subprocess_embed_deprecation_once(_Cfg())
        # Exactly one warning recorded across both calls.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_does_not_log_when_subprocess_embed_false(self, caplog):
        from lilbee.core import services as services_mod

        services_mod._DEPRECATED_SUBPROCESS_EMBED_LOGGED = False

        class _Cfg:
            subprocess_embed = False

        with caplog.at_level("WARNING", logger="lilbee.core.services"):
            services_mod._log_subprocess_embed_deprecation_once(_Cfg())
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_reset_services_clears_logged_flag(self):
        from lilbee.core import services as services_mod

        services_mod._DEPRECATED_SUBPROCESS_EMBED_LOGGED = True
        services_mod.reset_services()
        assert services_mod._DEPRECATED_SUBPROCESS_EMBED_LOGGED is False


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
