"""Tests for the services container."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole


@pytest.fixture(autouse=True)
def isolated_cfg():
    snapshot = cfg.model_copy()
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestSyncVisionOcrBackend:
    def _patch_kreuzberg(self, monkeypatch, *, listed):
        reg = MagicMock()
        unreg = MagicMock()
        monkeypatch.setattr("kreuzberg.list_ocr_backends", lambda: listed)
        monkeypatch.setattr("kreuzberg.register_ocr_backend", reg)
        monkeypatch.setattr("kreuzberg.unregister_ocr_backend", unreg)
        return reg, unreg

    def test_registers_when_model_set_and_absent(self, monkeypatch):
        from lilbee.app.services import sync_vision_ocr_backend

        monkeypatch.setattr(cfg, "vision_model", "vendor/glm-ocr")
        reg, unreg = self._patch_kreuzberg(monkeypatch, listed=["tesseract"])
        sync_vision_ocr_backend(MagicMock())
        reg.assert_called_once()
        unreg.assert_not_called()

    def test_rebinds_to_current_provider_when_already_registered(self, monkeypatch):
        """A rebuilt provider must replace the stale binding: unregister then re-register.

        ``reset_services`` shuts the old provider down; if sync left the prior
        registration in place, kreuzberg would keep routing OCR to the dead provider.
        """
        from lilbee.app.services import sync_vision_ocr_backend

        monkeypatch.setattr(cfg, "vision_model", "vendor/glm-ocr")
        reg, unreg = self._patch_kreuzberg(monkeypatch, listed=["lilbee-vision"])
        sync_vision_ocr_backend(MagicMock())
        unreg.assert_called_once_with("lilbee-vision")
        reg.assert_called_once()

    def test_unregisters_when_model_cleared(self, monkeypatch):
        from lilbee.app.services import sync_vision_ocr_backend

        monkeypatch.setattr(cfg, "vision_model", "")
        reg, unreg = self._patch_kreuzberg(monkeypatch, listed=["lilbee-vision"])
        sync_vision_ocr_backend(MagicMock())
        unreg.assert_called_once_with("lilbee-vision")
        reg.assert_not_called()

    def test_noop_when_no_model_and_absent(self, monkeypatch):
        from lilbee.app.services import sync_vision_ocr_backend

        monkeypatch.setattr(cfg, "vision_model", "")
        reg, unreg = self._patch_kreuzberg(monkeypatch, listed=["tesseract"])
        sync_vision_ocr_backend(MagicMock())
        reg.assert_not_called()
        unreg.assert_not_called()

    def test_settings_role_reload_syncs_vision_backend(self, monkeypatch):
        """A vision_model change via any settings path (REST/MCP/TUI/CLI) registers it."""
        from lilbee.app.services import set_services
        from lilbee.app.settings import _reload_changed_roles
        from tests.conftest import make_mock_services

        set_services(make_mock_services())
        try:
            monkeypatch.setattr(cfg, "vision_model", "org/V-GGUF/v-Q4_K_M.gguf")
            reg, _unreg = self._patch_kreuzberg(monkeypatch, listed=["tesseract"])
            _reload_changed_roles({"vision_model"})
            reg.assert_called_once()
        finally:
            set_services(None)


class TestServicesDataclass:
    def test_fields_are_immutable(self):
        from lilbee.app.services import CrawlerSyncState, Services

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
            known_models=MagicMock(),
        )
        with pytest.raises(AttributeError):
            services.clusterer = MagicMock()  # type: ignore[misc]


class TestCancelInference:
    def test_delegates_to_provider(self):
        """``cancel_inference`` forwards to the provider's cancel hook."""
        from tests.conftest import make_mock_services

        provider = MagicMock()
        services = make_mock_services(provider=provider)
        services.cancel_inference()
        provider.cancel_inference.assert_called_once_with()


class TestReloadRole:
    def test_delegates_to_provider_with_role(self):
        """``reload_role`` forwards the requested role (and wait flag) to the provider."""
        from tests.conftest import make_mock_services

        provider = MagicMock()
        services = make_mock_services(provider=provider)
        services.reload_role(WorkerRole.EMBED)
        provider.reload_role.assert_called_once_with(WorkerRole.EMBED, wait=False)

    def test_forwards_wait_flag_to_provider(self):
        """``wait=True`` (the chat-swap path) is forwarded to the provider."""
        from tests.conftest import make_mock_services

        provider = MagicMock()
        services = make_mock_services(provider=provider)
        services.reload_role(WorkerRole.CHAT, wait=True)
        provider.reload_role.assert_called_once_with(WorkerRole.CHAT, wait=True)


class TestAddPoolListener:
    def test_forwards_callbacks_to_provider(self):
        """``add_pool_listener`` forwards both callbacks to the provider."""
        from tests.conftest import make_mock_services

        provider = MagicMock()
        services = make_mock_services(provider=provider)

        def on_spawning(_role: WorkerRole) -> None: ...

        def on_spawned(_role: WorkerRole) -> None: ...

        services.add_pool_listener(on_spawning=on_spawning, on_spawned=on_spawned)
        provider.add_spawn_listener.assert_called_once_with(
            on_spawning=on_spawning, on_spawned=on_spawned
        )


class TestEagerStartBranch:
    """``get_services`` warms the fleet when ``cfg.worker_pool_eager_start`` is set."""

    def test_eager_start_warms_provider_when_flag_set(self, monkeypatch):
        cfg.worker_pool_eager_start = True

        from lilbee.app import services as services_mod

        services_mod.set_services(None)
        provider = MagicMock()
        monkeypatch.setattr("lilbee.providers.factory.create_provider", lambda _cfg: provider)
        try:
            services_mod.get_services()
        finally:
            services_mod.set_services(None)
        provider.warm_up_pool.assert_called_once_with()

    def test_eager_start_swallows_warm_up_failure(self, monkeypatch):
        """suppress(Exception) keeps get_services() resilient if warm-up raises."""
        cfg.worker_pool_eager_start = True

        from lilbee.app import services as services_mod

        services_mod.set_services(None)
        provider = MagicMock()
        provider.warm_up_pool.side_effect = RuntimeError("simulated warm-up failure")
        monkeypatch.setattr("lilbee.providers.factory.create_provider", lambda _cfg: provider)
        try:
            svc = services_mod.get_services()
            assert svc is not None
        finally:
            services_mod.set_services(None)

    def test_no_warm_up_when_flag_clear(self, monkeypatch):
        cfg.worker_pool_eager_start = False

        from lilbee.app import services as services_mod

        services_mod.set_services(None)
        provider = MagicMock()
        monkeypatch.setattr("lilbee.providers.factory.create_provider", lambda _cfg: provider)
        try:
            services_mod.get_services()
        finally:
            services_mod.set_services(None)
        provider.warm_up_pool.assert_not_called()


class TestResetServicesSwapBeforeClose:
    """reset_services clears the singleton before tearing the old one down."""

    def test_reference_cleared_before_teardown(self):
        from lilbee.app import services as services_mod
        from tests.conftest import make_mock_services

        observed: list[bool] = []
        store = MagicMock()
        provider = MagicMock()
        # When teardown runs, the module singleton must already be None so a
        # concurrent get_services() never hands out a closing container.
        store.close.side_effect = lambda: observed.append(services_mod.peek_services() is None)

        services_mod.set_services(make_mock_services(store=store, provider=provider))
        try:
            services_mod.reset_services()
            assert services_mod.peek_services() is None
            provider.shutdown.assert_called_once()
            store.close.assert_called_once()
            assert observed == [True]  # cleared before close
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
