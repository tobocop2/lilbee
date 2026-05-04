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
        )
        with pytest.raises(AttributeError):
            services.clusterer = MagicMock()  # type: ignore[misc]


class TestResetStore:
    def test_keeps_provider_and_embedder_replaces_store(self, tmp_path):
        """``reset_store`` rebuilds Store-bound services without unloading the provider."""
        from lilbee.core import services as services_mod
        from lilbee.core.services import get_services, reset_services, reset_store

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

            after = services_mod._svc
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
        from lilbee.core import services as services_mod
        from lilbee.core.services import reset_services, reset_store

        reset_services()
        assert services_mod._svc is None
        reset_store()
        assert services_mod._svc is None
