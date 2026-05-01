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
