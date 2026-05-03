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

    def test_does_not_log_when_attribute_missing(self, caplog):
        from lilbee.core import services as services_mod

        services_mod._DEPRECATED_SUBPROCESS_EMBED_LOGGED = False

        class _Cfg:
            pass

        with caplog.at_level("WARNING", logger="lilbee.core.services"):
            services_mod._log_subprocess_embed_deprecation_once(_Cfg())
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_reset_services_clears_logged_flag(self):
        from lilbee.core import services as services_mod

        services_mod._DEPRECATED_SUBPROCESS_EMBED_LOGGED = True
        services_mod.reset_services()
        assert services_mod._DEPRECATED_SUBPROCESS_EMBED_LOGGED is False
