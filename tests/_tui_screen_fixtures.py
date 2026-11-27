"""Autouse fixtures of the TUI screen tests: an isolated cfg, mock services, open gates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.app.services import set_services
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    """Snapshot and restore cfg for every test."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.chunk_size = 512
    # Simulate "already-initialized" state so the app does not read this as
    # a fresh install and route away from tests that exercise chat.
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def mock_svc():
    """Inject mock Services so TUI screens never touch real backends."""
    from lilbee.data.store import SourceSummary
    from tests.conftest import make_mock_services

    store = MagicMock()
    store.search.return_value = []
    store.bm25_probe.return_value = []
    store.get_sources.return_value = []
    store.source_summary.return_value = SourceSummary(0, 0, {}, [])
    store.partly_read_sources.return_value = ({}, 0)
    store.count_sources.return_value = 0
    store.add_chunks.side_effect = len
    store.delete_by_source.return_value = None
    store.delete_source.return_value = None
    services = make_mock_services(store=store)
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    """Patch out embedding model checks and model scanning so ChatScreen mounts cleanly."""
    from lilbee.cli.tui.widgets.model_bar import ModelBar

    with (
        patch("lilbee.cli.tui.app.chat_ready", return_value=True),
        patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
        patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=False,
        ),
        patch.object(ModelBar, "_scan_models"),
    ):
        yield


@pytest.fixture(autouse=True)
def _settings_gates_open():
    """Open all SettingsScreen feature gates so legacy tests can query
    every group's editors regardless of which optional extras are
    installed in the test environment."""
    with (
        patch(
            "lilbee.providers.availability.provider_sdk_available",
            return_value=True,
        ),
        patch(
            "lilbee.cli.tui.screens.settings_widgets.crawler_installed",
            return_value=True,
        ),
    ):
        cfg.wiki = True
        yield
