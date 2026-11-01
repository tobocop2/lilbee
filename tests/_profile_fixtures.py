"""Fixtures shared by the Profile tab and profile library tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import pytest

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import ready_services


@pytest.fixture(autouse=True)
def gate_releases_at_once() -> Iterator[None]:
    with ready_services():
        yield


@pytest.fixture(autouse=True)
def isolated_cfg(tmp_path) -> Iterator[None]:
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@contextmanager
def sources_totaling(count: int) -> Iterator[mock.MagicMock]:
    """Stand in for the store behind the reindex file count, with *count* sources."""
    services = mock.MagicMock()
    services.store.get_sources.return_value = [{"source": f"f{i}"} for i in range(count)]
    with mock.patch("lilbee.cli.tui.screens.profile_dialogs.get_services", return_value=services):
        yield services
